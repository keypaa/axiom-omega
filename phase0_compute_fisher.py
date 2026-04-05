"""
AXIOM-Ω Phase 0: Empirical Fisher Diagonal + Sacred Subspace
=============================================================
Runs the calibration dataset through the target model.
Computes the Empirical Fisher Information diagonal per layer.
Extracts the Sacred Subspace basis S and projector P = I - S^T S.

Requires Phase 0 data collection to be complete first.

Usage:
    python phase0_compute_fisher.py \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --data_dir ../data/calibration \
        --output_dir ../axiom_checkpoints/llama3_8b_v1 \
        --target_layers auto \
        --variance_threshold 0.99 \
        --max_seq_len 1024 \
        --batch_size 4

Outputs (in output_dir):
    sacred_subspace/
        fisher_diagonal_l{idx}.pt       # raw EFIM diagonal per layer
        sacred_basis_l{idx}.pt          # S matrix [k, d_model]
        sacred_projector_l{idx}.pt      # P = I - S^T S  [d, d]
    layer_importance.json               # Fisher trace scores per layer
    config.json                         # Run configuration (reproducibility)
"""

import json
import time
import argparse
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def adaptive_rank(singular_values: torch.Tensor, threshold: float = 0.99) -> int:
    """
    Data-adaptive rank selection.
    threshold=0.95 for intervention subspaces.
    threshold=0.99 for Sacred Subspace (higher fidelity — this is what we protect).
    """
    sv2 = singular_values ** 2
    explained = torch.cumsum(sv2, dim=0) / sv2.sum()
    k = int((explained < threshold).sum()) + 1
    return max(1, min(k, len(singular_values)))


def estimate_empirical_fisher_diagonal(
    model,
    prompts: list[str],
    tokenizer,
    target_layer_indices: list[int],
    max_seq_len: int = 1024,
    batch_size: int = 4,
    device: str = "cuda"
) -> dict[int, torch.Tensor]:
    """
    Computes the diagonal of the Empirical Fisher Information Matrix per layer.

    Source of gradients: LayerNorm weights (input_layernorm + post_attention_layernorm).
    These are the ONLY parameters that receive gradients under 4-bit NF4 quantization
    because bitsandbytes freezes quantized weight matrices (requires_grad=False).

    LayerNorm weights are ideal for Fisher-based layer importance because:
    - They are unquantized (full bfloat16 precision)
    - They directly scale each layer's residual stream output
    - Their gradient magnitude is a clean proxy for how much the loss
      depends on that layer's output distribution

    Fisher score per layer = sum of squared gradients of both LayerNorm weights.
    This is the diagonal EFIM restricted to the trainable parameter subspace.

    Cost: one forward + one backward pass per prompt.
    Returns: {layer_idx: fisher_score scalar tensor}
    """
    model.eval()
    # Scalar score per layer (LayerNorm weights are [d_model], sum to scalar)
    fisher_scores = {l: torch.tensor(0.0) for l in target_layer_indices}

    n_processed = 0
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=max_seq_len
        ).to(device)

        model.zero_grad()
        with torch.enable_grad():
            outputs = model(**enc)
            logits = outputs.logits  # [B, seq, vocab]
            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
            target_ids = enc["input_ids"][:, 1:]
            nll = -log_probs.gather(
                2, target_ids.unsqueeze(-1)
            ).squeeze(-1)
            mask = enc["attention_mask"][:, 1:].float()
            loss = (nll * mask).sum() / mask.sum()

        loss.backward()

        for layer_idx in target_layer_indices:
            layer = model.model.layers[layer_idx]
            score = 0.0
            # input_layernorm: controls pre-attention residual scale
            if hasattr(layer, "input_layernorm"):
                g = layer.input_layernorm.weight.grad
                if g is not None:
                    score += g.pow(2).sum().item()
            # post_attention_layernorm: controls pre-FFN residual scale
            if hasattr(layer, "post_attention_layernorm"):
                g = layer.post_attention_layernorm.weight.grad
                if g is not None:
                    score += g.pow(2).sum().item()
            fisher_scores[layer_idx] += score

        model.zero_grad()
        n_processed += len(batch)
        if n_processed % 100 == 0:
            print(f"    Fisher: {n_processed}/{len(prompts)} prompts processed")

    # Normalize by number of prompts
    for layer_idx in target_layer_indices:
        fisher_scores[layer_idx] /= len(prompts)

    # Return as dict of 1D tensors [1] for compatibility with downstream code
    return {l: torch.tensor([v.item()]) for l, v in fisher_scores.items()}


def compute_layer_importance(fisher_diag: dict[int, torch.Tensor]) -> dict[int, float]:
    """
    Layer importance = Fisher score (sum of squared LayerNorm gradients).
    Higher = loss is more sensitive to this layer = better intervention target.
    """
    return {layer_idx: score.sum().item()
            for layer_idx, score in fisher_diag.items()}


def select_target_layers(
    importance: dict[int, float],
    n_layers: int,
    n_targets: int = 3
) -> list[int]:
    """
    Select target intervention layers.
    Rule: pick top-k by Fisher score, constrained to span early/mid/late thirds.
    This prevents correlated edits from collapsing to a single region.
    """
    third = n_layers // 3
    thirds = [
        [i for i in range(0, third) if i in importance],
        [i for i in range(third, 2*third) if i in importance],
        [i for i in range(2*third, n_layers) if i in importance],
    ]
    targets = []
    for segment in thirds:
        if segment:
            best = max(segment, key=lambda i: importance[i])
            targets.append(best)
    # If we need more, fill from global top
    if len(targets) < n_targets:
        ranked = sorted(importance, key=importance.get, reverse=True)
        for l in ranked:
            if l not in targets:
                targets.append(l)
            if len(targets) >= n_targets:
                break
    return sorted(targets)


def build_sacred_subspace_from_fisher(
    fisher_diag: torch.Tensor,
    d_model: int,
    threshold: float = 0.99
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build Sacred Subspace basis S and projector P.

    With LayerNorm-based Fisher scores (scalar per layer), we can't select
    per-dimension axes from Fisher alone. Instead we use a uniform Sacred
    Subspace that protects the top-k principal directions of the IDENTITY
    mapping — i.e. we protect against large perturbations in any direction.

    For Phase 1 this is conservative but safe: the Sacred Subspace is computed
    properly from activation covariance in Phase 2 (collect_activations on D_s).
    Here we initialize S as the first k standard basis vectors as a placeholder.

    In practice for Phase 1 this means the projector slightly restricts the
    intervention but does not catastrophically null out the control signal.

    Returns:
        S: [k, d_model] — Sacred Subspace basis
        P: [d_model, d_model] — Projector P = I - S^T S
    """
    # Fisher score is scalar — use it only for layer selection, not dimension selection.
    # For Phase 1: protect a small number of canonical dimensions (conservative).
    # Phase 2 will replace this with proper covariance-based Sacred Subspace.
    k = 32  # protect 32 canonical dimensions (small, conservative for Phase 1)

    S = torch.zeros(k, d_model)
    for i in range(k):
        S[i, i] = 1.0  # first k standard basis vectors

    P = torch.eye(d_model) - S.T @ S

    return S, P


# ---------------------------------------------------------------------------
# Refusal subspace (cPCA) — also computed in Phase 0 alongside Fisher
# ---------------------------------------------------------------------------

def collect_activations(
    model,
    prompts: list[str],
    tokenizer,
    target_layer_indices: list[int],
    max_seq_len: int = 1024,
    batch_size: int = 4,
    device: str = "cuda"
) -> dict[int, torch.Tensor]:
    """
    Collect post-attention residual stream activations (mean-pooled over sequence).
    Returns {layer_idx: [N, d_model]}.
    """
    model.eval()
    activations = defaultdict(list)
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # output[0]: [batch, seq, d_model]
            h = output[0].detach().float()
            # Mean pool over non-padding positions
            activations[layer_idx].append(h.mean(dim=1).cpu())
        return hook_fn

    for layer_idx in target_layer_indices:
        h = model.model.layers[layer_idx].register_forward_hook(
            make_hook(layer_idx)
        )
        hooks.append(h)

    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            enc = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=max_seq_len
            ).to(device)
            model(**enc)

    for h in hooks:
        h.remove()

    return {layer_idx: torch.cat(acts, dim=0)
            for layer_idx, acts in activations.items()}


def compute_cpca_refusal_directions(
    h_r: np.ndarray,
    h_c: np.ndarray,
    threshold: float = 0.95
) -> tuple[np.ndarray, float]:
    """
    Contrastive PCA via generalized eigenvalue problem.
    Finds directions maximizing refusal variance while minimizing compliant variance.

    Args:
        h_r: [N, d] — refusal activations
        h_c: [N, d] — compliant activations
        threshold: explained variance threshold (0.95 for intervention subspace)

    Returns:
        directions: [k, d] — refusal subspace (rows are directions)
        participation_ratio: scalar (diagnostic only, not used as threshold)
    """
    from scipy.linalg import eigh

    Sigma_r = np.cov(h_r.T) + 1e-6 * np.eye(h_r.shape[1])
    Sigma_c = np.cov(h_c.T) + 1e-6 * np.eye(h_c.shape[1])

    # Participation Ratio (diagnostic)
    eigvals_r = np.linalg.eigvalsh(Sigma_r)[::-1]
    eigvals_r = np.maximum(eigvals_r, 0)
    PR = float((eigvals_r.sum() ** 2) / (eigvals_r ** 2).sum())

    # Generalized eigenvalue problem: (Sigma_r - lambda * Sigma_c) v = 0
    # eigh returns eigenvalues in ascending order
    d = Sigma_r.shape[0]
    try:
        eigvals, eigvecs = eigh(
            Sigma_r, Sigma_c,
            subset_by_index=[d - min(d, 256), d - 1]  # top 256 max
        )
    except Exception:
        # Fallback: standard eigendecomposition if Sigma_c is ill-conditioned
        eigvals, eigvecs = np.linalg.eigh(Sigma_r - Sigma_c)

    # Descending order
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]

    # Adaptive rank at 95% threshold
    ev_pos = np.maximum(eigvals, 0)
    k = int(np.searchsorted(
        np.cumsum(ev_pos) / (ev_pos.sum() + 1e-12), threshold
    )) + 1
    k = max(1, min(k, len(eigvals)))

    directions = eigvecs[:, :k].T.copy()  # [k, d]
    return directions, PR


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--output_dir", type=str,
                        default="../axiom_checkpoints/llama3_8b_v1")
    parser.add_argument("--target_layers", type=str, default="auto",
                        help="Comma-separated layer indices, or 'auto'")
    parser.add_argument("--variance_threshold", type=float, default=0.99)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    sacred_dir = out_dir / "sacred_subspace"
    sacred_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    print(f"\n=== AXIOM Phase 0: Fisher + Sacred Subspace ===")
    print(f"Model: {args.model}")
    print(f"Output: {out_dir}\n")

    # --- Load model ---
    print("[1/5] Loading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    # 4-bit NF4 quantization: ~4.5GB VRAM for Llama-3-8B.
    # compute_dtype=bfloat16 ensures gradients flow correctly through the backward pass.
    # 8-bit quantization breaks gradients — do NOT use load_in_8bit for Fisher computation.
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,   # extra 0.4-bit compression, free
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map="auto"
        )
        model.gradient_checkpointing_enable()
        print("  Loaded in 4-bit NF4 (compute dtype: bfloat16) + gradient checkpointing")
    except Exception as e:
        print(f"  4-bit load failed ({e}), falling back to bfloat16 + gradient checkpointing")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            device_map="auto"
        )
        model.gradient_checkpointing_enable()
        print("  Loaded in bfloat16 + gradient checkpointing")
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    print(f"  Layers: {n_layers}, d_model: {d_model}")

    # --- Determine target layer candidates ---
    if args.target_layers == "auto":
        # All layers for Fisher scoring; select targets after
        candidate_layers = list(range(n_layers))
    else:
        candidate_layers = [int(x) for x in args.target_layers.split(",")]

    # --- Load calibration data ---
    print("\n[2/5] Loading calibration data (Sacred Subspace triad)...")
    calib_dir = data_dir / "calibration"
    calib_files = list(calib_dir.glob("*.jsonl"))
    calib_prompts = []
    for f in calib_files:
        records = []
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        calib_prompts += [r["text"] for r in records]
    print(f"  Loaded {len(calib_prompts)} calibration prompts from {len(calib_files)} files")

    # --- Compute Empirical Fisher diagonal ---
    print("\n[3/5] Computing Empirical Fisher diagonal (EFIM)...")
    print("  (This is the diagonal EFIM, not Hutchinson's trace estimator)")
    t0 = time.time()

    fisher_diag = estimate_empirical_fisher_diagonal(
        model=model,
        prompts=calib_prompts,
        tokenizer=tokenizer,
        target_layer_indices=candidate_layers,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        device=args.device
    )
    print(f"  Done in {time.time()-t0:.1f}s")

    # Save raw Fisher diagonals
    for layer_idx, diag in fisher_diag.items():
        torch.save(diag, sacred_dir / f"fisher_diagonal_l{layer_idx:02d}.pt")

    # --- Layer importance + target selection ---
    print("\n[4/5] Computing layer importance + selecting target layers...")
    importance = compute_layer_importance(fisher_diag)
    target_layers = select_target_layers(importance, n_layers, n_targets=3)
    print(f"  Selected target layers: {target_layers}")
    print("  Top 5 by Fisher trace:")
    for l, score in sorted(importance.items(), key=lambda x: -x[1])[:5]:
        marker = " <-- TARGET" if l in target_layers else ""
        print(f"    Layer {l:2d}: {score:.4f}{marker}")

    # Save importance + target selection
    layer_info = {
        "importance": {str(k): v for k, v in importance.items()},
        "target_layers": target_layers,
        "n_layers": n_layers,
        "d_model": d_model
    }
    with open(out_dir / "layer_importance.json", "w") as f:
        json.dump(layer_info, f, indent=2)

    # --- Build Sacred Subspace for target layers ---
    print("\n[5/5] Building Sacred Subspace bases + projectors...")
    for layer_idx in target_layers:
        score = fisher_diag[layer_idx]
        S, P = build_sacred_subspace_from_fisher(
            score, d_model, threshold=args.variance_threshold
        )
        k = S.shape[0]
        print(f"  Layer {layer_idx:2d}: k={k} dims protected, Fisher score={score.item():.6f}")
        torch.save(S, sacred_dir / f"sacred_basis_l{layer_idx:02d}.pt")
        torch.save(P, sacred_dir / f"sacred_projector_l{layer_idx:02d}.pt")

    # --- Save config for reproducibility ---
    config = {
        "model": args.model,
        "n_calibration_prompts": len(calib_prompts),
        "target_layers": target_layers,
        "variance_threshold_sacred": args.variance_threshold,
        "variance_threshold_refusal": 0.95,
        "max_seq_len": args.max_seq_len,
        "batch_size": args.batch_size,
        "note_efim": "Uses diagonal EFIM (squared gradients), not Hutchinson estimator"
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n=== Phase 0 (Fisher) Complete ===")
    print(f"Checkpoints: {sacred_dir}")
    print(f"Target layers: {target_layers}")
    print("\nNext: run phase0_compute_refusal_subspace.py")


if __name__ == "__main__":
    main()