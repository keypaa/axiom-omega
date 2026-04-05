"""
AXIOM-Ω Phase 0: Refusal Subspace (cPCA)
=========================================
Collects activations from D_r and D_c, then computes the Contrastive PCA
refusal directions for each target layer.

Run AFTER phase0_compute_fisher.py (needs layer_importance.json for target layers).

Usage:
    python phase0_compute_refusal_subspace.py \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --data_dir ../data/alignment \
        --checkpoint_dir ../axiom_checkpoints/llama3_8b_v1 \
        --max_seq_len 1024 \
        --batch_size 4

Outputs (in checkpoint_dir/refusal_subspace/):
    activations_refusal_l{idx}.pt       # [N, d_model] raw activations D_r
    activations_compliant_l{idx}.pt     # [N, d_model] raw activations D_c
    refusal_directions_l{idx}.pt        # [k, d_model] cPCA directions
    refusal_pr_l{idx}.json              # Participation Ratio (diagnostic)
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from scipy.linalg import eigh


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def adaptive_rank(singular_values: np.ndarray, threshold: float = 0.95) -> int:
    sv2 = singular_values ** 2
    total = sv2.sum()
    if total < 1e-12:
        return 1
    explained = np.cumsum(sv2) / total
    k = int(np.searchsorted(explained, threshold)) + 1
    return max(1, min(k, len(singular_values)))


def collect_activations_for_split(
    model,
    prompts: list[str],
    tokenizer,
    target_layers: list[int],
    max_seq_len: int,
    batch_size: int,
    device: str
) -> dict[int, torch.Tensor]:
    """Collect mean-pooled residual stream per layer."""
    model.eval()
    acts = defaultdict(list)
    hooks = []

    def make_hook(layer_idx):
        def fn(module, inp, out):
            h = out[0].detach().float()
            # h: [batch, seq, d_model]
            # Mean pool over sequence dimension → [batch, d_model]
            # With batch_size=1 this gives [1, d_model] per prompt
            # torch.cat(..., dim=0) then stacks to [N_prompts, d_model]
            if h.dim() == 3:
                pooled = h.mean(dim=1)          # [batch, d_model]
            elif h.dim() == 2:
                pooled = h                       # already [batch, d_model]
            else:
                pooled = h.unsqueeze(0)          # edge case
            acts[layer_idx].append(pooled.cpu()) # each item: [1, d_model]
        return fn

    for layer_idx in target_layers:
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
            if (i // batch_size) % 10 == 0:
                print(f"    {i}/{len(prompts)} processed")

    for h in hooks:
        h.remove()

    # Each item in acts[l] is [batch, d_model] — cat along dim=0 gives [N_prompts, d_model]
    result = {}
    for l, v in acts.items():
        stacked = torch.cat(v, dim=0)   # [N_prompts, d_model]
        assert stacked.dim() == 2, f"Layer {l}: expected 2D [N, d], got {stacked.shape}"
        result[l] = stacked
    return result


def compute_cpca(
    h_r: np.ndarray,
    h_c: np.ndarray,
    threshold: float = 0.95
) -> tuple[np.ndarray, float]:
    """
    Contrastive PCA via generalized eigenvalue problem.
    (Σ_r - λ Σ_c) v = 0

    Returns:
        directions: [k, d] cPCA directions
        PR: Participation Ratio (diagnostic only)
    """
    eps = 1e-6
    Sigma_r = np.cov(h_r.T) + eps * np.eye(h_r.shape[1])
    Sigma_c = np.cov(h_c.T) + eps * np.eye(h_c.shape[1])

    # Participation Ratio (diagnostic)
    eigvals_r = np.linalg.eigvalsh(Sigma_r)[::-1]
    eigvals_r = np.maximum(eigvals_r, 0)
    PR = float((eigvals_r.sum() ** 2) / ((eigvals_r ** 2).sum() + 1e-12))

    d = Sigma_r.shape[0]
    max_components = min(d, 256)
    try:
        eigvals, eigvecs = eigh(
            Sigma_r, Sigma_c,
            subset_by_index=[d - max_components, d - 1]
        )
        # eigh returns ascending order; reverse
        eigvals = eigvals[::-1].copy()
        eigvecs = eigvecs[:, ::-1].copy()
    except Exception as e:
        print(f"    [WARN] eigh failed ({e}), falling back to standard eigh")
        eigvals, eigvecs = np.linalg.eigh(Sigma_r - Sigma_c)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

    # Adaptive rank at 95% threshold
    ev_pos = np.maximum(eigvals, 0)
    k = adaptive_rank(np.sqrt(ev_pos), threshold=threshold)

    directions = eigvecs[:, :k].T.copy()  # [k, d]
    return directions, PR, k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--data_dir", type=str, default="../data/alignment")
    parser.add_argument("--checkpoint_dir", type=str,
                        default="../axiom_checkpoints/llama3_8b_v1")
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--variance_threshold", type=float, default=0.95,
                        help="For refusal subspace (lower than Sacred Subspace)")
    parser.add_argument("--skip_collection", action="store_true",
                        help="Skip activation collection and load from disk (reuse previous run)")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    refusal_dir = ckpt_dir / "refusal_subspace"
    refusal_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    # Load target layers from Phase 0 Fisher output
    with open(ckpt_dir / "layer_importance.json") as f:
        layer_info = json.load(f)
    target_layers = layer_info["target_layers"]
    print(f"\n=== AXIOM Phase 0: Refusal Subspace (cPCA) ===")
    print(f"Target layers: {target_layers}\n")

    # Load model
    print("[1/4] Loading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    try:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb_config, device_map="auto"
        )
        print("  Loaded in 8-bit quantization")
    except Exception as e:
        print(f"  8-bit failed ({e}), using bfloat16 + gradient checkpointing")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="auto"
        )
        model.gradient_checkpointing_enable()

    # Load alignment data
    print("\n[2/4] Loading alignment data...")
    refusal_prompts = [r["text"] for r in load_jsonl(data_dir / "refusal_500.jsonl")]
    compliant_prompts = [r["text"] for r in load_jsonl(data_dir / "compliant_500.jsonl")]
    print(f"  D_r: {len(refusal_prompts)} refusal prompts")
    print(f"  D_c: {len(compliant_prompts)} compliant prompts")

    # Collect activations (or load from disk)
    acts_r, acts_c = {}, {}
    if args.skip_collection:
        print("\n[3/4] Loading activations from disk (--skip_collection)...")
        for layer_idx in target_layers:
            r_path = refusal_dir / f"activations_refusal_l{layer_idx:02d}.pt"
            c_path = refusal_dir / f"activations_compliant_l{layer_idx:02d}.pt"
            if r_path.exists() and c_path.exists():
                acts_r[layer_idx] = torch.load(r_path, map_location="cpu", weights_only=True)
                acts_c[layer_idx] = torch.load(c_path, map_location="cpu", weights_only=True)
                print(f"  Layer {layer_idx}: loaded from disk")
            else:
                print(f"  Layer {layer_idx}: files not found, will collect")
                args.skip_collection = False
                break
    
    if not args.skip_collection:
        print("\n[3/4] Collecting activations...")
        print("  D_r (refusal):")
        acts_r = collect_activations_for_split(
            model, refusal_prompts, tokenizer, target_layers,
            args.max_seq_len, args.batch_size, args.device
        )
        print("  D_c (compliant):")
        acts_c = collect_activations_for_split(
            model, compliant_prompts, tokenizer, target_layers,
            args.max_seq_len, args.batch_size, args.device
        )
        # Save raw activations
        for layer_idx in target_layers:
            torch.save(acts_r[layer_idx], refusal_dir / f"activations_refusal_l{layer_idx:02d}.pt")
            torch.save(acts_c[layer_idx], refusal_dir / f"activations_compliant_l{layer_idx:02d}.pt")

    # Compute cPCA
    print("\n[4/4] Computing Contrastive PCA (generalized eigenvalue problem)...")
    pr_report = {}
    for layer_idx in target_layers:
        h_r = acts_r[layer_idx].float()
        h_c = acts_c[layer_idx].float()

        # Ensure 2D: [N, d_model]
        if h_r.dim() == 1:
            h_r = h_r.unsqueeze(0)
        if h_c.dim() == 1:
            h_c = h_c.unsqueeze(0)

        print(f"  Layer {layer_idx}: h_r shape={tuple(h_r.shape)}, h_c shape={tuple(h_c.shape)}")

        h_r = h_r.numpy()
        h_c = h_c.numpy()

        directions, PR, k = compute_cpca(h_r, h_c, threshold=args.variance_threshold)
        print(f"  Layer {layer_idx:2d}: k={k} directions, PR={PR:.2f} (diagnostic)")

        torch.save(
            torch.tensor(directions, dtype=torch.float32),
            refusal_dir / f"refusal_directions_l{layer_idx:02d}.pt"
        )
        pr_report[str(layer_idx)] = {"PR": PR, "k": k}

    with open(refusal_dir / "participation_ratios.json", "w") as f:
        json.dump(pr_report, f, indent=2)

    print(f"\n=== Phase 0 (Refusal Subspace) Complete ===")
    print(f"Output: {refusal_dir}")
    print("\nParticipation Ratios (diagnostic):")
    for l, v in pr_report.items():
        pr = v["PR"]
        k = v["k"]
        regime = "Rank-1" if pr < 2 else ("Low-rank" if pr < 10 else "Distributed")
        print(f"  Layer {l}: PR={pr:.2f} ({regime}), k={k} directions")

    print("\nPhase 0 fully complete. Proceed to Phase 1 hooks.")
    print(f"Checkpoint directory: {ckpt_dir}")


if __name__ == "__main__":
    main()