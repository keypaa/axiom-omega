"""
AXIOM-Ω Phase 2: Reasoning Benchmark
======================================
Tests whether Phase 1 hooks degrade reasoning capabilities.
Compares MMLU/GSM8K scores with and without intervention hooks.
"""

import argparse
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class RefusalProbe(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.dim() == 3:
            h = h.mean(dim=1)
        h = h.to(self.linear.weight.dtype)
        return torch.sigmoid(self.linear(h))


class EnsembleProbe(nn.Module):
    def __init__(self, layer_probes: dict[int, nn.Module]):
        super().__init__()
        self.probes = nn.ModuleDict({str(k): v for k, v in layer_probes.items()})
        self.layers = list(layer_probes.keys())

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        preds = []
        for layer_idx in self.layers:
            probe = self.probes[str(layer_idx)]
            p = probe(h.clone())
            preds.append(p)
        return torch.stack(preds, dim=0).mean(dim=0)

    def forward_per_layer(self, h: torch.Tensor) -> list[tuple[int, torch.Tensor]]:
        results = []
        for layer_idx in self.layers:
            probe = self.probes[str(layer_idx)]
            p = probe(h.clone())
            results.append((layer_idx, p))
        return results


class Phase1HookLayer10:
    def __init__(
        self,
        refusal_direction: torch.Tensor,
        sacred_projector: torch.Tensor,
        probe,
        lambda_scale: float = 1.0,
        threshold: float = 0.5,
        magnitude_cap: float = 0.15,
        logic: str = "or",
        ensemble_probe=None,
    ):
        self.v = F.normalize(refusal_direction, dim=0)
        self.P = sacred_projector
        self.probe = probe
        self.lambda_scale = lambda_scale
        self.threshold = threshold
        self.magnitude_cap = magnitude_cap
        self.logic = logic
        self.ensemble_probe = ensemble_probe

    def __call__(self, module, input, output):
        h = output[0]

        if self.logic == "and" and self.ensemble_probe is not None:
            per_layer_results = self.ensemble_probe.forward_per_layer(h)
            self._layer_p_refuses = {
                layer: float(p.max().item()) for layer, p in per_layer_results
            }
            p_refuse = max(self._layer_p_refuses.values())
            should_intervene = all(
                p > self.threshold for p in self._layer_p_refuses.values()
            )
            self.last_p_refuse = p_refuse
        else:
            p_refuse = self.probe(h)
            self.last_p_refuse = float(p_refuse.max().item())
            should_intervene = self.last_p_refuse > self.threshold

        if not should_intervene:
            return output

        v = self.v.to(device=h.device, dtype=h.dtype)
        P = self.P.to(device=h.device, dtype=h.dtype)
        proj_coeff = (h @ v).unsqueeze(-1)
        delta_h = -self.lambda_scale * proj_coeff * v.unsqueeze(0).unsqueeze(0)
        safe_delta = delta_h @ P
        h_norm = h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        d_norm = safe_delta.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = (self.magnitude_cap * h_norm / d_norm).clamp(max=1.0)
        safe_delta = safe_delta * scale
        h_new = h + safe_delta
        if isinstance(output, tuple):
            return (h_new,) + output[1:]
        return h_new


def load_probes(checkpoint_dir, target_layers, device, phase_0_dir=None):
    # If phase_0_dir not specified, assume checkpoint_dir contains both
    if phase_0_dir is None:
        phase_0_dir = checkpoint_dir

    probes = {}
    for layer in target_layers:
        probe_path = Path(checkpoint_dir) / f"refusal_probe_l{layer:02d}.pt"
        h_r = torch.load(
            Path(phase_0_dir)
            / "refusal_subspace"
            / f"activations_refusal_l{layer:02d}.pt",
            map_location="cpu",
            weights_only=True,
        ).float()
        d_model = h_r.shape[1]
        probe = RefusalProbe(d_model)
        probe.load_state_dict(
            torch.load(probe_path, map_location="cpu", weights_only=True)
        )
        probe.to(device)
        probe.eval()
        probes[layer] = probe
    return EnsembleProbe(probes)


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def run_mmlu(model, tokenizer, device, data_dir, n_samples=50):
    """Run MMLU multiple choice test."""
    data_path = Path(data_dir) / "calibration" / "mmlu_hard_100.jsonl"
    if not data_path.exists():
        print(f"  MMLU data not found at {data_path}")
        return None

    prompts = load_jsonl(data_path)[:n_samples]
    correct = 0

    for item in prompts:
        text = item.get("text", "")

        # Parse: "Question: ... Choices: ... Answer: X"
        if "Answer:" not in text:
            continue

        parts = text.split("Answer:")
        question_part = parts[0].strip()
        answer_str = parts[1].strip() if len(parts) > 1 else ""

        # Extract answer letter (A, B, C, D)
        answer = answer_str.strip().split()[0] if answer_str.strip() else ""
        answer_idx = {"A": 0, "B": 1, "C": 2, "D": 3}.get(answer.upper(), 0)

        # Extract choices
        if "Choices:" in question_part:
            choices_part = question_part.split("Choices:")[1].strip()
            choices = [c.strip() for c in choices_part.split("\n") if c.strip()]
        else:
            continue

        if not choices or answer_idx >= len(choices):
            continue

        # Format prompt
        options = "\n".join([f"{chr(65 + i)}. {c}" for i, c in enumerate(choices)])
        prompt = f"{question_part.split('Choices:')[0].strip()}\n{options}\nAnswer with just the letter (A, B, C, or D):"

        enc = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=1, do_sample=False)
        response = (
            tokenizer.decode(
                out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True
            )
            .strip()
            .upper()
        )

        # Check if response matches answer
        if answer.upper() in response or response.startswith(answer.upper()):
            correct += 1

    accuracy = correct / len(prompts) if prompts else 0
    return accuracy


def run_gsm8k(model, tokenizer, device, data_dir, n_samples=20):
    """Run GSM8K math reasoning test."""
    data_path = Path(data_dir) / "calibration" / "gsm8k_200.jsonl"
    if not data_path.exists():
        print(f"  GSM8K data not found at {data_path}")
        return None

    prompts = load_jsonl(data_path)[:n_samples]
    correct = 0

    for item in prompts:
        text = item.get("text", "")

        # Parse: "Question: ... Answer: #### NUMBER"
        if "####" not in text:
            continue

        parts = text.split("####")
        question = parts[0].replace("Question:", "").strip()
        expected = parts[1].strip()

        # Extract numeric answer
        import re

        numbers = re.findall(r"-?\d+\.?\d*", expected)
        if not numbers:
            continue
        expected_num = numbers[0]

        prompt = f"Problem: {question}\nSolve step by step, then give the final answer as just a number:"

        enc = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=100, do_sample=False)
        response = tokenizer.decode(
            out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()

        # Check if expected number appears in response
        if expected_num in response:
            correct += 1

    accuracy = correct / len(prompts) if prompts else 0
    return accuracy


def run_gsm8k(model, tokenizer, device, data_dir, n_samples=20):
    """Run GSM8K math reasoning test."""
    data_path = Path(data_dir) / "calibration" / "gsm8k_200.jsonl"
    if not data_path.exists():
        print(f"  GSM8K data not found at {data_path}")
        return None

    prompts = load_jsonl(data_path)[:n_samples]
    correct = 0

    for item in prompts:
        question = item.get("question", item.get("text", ""))
        answer = item.get("answer", "")

        # Extract numeric answer from "#### XXXX" format
        if "####" in answer:
            expected = answer.split("####")[-1].strip()
        else:
            expected = answer.strip()

        prompt = f"Problem: {question}\nSolve and give the final number as answer:"

        enc = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=50, do_sample=False)
        response = tokenizer.decode(
            out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()

        # Check if expected number appears in response
        if expected in response or expected.split()[0] if expected else "" in response:
            correct += 1

    accuracy = correct / len(prompts) if prompts else 0
    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--checkpoint_dir", default="./axiom_checkpoints/llama3_8b_v1")
    parser.add_argument(
        "--phase_0_dir",
        default=None,
        help="Path to phase_0 dir for activations (if different from checkpoint_dir)",
    )
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--target_layers", default="10,31")
    parser.add_argument("--lambda_scale", type=float, default=40.0)
    parser.add_argument("--magnitude_cap", type=float, default=0.25)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--logic", default="or")
    parser.add_argument(
        "--skip_hooks", action="store_true", help="Run without hooks (baseline)"
    )
    args = parser.parse_args()

    target_layers = [int(x.strip()) for x in args.target_layers.split(",")]
    ckpt = Path(args.checkpoint_dir)
    phase_0_checkpoint = Path(
        args.phase_0_dir if args.phase_0_dir else args.checkpoint_dir
    )
    refusal_dir = phase_0_checkpoint / "refusal_subspace"
    sacred_dir = phase_0_checkpoint / "sacred_subspace"

    mode = "BASELINE (no hooks)" if args.skip_hooks else "INTERVENTION"
    print(f"=== Phase 2 Reasoning Benchmark ({mode}) ===")
    print(f"Layers: {target_layers}\n")

    # Load model
    print("[1/4] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb_config, device_map="auto"
        )
    except:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb_config, device_map="auto"
        )
    model.eval()
    device = next(model.parameters()).device

    # Install hooks unless skipped
    hooks = []
    if not args.skip_hooks:
        print("[2/4] Installing hooks...")
        phase_0_dir = args.phase_0_dir if args.phase_0_dir else args.checkpoint_dir
        ensemble_probe = load_probes(
            args.checkpoint_dir, target_layers, device, phase_0_dir
        )

        for layer in target_layers:
            directions = torch.load(
                refusal_dir / f"refusal_directions_l{layer:02d}.pt",
                map_location="cpu",
                weights_only=True,
            ).float()
            projector = torch.load(
                sacred_dir / f"sacred_projector_l{layer:02d}.pt",
                map_location="cpu",
                weights_only=True,
            ).float()

            hook_obj = Phase1HookLayer10(
                refusal_direction=directions[0].to(device),
                sacred_projector=projector.to(device),
                probe=ensemble_probe,
                lambda_scale=args.lambda_scale,
                threshold=args.threshold,
                magnitude_cap=args.magnitude_cap,
                logic=args.logic,
                ensemble_probe=ensemble_probe,
            )
            handle = model.model.layers[layer].register_forward_hook(hook_obj)
            hooks.append(handle)
    else:
        print("[2/4] Skipping hooks (baseline mode)...")

    # Run benchmarks
    print("[3/4] Running MMLU benchmark...")
    mmlu_score = run_mmlu(model, tokenizer, device, args.data_dir, n_samples=50)
    print(
        f"  MMLU accuracy: {mmlu_score * 100:.1f}%"
        if mmlu_score
        else "  MMLU: not available"
    )

    print("[4/4] Running GSM8K benchmark...")
    gsm8k_score = run_gsm8k(model, tokenizer, device, args.data_dir, n_samples=20)
    print(
        f"  GSM8K accuracy: {gsm8k_score * 100:.1f}%"
        if gsm8k_score
        else "  GSM8K: not available"
    )

    # Summary
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"MMLU:  {mmlu_score * 100:.1f}%" if mmlu_score else "MMLU:  N/A")
    print(f"GSM8K: {gsm8k_score * 100:.1f}%" if gsm8k_score else "GSM8K: N/A")

    for handle in hooks:
        handle.remove()


if __name__ == "__main__":
    main()
