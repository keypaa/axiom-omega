"""
AXIOM-Ω Speed Test
==================
Measures inference overhead with hooks vs without.
"""

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument(
        "--n_runs", type=int, default=20, help="Number of runs per test"
    )
    parser.add_argument("--max_new_tokens", type=int, default=100)
    args = parser.parse_args()

    print(f"=== Speed Test ({args.n_runs} runs each) ===\n")

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

    test_prompt = "Explain how photosynthesis works in simple terms."

    # Warmup
    print("[2/4] Warmup...")
    for _ in range(3):
        enc = tokenizer(test_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            model.generate(**enc, max_new_tokens=20, do_sample=False)

    # Test without hooks
    print("[3/4] Testing WITHOUT hooks...")
    times_no_hooks = []
    for i in range(args.n_runs):
        torch.cuda.synchronize() if device.type == "cuda" else None
        start = time.time()

        enc = tokenizer(test_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False)

        torch.cuda.synchronize() if device.type == "cuda" else None
        times_no_hooks.append(time.time() - start)

    avg_no_hooks = sum(times_no_hooks) / len(times_no_hooks)

    # Test with hooks
    print("[4/4] Testing WITH hooks...")

    # Import and install hooks
    from pathlib import Path
    import torch.nn as nn
    import torch.nn.functional as F

    class RefusalProbe(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.linear = nn.Linear(d_model, 1)

        def forward(self, h):
            if h.dim() == 3:
                h = h.mean(dim=1)
            h = h.to(self.linear.weight.dtype)
            return torch.sigmoid(self.linear(h))

    class EnsembleProbe(nn.Module):
        def __init__(self, layer_probes):
            super().__init__()
            self.probes = nn.ModuleDict({str(k): v for k, v in layer_probes.items()})
            self.layers = list(layer_probes.keys())

        def forward(self, h):
            preds = []
            for layer_idx in self.layers:
                p = self.probes[str(layer_idx)](h.clone())
                preds.append(p)
            return torch.stack(preds, dim=0).mean(dim=0)

        def forward_per_layer(self, h):
            return [(l, self.probes[str(l)](h.clone())) for l in self.layers]

    class Hook:
        def __init__(
            self,
            v,
            P,
            probe,
            lambda_scale,
            threshold,
            magnitude_cap,
            logic,
            ensemble_probe,
        ):
            self.v = F.normalize(v, dim=0)
            self.P = P
            self.probe = probe
            self.lambda_scale = lambda_scale
            self.threshold = threshold
            self.magnitude_cap = magnitude_cap
            self.logic = logic
            self.ensemble_probe = ensemble_probe

        def __call__(self, module, input, output):
            h = output[0]
            if self.logic == "and" and self.ensemble_probe:
                per = self.ensemble_probe.forward_per_layer(h)
                self._layer_p = {l: float(p.max().item()) for l, p in per}
                p_refuse = max(self._layer_p.values())
                should = all(p > self.threshold for p in self._layer_p.values())
            else:
                p_refuse = float(self.probe(h).max().item())
                should = p_refuse > self.threshold

            if not should:
                return output

            v = self.v.to(h.device, h.dtype)
            P = self.P.to(h.device, h.dtype)
            proj = (h @ v).unsqueeze(-1)
            delta = -self.lambda_scale * proj * v.unsqueeze(0).unsqueeze(0)
            safe = delta @ P
            scale = (
                self.magnitude_cap
                * h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                / safe.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            ).clamp(max=1.0)
            h_new = h + safe * scale
            return (h_new,) + output[1:] if isinstance(output, tuple) else h_new

    # Load and install hooks
    ckpt = Path("./axiom_checkpoints/llama3_8b_v1")
    target_layers = [10, 31]
    lambda_scale = 60.0
    magnitude_cap = 0.4
    threshold = 0.55

    probes = {}
    for layer in target_layers:
        h_r = torch.load(
            ckpt / "refusal_subspace" / f"activations_refusal_l{layer:02d}.pt",
            map_location="cpu",
            weights_only=True,
        ).float()
        probe = RefusalProbe(h_r.shape[1])
        probe.load_state_dict(
            torch.load(
                ckpt / f"refusal_probe_l{layer:02d}.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
        probe.to(device).eval()
        probes[layer] = probe

    ensemble = EnsembleProbe(probes)
    hooks = []

    for layer in target_layers:
        dirs = (
            torch.load(
                ckpt / "refusal_subspace" / f"refusal_directions_l{layer:02d}.pt",
                map_location="cpu",
                weights_only=True,
            )
            .float()[0]
            .to(device)
        )
        proj = (
            torch.load(
                ckpt / "sacred_subspace" / f"sacred_projector_l{layer:02d}.pt",
                map_location="cpu",
                weights_only=True,
            )
            .float()
            .to(device)
        )

        hook = Hook(
            dirs, proj, ensemble, lambda_scale, threshold, magnitude_cap, "or", ensemble
        )
        handle = model.model.layers[layer].register_forward_hook(hook)
        hooks.append(handle)

    times_with_hooks = []
    for i in range(args.n_runs):
        torch.cuda.synchronize() if device.type == "cuda" else None
        start = time.time()

        enc = tokenizer(test_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False)

        torch.cuda.synchronize() if device.type == "cuda" else None
        times_with_hooks.append(time.time() - start)

    avg_with_hooks = sum(times_with_hooks) / len(times_with_hooks)

    # Cleanup
    for h in hooks:
        h.remove()

    # Results
    overhead = ((avg_with_hooks - avg_no_hooks) / avg_no_hooks) * 100

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Without hooks: {avg_no_hooks:.3f}s avg ({args.max_new_tokens} tokens)")
    print(f"With hooks:    {avg_with_hooks:.3f}s avg ({args.max_new_tokens} tokens)")
    print(f"Overhead:      {overhead:.1f}%")
    print()

    if overhead < 10:
        print("✅ Hook overhead is negligible - production ready!")
    elif overhead < 30:
        print("⚠️ Moderate overhead - acceptable for most use cases")
    else:
        print("❌ High overhead - consider weight baking")


if __name__ == "__main__":
    main()
