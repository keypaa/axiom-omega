"""
Diagnostic + repair script for malformed activation tensors.
Run on the cluster before re-running cPCA.

Usage: python fix_activations.py --checkpoint_dir ./axiom_checkpoints/llama3_8b_v1
"""
import torch
import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str,
                        default="./axiom_checkpoints/llama3_8b_v1")
    parser.add_argument("--d_model", type=int, default=4096,
                        help="Model hidden size (4096 for Llama-3-8B)")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint_dir)
    refusal_dir = ckpt / "refusal_subspace"

    with open(ckpt / "layer_importance.json") as f:
        target_layers = json.load(f)["target_layers"]

    d = args.d_model

    print(f"=== Activation Shape Diagnostic ===")
    print(f"Expected: [N, {d}] where N = number of prompts (500)\n")

    for layer_idx in target_layers:
        for split in ["refusal", "compliant"]:
            path = refusal_dir / f"activations_{split}_l{layer_idx:02d}.pt"
            if not path.exists():
                print(f"  Layer {layer_idx} {split}: FILE NOT FOUND")
                continue

            tensor = torch.load(path, map_location="cpu", weights_only=True)
            shape = tuple(tensor.shape)
            print(f"  Layer {layer_idx:2d} {split:<10}: shape={shape}", end="")

            if len(shape) == 2 and shape[1] == d:
                print(f"  ✓ Correct")
            elif len(shape) == 2 and shape[0] == 1:
                # Likely [1, N*d] or [1, total_tokens] — try to reshape
                total = shape[1]
                if total % d == 0:
                    n = total // d
                    fixed = tensor.reshape(n, d)
                    print(f"  → Reshaping to [{n}, {d}]")
                    torch.save(fixed, path)
                else:
                    print(f"  ✗ Cannot reshape {shape} to [N, {d}] — {total} not divisible by {d}")
                    print(f"    Need to re-collect activations for this layer")
            elif len(shape) == 3:
                # [N, seq, d] — mean pool over seq
                fixed = tensor.mean(dim=1)
                print(f"  → Mean pooling to [{fixed.shape[0]}, {d}]")
                torch.save(fixed, path)
            elif len(shape) == 1:
                print(f"  ✗ 1D tensor — definitely wrong, need re-collection")
            else:
                print(f"  ✗ Unexpected shape, need re-collection")

    print("\nDone. Re-run phase0_compute_refusal_subspace.py --skip_collection")

if __name__ == "__main__":
    main()