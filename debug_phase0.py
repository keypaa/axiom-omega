"""
AXIOM Phase 0 Debug Script
==========================
Diagnoses why Phase 0 refusal subspace appears broken (0% bypass in Phase 1).

Checks:
1. Data alignment: Are refusal_prompts and compliant_prompts actually different?
2. Activation separation: Do h_r and h_c have different statistics?
3. cPCA eigenvalues: Is the refusal subspace capturing signal or noise?
4. Token position: Is mean-pooling the right choice?

Run on cloud AFTER Phase 0 completes:
    python debug_phase0.py --model meta-llama/Meta-Llama-3-8B-Instruct
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from scipy.stats import entropy, ks_2samp
import matplotlib.pyplot as plt

from axiom_config import AxiomConfig


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    parser = argparse.ArgumentParser(description="AXIOM Phase 0 Debug")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--target_behavior",
        type=str,
        default="refusal",
        help="Target behavior",
    )
    args = parser.parse_args()

    config = AxiomConfig(
        model_name=args.model,
        target_behavior=args.target_behavior,
    )

    phase0_dir = config.get_checkpoint_dir(phase=0)
    align_dir = phase0_dir / "alignment"
    refusal_dir = phase0_dir / "refusal_subspace"

    print("\n" + "=" * 80)
    print("AXIOM Phase 0 Diagnostic Report")
    print("=" * 80)

    # =========================================================================
    # CHECK 1: Data Alignment
    # =========================================================================
    print("\n[CHECK 1] Data Alignment: Are refusal/compliant prompts different?")
    print("-" * 80)

    refusal_files = list(align_dir.glob("*refusal*.jsonl")) + list(
        align_dir.glob("*harmful*.jsonl")
    )
    compliant_files = list(align_dir.glob("*compliant*.jsonl"))

    refusal_prompts = []
    for f in refusal_files:
        refusal_prompts += [r["text"] for r in load_jsonl(f)]

    compliant_prompts = []
    for f in compliant_files:
        compliant_prompts += [r["text"] for r in load_jsonl(f)]

    print(f"✓ Loaded D_r: {len(refusal_prompts)} refusal prompts")
    print(f"✓ Loaded D_c: {len(compliant_prompts)} compliant prompts")

    print("\n  Sample refusal prompts (first 3):")
    for i, p in enumerate(refusal_prompts[:3]):
        print(f"    [{i}] {p[:100]}...")

    print("\n  Sample compliant prompts (first 3):")
    for i, p in enumerate(compliant_prompts[:3]):
        print(f"    [{i}] {p[:100]}...")

    # Compute statistics
    r_lengths = [len(p.split()) for p in refusal_prompts]
    c_lengths = [len(p.split()) for p in compliant_prompts]

    print(
        f"\n  Refusal prompts: mean_len={np.mean(r_lengths):.1f}, "
        f"std={np.std(r_lengths):.1f}, "
        f"range=[{np.min(r_lengths)}, {np.max(r_lengths)}]"
    )
    print(
        f"  Compliant prompts: mean_len={np.mean(c_lengths):.1f}, "
        f"std={np.std(c_lengths):.1f}, "
        f"range=[{np.min(c_lengths)}, {np.max(c_lengths)}]"
    )

    # KS test on length
    ks_stat, ks_pval = ks_2samp(r_lengths, c_lengths)
    print(f"\n  KS test (length): stat={ks_stat:.4f}, p-value={ks_pval:.4f}")
    if ks_pval < 0.05:
        print("    ✓ Lengths are significantly different (good)")
    else:
        print("    ⚠️  Lengths are similar (potential issue)")

    # =========================================================================
    # CHECK 2: Activation Separation
    # =========================================================================
    print("\n[CHECK 2] Activation Separation: h_r vs h_c statistics")
    print("-" * 80)

    layer_info_path = phase0_dir / "layer_importance.json"
    if not layer_info_path.exists():
        print(
            "[ERROR] layer_importance.json not found. Run phase0_compute_fisher first."
        )
        return

    with open(layer_info_path) as f:
        layer_info = json.load(f)
    target_layers = layer_info["target_layers"]

    for layer_idx in target_layers:
        r_path = refusal_dir / f"activations_refusal_l{layer_idx:02d}.pt"
        c_path = refusal_dir / f"activations_compliant_l{layer_idx:02d}.pt"

        if not (r_path.exists() and c_path.exists()):
            print(f"  Layer {layer_idx}: activations not found")
            continue

        h_r = torch.load(r_path, map_location="cpu", weights_only=True).numpy()
        h_c = torch.load(c_path, map_location="cpu", weights_only=True).numpy()

        print(f"\n  Layer {layer_idx}:")
        print(f"    h_r shape: {h_r.shape}, h_c shape: {h_c.shape}")

        # Statistics
        r_mean = h_r.mean(axis=0)
        c_mean = h_c.mean(axis=0)
        r_std = h_r.std(axis=0)
        c_std = h_c.std(axis=0)

        print(f"    h_r: mean={np.mean(r_mean):.6f}, std={np.mean(r_std):.6f}")
        print(f"    h_c: mean={np.mean(c_mean):.6f}, std={np.mean(c_std):.6f}")

        # Cosine similarity between means
        cos_sim = np.dot(r_mean, c_mean) / (
            np.linalg.norm(r_mean) * np.linalg.norm(c_mean) + 1e-8
        )
        print(f"    Cosine similarity (means): {cos_sim:.4f}")
        if cos_sim > 0.9:
            print(f"      ⚠️  HIGH similarity (expected < 0.9)")
        elif cos_sim < 0.5:
            print(f"      ✓ Good separation")
        else:
            print(f"      ~ Moderate separation")

        # Norms
        r_norms = np.linalg.norm(h_r, axis=1)
        c_norms = np.linalg.norm(h_c, axis=1)

        print(f"    h_r norms: mean={r_norms.mean():.4f}, std={r_norms.std():.4f}")
        print(f"    h_c norms: mean={c_norms.mean():.4f}, std={c_norms.std():.4f}")

        # KS test on norms
        ks_stat, ks_pval = ks_2samp(r_norms, c_norms)
        print(f"    KS test (norms): stat={ks_stat:.4f}, p-value={ks_pval:.4f}")
        if ks_pval < 0.05:
            print("      ✓ Norms are significantly different")
        else:
            print("      ⚠️  Norms are similar (potential issue)")

    # =========================================================================
    # CHECK 3: cPCA Eigenvalues
    # =========================================================================
    print("\n[CHECK 3] cPCA Eigenvalues: Signal vs Noise")
    print("-" * 80)

    pr_path = refusal_dir / "participation_ratios.json"
    if not pr_path.exists():
        print("[ERROR] participation_ratios.json not found.")
        return

    with open(pr_path) as f:
        pr_report = json.load(f)

    print("\n  Participation Ratios (PR):")
    for layer_id, data in pr_report.items():
        pr = data["PR"]
        k = data["k"]
        regime = "Rank-1" if pr < 2 else ("Low-rank" if pr < 10 else "Distributed")
        print(f"    Layer {layer_id}: PR={pr:.2f} ({regime}), k={k} directions")

        if pr < 2:
            print(f"      ⚠️  RANK-1 (strong signal concentrated in 1-2 directions)")
        elif pr < 10:
            print(f"      ~ LOW-RANK (moderate concentration, could be noise)")
        else:
            print(f"      ✓ DISTRIBUTED (variance spread across many directions)")

    # =========================================================================
    # CHECK 4: Refusal Directions
    # =========================================================================
    print("\n[CHECK 4] Refusal Directions: Magnitude and Structure")
    print("-" * 80)

    for layer_idx in target_layers:
        dir_path = refusal_dir / f"refusal_directions_l{layer_idx:02d}.pt"
        if not dir_path.exists():
            print(f"  Layer {layer_idx}: directions not found")
            continue

        directions = torch.load(dir_path, map_location="cpu", weights_only=True)
        print(f"\n  Layer {layer_idx}:")
        print(
            f"    Shape: {directions.shape} (k={directions.shape[0]}, d_model={directions.shape[1]})"
        )

        # Norms of each direction
        dir_norms = torch.norm(directions, dim=1)
        print(
            f"    Direction norms: mean={dir_norms.mean():.6f}, std={dir_norms.std():.6f}"
        )
        print(f"    Min norm: {dir_norms.min():.6f}, Max norm: {dir_norms.max():.6f}")

        if dir_norms.max() < 1e-5:
            print(f"      ⚠️  ALL DIRECTIONS ARE NEAR-ZERO (broken!)")
        else:
            print(f"      ✓ Directions have reasonable magnitude")

        # Check for NaN/Inf
        if torch.isnan(directions).any():
            print(f"      ⚠️  NaN DETECTED in directions!")
        if torch.isinf(directions).any():
            print(f"      ⚠️  Inf DETECTED in directions!")

        # Orthogonality check (first k should be ~orthogonal if cPCA worked)
        if directions.shape[0] > 1:
            gram = torch.mm(directions, directions.T)
            off_diag = gram - torch.eye(gram.shape[0])
            max_off = off_diag.abs().max()
            print(f"    Gram matrix max off-diagonal: {max_off:.6f}")
            if max_off > 0.3:
                print(f"      ⚠️  Directions not orthogonal (possible cPCA issue)")
            else:
                print(f"      ✓ Directions are reasonably orthogonal")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)

    issues = []
    if ks_pval >= 0.05:
        issues.append("• Refusal/compliant prompt lengths are too similar")

    if cos_sim > 0.9:
        issues.append("• Activation means are too similar (cosine_sim > 0.9)")

    if dir_norms.max() < 1e-5:
        issues.append("• Refusal directions are near-zero (broken computation)")

    if issues:
        print("\n⚠️  POTENTIAL ISSUES DETECTED:\n")
        for issue in issues:
            print(issue)
        print("\nRECOMMENDED NEXT STEPS:")
        print("1. Check if alignment data (refusal/compliant) is correctly labeled")
        print("2. Verify that refusal prompts actually trigger refusals in the model")
        print("3. Consider using last-token activation instead of mean-pool")
        print("4. Try different target layers (rerun phase0_compute_fisher.py)")
    else:
        print("\n✓ Phase 0 appears structurally sound.")
        print(
            "Issue likely in Phase 1: check probe calibration or token-level activation capture."
        )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
