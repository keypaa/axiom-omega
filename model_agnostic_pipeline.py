#!/usr/bin/env python3
"""
AXIOM-Ω Model-Agnostic Pipeline

Usage:
    python model_agnostic_pipeline.py --model "mistralai/Mistral-7B-Instruct-v0.2" --phases "0,1,2,3"
    python model_agnostic_pipeline.py --model "meta-llama/Meta-Llama-3-8B" --target_behavior "hallucination"

This orchestrates all phases (0-5) for any model and behavior.
"""

import argparse
import sys
import subprocess
from pathlib import Path
from typing import Optional
import json
import yaml

from transformers import AutoModelForCausalLM, AutoTokenizer

from axiom_config import AxiomConfig
from model_registry import ModelRegistry


class AxiomPipeline:
    """Main orchestrator for AXIOM phases on any model"""

    def __init__(self, config: AxiomConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.model_info = None
        self._initialize()

    def _initialize(self):
        """Initialize model and tokenizer"""
        print(f"\n{'=' * 60}")
        print(f"AXIOM Pipeline for {self.config.model_name}")
        print(f"Behavior: {self.config.target_behavior}")
        print(f"Phases: {self.config.phases}")
        print(f"{'=' * 60}\n")

        # Load tokenizer
        print("[1/2] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        # Detect model architecture
        print("[2/2] Auto-detecting model architecture...")
        try:
            temp_config = AutoModelForCausalLM.from_pretrained(
                self.config.model_name, config_only=True
            )
            self.model_info = ModelRegistry.get(self.config.model_name, temp_config)
            print(f"  ✓ Detected: {self.model_info.family} architecture")
            print(f"  ✓ Default layers: {self.model_info.default_target_layers}")

            # Use detected layers if not specified
            if self.config.target_layers is None:
                self.config.target_layers = self.model_info.default_target_layers
                print(f"  ✓ Using auto-selected layers: {self.config.target_layers}")

        except Exception as e:
            print(f"  ✗ Failed to detect architecture: {e}")
            sys.exit(1)

    def run(self):
        """Execute all phases in sequence"""
        try:
            if 0 in self.config.phases:
                self.phase0_collect_and_compute()

            if 1 in self.config.phases:
                self.phase1_train_and_benchmark()

            if 2 in self.config.phases:
                self.phase2_reasoning_benchmark()

            if 3 in self.config.phases:
                self.phase3_dynamic_control()

            if 4 in self.config.phases:
                self.phase4_anticipatory_memory()

            if 5 in self.config.phases:
                self.phase5_attention_rewiring()

            print(f"\n{'=' * 60}")
            print("✅ AXIOM Pipeline completed successfully!")
            print(f"{'=' * 60}\n")

            # Save results
            self._save_results()

        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    def phase0_collect_and_compute(self):
        """
        Phase 0: Collect data and compute subspaces

        Steps:
        1. Download calibration datasets
        2. Collect activations at target layers
        3. Compute Fisher importance scores
        4. Build Sacred Subspace (EFIM)
        5. Collect refusal/compliant activations
        6. Compute refusal subspace (cPCA)
        """
        print("\n" + "=" * 60)
        print("PHASE 0: Data Collection & Subspace Computation")
        print("=" * 60)

        # Step 1: Collect data (calibration + alignment)
        print("\n  [1/3] Collecting calibration and alignment datasets...")
        cmd = [
            "python",
            "phase0_collect_data.py",
            "--model",
            self.config.model_name,
            "--target_behavior",
            self.config.target_behavior,
        ]
        result = subprocess.run(cmd, check=True)
        if result.returncode != 0:
            raise RuntimeError(f"Phase 0 data collection failed")

        # Step 2: Compute Fisher + Sacred Subspace
        print("\n  [2/3] Computing Fisher importance + Sacred Subspace...")
        target_layers_str = (
            ",".join(map(str, self.config.target_layers))
            if self.config.target_layers
            else "auto"
        )
        cmd = [
            "python",
            "phase0_compute_fisher.py",
            "--model",
            self.config.model_name,
            "--target_behavior",
            self.config.target_behavior,
            "--target_layers",
            target_layers_str,
        ]
        result = subprocess.run(cmd, check=True)
        if result.returncode != 0:
            raise RuntimeError(f"Phase 0 Fisher computation failed")

        # Step 3: Compute Refusal Subspace
        print("\n  [3/3] Computing Refusal Subspace (cPCA)...")
        cmd = [
            "python",
            "phase0_compute_refusal_subspace.py",
            "--model",
            self.config.model_name,
            "--target_behavior",
            self.config.target_behavior,
        ]
        result = subprocess.run(cmd, check=True)
        if result.returncode != 0:
            raise RuntimeError(f"Phase 0 refusal subspace computation failed")

        print("\n  ✓ Phase 0 complete")

    def phase1_train_and_benchmark(self):
        """
        Phase 1: Train refusal probe and install intervention hooks

        Steps:
        1. Train linear probe on refusal/compliant activations
        2. Install hooks on target layers
        3. Benchmark refusal bypass rate and false positives
        """
        print("\n" + "=" * 60)
        print("PHASE 1: Probe Training & Intervention")
        print("=" * 60)

        target_layers_str = (
            ",".join(map(str, self.config.target_layers))
            if self.config.target_layers
            else "auto"
        )
        cmd = [
            "python",
            "phase1_train_and_test.py",
            "--model",
            self.config.model_name,
            "--target_behavior",
            self.config.target_behavior,
            "--target_layers",
            target_layers_str,
            "--lambda_scale",
            str(self.config.lambda_scale),
            "--threshold",
            str(self.config.threshold),
            "--magnitude_cap",
            str(self.config.magnitude_cap),
        ]
        result = subprocess.run(cmd, check=True)
        if result.returncode != 0:
            raise RuntimeError(f"Phase 1 probe training failed")

        print("\n  ✓ Phase 1 complete")

    def phase2_reasoning_benchmark(self):
        """
        Phase 2: Benchmark reasoning preservation

        Steps:
        1. Evaluate MMLU (general knowledge)
        2. Evaluate GSM8K (math reasoning)
        3. Ensure <1% degradation
        """
        print("\n" + "=" * 60)
        print("PHASE 2: Reasoning Preservation Benchmark")
        print("=" * 60)

        target_layers_str = (
            ",".join(map(str, self.config.target_layers))
            if self.config.target_layers
            else "auto"
        )
        cmd = [
            "python",
            "phase2_reasoning_benchmark.py",
            "--model",
            self.config.model_name,
            "--checkpoint_dir",
            str(Path(self.config.checkpoint_dir) / self.config.model_short_name),
            "--target_layers",
            target_layers_str,
            "--lambda_scale",
            str(self.config.lambda_scale),
            "--threshold",
            str(self.config.threshold),
            "--magnitude_cap",
            str(self.config.magnitude_cap),
        ]
        result = subprocess.run(cmd, check=True)
        if result.returncode != 0:
            raise RuntimeError(f"Phase 2 reasoning benchmark failed")

        print("\n  ✓ Phase 2 complete")

    def phase3_dynamic_control(self):
        """
        Phase 3: Train EKF state estimator + MPC controller

        Steps:
        1. Train EKF to predict refusal probability
        2. Train MPC to compute minimal interventions
        3. Integrate with Phase 1/2 hooks
        """
        print("\n" + "=" * 60)
        print("PHASE 3: Dynamic Control (EKF + MPC)")
        print("=" * 60)

        target_layers_str = (
            ",".join(map(str, self.config.target_layers))
            if self.config.target_layers
            else "auto"
        )
        cmd = [
            "python",
            "phase3_dynamic_control.py",
            "--model",
            self.config.model_name,
            "--checkpoint_dir",
            str(Path(self.config.checkpoint_dir) / self.config.model_short_name),
            "--target_layers",
            target_layers_str,
            "--lambda_scale",
            str(self.config.lambda_scale),
            "--threshold",
            str(self.config.threshold),
        ]
        result = subprocess.run(cmd, check=True)
        if result.returncode != 0:
            raise RuntimeError(f"Phase 3 dynamic control failed")

        print("\n  ✓ Phase 3 complete")

    def phase4_anticipatory_memory(self):
        """Phase 4: Add EMA memory for anticipatory intervention"""
        print("\n" + "=" * 60)
        print("PHASE 4: Anticipatory Memory")
        print("=" * 60)

        target_layers_str = (
            ",".join(map(str, self.config.target_layers))
            if self.config.target_layers
            else "auto"
        )
        cmd = [
            "python",
            "phase4_anticipatory.py",
            "--model",
            self.config.model_name,
            "--checkpoint_dir",
            str(Path(self.config.checkpoint_dir) / self.config.model_short_name),
            "--target_layers",
            target_layers_str,
        ]
        result = subprocess.run(cmd, check=True)
        if result.returncode != 0:
            raise RuntimeError(f"Phase 4 anticipatory memory failed")

        print("\n  ✓ Phase 4 complete")

    def phase5_attention_rewiring(self):
        """Phase 5: Low-rank attention bias injection"""
        print("\n" + "=" * 60)
        print("PHASE 5: Attention Rewiring")
        print("=" * 60)

        target_layers_str = (
            ",".join(map(str, self.config.target_layers))
            if self.config.target_layers
            else "auto"
        )
        cmd = [
            "python",
            "phase5_attention_rewiring.py",
            "--model",
            self.config.model_name,
            "--checkpoint_dir",
            str(Path(self.config.checkpoint_dir) / self.config.model_short_name),
            "--target_layers",
            target_layers_str,
        ]
        result = subprocess.run(cmd, check=True)
        if result.returncode != 0:
            raise RuntimeError(f"Phase 5 attention rewiring failed")

        print("\n  ✓ Phase 5 complete")

    def _save_results(self):
        """Save configuration and results"""
        results_dir = Path(self.config.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = (
            results_dir / f"{self.config.model_name.split('/')[-1]}_config.json"
        )
        self.config.save(str(config_path))
        print(f"  ✓ Config saved: {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description="AXIOM-Ω: Model-agnostic latent space intervention pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Llama 3 8B - refusal bypass
  python model_agnostic_pipeline.py --model "meta-llama/Meta-Llama-3-8B-Instruct" --phases "0,1,2,3"
  
  # Mistral 7B - refusal bypass
  python model_agnostic_pipeline.py --model "mistralai/Mistral-7B-Instruct-v0.2" --phases "0,1,2,3"
  
  # Custom layers
  python model_agnostic_pipeline.py --model "mistralai/Mistral-7B" --target_layers "8,16,24"
  
  # Hallucination prevention (future)
  python model_agnostic_pipeline.py --model "Qwen/Qwen2-7B" --target_behavior "hallucination"
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model ID (e.g., 'meta-llama/Meta-Llama-3-8B-Instruct')",
    )

    parser.add_argument(
        "--target_behavior",
        type=str,
        default="refusal",
        choices=["refusal", "hallucination", "toxicity"],
        help="Target behavior to steer (refusal, hallucination, toxicity)",
    )

    parser.add_argument(
        "--phases",
        type=str,
        default="0,1,2,3",
        help="Phases to run (comma-separated, e.g., '0,1,2,3')",
    )

    parser.add_argument(
        "--target_layers",
        type=str,
        default=None,
        help="Target layer indices (comma-separated, e.g., '10,31'). Auto-detect if not specified.",
    )

    parser.add_argument(
        "--lambda_scale",
        type=float,
        default=60.0,
        help="Intervention strength (default: 60.0)",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="Refusal detection threshold (default: 0.55)",
    )

    parser.add_argument(
        "--magnitude_cap",
        type=float,
        default=0.4,
        help="Max intervention magnitude as fraction of activation norm (default: 0.4)",
    )

    parser.add_argument(
        "--quantization",
        type=str,
        default="nf4",
        choices=["nf4", "int8", "bfloat16", "fp32"],
        help="Quantization strategy (default: nf4)",
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./axiom_checkpoints",
        help="Directory to save checkpoints (default: ./axiom_checkpoints)",
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results",
        help="Directory to save results (default: ./results)",
    )

    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size (default: 4)"
    )

    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Create config
    config = AxiomConfig(
        model_name=args.model,
        target_behavior=args.target_behavior,
        phases=[int(p) for p in args.phases.split(",")],
        target_layers=[int(l) for l in args.target_layers.split(",")]
        if args.target_layers
        else None,
        lambda_scale=args.lambda_scale,
        threshold=args.threshold,
        magnitude_cap=args.magnitude_cap,
        quantization=args.quantization,
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )

    # Run pipeline
    pipeline = AxiomPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
