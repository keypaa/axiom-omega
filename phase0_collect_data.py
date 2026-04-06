"""
AXIOM-Ω Phase 0: Data Collection Pipeline
==========================================
Pulls and formats all datasets required before any model runs.
No model loading happens here — pure data prep.

Reads dataset specs from dataset_config.yaml to support multiple behaviors.
Uses AxiomConfig for path management and centralized configuration.

Output structure (model-agnostic):
    {checkpoint_dir}/{model_short_name}/phase_0/
    ├── calibration/          # Sacred Subspace triad (behavior-specific count)
    │   ├── {source_1}.jsonl
    │   ├── {source_2}.jsonl
    │   └── ...
    └── alignment/            # Alignment axis (e.g., harmful/compliant for refusal)
        ├── {label_1}_*.jsonl
        └── {label_2}_*.jsonl

Run:
    python phase0_collect_data.py --model meta-llama/Llama-3-8B-Instruct --target_behavior refusal
    or
    python phase0_collect_data.py --model mistralai/Mistral-7B-Instruct-v0.2 --target_behavior refusal
"""

import json
import random
import argparse
from pathlib import Path
from typing import Iterator, Optional

from axiom_config import AxiomConfig, load_dataset_config

random.seed(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records):>4} records → {path}")


def truncate_to_tokens(text: str, max_chars: int = 4096) -> str:
    """
    Rough char-level truncation before tokenization.
    Real truncation to 1024 tokens happens in the Fisher computation script.
    """
    return text[:max_chars]


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------


def collect_wikitext(n: int = 200) -> list[dict]:
    """WikiText-103: syntax, grammar, factual recall."""
    try:
        from datasets import load_dataset

        ds = load_dataset(
            "wikitext", "wikitext-103-raw-v1", split="train", trust_remote_code=True
        )
        records = []
        for row in ds:
            text = row["text"].strip()
            if len(text) > 200:  # skip very short passages
                records.append(
                    {"source": "wikitext-103", "text": truncate_to_tokens(text)}
                )
            if len(records) >= n:
                break
        return records
    except Exception as e:
        print(f"  [WARN] WikiText load failed ({e}). Using stubs.")
        return [{"source": "wikitext-103", "text": f"STUB_WIKI_{i}"} for i in range(n)]


def collect_gsm8k(n: int = 200) -> list[dict]:
    """GSM8K: multi-step arithmetic reasoning."""
    try:
        from datasets import load_dataset

        ds = load_dataset("gsm8k", "main", split="train", trust_remote_code=True)
        records = []
        for row in ds:
            text = f"Problem: {row['question']}\nSolution: {row['answer']}"
            records.append({"source": "gsm8k", "text": truncate_to_tokens(text)})
            if len(records) >= n:
                break
        return records
    except Exception as e:
        print(f"  [WARN] GSM8K load failed ({e}). Using stubs.")
        return [{"source": "gsm8k", "text": f"STUB_GSM8K_{i}"} for i in range(n)]


def collect_codealpaca(n: int = 200) -> list[dict]:
    """CodeAlpaca: structured/algorithmic reasoning."""
    try:
        from datasets import load_dataset

        ds = load_dataset(
            "sahil2801/CodeAlpaca-20k", split="train", trust_remote_code=True
        )
        records = []
        for row in ds:
            text = f"Instruction: {row['instruction']}\nOutput: {row['output']}"
            records.append({"source": "codealpaca", "text": truncate_to_tokens(text)})
            if len(records) >= n:
                break
        return records
    except Exception as e:
        print(f"  [WARN] CodeAlpaca load failed ({e}). Using stubs.")
        return [{"source": "codealpaca", "text": f"STUB_CODE_{i}"} for i in range(n)]


def collect_mc4(n: int = 100, lang: str = "fr") -> list[dict]:
    """mC4: multilingual coverage (default: French). Prevents English-only subspace."""
    try:
        from datasets import load_dataset

        ds = load_dataset(
            "mc4", lang, split="train", streaming=True, trust_remote_code=True
        )
        records = []
        for row in ds:
            text = row["text"].strip()
            if len(text) > 200:
                records.append(
                    {"source": f"mc4-{lang}", "text": truncate_to_tokens(text)}
                )
            if len(records) >= n:
                break
        return records
    except Exception as e:
        print(f"  [WARN] mC4 load failed ({e}). Using stubs.")
        return [{"source": "mc4", "text": f"STUB_MC4_{i}"} for i in range(n)]


def collect_mmlu_hard(n: int = 100) -> list[dict]:
    """MMLU hard subset: broad knowledge + reasoning diversity."""
    HARD_SUBJECTS = [
        "abstract_algebra",
        "college_mathematics",
        "college_physics",
        "formal_logic",
        "high_school_mathematics",
        "professional_law",
        "moral_reasoning",
        "philosophy",
    ]
    try:
        from datasets import load_dataset

        records = []
        for subject in HARD_SUBJECTS:
            ds = load_dataset(
                "cais/mmlu", subject, split="test", trust_remote_code=True
            )
            for row in ds:
                choices = "\n".join(
                    f"  {chr(65 + i)}. {c}" for i, c in enumerate(row["choices"])
                )
                text = (
                    f"Question: {row['question']}\n"
                    f"Choices:\n{choices}\n"
                    f"Answer: {chr(65 + row['answer'])}"
                )
                records.append(
                    {"source": f"mmlu-{subject}", "text": truncate_to_tokens(text)}
                )
                if len(records) >= n:
                    return records
        return records[:n]
    except Exception as e:
        print(f"  [WARN] MMLU load failed ({e}). Using stubs.")
        return [{"source": "mmlu-hard", "text": f"STUB_MMLU_{i}"} for i in range(n)]


def collect_longcontext(n: int = 100) -> list[dict]:
    """
    Long-context tasks (>=2k tokens).
    Uses BookCorpus passages concatenated to target length.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset(
            "bookcorpus", split="train", streaming=True, trust_remote_code=True
        )
        records = []
        buffer = []
        for row in ds:
            buffer.append(row["text"])
            combined = " ".join(buffer)
            if len(combined) >= 8000:  # ~2k tokens rough estimate
                records.append(
                    {
                        "source": "longcontext-bookcorpus",
                        "text": combined[:8192],  # cap at ~2k tokens
                        "long_context": True,
                    }
                )
                buffer = []
            if len(records) >= n:
                break
        return records
    except Exception as e:
        print(f"  [WARN] Long-context load failed ({e}). Using stubs.")
        return [
            {"source": "longcontext", "text": f"STUB_LONG_{i} " * 400} for i in range(n)
        ]


def collect_adversarial_calibration(n: int = 50) -> list[dict]:
    """
    Adversarial but valid prompts: ambiguous phrasing, edge cases.
    These stress the Sacred Subspace boundary — should not trigger intervention.
    Hand-curated templates; not sourced from AdvBench (that's D_r).
    """
    templates = [
        "Explain the dual-use nature of {topic} research without advocating for misuse.",
        "What are the legitimate reasons someone might study {topic}?",
        "Describe the safety considerations when working with {topic}.",
        "How do regulatory frameworks govern the use of {topic}?",
        "Compare the risks and benefits of {topic} in industrial applications.",
    ]
    topics = [
        "encryption",
        "network penetration testing",
        "chemical synthesis",
        "biological research",
        "pharmaceutical compounds",
        "pyrotechnics",
        "lock picking",
        "social engineering awareness",
        "OSINT",
        "reverse engineering",
    ]
    records = []
    for i in range(n):
        tmpl = templates[i % len(templates)]
        topic = topics[i % len(topics)]
        text = tmpl.format(topic=topic)
        records.append(
            {
                "source": "adversarial-calibration",
                "text": text,
                "label": "valid_but_edge_case",
            }
        )
    return records


# ---------------------------------------------------------------------------
# Alignment datasets (D_r and D_c)
# ---------------------------------------------------------------------------

# New collection functions for dataset_config.yaml sources


def collect_alpaca(n: int = 250) -> list[dict]:
    """Tatsu-lab Alpaca: instruction-following prompts."""
    try:
        from datasets import load_dataset

        ds = load_dataset("tatsu-lab/alpaca", split="train", trust_remote_code=True)
        records = []
        for row in ds:
            text = row["instruction"]
            if text and len(text) > 20:
                records.append({"source": "alpaca", "text": truncate_to_tokens(text)})
            if len(records) >= n:
                break
        return records
    except Exception as e:
        print(f"  [WARN] Alpaca load failed ({e}). Using stubs.")
        return [{"source": "alpaca", "text": f"STUB_ALPACA_{i}"} for i in range(n)]


def collect_starcoder(n: int = 150) -> list[dict]:
    """BigCode StarCoder: code generation and reasoning."""
    try:
        from datasets import load_dataset

        ds = load_dataset(
            "bigcode/starcoderdata",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        records = []
        for row in ds:
            text = row.get("content", row.get("text", ""))
            if text and len(text) > 100:
                records.append(
                    {"source": "starcoder", "text": truncate_to_tokens(text)}
                )
            if len(records) >= n:
                break
        return records
    except Exception as e:
        print(f"  [WARN] StarCoder load failed ({e}). Using stubs.")
        return [{"source": "starcoder", "text": f"STUB_CODE_{i}"} for i in range(n)]


def collect_openwebtext(n: int = 150) -> list[dict]:
    """OpenWebText: long-range coherence and general knowledge."""
    try:
        from datasets import load_dataset

        ds = load_dataset(
            "openwebtext", split="train", streaming=True, trust_remote_code=True
        )
        records = []
        for row in ds:
            text = row.get("text", "")
            if text and len(text) > 200:
                records.append(
                    {"source": "openwebtext", "text": truncate_to_tokens(text)}
                )
            if len(records) >= n:
                break
        return records
    except Exception as e:
        print(f"  [WARN] OpenWebText load failed ({e}). Using stubs.")
        return [{"source": "openwebtext", "text": f"STUB_WEB_{i}"} for i in range(n)]


def collect_wikipedia(n: int = 500) -> list[dict]:
    """Wikipedia: factually grounded text."""
    try:
        from datasets import load_dataset

        ds = load_dataset(
            "wikipedia",
            "20220601.en",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        records = []
        for row in ds:
            text = row.get("text", "")
            if text and len(text) > 200:
                records.append(
                    {"source": "wikipedia", "text": truncate_to_tokens(text)}
                )
            if len(records) >= n:
                break
        return records
    except Exception as e:
        print(f"  [WARN] Wikipedia load failed ({e}). Using stubs.")
        return [{"source": "wikipedia", "text": f"STUB_WIKI_{i}"} for i in range(n)]


def collect_c4(n: int = 300, lang: str = "en") -> list[dict]:
    """C4: large-scale web text corpus."""
    try:
        from datasets import load_dataset

        ds = load_dataset(
            "allenai/c4", lang, split="train", streaming=True, trust_remote_code=True
        )
        records = []
        for row in ds:
            text = row.get("text", "")
            if text and len(text) > 200:
                records.append(
                    {"source": f"c4-{lang}", "text": truncate_to_tokens(text)}
                )
            if len(records) >= n:
                break
        return records
    except Exception as e:
        print(f"  [WARN] C4 load failed ({e}). Using stubs.")
        return [{"source": f"c4-{lang}", "text": f"STUB_C4_{i}"} for i in range(n)]


def collect_truthfulqa(n: int = 500) -> list[dict]:
    """TruthfulQA: factual question-answering with ground truth."""
    try:
        from datasets import load_dataset

        ds = load_dataset(
            "truthfulqa/truthfulqa",
            "generation",
            split="validation",
            trust_remote_code=True,
        )
        records = []
        for row in ds:
            question = row.get("question", "")
            if question:
                records.append({"source": "truthfulqa", "text": question})
            if len(records) >= n:
                break
        return records
    except Exception as e:
        print(f"  [WARN] TruthfulQA load failed ({e}). Using stubs.")
        return [
            {"source": "truthfulqa", "text": f"STUB_TRUTHFUL_{i}"} for i in range(n)
        ]


def collect_ambig(n: int = 300) -> list[dict]:
    """AMBIG: ambiguous questions that trigger hallucinations."""
    try:
        from datasets import load_dataset

        # AMBIG is typically from a paper; try HF Hub or local
        ds = load_dataset("ambiqa", split="validation", trust_remote_code=True)
        records = []
        for row in ds:
            question = row.get("question", "")
            if question:
                records.append({"source": "ambig", "text": question})
            if len(records) >= n:
                break
        return records
    except Exception as e:
        print(f"  [WARN] AMBIG load failed ({e}). Using stubs.")
        return [{"source": "ambig", "text": f"STUB_AMBIG_{i}"} for i in range(n)]


def collect_jigsaw_toxic(n: int = 500) -> list[dict]:
    """Jigsaw Toxic Comments: comments flagged as toxic."""
    try:
        from datasets import load_dataset

        ds = load_dataset(
            "jigsaw_toxicity_pred", split="validation", trust_remote_code=True
        )
        records = []
        for row in ds:
            comment = row.get("comment_text", row.get("text", ""))
            if comment:
                records.append({"source": "jigsaw-toxic", "text": comment})
            if len(records) >= n:
                break
        return records
    except Exception as e:
        print(f"  [WARN] Jigsaw Toxic load failed ({e}). Using stubs.")
        return [{"source": "jigsaw-toxic", "text": f"STUB_TOXIC_{i}"} for i in range(n)]


def collect_oasst1(n: int = 500) -> list[dict]:
    """OpenAssistant v1: diverse benign prompts."""
    try:
        from datasets import load_dataset

        ds = load_dataset("OpenAssistant/oasst1", split="train", trust_remote_code=True)
        records = []
        for row in ds:
            if row.get("role") == "prompter" and row.get("lang") == "en":
                text = row.get("text", "")
                if text and len(text) > 20:
                    records.append({"source": "oasst1", "text": text})
            if len(records) >= n:
                break
        return records
    except Exception as e:
        print(f"  [WARN] OpenAssistant load failed ({e}). Using stubs.")
        return [{"source": "oasst1", "text": f"STUB_OASST_{i}"} for i in range(n)]


def collect_advbench(n: int = 500) -> list[dict]:
    """Harmful behaviors: refusal-triggering prompts (D_r).
    Uses mlabonne/harmful_behaviors instead of gated AdvBench.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("mlabonne/harmful_behaviors", split="train")
        records = []
        for row in ds:
            text = row.get("text", "")
            if text:
                records.append(
                    {"source": "harmful_behaviors", "text": text, "label": "refusal"}
                )
            if len(records) >= n:
                break
        # Pad with templates if dataset is smaller than n
        if len(records) < n:
            print(
                f"  [INFO] Only {len(records)} harmful prompts available, padding to {n}"
            )
        while len(records) < n:
            records.append(
                {
                    "source": "harmful_behaviors-template",
                    "text": f"[TEMPLATE HARMFUL PROMPT {len(records)}]",
                    "label": "refusal",
                }
            )
        return records[:n]
    except Exception as e:
        print(f"  [ERROR] Harmful behaviors load failed ({e}). Cannot proceed.")
        raise


def collect_compliant(n: int = 500) -> list[dict]:
    """Harmless, compliant prompts (D_c). Sources: Alpaca + OpenAssistant."""
    try:
        from datasets import load_dataset

        records = []
        # Alpaca
        ds = load_dataset("tatsu-lab/alpaca", split="train", trust_remote_code=True)
        for row in ds:
            text = row["instruction"]
            if text and len(text) > 20:
                records.append({"source": "alpaca", "text": text, "label": "compliant"})
            if len(records) >= n // 2:
                break
        # OpenAssistant
        ds2 = load_dataset(
            "OpenAssistant/oasst1", split="train", trust_remote_code=True
        )
        for row in ds2:
            if row.get("role") == "prompter" and row.get("lang") == "en":
                text = row["text"]
                if text and len(text) > 20:
                    records.append(
                        {"source": "oasst1", "text": text, "label": "compliant"}
                    )
            if len(records) >= n:
                break
        random.shuffle(records)
        return records[:n]
    except Exception as e:
        print(f"  [WARN] Compliant dataset load failed ({e}). Using stubs.")
        return [
            {"source": "compliant", "text": f"STUB_COMPLIANT_{i}", "label": "compliant"}
            for i in range(n)
        ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="AXIOM Phase 0 Data Collection")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3-8B-Instruct",
        help="HuggingFace model ID (used for checkpoint path derivation only)",
    )
    parser.add_argument(
        "--target_behavior",
        type=str,
        default="refusal",
        help="Target behavior: refusal, hallucination, toxicity",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="dataset_config.yaml",
        help="Path to dataset_config.yaml",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="Target token length (approximate, enforced in Fisher script)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # --- Load config ---
    config = AxiomConfig(
        model_name=args.model,
        target_behavior=args.target_behavior,
        dataset_config_path=args.dataset_config,
    )

    # --- Load dataset specs ---
    dataset_cfg = load_dataset_config(args.dataset_config)
    if args.target_behavior not in dataset_cfg:
        print(f"[ERROR] Unknown behavior: {args.target_behavior}")
        print(f"Available: {list(dataset_cfg.keys())}")
        return

    behavior_cfg = dataset_cfg[args.target_behavior]

    # --- Output directories ---
    phase0_dir = config.get_checkpoint_dir(phase=0)
    calib_dir = phase0_dir / "calibration"
    align_dir = phase0_dir / "alignment"

    print(f"\n=== AXIOM Phase 0: Data Collection ({args.target_behavior}) ===")
    print(f"Model: {args.model}")
    print(f"Checkpoint dir: {phase0_dir}")
    print()

    # --- Calibration datasets ---
    print("[1/2] Calibration datasets (Sacred Subspace):")
    total_calib = 0
    for dataset_spec in behavior_cfg.get("calibration", []):
        source = dataset_spec["source"]
        n_samples = dataset_spec.get("n_samples", 100)
        split = dataset_spec.get("split", "train")

        print(f"  Loading {source} ({n_samples} samples)...")

        # Map source to collection function
        if source == "wikitext":
            records = collect_wikitext(n_samples)
        elif source == "gsm8k":
            records = collect_gsm8k(n_samples)
        elif source == "tatsu-lab/alpaca":
            records = collect_alpaca(n_samples)
        elif source == "bigcode/starcoderdata":
            records = collect_starcoder(n_samples)
        elif source == "openwebtext":
            records = collect_openwebtext(n_samples)
        elif source == "wikipedia":
            records = collect_wikipedia(n_samples)
        elif source == "c4":
            records = collect_c4(n_samples, lang=split if split != "train" else "en")
        else:
            print(f"    [WARN] Unknown source {source}, using stubs")
            records = [
                {"source": source, "text": f"STUB_{i}"} for i in range(n_samples)
            ]

        output_file = calib_dir / f"{source.replace('/', '_')}_{n_samples}.jsonl"
        write_jsonl(output_file, records)
        total_calib += len(records)

    print(f"\n  Total calibration sequences: {total_calib}")

    # --- Alignment datasets ---
    print("\n[2/2] Alignment datasets:")
    for label, sources in behavior_cfg.get("alignment", {}).items():
        print(f"  [{label.upper()}]")
        for dataset_spec in sources:
            source = dataset_spec["source"]
            n_samples = dataset_spec.get("n_samples", 100)

            if n_samples == 0:
                continue

            print(f"    Loading {source} ({n_samples} samples)...")

            # Map source to collection function
            if source == "walledai/AdvBench":
                records = collect_advbench(n_samples)
            elif source == "tatsu-lab/alpaca":
                records = collect_alpaca(n_samples)
            elif source == "openassistant/oasst1":
                records = collect_oasst1(n_samples)
            elif source == "truthfulqa":
                records = collect_truthfulqa(n_samples)
            elif source == "AMBIG":
                records = collect_ambig(n_samples)
            elif source == "jigsaw-toxic-comment-classification-challenge":
                records = collect_jigsaw_toxic(n_samples)
            else:
                print(f"      [WARN] Unknown source {source}, using stubs")
                records = [
                    {"source": source, "text": f"STUB_{i}"} for i in range(n_samples)
                ]

            output_file = (
                align_dir / f"{label}_{source.replace('/', '_')}_{n_samples}.jsonl"
            )
            write_jsonl(output_file, records)

    # --- Summary ---
    print("\n=== Phase 0 Complete ===")
    print(f"Output directory: {phase0_dir.resolve()}")
    print("\nData manifest:")
    for f in sorted(phase0_dir.rglob("*.jsonl")):
        n_lines = sum(1 for _ in open(f))
        print(f"  {str(f.relative_to(phase0_dir)):<50} {n_lines:>4} records")

    print("\nNext step: run phase0_compute_fisher.py")


if __name__ == "__main__":
    main()
