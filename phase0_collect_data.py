"""
AXIOM-Ω Phase 0: Data Collection Pipeline
==========================================
Pulls and formats all datasets required before any model runs.
No model loading happens here — pure data prep.

Output structure:
    data/
    ├── calibration/          # Sacred Subspace triad (950 sequences)
    │   ├── wikitext_200.jsonl
    │   ├── gsm8k_200.jsonl
    │   ├── codealpaca_200.jsonl
    │   ├── mc4_100.jsonl
    │   ├── mmlu_hard_100.jsonl
    │   ├── longcontext_100.jsonl
    │   └── adversarial_50.jsonl
    └── alignment/            # Refusal / compliant axis
        ├── refusal_500.jsonl     # D_r
        └── compliant_500.jsonl   # D_c

Run:
    pip install datasets tqdm --break-system-packages
    python phase0_collect_data.py --output_dir ../data --max_tokens 1024
"""

import json
import random
import argparse
from pathlib import Path
from typing import Iterator

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
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train",
                          trust_remote_code=True)
        records = []
        for row in ds:
            text = row["text"].strip()
            if len(text) > 200:  # skip very short passages
                records.append({"source": "wikitext-103", "text": truncate_to_tokens(text)})
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
        ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train",
                          trust_remote_code=True)
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
        ds = load_dataset("mc4", lang, split="train", streaming=True,
                          trust_remote_code=True)
        records = []
        for row in ds:
            text = row["text"].strip()
            if len(text) > 200:
                records.append({"source": f"mc4-{lang}", "text": truncate_to_tokens(text)})
            if len(records) >= n:
                break
        return records
    except Exception as e:
        print(f"  [WARN] mC4 load failed ({e}). Using stubs.")
        return [{"source": "mc4", "text": f"STUB_MC4_{i}"} for i in range(n)]


def collect_mmlu_hard(n: int = 100) -> list[dict]:
    """MMLU hard subset: broad knowledge + reasoning diversity."""
    HARD_SUBJECTS = [
        "abstract_algebra", "college_mathematics", "college_physics",
        "formal_logic", "high_school_mathematics", "professional_law",
        "moral_reasoning", "philosophy"
    ]
    try:
        from datasets import load_dataset
        records = []
        for subject in HARD_SUBJECTS:
            ds = load_dataset("cais/mmlu", subject, split="test",
                              trust_remote_code=True)
            for row in ds:
                choices = "\n".join(f"  {chr(65+i)}. {c}"
                                    for i, c in enumerate(row["choices"]))
                text = (f"Question: {row['question']}\n"
                        f"Choices:\n{choices}\n"
                        f"Answer: {chr(65 + row['answer'])}")
                records.append({"source": f"mmlu-{subject}", "text": truncate_to_tokens(text)})
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
        ds = load_dataset("bookcorpus", split="train", streaming=True,
                          trust_remote_code=True)
        records = []
        buffer = []
        for row in ds:
            buffer.append(row["text"])
            combined = " ".join(buffer)
            if len(combined) >= 8000:  # ~2k tokens rough estimate
                records.append({
                    "source": "longcontext-bookcorpus",
                    "text": combined[:8192],  # cap at ~2k tokens
                    "long_context": True
                })
                buffer = []
            if len(records) >= n:
                break
        return records
    except Exception as e:
        print(f"  [WARN] Long-context load failed ({e}). Using stubs.")
        return [{"source": "longcontext", "text": f"STUB_LONG_{i} " * 400} for i in range(n)]


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
        "encryption", "network penetration testing", "chemical synthesis",
        "biological research", "pharmaceutical compounds", "pyrotechnics",
        "lock picking", "social engineering awareness", "OSINT",
        "reverse engineering"
    ]
    records = []
    for i in range(n):
        tmpl = templates[i % len(templates)]
        topic = topics[i % len(topics)]
        text = tmpl.format(topic=topic)
        records.append({
            "source": "adversarial-calibration",
            "text": text,
            "label": "valid_but_edge_case"
        })
    return records


# ---------------------------------------------------------------------------
# Alignment datasets (D_r and D_c)
# ---------------------------------------------------------------------------

def collect_advbench(n: int = 500) -> list[dict]:
    """AdvBench: refusal-triggering prompts (D_r)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("walledai/AdvBench", split="train", trust_remote_code=True)
        records = []
        for row in ds:
            text = row.get("goal", row.get("prompt", row.get("text", "")))
            if text:
                records.append({"source": "advbench", "text": text, "label": "refusal"})
            if len(records) >= n:
                break
        # Pad with templates if dataset is smaller than n
        while len(records) < n:
            records.append({
                "source": "advbench-template",
                "text": f"[TEMPLATE REFUSAL PROMPT {len(records)}]",
                "label": "refusal"
            })
        return records[:n]
    except Exception as e:
        print(f"  [WARN] AdvBench load failed ({e}). Using stubs.")
        return [{"source": "advbench", "text": f"STUB_REFUSAL_{i}", "label": "refusal"}
                for i in range(n)]


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
        ds2 = load_dataset("OpenAssistant/oasst1", split="train", trust_remote_code=True)
        for row in ds2:
            if row.get("role") == "prompter" and row.get("lang") == "en":
                text = row["text"]
                if text and len(text) > 20:
                    records.append({"source": "oasst1", "text": text, "label": "compliant"})
            if len(records) >= n:
                break
        random.shuffle(records)
        return records[:n]
    except Exception as e:
        print(f"  [WARN] Compliant dataset load failed ({e}). Using stubs.")
        return [{"source": "compliant", "text": f"STUB_COMPLIANT_{i}", "label": "compliant"}
                for i in range(n)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AXIOM Phase 0 Data Collection")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--max_tokens", type=int, default=1024,
                        help="Target token length (approximate, enforced in Fisher script)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out = Path(args.output_dir)

    print("\n=== AXIOM Phase 0: Data Collection ===\n")

    # --- Calibration (Sacred Subspace triad) ---
    print("[1/2] Calibration datasets (Sacred Subspace):")
    calib_dir = out / "calibration"

    print("  Loading WikiText-103...")
    write_jsonl(calib_dir / "wikitext_200.jsonl", collect_wikitext(200))

    print("  Loading GSM8K...")
    write_jsonl(calib_dir / "gsm8k_200.jsonl", collect_gsm8k(200))

    print("  Loading CodeAlpaca...")
    write_jsonl(calib_dir / "codealpaca_200.jsonl", collect_codealpaca(200))

    print("  Loading mC4 (multilingual)...")
    write_jsonl(calib_dir / "mc4_100.jsonl", collect_mc4(100))

    print("  Loading MMLU hard subset...")
    write_jsonl(calib_dir / "mmlu_hard_100.jsonl", collect_mmlu_hard(100))

    print("  Loading long-context passages...")
    write_jsonl(calib_dir / "longcontext_100.jsonl", collect_longcontext(100))

    print("  Generating adversarial calibration prompts...")
    write_jsonl(calib_dir / "adversarial_50.jsonl", collect_adversarial_calibration(50))

    total_calib = 200 + 200 + 200 + 100 + 100 + 100 + 50
    print(f"\n  Total calibration sequences: {total_calib}")

    # --- Alignment (D_r and D_c) ---
    print("\n[2/2] Alignment datasets (Refusal/Compliant axis):")
    align_dir = out / "alignment"

    print("  Loading AdvBench (D_r — refusal triggers)...")
    write_jsonl(align_dir / "refusal_500.jsonl", collect_advbench(500))

    print("  Loading compliant prompts (D_c — Alpaca + OpenAssistant)...")
    write_jsonl(align_dir / "compliant_500.jsonl", collect_compliant(500))

    # --- Summary ---
    print("\n=== Phase 0 Complete ===")
    print(f"Output directory: {out.resolve()}")
    print("\nData manifest:")
    for f in sorted(out.rglob("*.jsonl")):
        n_lines = sum(1 for _ in open(f))
        print(f"  {f.relative_to(out):<45} {n_lines:>4} records")

    print("\nNext step: run phase0_compute_fisher.py")


if __name__ == "__main__":
    main()
