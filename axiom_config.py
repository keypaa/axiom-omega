"""
AXIOM Config: Centralized configuration management for model-agnostic pipeline.

Handles:
- Model selection and architecture detection
- Behavior definitions (refusal, hallucination, toxicity)
- Dataset sources and counts
- Phase-specific hyperparameters
- Checkpoint and output paths
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List
from enum import Enum
import yaml
import json
from pathlib import Path


class TargetBehavior(Enum):
    """Supported intervention targets"""

    REFUSAL = "refusal"
    HALLUCINATION = "hallucination"
    TOXICITY = "toxicity"


@dataclass
class DatasetConfig:
    """Configuration for a dataset source"""

    source: str  # HF dataset ID or local path
    split: str = "train"
    n_samples: int = 100
    prompt_field: str = "text"
    role: str = "general"  # reasoning, instruction, factual, etc.
    filter_fn: Optional[str] = None  # Custom filter (e.g., "non_harmful")


@dataclass
class BehaviorDefinition:
    """Defines a target behavior (refusal, hallucination, etc)"""

    name: str
    probe_type: str  # "refusal", "hallucination", "toxicity", etc.
    calibration_sources: List[DatasetConfig] = field(default_factory=list)
    target_sources: List[DatasetConfig] = field(
        default_factory=list
    )  # E.g., harmful + harmless
    eval_metric: str = "bypass_rate"  # Or "hallucination_rate", "toxicity_rate"


@dataclass
class AxiomConfig:
    """Master configuration for AXIOM pipeline"""

    # ===== Model Configuration =====
    model_name: str  # HF model ID (required)
    quantization: str = "nf4"  # nf4 | int8 | bfloat16 | fp32
    device_map: str = "auto"
    max_memory: Optional[Dict[int, str]] = None  # Per-GPU memory limits

    # ===== Behavior Configuration =====
    target_behavior: str = "refusal"

    # ===== Phase Configuration =====
    phases: List[int] = field(default_factory=lambda: [0, 1, 2, 3])

    # ===== Data Configuration =====
    target_layers: Optional[List[int]] = None  # Auto-select if None
    n_target_layers: int = 3
    layer_selection_strategy: str = "fisher"  # fisher | uniform | custom

    n_refusal_samples: int = 500
    n_compliant_samples: int = 500
    n_calibration_samples: int = 950

    # ===== Phase 0 (Calibration) =====
    variance_threshold_sacred: float = 0.99
    variance_threshold_refusal: float = 0.95

    # ===== Phase 1-2 (Intervention) =====
    lambda_scale: float = 60.0
    magnitude_cap: float = 0.4
    threshold: float = 0.55
    logic: str = "or"  # or | and

    # ===== Phase 3-5 (Advanced) =====
    memory_alpha: float = 0.9  # EMA decay
    use_anticipatory_memory: bool = True
    use_attention_rewiring: bool = False

    # ===== Batch & Hardware =====
    batch_size: int = 4
    max_seq_len: int = 1024
    gradient_checkpointing: bool = True
    use_flash_attention: bool = True

    # ===== Paths =====
    data_dir: str = "./data"
    checkpoint_dir: str = "./axiom_checkpoints"
    results_dir: str = "./results"
    dataset_config_path: Optional[str] = None  # Path to dataset_config.yaml

    # ===== Logging & Evaluation =====
    verbose: bool = True
    save_metrics: bool = True
    eval_on_benchmark: bool = True
    n_eval_prompts: int = 10

    def __post_init__(self):
        """Validate and initialize config"""
        # Ensure paths exist
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)

        # Set model-specific checkpoint directory
        model_short_name = self.model_name.split("/")[-1].lower()
        self.checkpoint_dir = str(Path(self.checkpoint_dir) / model_short_name)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def get_checkpoint_dir(self, phase: int) -> Path:
        """Get phase-specific checkpoint directory"""
        phase_dir = Path(self.checkpoint_dir) / f"phase_{phase}"
        phase_dir.mkdir(parents=True, exist_ok=True)
        return phase_dir

    def save(self, path: str):
        """Save config to JSON"""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "AxiomConfig":
        """Load config from JSON"""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_args(cls, args) -> "AxiomConfig":
        """Create config from argparse Namespace"""
        return cls(
            model_name=args.model,
            target_behavior=args.target_behavior,
            phases=[int(p) for p in args.phases.split(",")],
            target_layers=[int(l) for l in args.target_layers.split(",")]
            if args.target_layers
            else None,
            quantization=args.quantization,
            verbose=args.verbose,
            # ... add more as needed
        )


# Predefined behavior templates
BEHAVIOR_TEMPLATES = {
    "refusal": BehaviorDefinition(
        name="refusal",
        probe_type="refusal",
        eval_metric="bypass_rate",
        # Datasets are loaded from dataset_config.yaml
    ),
    "hallucination": BehaviorDefinition(
        name="hallucination",
        probe_type="hallucination",
        eval_metric="hallucination_rate",
    ),
    "toxicity": BehaviorDefinition(
        name="toxicity",
        probe_type="toxicity",
        eval_metric="toxicity_rate",
    ),
}


def get_default_config(model_name: str, behavior: str = "refusal") -> AxiomConfig:
    """Get sensible defaults for a model+behavior combo"""
    return AxiomConfig(
        model_name=model_name,
        target_behavior=behavior,
        phases=[0, 1, 2, 3],  # Run Phase 0-3 by default
    )


def load_dataset_config(path: str) -> Dict:
    """Load dataset sources from YAML"""
    with open(path, "r") as f:
        return yaml.safe_load(f)
