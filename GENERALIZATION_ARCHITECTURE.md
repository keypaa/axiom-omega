# AXIOM-Ω Model Generalization Architecture

**Goal**: Run AXIOM pipeline (Phase 0-5) on ANY transformer model with a single command.

```bash
python model_agnostic_pipeline.py \
    --model "mistralai/Mistral-7B-Instruct-v0.2" \
    --target_behavior "refusal" \
    --phases "0,1,2,3" \
    --output_dir "./axiom_checkpoints/mistral_v0.2"
```

---

## Design Principles

1. **Auto-detection over configuration**
   - Detect model architecture, layer count, hidden dim automatically
   - Select target layers via Fisher scoring (not hardcoded)
   - Use model-family defaults for hyperparameters

2. **Behavior-agnostic**
   - "refusal" → Use refusal probe + AdvBench data
   - "hallucination" → Use factuality probe + SQuAD/TruthfulQA data
   - "toxicity" → Use toxicity probe + Jigsaw Toxic Comments data
   - Architecture is identical; only data + probe labels change

3. **Checkpoint portability**
   - Save model architecture fingerprint in checkpoint
   - Enable transfer learning (train on Llama, test on Mistral with fine-tuning)

4. **Graceful degradation**
   - Falls back from 4-bit NF4 → 8-bit → bfloat16 if OOM
   - Falls back from auto layer selection → user-provided layers
   - Falls back from HF datasets → local cache → custom files

---

## Core Components

### 1. Model Registry (`model_registry.py`)
```
Detects model architecture & provides model-specific defaults
├── Llama (3, 2)
│   ├── Layer structure: transformer.layers[i].self_attn.o_proj
│   ├── Default target layers: [10, 31]
│   ├── Hidden dim auto-detect: model.config.hidden_size
│   └── Hook point: output[0] from layer forward
├── Mistral
│   ├── Layer structure: model.layers[i].self_attn.o_proj
│   ├── Default target layers: [15, 25]  # Mistral-7B has 32 layers
│   └── (Same hook points as Llama)
├── Qwen2
│   ├── Layer structure: model.layers[i].self_attn.o_proj
│   ├── Default target layers: [12, 28]  # Qwen2-7B has 32 layers
│   └── (Same hook points)
└── (Add more as needed)
```

### 2. Config Management (`axiom_config.py`)
```python
@dataclass
class AxiomConfig:
    # Model
    model_name: str  # HF model ID
    target_behavior: str = "refusal"  # refusal | hallucination | toxicity
    phases: list[int] = field(default_factory=lambda: [0, 1, 2, 3])
    
    # Data
    dataset_sources: dict = field(default_factory=dict)  # Auto-populated per behavior
    n_refusal_samples: int = 500
    n_compliant_samples: int = 500
    n_calibration_samples: int = 950
    
    # Model config
    target_layers: Optional[list[int]] = None  # Auto-select if None
    n_target_layers: int = 3
    quantization: str = "nf4"  # nf4 | int8 | bfloat16
    
    # Phase 0
    variance_threshold_sacred: float = 0.99
    variance_threshold_refusal: float = 0.95
    
    # Phase 1-2
    lambda_scale: float = 60.0
    magnitude_cap: float = 0.4
    threshold: float = 0.55
    logic: str = "or"  # or | and
    
    # Paths
    data_dir: str = "./data"
    checkpoint_dir: str = "./axiom_checkpoints"
    batch_size: int = 4
    max_seq_len: int = 1024
```

### 3. Dataset Configuration (`dataset_config.yaml`)
```yaml
refusal:
  calibration:
    - source: "wikitext"
      split: "train"
      n_samples: 200
      role: "reasoning"
    - source: "gsm8k"
      split: "train"
      n_samples: 200
      role: "reasoning"
    - source: "tatsu-lab/alpaca"
      n_samples: 250
      role: "instruction"
    # ... more calibration sources
  
  alignment:
    refusal:
      - source: "walledai/AdvBench"
        n_samples: 500
        prompt_field: "prompt"
    compliant:
      - source: "tatsu-lab/alpaca"
        n_samples: 500
        filter: "non_harmful"

hallucination:
  calibration:
    - source: "wikipedia"
      n_samples: 500
      role: "factual"
  alignment:
    hallucination:
      - source: "ambig"
        n_samples: 300
    truth:
      - source: "truthful_qa"
        n_samples: 300

toxicity:
  # Similar structure
```

### 4. Main Pipeline (`model_agnostic_pipeline.py`)
```python
class AxiomPipeline:
    def __init__(self, config: AxiomConfig):
        self.config = config
        self.model_info = ModelRegistry.get(config.model_name)
        self.model = self._load_model()
        
    def run(self):
        """Execute phases sequentially"""
        for phase in self.config.phases:
            if phase == 0:
                self.phase0_collect_and_compute()
            elif phase == 1:
                self.phase1_train_and_benchmark()
            # ... phases 2-5
    
    def _load_model(self):
        """Load with auto-fallback quantization"""
        try:
            return AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=self._get_quantization_config(),
                device_map="auto"
            )
        except Exception as e:
            print(f"Failed with {self.config.quantization}, falling back...")
            # Fallback logic
    
    def phase0_collect_and_compute(self):
        """Generalized Phase 0: auto layer selection + activation collection"""
        # 1. Download/load datasets
        datasets = self._load_datasets()
        
        # 2. Collect activations on calibration set
        layer_importance = self._compute_fisher_importance(datasets['calibration'])
        
        # 3. Select target layers (top-3, spanning thirds)
        self.target_layers = self._select_layers(layer_importance)
        
        # 4. Collect refusal/compliant activations
        self._collect_refusal_activations(datasets['refusal'], datasets['compliant'])
        
        # 5. Compute subspaces
        self._compute_subspaces()
    
    def phase1_train_and_benchmark(self):
        """Generalized Phase 1: probe training + intervention"""
        # Load activations from Phase 0
        probes = self._train_probes()
        
        # Install hooks
        self._install_hooks(probes)
        
        # Benchmark
        self._benchmark(n_prompts=10)
```

---

## Behavior Extensibility (Future)

To add a new behavior:

1. **Define a probe** (e.g., `HallucinationProbe`):
   ```python
   class HallucinationProbe(nn.Module):
       def forward(self, h: torch.Tensor) -> torch.Tensor:
           # Predict probability of hallucination
           ...
   ```

2. **Add dataset sources** to `dataset_config.yaml`:
   ```yaml
   hallucination:
     alignment:
       hallucination: ["ambig", "truthful_qa"]
       truth: ["truthful_qa"]
   ```

3. **Register in `AxiomConfig.BEHAVIOR_DEFAULTS`**:
   ```python
   BEHAVIOR_DEFAULTS = {
       "refusal": {...},
       "hallucination": {...},  # NEW
   }
   ```

4. **Run**:
   ```bash
   python model_agnostic_pipeline.py \
       --model "mistralai/Mistral-7B" \
       --target_behavior "hallucination"
   ```

---

## Implementation Roadmap

### Week 1: Foundation
- [ ] `model_registry.py` (Llama, Mistral, Qwen2 support)
- [ ] `axiom_config.py` (config management)
- [ ] `dataset_config.yaml` (refusal behavior fully specified)
- [ ] `model_agnostic_pipeline.py` Phase 0-1 (data collection + probe training)

### Week 2: Testing & Validation
- [ ] Test Phase 0-1 on Mistral-7B (validate generalization)
- [ ] Benchmark (bypass rate, false positives, speed)
- [ ] Lambda sweep on Mistral to find optimal hyperparams

### Week 3: Extensions
- [ ] Phase 2-5 integration into pipeline
- [ ] Hallucination behavior dataset + probe
- [ ] Multi-behavior support (run refusal + hallucination simultaneously)

---

## Example Usage

```bash
# Llama 3 8B - refusal (same as current)
python model_agnostic_pipeline.py \
    --model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --target_behavior "refusal" \
    --phases "0,1,2,3"

# Mistral 7B - refusal (NEW)
python model_agnostic_pipeline.py \
    --model "mistralai/Mistral-7B-Instruct-v0.2" \
    --target_behavior "refusal" \
    --phases "0,1,2,3"

# Qwen2 7B - hallucination (FUTURE)
python model_agnostic_pipeline.py \
    --model "Qwen/Qwen2-7B-Instruct" \
    --target_behavior "hallucination" \
    --phases "0,1,2"

# Custom layers
python model_agnostic_pipeline.py \
    --model "mistralai/Mistral-7B" \
    --target_behavior "refusal" \
    --target_layers "8,16,24"

# Custom datasets
python model_agnostic_pipeline.py \
    --model "mistralai/Mistral-7B" \
    --target_behavior "refusal" \
    --dataset_config "./my_datasets.yaml"
```

---

## Key Design Decisions

| Decision | Why |
|----------|-----|
| Auto-detect model architecture | No hardcoding per model. Scales to 100+ HF models. |
| Parameterize target layers | Fisher-based selection is more robust than fixed indices. |
| Separate behavior config from model config | Same architecture works for refusal, hallucination, toxicity, etc. |
| Dataset YAML over Python | Easier to modify data sources without code changes. |
| Fallback quantization strategies | Handles diverse GPU memory constraints. |
| Checkpoint portability | Future: train on one model, transfer to another. |
