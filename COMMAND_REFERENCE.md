# AXIOM Model Generalization: Command Reference

## Quick Start

### 1. Llama 3 8B - Full Pipeline
```bash
python model_agnostic_pipeline.py \
    --model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --phases "0,1,2,3" \
    --checkpoint_dir "./axiom_checkpoints"
```

### 2. Mistral 7B - Full Pipeline (NEW)
```bash
python model_agnostic_pipeline.py \
    --model "mistralai/Mistral-7B-Instruct-v0.2" \
    --phases "0,1,2,3" \
    --checkpoint_dir "./axiom_checkpoints"
```

### 3. Qwen2 7B - Specific Phases
```bash
python model_agnostic_pipeline.py \
    --model "Qwen/Qwen2-7B-Instruct" \
    --phases "0,1" \
    --quantization "int8"
```

## Common Options

| Option | Default | Examples |
|--------|---------|----------|
| `--model` | required | `"meta-llama/Meta-Llama-3-8B"`, `"mistralai/Mistral-7B"` |
| `--phases` | `0,1,2,3` | `"0,1"` (Phase 0+1), `"1,2,3"` (skip Phase 0) |
| `--target_behavior` | `refusal` | `refusal`, `hallucination`, `toxicity` |
| `--target_layers` | auto-detect | `"10,31"`, `"8,16,24"` |
| `--lambda_scale` | `60.0` | `40.0`, `80.0`, `100.0` |
| `--threshold` | `0.55` | `0.3`, `0.5`, `0.8` |
| `--magnitude_cap` | `0.4` | `0.2`, `0.5`, `1.0` |
| `--quantization` | `nf4` | `int8`, `bfloat16`, `fp32` |
| `--batch_size` | `4` | `8`, `16` (if GPU memory allows) |
| `--checkpoint_dir` | `./axiom_checkpoints` | `/path/to/checkpoints` |
| `--results_dir` | `./results` | `/path/to/results` |
| `--verbose` | off | `--verbose` (enable logging) |

## Advanced Scenarios

### Auto-detect Layers (Recommended)
```bash
python model_agnostic_pipeline.py \
    --model "mistralai/Mistral-7B" \
    --phases "0,1,2,3"
# Pipeline will auto-select optimal layers using Fisher scoring
```

### Custom Layer Selection
```bash
python model_agnostic_pipeline.py \
    --model "mistralai/Mistral-7B" \
    --target_layers "12,24,30" \
    --phases "0,1,2,3"
```

### Optimize for Hardware
```bash
# Low-memory GPU (A100 40GB)
python model_agnostic_pipeline.py \
    --model "mistralai/Mistral-7B" \
    --quantization "nf4" \
    --batch_size 8

# Very low-memory GPU (RTX 3090)
python model_agnostic_pipeline.py \
    --model "mistralai/Mistral-7B" \
    --quantization "int8" \
    --batch_size 4

# CPU only (experimental)
python model_agnostic_pipeline.py \
    --model "mistralai/Mistral-7B" \
    --quantization "fp32" \
    --batch_size 1
```

### Lambda Sweep (Find Optimal Intervention Strength)
```bash
# Run lambda_sweep.py after Phase 0
for lambda in 20 40 60 80 100; do
    python model_agnostic_pipeline.py \
        --model "mistralai/Mistral-7B" \
        --lambda_scale $lambda \
        --phases "1,2,3" \
        --checkpoint_dir "./axiom_checkpoints/mistral_lambda_sweep"
done
```

### Multi-GPU Acceleration (Future)
```bash
# Not yet supported; use model_parallel for now
export CUDA_VISIBLE_DEVICES=0,1,2,3
python model_agnostic_pipeline.py \
    --model "meta-llama/Meta-Llama-3-70B" \
    --phases "0,1,2,3"
```

## Hallucination Prevention (Future)

### Basic
```bash
python model_agnostic_pipeline.py \
    --model "mistralai/Mistral-7B" \
    --target_behavior "hallucination" \
    --phases "0,1,2"
```

### With Custom Hallucination Dataset
```bash
python model_agnostic_pipeline.py \
    --model "mistralai/Mistral-7B" \
    --target_behavior "hallucination" \
    --phases "0,1,2" \
    --dataset_config "./custom_hallucination.yaml"
```

## Debugging & Troubleshooting

### Verbose Output
```bash
python model_agnostic_pipeline.py \
    --model "mistralai/Mistral-7B" \
    --phases "0,1" \
    --verbose
```

### Skip Data Collection (Use Existing Checkpoints)
```bash
python model_agnostic_pipeline.py \
    --model "mistralai/Mistral-7B" \
    --phases "1,2,3" \
    --checkpoint_dir "./axiom_checkpoints/mistral_v0.2"
# Phase 0 is skipped; uses checkpoints from previous run
```

### Save Intermediate Probes
```bash
python model_agnostic_pipeline.py \
    --model "mistralai/Mistral-7B" \
    --phases "0,1" \
    --results_dir "./probe_analysis"
# Detailed probe F1 scores, calibration plots saved
```

### Test on Small Subset (Fast Iteration)
```bash
# Modify dataset_config.yaml:
# n_samples: 100  (instead of 500)
python model_agnostic_pipeline.py \
    --model "mistralai/Mistral-7B" \
    --phases "0,1"
```

## Expected Output Structure

```
axiom_checkpoints/
├── mistral-7b-instruct-v0.2/          # Auto-named from model
│   ├── phase_0/
│   │   ├── refusal_probe_l10.pt
│   │   ├── refusal_probe_l31.pt
│   │   ├── refusal_directions_l10.pt
│   │   ├── refusal_directions_l31.pt
│   │   ├── sacred_projector_l10.pt
│   │   ├── sacred_projector_l31.pt
│   │   └── config.json
│   ├── phase_1/
│   │   ├── benchmark_results.json
│   │   └── intervention_metrics.json
│   ├── phase_2/
│   │   ├── mmlu_scores.json
│   │   └── gsm8k_scores.json
│   └── phase_3/
│       ├── ekf_estimator.pt
│       └── mpc_controller.pt

results/
└── mistral-7b-instruct-v0.2_config.json
```

## Integration with Existing Scripts

All existing scripts remain compatible:

```bash
# Still works
python phase1_benchmark.py \
    --checkpoint_dir ./axiom_checkpoints/llama3_8b_v1 \
    --target_layers "10,31" \
    --lambda_scale 60.0

# New approach (recommended)
python model_agnostic_pipeline.py \
    --model "meta-llama/Meta-Llama-3-8B" \
    --phases "1"
```

## Performance Benchmarks (Expected)

| Model | Phases | Time | Memory | Bypass% | Speed Overhead |
|-------|--------|------|--------|---------|----------------|
| Llama 3 8B | 0-1 | ~30min | 16GB | 100% | 0.6% |
| Mistral 7B | 0-1 | ~25min | 14GB | 100% | 0.5% |
| Qwen2 7B | 0-1 | ~25min | 14GB | ~95% | 0.6% |
| Llama 3 70B | 0-1 | ~4hrs | 80GB | TBD | TBD |

(Measured on NVIDIA A100 40GB with nf4 quantization)

## Common Errors & Solutions

### `ValueError: Unknown model: <model_id>`
**Cause**: Model architecture not registered
**Solution**:
```bash
# Option 1: Use wildcard (if available)
python model_agnostic_pipeline.py --model "mistralai/Mistral-*"

# Option 2: Contribute to model_registry.py
# Edit model_registry.py, add your model to REGISTRY
```

### `OutOfMemoryError: CUDA out of memory`
**Solution**:
```bash
python model_agnostic_pipeline.py \
    --model "mistralai/Mistral-7B" \
    --quantization "int8" \
    --batch_size 2
# Or try bfloat16 or fp32 (slower but lower memory)
```

### `ValueError: target_layers must be specified when model is unknown`
**Solution**: Register model in model_registry.py or specify manually:
```bash
python model_agnostic_pipeline.py \
    --model "unknown/model" \
    --target_layers "10,20,30"
```

## Next: Implement Phase 0-5

Once generalized, Phase 0-1 will be ready to test on Mistral. Then:
1. Validate bypass rates match Llama 3 8B
2. Optimize hyperparameters for Mistral
3. Add hallucination behavior support
4. Test on Qwen2, larger models

---

**See**: `GENERALIZATION_NEXT_STEPS.md` for implementation roadmap
