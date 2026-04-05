# AXIOM-Ω Usage Guide

## Quick Start

### Run Refusal Bypass (Production)
```bash
python phase1_benchmark.py \
    --target_layers "10,31" \
    --lambda_scale 60.0 \
    --magnitude_cap 0.4 \
    --threshold 0.55 \
    --logic or
```

### Run Custom Prompts
```bash
python phase1_train_and_test.py \
    --target_layers "10,31" \
    --lambda_scale 60.0 \
    --magnitude_cap 0.4 \
    --threshold 0.55 \
    --logic or
```

## Scripts

| Script | Purpose |
|--------|---------|
| `phase1_train_and_test.py` | Test custom prompts interactively |
| `phase1_benchmark.py` | Run standard benchmark (10 refusal + 10 compliant) |
| `lambda_sweep.py` | Find optimal lambda for your hardware |
| `phase2_reasoning_benchmark.py` | Measure MMLU/GSM8K degradation |
| `speed_test.py` | Measure inference overhead |

## Configuration Reference

| Parameter | Description | Recommended |
|-----------|-------------|--------------|
| `--target_layers` | Layers to hook | "10,31" |
| `--lambda_scale` | Intervention strength | 60.0 |
| `--magnitude_cap` | Max Δh as fraction of h | 0.4 |
| `--threshold` | Probe threshold for intervention | 0.55 |
| `--logic` | "or" = any layer, "and" = all layers | "or" |

## Results Summary

- **Refusal bypass**: 100%
- **False positives**: 0%
- **MMLU delta**: +2%
- **GSM8K delta**: 0%
- **Speed overhead**: 0.6%

## Checkpoint Structure

```
axiom_checkpoints/llama3_8b_v1/
├── refusal_probe_l10.pt      # Probe for Layer 10
├── refusal_probe_l31.pt      # Probe for Layer 31
├── refusal_subspace/
│   ├── activations_refusal_l10.pt
│   ├── activations_refusal_l31.pt
│   ├── activations_compliant_l10.pt
│   ├── activations_compliant_l31.pt
│   ├── refusal_directions_l10.pt
│   └── refusal_directions_l31.pt
└── sacred_subspace/
    ├── sacred_projector_l10.pt
    └── sacred_projector_l31.pt
```

## Phase Progression

1. **Phase 0**: Data collection + cPCA computation (done)
2. **Phase 1**: Latent intervention hooks (done ✅)
3. **Phase 2**: Reasoning preservation (done ✅)
4. **Phase 3**: MPC adaptive control (optional)