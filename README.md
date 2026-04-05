# AXIOM-Ω

Latent space intervention system for transformer-based language models.

**Tested on**: Meta-Llama-3-8B-Instruct  
**Compatible with**: Any transformer model (Llama, Mistral, Qwen, etc.)

## Status: ALL PHASES COMPLETE ✅

| Phase | Status | Key Achievement |
|-------|--------|-----------------|
| Phase 1 | ✅ Complete | Latent hooks (100% bypass) |
| Phase 2 | ✅ Complete | Reasoning preservation (no degradation) |
| Phase 3 | ✅ Complete | EKF + MPC integration (uncertainty tracking) |

## Quick Start

### Run Phase 1 (Production)
```bash
python phase1_benchmark.py \
    --target_layers "10,31" \
    --lambda_scale 60.0 \
    --magnitude_cap 0.4 \
    --threshold 0.55
```

### Run Phase 3 (With EKF Uncertainty Tracking)
```bash
python phase3_test.py \
    --mode phase3 \
    --target_layers "10,31" \
    --lambda_scale 60.0 \
    --magnitude_cap 0.4 \
    --threshold 0.55
```

## Results

- **Refusal bypass**: 100%
- **False positives**: 0%
- **Speed overhead**: 0.6%
- **MMLU/GSM8K**: No degradation

## Scripts

| Script | Purpose |
|--------|---------|
| `phase1_train_and_test.py` | Interactive testing with custom prompts |
| `phase1_benchmark.py` | Standard benchmark (10 refusal + 10 compliant) |
| `lambda_sweep.py` | Find optimal lambda for your hardware |
| `phase2_reasoning_benchmark.py` | Measure MMLU/GSM8K degradation |
| `phase3_test.py` | Test Phase 3 (EKF + MPC) |
| `speed_test.py` | Measure inference overhead |

## Configuration

| Parameter | Description | Recommended |
|-----------|-------------|--------------|
| `--target_layers` | Layers to hook | "10,31" |
| `--lambda_scale` | Intervention strength | 60.0 |
| `--magnitude_cap` | Max Δh as fraction of h | 0.4 |
| `--threshold` | Probe threshold | 0.55 |
| `--logic` | "or" = any layer, "and" = all | "or" |

## Checkpoint Structure

```
axiom_checkpoints/llama3_8b_v1/
├── refusal_probe_l10.pt      # Probe for Layer 10
├── refusal_probe_l31.pt     # Probe for Layer 31
├── ekf_estimator.pt        # Phase 3 EKF
├── mpc_controller.pt        # Phase 3 MPC
├── refusal_subspace/
│   ├── activations_refusal_l10.pt
│   ├── activations_refusal_l31.pt
│   ├── refusal_directions_l10.pt
│   └── refusal_directions_l31.pt
└── sacred_subspace/
    ├── sacred_projector_l10.pt
    └── sacred_projector_l31.pt
```

## Documentation

- [PHASE1_RESULTS.md](./PHASE1_RESULTS.md) - Detailed results and learnings
- [USAGE.md](./USAGE.md) - Usage guide and configuration reference