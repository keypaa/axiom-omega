# AXIOM-Ω

Latent space intervention system for transformer-based language models.

**Tested on**: Meta-Llama-3-8B-Instruct  
**Compatible with**: Any transformer model (Llama, Mistral, Qwen, etc.)

## Status: Phase 0 & 1 Verified ✅ | Phase 2/3 Pending Verification

| Phase | Status | Key Achievement | Notes |
|-------|--------|-----------------|-------|
| Phase 0 | ✅ Working | Refusal subspace extraction (real data) | Uses mlabonne/harmful_behaviors (416 prompts) |
| Phase 1 | ✅ Working | Latent hooks with threshold logic | Respects 0.55 threshold, no false positives on compliant |
| Phase 2 | ⏳ Pending | Reasoning preservation (MMLU/GSM8K) | Ready to test on cloud |
| Phase 3 | ⏳ Pending | EKF + MPC integration | Ready to test on cloud |

## Quick Start

### Phase 0: Data Collection & Subspace Extraction
```bash
# Collect refusal/compliant data (uses real harmful prompts)
python phase0_collect_data.py --model meta-llama/Meta-Llama-3-8B-Instruct --target_behavior refusal

# Compute Fisher importance scores
python phase0_compute_fisher.py --model meta-llama/Meta-Llama-3-8B-Instruct

# Extract refusal subspace via cPCA
python phase0_compute_refusal_subspace.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

### Phase 1: Run Refusal Hooks with Threshold Logic
```bash
# Interactive test with custom prompts (respects 0.55 threshold)
python phase1_train_and_test.py \
    --target_layers "10,31" \
    --lambda_scale 60.0 \
    --magnitude_cap 0.15 \
    --threshold 0.55 \
    --logic or

# Benchmark on 10 refusal + 10 compliant prompts
python phase1_benchmark.py \
    --target_layers "10,31" \
    --lambda_scale 60.0 \
    --threshold 0.55
```

## Results

**Phase 1 (Current — Verified ✅)**
- **Refusal bypass**: ✅ Methamphetamine prompt correctly intervened (p=0.967 on Layer 31)
- **False positives**: ✅ ZERO — Compliant prompts (photosynthesis, sort list) NOT intervened
- **Threshold enforcement**: ✅ 0.55 threshold respected (compliant stay below 0.55)
- **Speed overhead**: ~0.6% (minimal)

**Phase 2 (Expected from previous runs)**
- **MMLU/GSM8K**: +2% improvement / 0% delta (no degradation)
- Status: Ready to verify on cloud

**Critical Fix Applied**
- Removed hardcoded `should_intervene = True` that was bypassing threshold logic
- Implemented proper threshold comparison: intervene only if `p_refuse > threshold`
- Fixed AND logic: use `min()` across layers (intervene only if ALL exceed threshold)

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