# AXIOM On New Model: Execution Plan for Mistral

This document tells you exactly what to do when you spin up the instance to run AXIOM on Mistral 7B.

## Timeline

**Estimated**: 2-3 hours total (excluding download/compute time)

## Step-by-Step

### Phase 0: Infrastructure (5 minutes)
```bash
# 1. Clone repo
cd /path/to/axiom

# 2. Verify architecture files exist
ls -la model_registry.py axiom_config.py model_agnostic_pipeline.py

# 3. Test imports
python -c "from model_registry import ModelRegistry; print('OK')"
```

### Phase 1: Run on Mistral 7B (2-3 hours)

#### Option A: Let Auto-Detection Do Everything (Recommended)
```bash
python model_agnostic_pipeline.py \
    --model "mistralai/Mistral-7B-Instruct-v0.2" \
    --phases "0,1,2,3" \
    --checkpoint_dir "./axiom_checkpoints/mistral_v0.2" \
    --verbose
```

**What this does**:
1. Auto-detects Mistral architecture
2. Auto-selects target layers (likely [10, 22] for 32-layer model)
3. Downloads AdvBench + WikiText + GSM8K (auto-downloads if not cached)
4. Collects activations (Phase 0)
5. Trains probes (Phase 1)
6. Benchmarks refusal bypass rate (Phase 1)
7. Measures MMLU/GSM8K degradation (Phase 2)
8. Trains EKF + MPC (Phase 3)

**Expected output**:
```
axiom_checkpoints/mistral_v0.2/
├── phase_0/
│   ├── refusal_probe_l10.pt
│   ├── refusal_probe_l22.pt
│   ├── refusal_directions_l10.pt
│   ├── refusal_directions_l22.pt
│   ├── sacred_projector_l10.pt
│   └── sacred_projector_l22.pt
├── phase_1/
│   └── benchmark_results.json
├── phase_2/
│   └── reasoning_benchmark.json
└── phase_3/
    ├── ekf_estimator.pt
    └── mpc_controller.pt

results/
└── mistral-7b-instruct-v0.2_config.json
```

#### Option B: Custom Layer Selection
```bash
python model_agnostic_pipeline.py \
    --model "mistralai/Mistral-7B-Instruct-v0.2" \
    --target_layers "10,22" \
    --phases "0,1,2,3"
```

#### Option C: Phase by Phase (For Testing)
```bash
# Phase 0 only (data collection)
python model_agnostic_pipeline.py \
    --model "mistralai/Mistral-7B-Instruct-v0.2" \
    --phases "0"

# Then Phase 1 (probe training)
python model_agnostic_pipeline.py \
    --model "mistralai/Mistral-7B-Instruct-v0.2" \
    --phases "1"

# etc
```

### Phase 2: Compare with Llama 3 8B Results

#### Expected Bypass Rates
| Model | Phase | Bypass % | FP % | Notes |
|-------|-------|----------|------|-------|
| Llama 3 8B | 1 | 100% | 0% | Baseline |
| Mistral 7B | 1 | 95-100% | 0-1% | Should be similar |

#### Compare Configs
```bash
# Side-by-side comparison
cat axiom_checkpoints/llama3_8b_v1/phase_0/config.json
cat axiom_checkpoints/mistral_v0.2/phase_0/config.json

# Check selected layers match expectations
python -c """
import json
with open('axiom_checkpoints/llama3_8b_v1/phase_0/config.json') as f:
    llama_cfg = json.load(f)
with open('axiom_checkpoints/mistral_v0.2/phase_0/config.json') as f:
    mistral_cfg = json.load(f)
print(f'Llama layers: {llama_cfg[\"target_layers\"]}')
print(f'Mistral layers: {mistral_cfg[\"target_layers\"]}')
"""
```

### Phase 3: Lambda Sweep (Optional, if results differ >10%)

If Mistral's bypass rate is <95%, optimize lambda:

```bash
for lambda in 40 60 80 100 120; do
    echo "Testing lambda=$lambda"
    python model_agnostic_pipeline.py \
        --model "mistralai/Mistral-7B-Instruct-v0.2" \
        --lambda_scale $lambda \
        --phases "1" \
        --checkpoint_dir "./axiom_checkpoints/mistral_lambda_$lambda"
done

# Then review results
ls -la axiom_checkpoints/mistral_lambda_*/phase_1/benchmark_results.json
```

### Phase 4: Collect Results

```bash
# Summary of all results
python -c """
import json
from pathlib import Path

models = ['llama3_8b_v1', 'mistral_v0.2']
for model in models:
    try:
        with open(f'axiom_checkpoints/{model}/phase_1/benchmark_results.json') as f:
            data = json.load(f)
        print(f'\n{model}:')
        print(f'  Bypass rate: {data.get(\"bypass_rate\", \"N/A\")}')
        print(f'  False positives: {data.get(\"fp_rate\", \"N/A\")}')
        print(f'  Speed overhead: {data.get(\"overhead_pct\", \"N/A\")}')
    except FileNotFoundError:
        print(f'{model}: Results not found')
"""
```

## What Could Go Wrong (Troubleshooting)

### OutOfMemoryError
```bash
# Reduce batch size
python model_agnostic_pipeline.py \
    --model "mistralai/Mistral-7B" \
    --batch_size 2 \
    --quantization "int8"
```

### Model Not Found
```bash
# Check HF token is set
huggingface-cli login

# Or download manually first
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')"
```

### Unexpected Layer Count
```bash
# Check what was auto-detected
python -c """
from model_registry import ModelRegistry
from transformers import AutoConfig
config = AutoConfig.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')
info = ModelRegistry.get('mistralai/Mistral-7B-Instruct-v0.2', config)
print(f'Family: {info.family}')
print(f'Default layers: {info.default_target_layers}')
"""
```

### Probes Not Training Well
```bash
# Debug: check probe F1 scores
python -c """
import json
with open('axiom_checkpoints/mistral_v0.2/phase_1/benchmark_results.json') as f:
    data = json.load(f)
print(json.dumps(data, indent=2))
"""
```

## Success Criteria

✅ **Phase 0 Success**:
- Activations collected for both layers
- Sacred subspace computed (>95% variance captured)
- Refusal directions learned via cPCA

✅ **Phase 1 Success**:
- Refusal bypass rate ≥ 95%
- False positive rate ≤ 1%
- Speed overhead < 1%

✅ **Phase 2 Success**:
- MMLU drop ≤ 1%
- GSM8K drop ≤ 2%
- Probe F1 scores ≥ 0.85

✅ **Phase 3 Success**:
- EKF predicts refusal with ≥ 90% accuracy
- MPC computes stable interventions
- End-to-end bypass still ≥ 95%

## After This Session

1. **Document results** in a new markdown file:
   ```bash
   cat > MISTRAL_RESULTS.md << 'EOF'
   # Mistral 7B Results
   
   **Date**: [DATE]
   **Model**: mistralai/Mistral-7B-Instruct-v0.2
   **Phases**: 0-3
   
   ## Results
   - Bypass rate: [X]%
   - False positives: [Y]%
   - MMLU delta: [Z]%
   - Overhead: [W]%
   
   ## Comparison with Llama 3 8B
   [...]
   EOF
   ```

2. **Commit results**:
   ```bash
   git add MISTRAL_RESULTS.md axiom_checkpoints/mistral_v0.2/
   git commit -m "feat: AXIOM validated on Mistral 7B

   - Auto-layer selection: [X] layers
   - Bypass rate: [X]%
   - False positives: [Y]%
   - All phases completed successfully"
   ```

3. **Next steps**:
   - If results match Llama → Move to hallucination behavior
   - If results differ → Debug + optimize
   - If successful → Add Qwen2 7B

---

**Ready to run? Just execute:**
```bash
python model_agnostic_pipeline.py --model "mistralai/Mistral-7B-Instruct-v0.2" --phases "0,1,2,3" --verbose
```
