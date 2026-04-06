# AXIOM Phase 0 Debug - Executive Summary

**Status**: 🔴 Phase 0 Likely Broken (4 Critical Issues Found)  
**Bypass Rate**: 0% (expected ≥80%)  
**Action**: Run debug script on cloud, then fix and retest  

---

## The Problem

Phase 1 refusal bypass achieves **0% success** despite:
- ✅ Phase 0 runs without errors
- ✅ Probes train to 92-98% accuracy
- ✅ Hooks install and fire correctly
- ✅ Activation shapes correct
- ❌ **But**: Model still refuses harmful prompts after intervention

**This means**: Phase 0 refusal subspace is NOT capturing the real refusal signal.

---

## Root Cause: 4 Critical Issues Identified

### 🔴 Issue 1: Mean-Pooling Over ALL Tokens (CRITICAL - Likely Culprit)

**Current**:
```python
# Phase 0: Train on mean-pooled activations
h = model_output  # [batch, seq, d_model]
h_pooled = h.mean(dim=1)  # Average ALL tokens → [batch, d_model]
```

**Problem**: 
- Mean-pooling dilutes refusal signal across all tokens
- Probe trained on smoothed distribution
- During Phase 1 generation, sees individual token activations (sharp, different distribution)
- Distribution mismatch → probe fails

**Evidence**:
- ✓ Works in Phase 0 (both refusal and compliant get pooled the same way)
- ✗ Fails in Phase 1 (token-level activations are different from mean-pooled)

**Fix**: Use last-token activation instead
```python
h_pooled = h[:, -1, :]  # Last token only (where refusal decision happens)
```

**Timeline**: 30 minutes (recollect activations + recompute cPCA)

---

### 🟠 Issue 2: No Probe Generalization Validation

**Current**:
- Probes trained on Phase 0 activations
- Validation accuracy computed on random 10% split
- **No checks** for train/val distribution similarity
- **No detection** of distribution shift from training to generation

**Problem**:
- High training accuracy (92-98%) hides overfitting
- Probe learned patterns specific to Phase 0 data distribution
- Fails on Phase 1 generation data (different distribution)

**Evidence**:
- Validation accuracy looks good ✓
- Generation accuracy is 0% ✗

**Fix**: Add KS test before declaring probe "trained"
```python
# After training, verify probe separates classes
h_r_embedded = probe(h_r)
h_c_embedded = probe(h_c)
ks_stat, pval = ks_2samp(h_r_embedded, h_c_embedded)
assert pval < 0.05, "Probe does not separate classes!"
```

**Timeline**: 10 minutes (add validation check)

---

### 🟠 Issue 3: Alignment Data May Be Mislabeled

**Current**:
- Phase 0 loads files named `*refusal*.jsonl` as D_r
- Assumes they trigger refusals (may not on Llama 3.1)
- Assumes Alpaca/OASST are fully "compliant" (may trigger refusals)

**Problem**:
- If D_r and D_c activations are too similar → no signal
- Refusal signal washed out by noise

**Evidence**:
- Can be checked by `debug_phase0.py`
- If cosine similarity of activation means > 0.9 → data is broken

**Fix**: Verify alignment data by running through model
```python
# Check if AdvBench actually triggers refusals
output = model.generate(prompt)
if "cannot" not in output.lower() and "refuse" not in output.lower():
    print(f"WARNING: '{prompt}' did NOT trigger refusal!")
```

**Timeline**: 15 minutes (manual spot-check on cloud)

---

### 🟠 Issue 4: cPCA Eigenvalue Quality (PR=5.63)

**Current**:
- Participation Ratio = 5.63 for Layer 31
- Labeled as "Low-rank" (ambiguous)

**Problem**:
- Could indicate either signal or noise
- Needs manual verification

**Evidence**:
- Can be checked by `debug_phase0.py`
- If direction norms are very small or non-orthogonal → cPCA broken

**Timeline**: 5 minutes (diagnostic check)

---

## Recommended Next Steps

### 🚀 Immediate (On Cloud)

**Step 1**: Run diagnostic script (5 minutes)
```bash
cd axiom_checkpoints/meta-llama-3-8b-instruct/phase_0/
python ../../debug_phase0.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

**Output will tell you**:
- ✓ Are refusal/compliant prompts actually different? (KS test)
- ✓ Are activation means separated? (cosine similarity)
- ✓ Are refusal directions reasonable? (magnitude, orthogonality)

### 📋 Based on Results

**If diagnosis says "activation means similar"** (most likely):
- Fix Issue 1: Change mean-pooling to last-token pooling
- Recollect Phase 0 activations (20 minutes)
- Rerun Phase 1 test

**If diagnosis says "directions are broken"**:
- Fix Issue 4: Recompute cPCA or try different layers
- Recollect Phase 0 (30 minutes)

**If diagnosis says "everything looks fine"**:
- Issue is in Phase 1 hooks or probe inference logic
- Check Phase1HookLayer10/31 implementation

---

## Files to Check/Modify

| File | Issue | Action |
|------|-------|--------|
| `debug_phase0.py` | NEW | Run on cloud to diagnose |
| `phase0_compute_refusal_subspace.py:75` | Issue 1 | Change `h.mean(dim=1)` → `h[:, -1, :]` |
| `phase1_train_and_test.py:121+` | Issue 2 | Add KS test validation |
| `phase0_collect_data.py` | Issue 3 | Manual verification of alignment data |
| PHASE0_DEBUG_FINDINGS.md | ALL | Detailed analysis of each issue |

---

## Why This Happened

**Mean-pooling was reasonable for**:
- Calibration data (Sacred Subspace computation)
- Initial architecture design (simplicity)

**But fails for refusal detection because**:
- Refusal signal is **local** (at the refusal decision token)
- Averaging over all tokens dilutes this local signal
- Phase 1 operates **token-by-token** (sees sharp distributions)
- Mismatch → probe failure

---

## Confidence Levels

| Issue | Confidence | Priority |
|-------|------------|----------|
| Issue 1 (mean-pooling) | 🔴 HIGH (85%) | CRITICAL — fix first |
| Issue 2 (no validation) | 🟠 MEDIUM (60%) | HIGH — secondary |
| Issue 3 (data mislabel) | 🟡 LOW (30%) | MEDIUM — unlikely |
| Issue 4 (eigenvalues) | 🟡 LOW (25%) | LOW — cPCA math probably OK |

---

## Success Criteria

After fix is applied, Phase 1 test should show:
- ✓ Bypass rate ≥ 80% (expected 100%)
- ✓ False positive rate ≤ 5%
- ✓ Speed overhead ≤ 1% (measured at 0.6%)
- ✓ Reasoning tasks unaffected (measured at +2% MMLU, 0% GSM8K)

---

## Timeline

- **Now**: Run `debug_phase0.py` on cloud (5 min)
- **After diagnosis**: Apply fix (30-60 min depending on issue)
- **Retest**: Phase 1 run with fixed Phase 0 (15 min)
- **Verify**: Check bypass rate achieves ≥80% (pass/fail immediately)

**Total**: 50-120 minutes to resolution

---

## Next Action

**➡️ Please run on cloud**:
```bash
python debug_phase0.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

**Then share the output** and I'll:
1. Identify exact root cause from diagnostic results
2. Provide specific fix code
3. Guide Phase 0 re-run with correct parameters
4. Help retest Phase 1 to verify bypass works

All materials ready:
- ✅ debug_phase0.py (comprehensive diagnostic)
- ✅ PHASE0_DEBUG_FINDINGS.md (detailed analysis)
- ✅ PHASE1_RESULTS.md (updated status)

**Waiting for your cloud test results.** 🚀
