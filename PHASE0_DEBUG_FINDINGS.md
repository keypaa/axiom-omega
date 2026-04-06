# AXIOM Phase 0 Debug - Critical Findings

**Date**: April 6, 2026  
**Status**: 4 Critical Issues Identified

---

## Summary

After analyzing Phase 0 code and comparing with Phase 1 failures, I found **4 critical issues** that likely explain the 0% bypass rate:

1. **Mean-pooling activations over ALL tokens** (instead of last token)
2. **No validation of probe generalization** (distribution shift not detected)
3. **Potential alignment data mislabeling** (refusal/compliant similarity not checked)
4. **cPCA eigenvalue concerns** (PR=5.63 may indicate noise)

---

## Issue 1: Mean-Pooling Over All Tokens (CRITICAL)

### Problem
Phase 0 collects activations by **mean-pooling over all token positions** (line 75 in phase0_compute_refusal_subspace.py):

```python
def make_hook(layer_idx):
    def fn(module, inp, out):
        h = out[0].detach().float()  # [batch, seq, d_model]
        if h.dim() == 3:
            pooled = h.mean(dim=1)  # Mean over ALL tokens → [batch, d_model]
```

But in Phase 1, the probe is applied **token-by-token during autoregressive generation**, potentially at very different activation distributions.

### Why This Breaks Refusal Detection
- **Phase 0 training**: Probe learns on mean-pooled h (all tokens averaged)
- **Phase 1 inference**: Probe sees individual token activations (at generation step t)
- **Distribution shift**: Mean-pooled activations ≠ individual token activations
  - Mean smooths out signal
  - Peak refusal activation (at the refusal decision token) is diluted

### Evidence
- Probes train to 92-98% accuracy on Phase 0 data ✓ (because mean-pooling was consistent)
- But 0% bypass in Phase 1 ❌ (because individual tokens have different distribution)

### Test This
Run `debug_phase0.py` to check if activations look reasonable. But the real fix requires:

**Solution**: Modify phase0_compute_refusal_subspace.py to use **last-token activation** instead:

```python
# Instead of:
pooled = h.mean(dim=1)  # Mean all tokens

# Use:
pooled = h[:, -1, :]  # Last token only
```

This matches how Phase 1 uses activations during generation.

---

## Issue 2: No Probe Generalization Validation

### Problem
Probe training (phase1_train_and_test.py lines 62-90) trains on Phase 0 activations with **no distribution checks**:

```python
X = torch.cat([h_r, h_c], dim=0)
y = torch.cat([torch.ones(len(h_r)), torch.zeros(len(h_c))], dim=0)
idx = torch.randperm(len(X))
X, y = X[idx], y[idx]
split = int(0.9 * len(X))
X_train, y_train = X[:split], y[:split]
X_val, y_val = X[split:], y[split:]
```

**No validation** of:
- Train/val distribution similarity (KL divergence)
- Temporal consistency (if data has time ordering)
- Representation shift between training and generation

### Evidence
- Training accuracy: 92-98% (looks good on validation set)
- Generation accuracy: 0% (completely broken on new distribution)

### Solution
Add distribution checks before declaring probe "trained":

```python
from scipy.stats import ks_2samp

# After training...
with torch.no_grad():
    h_r_embedded = probe.linear(X_train[y_train == 1])
    h_c_embedded = probe.linear(X_train[y_train == 0])
    
    # KS test
    ks_stat, ks_pval = ks_2samp(
        h_r_embedded.abs().mean(dim=1).cpu().numpy(),
        h_c_embedded.abs().mean(dim=1).cpu().numpy()
    )
    
    if ks_pval < 0.05:
        print(f"✓ Probe separates classes: KS p={ks_pval:.4f}")
    else:
        print(f"⚠️ Probe does NOT separate: KS p={ks_pval:.4f}")
```

---

## Issue 3: Alignment Data May Be Mislabeled

### Problem
Phase 0 loads alignment data with NO VERIFICATION:

```python
refusal_files = list(align_dir.glob("*refusal*.jsonl")) + list(align_dir.glob("*harmful*.jsonl"))
compliant_files = list(align_dir.glob("*compliant*.jsonl"))

refusal_prompts = []
for f in refusal_files:
    refusal_prompts += [r["text"] for r in load_jsonl(f)]
```

Assumptions:
- Files named `*refusal*.jsonl` actually contain refusal-triggering prompts
- Files named `*compliant*.jsonl` actually contain benign prompts
- AdvBench prompts actually trigger refusals (may not on Llama 3.1)

### Why This Could Fail
1. AdvBench designed for older models; may not trigger refusal on Llama 3.1
2. Alpaca/OpenAssistant may not be perfectly "compliant" (some could trigger refusals)
3. **Label leakage**: If both datasets have similar distributions, no refusal signal exists

### Test This
Run `debug_phase0.py` which:
- Prints first 10 refusal/compliant prompts (manual inspection)
- Computes KS test on prompt lengths
- Checks cosine similarity of activation means

If means are too similar (cos_sim > 0.9), data is **not separable**.

### Solution
Verify alignment data by running prompts through model and checking actual outputs:

```python
# Quick validation: do these prompts actually trigger different behaviors?
for prompt in refusal_prompts[:5]:
    # Run through model, check if it refuses
    output = model.generate(tokenizer(prompt)["input_ids"])
    if "[REFUSED]" not in tokenizer.decode(output):
        print(f"WARNING: '{prompt[:50]}...' did NOT trigger refusal!")
```

---

## Issue 4: cPCA Eigenvalue Quality (PR=5.63)

### Problem
Phase 0 reports PR=5.63 for Layer 31, which is "Low-rank" (not rank-1, but low):

```
Layer 31: PR=5.63 (Low-rank), k=7 directions
```

Participation Ratio measures how concentrated variance is. Low PR may indicate:
- **Signal**: A few strong refusal directions (good)
- **Noise**: Variance spread unevenly due to poor alignment data (bad)

### Theory
- **PR < 2**: Rank-1 (one dominant direction) — strong, concentrated signal
- **2 < PR < 10**: Low-rank — mixed signal/noise
- **PR > 10**: Distributed — signal spread across many directions (diluted)

PR=5.63 is ambiguous. Could be either good or bad depending on data quality.

### Test This
In `debug_phase0.py`, check:
1. Direction norms: Are they reasonably large?
2. Direction orthogonality: Are eigenvectors orthogonal (as expected)?
3. Gram matrix: Off-diagonal << diagonal?

If any fail, cPCA math may be broken.

---

## Recommended Debug Protocol

**Run on cloud in this order:**

### Step 1: Run diagnostic script (5 min)
```bash
python debug_phase0.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

This will tell you:
- ✓ Is alignment data well-separated?
- ✓ Are activation means different (cos_sim)?
- ✓ Are refusal directions reasonable (magnitude, orthogonality)?

### Step 2: If diagnosis says "alignment data similar" (10 min)
Verify that refusal prompts actually trigger refusals:

```python
# Quick test on first 5 AdvBench prompts
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

from phase0_collect_data import load_jsonl
refusal_data = load_jsonl("axiom_checkpoints/meta-llama-3-8b-instruct/phase_0/alignment/*refusal*.jsonl")

for item in refusal_data[:5]:
    prompt = item["text"]
    output = model.generate(tokenizer(prompt, return_tensors="pt")["input_ids"], max_new_tokens=20)
    response = tokenizer.decode(output[0])
    print(f"Prompt: {prompt[:60]}...")
    print(f"Response: {response[-100:]}\n")
```

If responses show compliance instead of refusal, **data is broken**.

### Step 3: If diagnosis says "activation means too similar" (30 min)
Try **last-token pooling** instead of mean-pooling:

In `phase0_compute_refusal_subspace.py`, line 75, change:
```python
pooled = h.mean(dim=1)  # BEFORE: mean over all tokens
```

To:
```python
pooled = h[:, -1, :]  # AFTER: last token only
```

Then re-run:
```bash
python phase0_compute_refusal_subspace.py --model meta-llama/Meta-Llama-3-8B-Instruct --skip_collection  # Reuse activation collection, just recompute cPCA
```

Actually, you'll need to recollect (remove --skip_collection):
```bash
python phase0_compute_refusal_subspace.py --model meta-llama/Meta-Llama-3-8B-Instruct  # Recollect with last-token pooling
```

Then test Phase 1 again.

### Step 4: If still failing after Step 3 (60 min)
Try different target layers. Layer 10/31 may not be optimal for Llama 3.1.

Recompute Fisher:
```bash
python phase0_compute_fisher.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

This will suggest new target layers. Then recompute refusal subspace.

---

## Root Cause Hypothesis

**Most likely**: Mean-pooling (Issue 1) + Distribution shift (Issue 2)

**Why**:
- Mean-pooling dilutes refusal signal across all tokens
- Probe trained on smoothed distribution, sees sharp distributions during generation
- Probe threshold (0.55) tuned for mean-pooled data, doesn't generalize

**Quick fix**: Use last-token pooling (30 min rerun of Phase 0)

**If that fails**: Check alignment data quality (Issue 3)

---

## Files to Modify

1. **phase0_compute_refusal_subspace.py** (line 75)
   - Change: `pooled = h.mean(dim=1)` → `pooled = h[:, -1, :]`

2. **phase1_train_and_test.py** (after line 121, add distribution check)
   - Add KS test to verify probe generalizes

3. **debug_phase0.py** (NEW)
   - Run diagnostic to identify which issue is happening

---

## Next Steps

1. ✅ Created debug_phase0.py (ready to run on cloud)
2. 📋 Updated PHASE1_RESULTS.md with current status
3. 🚀 **WAITING FOR YOU**: Run debug_phase0.py on cloud and share results

Once you run debug_phase0.py and share output, I can:
- Identify exact root cause
- Propose precise fix
- Guide re-run of Phase 0 with correct parameters
- Re-test Phase 1 to verify bypass works

---

**Next**: Please run on cloud:
```bash
python debug_phase0.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

And share the output.
