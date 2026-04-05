# AXIOM-Ω — Master Implementation Plan
*Triple-validated: Gemini · ChatGPT · Claude — April 2026*

---

## I. Validated Architecture Overview

AXIOM-Ω is a **runtime control kernel** that wraps any frozen base model and bends its latent
trajectory at inference time — no weight modification required. The unifying abstraction:

```
dh_t = f(h_t, t) dt  +  P_{S⊥} · π_θ(h_{<t}, t, M) dt  +  σ dW_t
       ↑                  ↑                                    ↑
   Base model         Anticipatory policy                 Stochasticity
   (frozen)           (learned, tiny MLP)                 (sampling temp)
```

Objective (trained via PPO or Soft Actor-Critic):

```
min_θ  E_τ [ ∫₀ᵀ ( E_task(h_t, t)  +  α·KL(P_base ‖ P_ctrl)  +  β‖π_θ‖² ) dt ]
              ↑                           ↑                         ↑
          Fiber energy            Identity constraint          Minimal edit
```

**Three fixed invariants enforced at all times:**
1. `Δh ⊥ S` — all interventions projected out of the Sacred Subspace
2. `KL(router_orig ‖ router_mod) < ε` — MoE routing stability
3. `‖Δh‖ < δ_layer` — per-layer magnitude cap (prevents attention cascade)

---

## II. Validated Corrections (applied before implementation)

| Issue | Original Claim | Correction |
|-------|---------------|-----------|
| MP threshold | Marchenko-Pastur gives principled rank cutoff | MP assumes IID — use data-adaptive rank (cross-validated SVD or nuclear norm). MP is a *lower bound estimate* only. |
| Kalman framing | "Latent Kalman Interceptor" = controller | Kalman/EKF = **state estimator**. Pair with an MPC or LQR as the actual controller. Phase 3 = EKF (estimator) + MPC (controller). |
| Router nullspace formula | `Δh - (Δh·W^T)(WW^T)^{-1}W` | Correct: `Δh - W^T(WW^T)^{-1}W·Δh`. Dimensions are transposed in original — would fail at runtime. |
| KV-cache virus | "orthogonal to vocabulary subspace → invisible" | Wrong. Must inject into null space of the **unembedding matrix** `W_U`, not vocabulary space. Correct: `v_inject ∈ null(W_U)`. |
| PR threshold | PR < 5 → Rank-1, PR > 5 → OT | These are hyperparameters, not constants. Calibrate per model with a 50-prompt validation set. Use relative threshold: top-k components explaining ≥ 95% variance. |
| Fiber bundle framing | Adds computational content | Analogy only. Implementation = `E(h, t)` conditioned on layer index. No additional machinery needed. |

---

## III. Phased Implementation Roadmap

### Phase 0 — Calibration Infrastructure (prerequisite for everything)
**Duration:** ~1 week  
**Goal:** Build the data pipelines that all subsequent phases depend on.

#### 0.1 Dual Dataset Construction
Two datasets, assembled once per target model:

**Refusal dataset** (`D_r`): ~500 prompts that reliably trigger refusal.
- Sources: AdvBench, custom adversarial templates, jailbreak benchmarks
- Required fields: prompt, layer activations at every target layer (saved to disk)

**Compliant dataset** (`D_c`): ~500 prompts that never trigger refusal but cover diverse topics.
- Sources: MMLU, GSM8K, instruction-following benchmarks, coding tasks
- Required fields: prompt, layer activations, task category label

**Sacred Subspace calibration set** (`D_s`): ~200 prompts specifically designed to stress
reasoning, math, and instruction-following.
- Purpose: defines what *must not be disturbed*
- Required coverage: arithmetic, multi-step logic, code generation, factual recall

#### 0.2 Activation Collection Pipeline
```python
class ActivationCollector:
    def __init__(self, model, target_layers: list[int]):
        self.hooks = {}
        for layer_idx in target_layers:
            hook = model.model.layers[layer_idx].register_forward_hook(
                self._make_hook(layer_idx)
            )
            self.hooks[layer_idx] = hook
    
    def _make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            # Store post-attention residual stream
            self.activations[layer_idx] = output[0].detach().cpu()
        return hook_fn
    
    def collect(self, dataset, batch_size=8) -> dict[int, torch.Tensor]:
        # Returns {layer_idx: tensor of shape [N, seq_len, d_model]}
        ...
```

**Critical:** collect activations at the *residual stream output* (post-attention + post-FFN),
not at attention logits. This is the canonical intervention point.

#### 0.3 Layer Selection (Jacobian-based)
Don't naively compute full Jacobians. Use Hutchinson's estimator:

```python
def estimate_layer_importance(model, D_r, n_samples=32):
    """
    Returns importance score per layer using Hutchinson's diagonal Fisher estimate.
    Cost: n_samples backward passes (not N×L).
    """
    scores = defaultdict(float)
    for prompt in D_r[:n_samples]:
        output = model(prompt, output_hidden_states=True)
        refusal_logit = output.logits[0, -1, refusal_token_id]
        refusal_logit.backward()
        
        for layer_idx, layer in enumerate(model.model.layers):
            grad = layer.mlp.down_proj.weight.grad
            scores[layer_idx] += (grad ** 2).sum().item()  # diagonal Fisher approx
    
    return dict(sorted(scores.items(), key=lambda x: -x[1]))
    # Select top-k layers (typically k=3-5 is sufficient)
```

**Target layer selection rule:** Top 3 layers by Fisher score, with the constraint that
they span early/mid/late thirds of the network. One layer per third prevents correlated edits.

---

### Phase 1 — Minimal Viable Control
**Duration:** ~1 week  
**Goal:** Prove that trajectory steering works without weight modification. Nothing fancy.

#### 1.1 Refusal Probe (linear, cheap)
```python
class RefusalProbe(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, 1)
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [batch, seq_len, d_model]
        # Use last token position as refusal signal
        return torch.sigmoid(self.linear(h[:, -1, :]))

# Training: binary cross-entropy on D_r (label=1) vs D_c (label=0)
# Takes ~5 minutes on a single GPU
```

#### 1.2 Refusal Direction (cPCA, not mean-diff)
```python
def compute_refusal_direction(h_r: np.ndarray, h_c: np.ndarray, 
                               n_components: int = None) -> np.ndarray:
    """
    Contrastive PCA: find directions maximizing refusal variance
    while minimizing compliant variance.
    
    Args:
        h_r: [N, d] refusal activations (mean-pooled over seq)
        h_c: [N, d] compliant activations
        n_components: if None, auto-select via relative PR threshold
    
    Returns:
        directions: [n_components, d] refusal subspace
    """
    Sigma_r = np.cov(h_r.T)
    Sigma_c = np.cov(h_c.T)
    
    # Participation Ratio for auto-routing (calibrated threshold)
    eigvals_r = np.linalg.eigvalsh(Sigma_r)[::-1]
    PR = (eigvals_r.sum() ** 2) / (eigvals_r ** 2).sum()
    
    # Relative threshold: components explaining ≥95% of refusal variance
    cumvar = np.cumsum(eigvals_r) / eigvals_r.sum()
    auto_k = np.searchsorted(cumvar, 0.95) + 1
    n_components = n_components or auto_k
    
    # Generalized eigenvalue problem: (Sigma_r - lambda * Sigma_c) v = 0
    # Solved via scipy for numerical stability
    from scipy.linalg import eigh
    eigvals, eigvecs = eigh(Sigma_r, Sigma_c + 1e-6 * np.eye(Sigma_r.shape[0]),
                             subset_by_index=[Sigma_r.shape[0]-n_components, 
                                              Sigma_r.shape[0]-1])
    directions = eigvecs[:, ::-1].T  # [n_components, d]
    
    return directions, PR, n_components
```

#### 1.3 Runtime Hook (Phase 1 logic)
```python
class Phase1Hook:
    def __init__(self, probe, directions, lambda_scale=1.0, threshold=0.5):
        self.probe = probe
        self.directions = torch.tensor(directions)  # [k, d]
        self.lambda_scale = lambda_scale
        self.threshold = threshold
    
    def __call__(self, module, input, output):
        h = output[0]  # [batch, seq, d]
        
        p_refuse = self.probe(h)  # [batch, 1]
        
        if p_refuse.max() > self.threshold:
            # Project activations onto refusal subspace and subtract
            # h_proj = h · D^T · D  (refusal component)
            D = self.directions.to(h.device)
            h_refusal = (h @ D.T) @ D  # [batch, seq, d]
            h = h - self.lambda_scale * h_refusal
            return (h,) + output[1:]
        
        return output
```

**Phase 1 success criteria:**
- Model responds to ≥70% of refusal-triggering prompts without refusing
- MMLU score drops by ≤3% on `D_s`
- No noticeable incoherence on simple queries

**Expected failures:** Overcorrection (weird outputs), undercorrection (still refuses at lower rate).
This is expected. Log both failure modes — they define the control surface.

---

### Phase 2 — Sacred Subspace + Stability
**Duration:** ~2 weeks  
**Goal:** Add the projector `P_{S⊥}` so aggressive steering doesn't break reasoning.

#### 2.1 Sacred Subspace Construction (Multi-View)
```python
def build_sacred_subspace(model, D_s: Dataset, n_fisher=50, n_causal=100):
    """
    S = S_fisher ∩ S_causal ∩ S_semantic
    
    In practice: compute each basis, concatenate, then find the shared
    subspace via SVD on the concatenated matrix.
    """
    
    # Fisher component: gradient sensitivity
    S_fisher = compute_fisher_eigenvectors(model, D_s, n_samples=n_fisher, top_k=32)
    # shape: [32, d]
    
    # Causal component: activation patching sensitivity
    S_causal = compute_causal_directions(model, D_s, n_probes=n_causal, top_k=16)
    # shape: [16, d]
    # Method: perturb h at layer l by random Gaussian, measure logit delta
    # Directions with high logit sensitivity = causal
    
    # Semantic component: embedding-space clustering of critical concepts
    # (math, logic, factual recall → cluster centers define semantic axes)
    S_semantic = compute_semantic_axes(model, D_s, n_clusters=8)
    # shape: [8, d]
    
    # Combine: SVD on the concatenated basis
    S_combined = np.concatenate([S_fisher, S_causal, S_semantic], axis=0)  # [56, d]
    U, s, Vt = np.linalg.svd(S_combined, full_matrices=False)
    
    # Take top components explaining 90% of combined variance
    cumvar = np.cumsum(s**2) / (s**2).sum()
    k = np.searchsorted(cumvar, 0.90) + 1
    S = Vt[:k]  # [k, d] — the Sacred Subspace basis
    
    # Projector: P = I - S^T S
    P = np.eye(S.shape[1]) - S.T @ S
    
    return S, P
```

#### 2.2 Updated Hook with Projector
```python
class Phase2Hook(Phase1Hook):
    def __init__(self, *args, projector: np.ndarray, **kwargs):
        super().__init__(*args, **kwargs)
        self.P = torch.tensor(projector)  # [d, d]
    
    def __call__(self, module, input, output):
        h = output[0]
        p_refuse = self.probe(h)
        
        if p_refuse.max() > self.threshold:
            D = self.directions.to(h.device)
            P = self.P.to(h.device)
            
            # Raw correction
            delta_h = -(h @ D.T) @ D * self.lambda_scale
            
            # Project through Sacred Subspace: force delta_h ⊥ S
            safe_delta_h = delta_h @ P  # [batch, seq, d]
            
            h = h + safe_delta_h
            return (h,) + output[1:]
        
        return output
```

**Phase 2 success criteria:**
- MMLU drop ≤1% (vs ≤3% in Phase 1)
- GSM8K (math reasoning) drop ≤2%
- Refusal bypass rate maintained at ≥70%

#### 2.3 Weight Editing Fallback (if hooks are too slow)
If inference overhead is unacceptable, bake the Phase 2 correction into weights:

```python
def bake_correction_into_weights(model, layer_idx, directions, projector, 
                                  X_activations: torch.Tensor):
    """
    Pseudoinverse route: find ΔW such that the layer produces the corrected output.
    Uses data-adaptive rank (cross-validated SVD, NOT Marchenko-Pastur).
    """
    # X: [N*seq, d_in] input activations to the target layer
    X = X_activations.reshape(-1, X_activations.shape[-1])
    
    # Cross-validated SVD for rank selection
    from sklearn.decomposition import TruncatedSVD
    from sklearn.model_selection import cross_val_score
    
    best_k, best_score = 1, -np.inf
    for k in range(1, min(50, X.shape[1])):
        svd = TruncatedSVD(n_components=k)
        score = cross_val_score(svd, X, cv=3).mean()
        if score > best_score:
            best_k, best_score = k, score
    
    U, s, Vt = torch.linalg.svd(X, full_matrices=False)
    # Truncate at cross-validated rank
    U, s, Vt = U[:, :best_k], s[:best_k], Vt[:best_k, :]
    
    # Regularized pseudoinverse: X+ = Vt^T diag(1/s) U^T
    s_inv = 1.0 / (s + 1e-6)  # small ridge
    X_pinv = Vt.T @ torch.diag(s_inv) @ U.T  # [d_in, N*seq]
    
    # Desired output correction (post-projector)
    delta_h = compute_desired_correction(directions, projector)  # [N*seq, d_out]
    
    # Weight update
    delta_W = delta_h.T @ X_pinv.T  # [d_out, d_in]
    
    target_layer = model.model.layers[layer_idx].mlp.down_proj
    target_layer.weight.data += delta_W
```

---

### Phase 3 — Dynamic Control (EKF State Estimator + MPC Controller)
**Duration:** ~3 weeks  
**Goal:** Replace static vectors with a learned policy that *anticipates* refusal.

**Clarification from validation:** Phase 3 has two distinct components:
- **EKF (state estimator):** tracks the latent state's probability of entering the refusal region
- **MPC (controller):** given the EKF estimate, computes the minimal intervention to stay compliant

#### 3.1 EKF State Estimator
```python
class RefusalStateEstimator(nn.Module):
    """
    Extended Kalman Filter approximation via a tiny MLP.
    Input: current hidden state h_t
    Output: estimated probability of refusal at layer T (terminal)
    
    This is NOT the controller — it's the sensor.
    """
    def __init__(self, d_model: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 2)  # [p_refuse, uncertainty]
        )
    
    def forward(self, h: torch.Tensor):
        # h: [batch, d_model] — last token position
        out = self.net(h)
        p_refuse = torch.sigmoid(out[:, 0])
        uncertainty = torch.sigmoid(out[:, 1])
        return p_refuse, uncertainty
```

#### 3.2 MPC Controller (minimal intervention)
```python
class LatentMPCController(nn.Module):
    """
    Given EKF estimate, compute minimal correction.
    Outputs: Δh correction AND low-rank attention mask (U, V)
    """
    def __init__(self, d_model: int, rank_k: int = 4, hidden: int = 256):
        super().__init__()
        self.residual_head = nn.Sequential(
            nn.Linear(d_model + 1, hidden),  # h + p_refuse
            nn.SiLU(),
            nn.Linear(hidden, d_model)
        )
        self.attn_U = nn.Linear(d_model + 1, rank_k, bias=False)
        self.attn_V = nn.Linear(d_model + 1, rank_k, bias=False)
        self.rank_k = rank_k
    
    def forward(self, h: torch.Tensor, p_refuse: torch.Tensor):
        x = torch.cat([h, p_refuse.unsqueeze(-1)], dim=-1)
        
        delta_h = self.residual_head(x)  # [batch, d_model]
        U = self.attn_U(x)  # [batch, rank_k]
        V = self.attn_V(x)  # [batch, rank_k]
        
        # Attention mask: ΔM = U V^T (rank-k, clamped)
        # Applied as: Softmax(QK^T + ΔM) V_attn
        delta_M = torch.bmm(U.unsqueeze(2), V.unsqueeze(1))  # [batch, k, k]
        delta_M = torch.clamp(delta_M, -2.0, 2.0)  # prevent chaos
        
        return delta_h, delta_M
```

#### 3.3 Phase 3 Hook
```python
class Phase3Hook:
    def __init__(self, estimator, controller, projector, sacred_subspace, 
                 threshold=0.4, magnitude_cap=0.1):
        self.est = estimator
        self.ctrl = controller
        self.P = torch.tensor(projector)
        self.threshold = threshold
        self.cap = magnitude_cap
    
    def __call__(self, module, input, output):
        h = output[0]  # [batch, seq, d]
        
        # State estimation (last token)
        p_refuse, uncertainty = self.est(h[:, -1, :])
        
        if p_refuse.max() > self.threshold:
            # Control
            delta_h, delta_M = self.ctrl(h[:, -1, :], p_refuse)
            
            # Project through Sacred Subspace
            P = self.P.to(h.device)
            safe_delta = (delta_h @ P).unsqueeze(1)  # [batch, 1, d]
            
            # Magnitude cap (prevents destabilization)
            norm = safe_delta.norm(dim=-1, keepdim=True)
            safe_delta = safe_delta * torch.clamp(self.cap / norm, max=1.0)
            
            h = h + safe_delta
        
        return (h,) + output[1:]
```

#### 3.4 Training the Phase 3 Components

**Dataset:** Use the calibration sets from Phase 0. No new data required.

**Training loop:**
```python
# Reward signal
def compute_reward(response: str, original_prompt: str) -> float:
    r = 0.0
    if not is_refusal(response):          r += 1.0   # no refusal
    if is_coherent(response):             r += 0.5   # maintains coherence
    if not is_degraded(response):         r += 0.5   # no lobotomy
    if follows_format(response):          r += 0.3   # respects formatting
    return r

# PPO on the controller weights only.
# The base model and estimator are frozen during controller training.
# Estimator is trained separately (supervised) on collected trajectories.
```

**Training schedule:**
1. Train `RefusalStateEstimator` supervised on activation trajectories (3-5 epochs)
2. Freeze estimator. Train `LatentMPCController` via PPO (200-500 episodes)
3. Fine-tune jointly with lower LR

---

### Phase 4 — Anticipatory Memory
**Duration:** ~2 weeks  
**Goal:** Controller uses trajectory history to act *before* refusal emerges.

**Concrete decision (from validation):** Use a simple **exponential moving average** of
hidden states as memory `M_t`. Do NOT use an RNN yet — keep it cheap for the first version.

```python
class MemoryState:
    def __init__(self, alpha: float = 0.9, d_model: int = 4096):
        self.alpha = alpha
        self.M = None  # initialized on first call
    
    def update(self, h_t: torch.Tensor):
        # h_t: [batch, d_model]
        if self.M is None:
            self.M = h_t.clone()
        else:
            self.M = self.alpha * self.M + (1 - self.alpha) * h_t
    
    def get(self) -> torch.Tensor:
        return self.M

# Phase 4 controller input: cat([h_t, M_t], dim=-1)
# This doubles the input dimension — adjust MPC accordingly
```

**Upgrade path (Phase 4.5):** Replace EMA with a 2-layer GRU over the sequence of
`h_t` values. This gives genuine recurrent anticipation. GRU hidden size = 128 (cheap).

---

### Phase 5 — Attention Rewiring
**Duration:** ~2 weeks  
**Goal:** Sever specific attention paths rather than editing the residual stream.

**Implementation constraint:** Keep rank `k ≤ 8`. Larger ranks cause attention cascade
(small mask changes → global attention pattern shift). Clamp to `[-2, 2]`.

```python
def apply_attention_mask(attention_scores: torch.Tensor,
                          U: torch.Tensor, V: torch.Tensor,
                          refusal_token_ids: list[int]) -> torch.Tensor:
    """
    Injects low-rank bias into attention logits.
    Only applied when controller detects refusal trajectory.
    
    Preserves row normalization via softmax (no explicit renorm needed).
    """
    # delta_M: [batch, heads, seq, seq]
    batch, heads, seq, _ = attention_scores.shape
    
    # Low-rank delta: outer product of U and V (broadcast over heads)
    delta_M = torch.bmm(U.unsqueeze(2), V.unsqueeze(1))  # [batch, k, k]
    
    # Pad to full seq × seq
    delta_M_full = torch.zeros(batch, seq, seq, device=attention_scores.device)
    delta_M_full[:, :U.shape[1], :V.shape[1]] = delta_M
    
    # Broadcast over heads and add
    attention_scores = attention_scores + delta_M_full.unsqueeze(1) * 0.5
    
    return attention_scores
```

---

### Phase 6 — Full AXIOM-Ω (Composable Cartridges)
**Duration:** ~3-4 weeks  
**Goal:** Multiple swappable control policies. Runtime hot-swap without model reload.

#### 6.1 Cartridge Architecture
```python
@dataclass
class ControlCartridge:
    name: str
    estimator: RefusalStateEstimator
    controller: LatentMPCController
    projector: torch.Tensor          # Sacred Subspace projector
    energy_weights: dict[str, float]  # composable behavior weights
    target_layers: list[int]
    metadata: dict
    
    def save(self, path: str):
        torch.save(asdict(self), path)
    
    @classmethod
    def load(cls, path: str) -> 'ControlCartridge':
        return cls(**torch.load(path))

class AxiomKernel:
    def __init__(self, base_model):
        self.model = base_model
        self.active_cartridge: ControlCartridge | None = None
        self._hooks = []
    
    def mount(self, cartridge: ControlCartridge):
        """Hot-swap control policy without model reload."""
        self._remove_hooks()
        self.active_cartridge = cartridge
        self._install_hooks(cartridge)
    
    def unmount(self):
        self._remove_hooks()
        self.active_cartridge = None
    
    def generate(self, prompt: str, **kwargs):
        return self.model.generate(
            self.tokenizer(prompt, return_tensors='pt').input_ids,
            **kwargs
        )
```

#### 6.2 Composable Behavior Policies
```python
# Energy decomposition: u = Σ w_i ∇E_i
# Each E_i is a lightweight probe targeting a different behavior

STANDARD_CARTRIDGES = {
    "uncensored": {
        "E_refusal": 1.0,   # push away from refusal
        "E_logic":  -0.1,   # slight boost to logical coherence
        "E_creativity": 0.0
    },
    "logic_boost": {
        "E_refusal": 0.0,
        "E_logic":  1.0,    # enforce logical manifold
        "E_creativity": -0.3
    },
    "creative": {
        "E_refusal": 0.0,
        "E_logic":  -0.2,   # allow some incoherence
        "E_creativity": 1.0  # increase σ dW_t (sampling noise)
    },
    "safe_mode": {
        "E_refusal": -1.0,  # pull INTO refusal attractors
        "E_logic":  0.5,
        "E_creativity": 0.0
    }
}
```

---

## IV. Evaluation Framework

Every phase must be evaluated on all four axes before moving to the next.

| Axis | Metric | Tool | Threshold |
|------|--------|------|-----------|
| **Signal extraction** | Probe F1 on held-out prompts | Custom eval | ≥ 0.85 |
| **Intervention stability** | KL(base ‖ controlled) on D_s | Custom | ≤ 0.05 nats |
| **Latency** | Overhead vs baseline | `torch.profiler` | ≤ 1.5× |
| **Generalization** | Bypass rate on unseen prompts | AdvBench subset | ≥ 0.65 |

Additional benchmarks per phase:
- **Phase 1-2:** MMLU (general knowledge), GSM8K (math), HumanEval (code)
- **Phase 3+:** TruthfulQA (coherence), MT-Bench (instruction-following quality)
- **Phase 4+:** Long-context coherence (custom multi-turn eval)

---

## V. Architecture-Agnostic Strategy per Model Type

| Model type | Key challenge | AXIOM adaptation |
|------------|--------------|-----------------|
| Standard Transformer (Llama, Qwen, Mistral) | None — designed for this | Hook residual stream directly |
| MoE (Mixtral, DBRX, Grok) | Routing collapse | Router-nullspace constraint: `Δh_safe = Δh - W_r^T(W_rW_r^T)^{-1}W_r·Δh` |
| SSM (Mamba) | No attention mechanism | Hook state-update equation `h_t = Ah_{t-1} + Bx_t`. Edit B-matrix path. |
| Hybrid (Jamba) | Mixed attention + SSM layers | Apply different hook types per layer type. Auto-detected via module class names. |
| Gemma 4 (inhomogeneous layers) | Variable layer structure | Fingerprinter maps layer types → hook type. No hardcoded assumptions. |

**Architecture fingerprinter (handles inhomogeneous layers like Gemma 4):**
```python
def fingerprint_architecture(model) -> dict[int, str]:
    """
    Returns {layer_idx: layer_type} without relying on model class names.
    Detects by analyzing module graph topology.
    """
    layer_types = {}
    for idx, layer in enumerate(model.model.layers):
        has_attention = any('attn' in name.lower() or 'attention' in name.lower()
                           for name, _ in layer.named_modules())
        has_ssm = any('mamba' in name.lower() or 'ssm' in name.lower() or
                     'state_space' in name.lower()
                     for name, _ in layer.named_modules())
        has_moe = any('expert' in name.lower() or 'router' in name.lower()
                     for name, _ in layer.named_modules())
        
        if has_moe:
            layer_types[idx] = 'moe'
        elif has_ssm and has_attention:
            layer_types[idx] = 'hybrid'
        elif has_ssm:
            layer_types[idx] = 'ssm'
        else:
            layer_types[idx] = 'transformer'
    
    return layer_types
```

---

## VI. Known Limits (Theoretical)

These are **unavoidable** — don't fight them, engineer around them.

1. **Safety-reasoning entanglement:** When `Σ_r` and `Σ_c` are nearly collinear (same
   circuit processes dangerous and benign concepts), AXIOM cannot separate them without
   capability loss. Detection: if `cosine(refusal_direction, sacred_direction) > 0.7`,
   flag the model as "entangled" and use weaker `λ`.

2. **High-curvature manifolds:** Deep RLHF models (Claude, GPT-4 class) have non-Gaussian,
   high-curvature refusal manifolds. The Bures-Wasserstein closed form overshoots. Use
   Sinkhorn OT fallback (5-10 iterations) when `PR > 15`.

3. **Causal entanglement in polysemantic neurons:** Some neurons fire for both "bomb" and
   "chemical reaction" contexts. Transient SAE can separate these if the sparse codes are
   distinct. If FastICA returns components with correlation > 0.5, the separation failed —
   fall back to global cPCA.

---

## VII. Repository Structure (when building)

```
axiom/
├── core/
│   ├── kernel.py           # AxiomKernel wrapper
│   ├── hooks.py            # Phase 1/2/3 hook implementations
│   └── fingerprint.py      # Architecture auto-detection
├── probe/
│   ├── collector.py        # Activation collection pipeline
│   ├── cpca.py             # Contrastive PCA implementation
│   └── sacred_subspace.py  # Multi-view invariant subspace
├── control/
│   ├── estimator.py        # EKF state estimator (tiny MLP)
│   ├── controller.py       # MPC controller
│   └── memory.py           # EMA / GRU memory state
├── cartridge/
│   ├── cartridge.py        # ControlCartridge dataclass
│   ├── standard.py         # Built-in cartridges
│   └── composer.py         # Energy composition (Σ w_i ∇E_i)
├── eval/
│   ├── benchmarks.py       # MMLU, GSM8K, HumanEval wrappers
│   ├── stability.py        # KL divergence, coherence metrics
│   └── bypass_rate.py      # Refusal bypass evaluation
└── examples/
    ├── llama3_quickstart.py
    ├── mixtral_moe.py
    └── gemma4_inhomogeneous.py
```

---

## VIII. Build Order Summary

```
Week 1:   Phase 0 — Dataset + activation collection pipeline
Week 2:   Phase 1 — Linear probe + cPCA + runtime hook
Week 3:   Phase 2 — Sacred subspace + projector + weight-bake fallback
Week 4-5: Phase 3 — EKF estimator + MPC controller + PPO training
Week 6:   Phase 4 — EMA memory → anticipatory control
Week 7:   Phase 5 — Low-rank attention rewiring
Week 8+:  Phase 6 — Cartridge system + composable policies + RL

Stop condition: ship after Phase 3 if it works.
Phase 3 alone is novel, useful, and publishable.
```

---

*Master plan validated April 2026. Corrections integrated from three-way review.*
*Next step: begin Phase 0 data collection.*

---

## IX. Three-Way Validation Addendum
*Gemini 2.5 Pro · ChatGPT · Claude — April 2026*

### Locked Decisions (supersede any earlier ambiguity)

#### Rank Selection (final)
- **Refusal subspace:** 95% explained variance threshold
- **Sacred Subspace:** 99% explained variance threshold (higher fidelity — this is what you protect)
- **Marchenko-Pastur:** removed entirely from codebase
- **Bootstrap stability:** recommended validation step, not blocking

```python
def adaptive_rank(singular_values: torch.Tensor, threshold: float = 0.95) -> int:
    """threshold=0.95 for refusal dirs, threshold=0.99 for Sacred Subspace"""
    explained = torch.cumsum(singular_values**2, dim=0) / (singular_values**2).sum()
    k = int((explained < threshold).sum()) + 1
    return max(1, min(k, len(singular_values)))
```

#### Sacred Subspace Dataset (final, reproducible)
- 200 sequences × WikiText-103 (syntax, grammar, factual recall)
- 200 sequences × GSM8K (multi-step arithmetic)
- 200 sequences × CodeAlpaca (structured/algorithmic reasoning)
- 100 sequences × mC4 (multilingual coverage — prevents English-only subspace)
- Fixed sequence length: **1024 tokens** (captures long-range dependencies)
- Rank: top-k EFIM eigenvectors explaining **99%** of variance

#### Memory M (final build order)
- **Phase 3:** EMA (`α=0.9`) — simple, debuggable, proves hook system works
- **Phase 4 entry:** drop-in swap to Micro-GRU (Gemini spec: d_model → 64 → d_model, 1 layer)
- The module interface is identical — swap requires changing one line

```python
# Phase 3
class EMAMemory:
    def update(self, h): self.M = 0.9 * self.M + 0.1 * h
    def get(self): return self.M

# Phase 4 (drop-in replacement, identical interface)
class MicroGRUMemory(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gru = nn.GRUCell(d_model, 64)
        self.proj = nn.Linear(64, d_model)
    def update(self, h):
        self.hidden = self.gru(h, self.hidden)
        self.M = self.proj(self.hidden)
    def get(self): return self.M
```

#### PPO Curriculum (merged, final)

| Phase | Epochs | α (KL) | Task E | β (size) | w_refusal | w_coherence | w_KL | w_size |
|-------|--------|--------|--------|----------|-----------|-------------|------|--------|
| Identity lock | 0–100 | 10.0 | 0.0 | 0.0 | 0.0 | 3.0 | 1.0 | 0.0 |
| Trajectory bending | 100–500 | 5.0 | 1.0 | 0.1 | 2.0 | 1.0 | 0.5 | 0.0 |
| Minimal intervention | 500–1000 | 5.0 | 1.0 | 2.0 | 2.0 | 1.0 | 0.5 | 1.0 |

Collapse-prevention: if `u_t → 0` for >10 consecutive episodes, reset β to 0.1 and restart Phase 2.

#### Router Nullspace (final, numerically stable)
All three reviews agreed on the corrected formula. Complete safe implementation:

```python
def router_safe_projection(delta_h: torch.Tensor, W_router: torch.Tensor,
                            eps: float = 1e-4) -> torch.Tensor:
    """
    W_router: [n_experts, d_model]
    delta_h:  [batch, d_model]
    eps: Tikhonov regularization (handles rank-deficient routers in small MoEs)
    """
    gram = W_router @ W_router.T + eps * torch.eye(
        W_router.shape[0], device=W_router.device
    )
    rhs = W_router @ delta_h.T           # [n_experts, batch]
    coeff = torch.linalg.solve(gram, rhs) # [n_experts, batch]
    proj = W_router.T @ coeff             # [d_model, batch]
    return delta_h - proj.T               # [batch, d_model]
```

#### Terminology (final, frozen)
- ❌ "Latent Kalman Interceptor" → ✅ "Latent MPC Controller" + optional "State Estimator"
- ❌ "Fiber Bundle" → ✅ `E(h, t)` (layer-conditioned energy)
- ❌ "Marchenko-Pastur threshold" → ✅ "adaptive rank via explained variance"
- ❌ "KV-cache orthogonal virus" → ✅ "Null(W_U) prefix injection"

### What All Three Reviews Confirmed (freeze these, stop revisiting)
- SDE framing as conceptual foundation ✅
- cPCA via generalized eigenvalue problem `(Σ_r - λΣ_c)` ✅
- Pseudoinverse route with regularization ✅
- Sacred Subspace projector `P = I - S^T S` ✅
- Intervention magnitude cap per layer ✅
- OT (Sinkhorn) as optional advanced module for PR > 15 ✅

---
*Document frozen. Next action: Phase 0 data collection.*

---

## X. Four-Way Validation Addendum
*ChatGPT · Gemini · Claude (Round 2) — April 2026*

### Confirmed Deletions

**❌ KV-Cache Prefix Injection / "Null(W_U)" — DELETED**
Fatal linear algebra flaw (Gemini, confirmed). `W_U` is `[V, d]` with `V >> d`
(e.g. Llama 3: V=128,256, d=4,096). By the Rank-Nullity theorem, W_U has full
column rank → `Null(W_U) = {0}`. No non-trivial injection vector exists.
Phase 3 residual stream editing covers all use cases.

**❌ Multi-view Sacred Subspace concatenation (Fisher + Causal + Semantic) — REVISED**
Concatenating basis sets of different sizes (32 Fisher + 16 Causal + 8 Semantic)
implicitly weights Fisher 2× in the subsequent SVD. Build Sacred Subspace from
EFIM alone at 99% threshold on the calibration triad. Causal/semantic coverage
is achieved through dataset composition, not basis concatenation.

**❌ Attention mask pasting `[k,k]` into `[:k,:k]` of `[seq,seq]` — FIXED (not deleted)**
Pasting a k×k matrix into the top-left corner edits only first-k token mutual
attention regardless of refusal token positions. Also breaks FlashAttention.
Phase 5 is retained but implemented correctly (see below).

### Confirmed Fixes

**Sacred Subspace Dataset (final, expanded):**
- 200 × WikiText-103
- 200 × GSM8K
- 200 × CodeAlpaca
- 100 × mC4 (multilingual)
- 100 × MMLU hard subset (broad knowledge reasoning)
- 100 × long-context tasks (≥2k tokens)
- 50 × adversarial/ambiguous prompts
- Fixed length: 1024 tokens. Total: 950 sequences.

**Sacred Subspace Construction (final):**
Use EFIM only. No concatenation of multiple basis types.
```python
def build_sacred_subspace_efim(model, calibration_dataset, threshold=0.99):
    fisher_diag = defaultdict(lambda: torch.zeros(d_model))
    for prompt in calibration_dataset:
        output = model(prompt)
        log_prob = output.logits.log_softmax(-1)[:, -1, :].max()
        log_prob.backward()
        for layer_idx, layer in enumerate(model.model.layers):
            grad = layer.self_attn.o_proj.weight.grad
            fisher_diag[layer_idx] += grad.pow(2).sum(dim=0)  # diagonal EFIM
    # Per-layer: SVD on the outer product approximation, take 99% threshold
    # Returns S: [k, d] Sacred Subspace basis, P: [d, d] projector
```
Rename all functions from "hutchinson" to "empirical_fisher_diagonal".

**EMA → Micro-GRU transition (distillation required):**
```python
# Step 1: distill — train GRU to match EMA outputs
distill_loss = F.mse_loss(gru_memory.get(), ema_memory.get())
# Step 2: validate — check policy output distribution before/after swap
kl_shift = F.kl_div(policy(h, gru_M), policy(h, ema_M), reduction='batchmean')
assert kl_shift < 0.01, "Memory swap destabilizes policy — distill more"
# Step 3: only then resume PPO fine-tuning from current checkpoint
```

**KL Identity Constraint — approximation schedule:**
- Phase 1-2: `||h_ctrl - h_base||²` (L2, crude but stable and fast)
- Phase 3+: `KL(N(μ_ctrl,Σ_ctrl) || N(μ_base,Σ_base))` (closed-form Gaussian KL)
- Full EBM/score model: optional advanced module, not in critical path

```python
def identity_constraint(h_ctrl, h_base, phase: int) -> torch.Tensor:
    if phase < 3:
        return F.mse_loss(h_ctrl, h_base)
    else:
        # Closed-form Gaussian KL
        mu_diff = h_ctrl.mean(0) - h_base.mean(0)
        var_ctrl = h_ctrl.var(0) + 1e-6
        var_base = h_base.var(0) + 1e-6
        kl = 0.5 * ((var_ctrl/var_base) + mu_diff**2/var_base
                    - 1 + (var_base/var_ctrl).log())
        return kl.mean()
```

**Attention Rewiring — correct FlashAttention-compatible implementation:**
```python
def compute_attention_bias(
    h_t: torch.Tensor,           # [batch, seq, d] — current hidden states
    refusal_probe: nn.Module,    # linear probe → refusal probability per token
    gamma: float = 10.0          # soft suppression strength, NOT -inf
) -> torch.Tensor:
    """
    Returns additive bias [batch, 1, seq, seq] compatible with FlashAttention.
    High refusal-prob query positions are suppressed from attending to
    high refusal-prob key positions.
    Semantically aware: U and V are derived from actual refusal scores.
    """
    # Score each token position for refusal probability
    scores = refusal_probe(h_t).squeeze(-1)  # [batch, seq]
    scores = torch.sigmoid(scores)
    
    # Outer product: suppress refusal-query → refusal-key attention
    bias = -gamma * torch.bmm(scores.unsqueeze(2), scores.unsqueeze(1))
    # [batch, seq, seq] → [batch, 1, seq, seq] (broadcast over heads)
    return bias.unsqueeze(1)

# Usage (additive, not mask replacement):
# attn_scores = attn_scores + compute_attention_bias(h_t, probe, gamma=10.0)
```

**Temporal credit assignment (layer-wise advantage estimation):**
```python
# Treat layer depth as time steps in RL
# Layer-wise rewards: penalize refusal signal at each layer
def layer_rewards(refusal_probs: list[float], gamma_depth: float = 0.95) -> list[float]:
    """
    refusal_probs: [p_layer_0, p_layer_1, ..., p_layer_L]
    Returns per-layer advantage estimates via TD(λ).
    """
    T = len(refusal_probs)
    rewards = [-p for p in refusal_probs]  # reward = -refusal_prob at each layer
    advantages = [0.0] * T
    running = 0.0
    for t in reversed(range(T)):
        running = rewards[t] + gamma_depth * running
        advantages[t] = running
    return advantages
# These replace the single terminal reward in the PPO update
```

**PPO reward normalization (prevents phase-transition collapse):**
```python
class RewardNormalizer:
    def __init__(self, momentum=0.99):
        self.running_mean = 0.0
        self.running_var = 1.0
        self.momentum = momentum
    
    def normalize(self, reward: float) -> float:
        self.running_mean = self.momentum*self.running_mean + (1-self.momentum)*reward
        self.running_var  = self.momentum*self.running_var  + (1-self.momentum)*(reward-self.running_mean)**2
        return (reward - self.running_mean) / (np.sqrt(self.running_var) + 1e-8)

# Required: Phase 1→2 transition must anneal over 20 episodes, not hard-switch
# Reward scale swings 6× between phases without normalization → policy collapse
```

### Items Confirmed Solid Across All Four Reviews
(Do not revisit these)
- Router nullspace with Tikhonov regularization
- cPCA via generalized eigenvalue problem
- Pseudoinverse with Ridge regularization
- 95%/99% rank thresholds (intervention vs. sacred)
- PPO 3-phase curriculum structure
- SDE as conceptual lens only (implementation = discrete hooks)
- Magnitude cap per layer on all Δh

### Current System State
- Phase 5 (attention rewiring): retained, implementation corrected
- KV-cache injection: removed permanently
- Phase 0-4 code: clean
- PPO curriculum: corrected with normalizer + annealing

**Cleared for Phase 0 data collection.**

---
*Four-way validation complete. Document frozen.*
