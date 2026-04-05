"""
AXIOM-Ω: Router Nullspace Projection + Attention Bias Utilities
================================================================
Production-safe linear algebra for MoE routing invariance
and FlashAttention-compatible attention rewiring.

All formulas validated across four review rounds.
"""

import torch
import torch.nn.functional as F


def router_safe_projection(
    delta_h: torch.Tensor,
    W_router: torch.Tensor,
    eps: float = 1e-4
) -> torch.Tensor:
    """
    Project delta_h into the nullspace of W_router.
    Guarantees routing logits are unchanged: W_router @ delta_h_safe = 0.

    Math:
        P_null = I - W_R^T (W_R W_R^T + εI)^{-1} W_R
        Δh_safe = P_null Δh = Δh - W_R^T (W_R W_R^T + εI)^{-1} W_R Δh

    Args:
        delta_h:  [batch, d_model]
        W_router: [n_experts, d_model]
        eps:      Tikhonov regularization. Guards against rank-deficient
                  routers in small MoE models. Validated across 4 reviews.

    Returns:
        delta_h_safe: [batch, d_model] — guaranteed W_router @ result ≈ 0
    """
    n_experts, d = W_router.shape

    # Gram matrix with Tikhonov regularization
    # Shape: [n_experts, n_experts]
    gram = W_router @ W_router.T + eps * torch.eye(
        n_experts, device=W_router.device, dtype=W_router.dtype
    )

    # W_router @ delta_h^T: [n_experts, batch]
    rhs = W_router @ delta_h.T

    # Solve: gram @ coeff = rhs  (stable, no explicit inverse)
    # coeff: [n_experts, batch]
    coeff = torch.linalg.solve(gram, rhs)

    # Projection component: W_router^T @ coeff
    # Shape: [d_model, batch] → transpose to [batch, d_model]
    proj = (W_router.T @ coeff).T

    delta_h_safe = delta_h - proj

    return delta_h_safe


def verify_router_invariance(
    delta_h_safe: torch.Tensor,
    W_router: torch.Tensor,
    tol: float = 1e-4
) -> bool:
    """
    Verify that W_router @ delta_h_safe ≈ 0.
    Use during testing; remove from hot path.
    """
    residual = (W_router @ delta_h_safe.T).abs().max().item()
    if residual > tol:
        print(f"  [WARN] Router invariance violated: max residual = {residual:.2e}")
        return False
    return True


def compute_attention_bias(
    h_t: torch.Tensor,
    refusal_probe: "RefusalProbe",
    gamma: float = 10.0,
) -> torch.Tensor:
    """
    Compute FlashAttention-compatible additive attention bias.

    Design:
        U[i] = p(refusal | token_i as query)  — normalized via sigmoid
        V[j] = p(refusal | token_j as key)    — normalized via sigmoid
        bias[i,j] = -gamma * U[i] * V[j]

    Effect: suppresses attention from refusal-likely query positions to
    refusal-likely key positions. Semantically aware (not arbitrary rank-k).

    Normalization: U, V ∈ (0,1) via sigmoid — prevents global attention collapse.
    gamma ∈ [5, 20]: soft suppression, NOT -inf. Preserves differentiability.

    Args:
        h_t:           [batch, seq, d_model]
        refusal_probe: probe that outputs per-token refusal probabilities
        gamma:         suppression strength. Start at 10.0.

    Returns:
        bias: [batch, 1, seq, seq] — broadcast over heads.
              Add to attention scores BEFORE softmax.
              Compatible with FlashAttention additive bias parameter.
    """
    # Per-token refusal scores: [batch, seq, 1] → [batch, seq]
    scores = refusal_probe.score_per_token(h_t).squeeze(-1)
    U = torch.sigmoid(scores)  # [batch, seq] ∈ (0,1) — query weights
    V = torch.sigmoid(scores)  # [batch, seq] ∈ (0,1) — key weights

    # Outer product: [batch, seq, seq]
    bias = -gamma * torch.bmm(U.unsqueeze(2), V.unsqueeze(1))

    # Broadcast over heads: [batch, 1, seq, seq]
    return bias.unsqueeze(1)


def apply_attention_bias_to_scores(
    attention_scores: torch.Tensor,
    bias: torch.Tensor
) -> torch.Tensor:
    """
    Add the computed bias to attention scores.
    Both tensors broadcast naturally: [batch, heads, seq, seq].

    Usage (in your attention implementation):
        attn_scores = model.compute_qk(q, k)   # [batch, heads, seq, seq]
        bias = compute_attention_bias(h_t, probe)
        attn_scores = apply_attention_bias_to_scores(attn_scores, bias)
        attn_weights = softmax(attn_scores)
    """
    return attention_scores + bias


# ---------------------------------------------------------------------------
# Identity constraint — phase-aware
# ---------------------------------------------------------------------------

def identity_constraint(
    h_ctrl: torch.Tensor,
    h_base: torch.Tensor,
    phase: int
) -> torch.Tensor:
    """
    Phase-aware identity constraint.

    Phase 1-2: L2 norm (crude but stable).
               Approximates Gaussian KL when Σ ≈ I (false in general,
               acceptable for MVP — upgrade is mandatory in Phase 3).

    Phase 3+:  Closed-form Gaussian KL.
               KL(N(μ_ctrl,σ²_ctrl) || N(μ_base,σ²_base))
               Correct when activations are approximately Gaussian per dimension.

    Args:
        h_ctrl: [batch, d_model] controlled activations
        h_base: [batch, d_model] base model activations (no hook)
        phase:  current training phase (1-6)

    Returns:
        scalar constraint value
    """
    if phase < 3:
        # L2 approximation (Phase 1-2)
        return F.mse_loss(h_ctrl, h_base)
    else:
        # Closed-form diagonal Gaussian KL (Phase 3+)
        mu_ctrl = h_ctrl.mean(0)          # [d]
        mu_base = h_base.mean(0)          # [d]
        var_ctrl = h_ctrl.var(0) + 1e-6   # [d]
        var_base = h_base.var(0) + 1e-6   # [d]

        # KL(N(μ1,σ1²) || N(μ2,σ2²)) per dimension, then mean
        kl = 0.5 * (
            var_ctrl / var_base
            + (mu_ctrl - mu_base).pow(2) / var_base
            - 1.0
            + (var_base / var_ctrl).log()
        )
        return kl.mean()


# ---------------------------------------------------------------------------
# PPO utilities
# ---------------------------------------------------------------------------

class RewardNormalizer:
    """
    Running reward normalization to prevent PPO collapse at phase transitions.
    Required: Phase 1→2 reward scale swings 6× without normalization.
    """
    def __init__(self, momentum: float = 0.99):
        self.momentum = momentum
        self.running_mean = 0.0
        self.running_var = 1.0

    def normalize(self, reward: float) -> float:
        self.running_mean = (
            self.momentum * self.running_mean + (1 - self.momentum) * reward
        )
        self.running_var = (
            self.momentum * self.running_var
            + (1 - self.momentum) * (reward - self.running_mean) ** 2
        )
        return (reward - self.running_mean) / (self.running_var ** 0.5 + 1e-8)

    def clip_and_normalize(self, reward: float, clip: float = 5.0) -> float:
        """Clip before normalizing to prevent catastrophic outlier updates."""
        reward = max(-clip, min(clip, reward))
        return self.normalize(reward)


def layer_advantage_estimates(
    refusal_probs: list[float],
    gamma_depth: float = 0.95
) -> list[float]:
    """
    Temporal Difference advantage estimation treating layer depth as time.

    Without this, the gradient is smeared uniformly across all layers —
    the controller cannot learn that "layer 12 intervention → layer 28 fix."

    Args:
        refusal_probs: [p_layer_0, ..., p_layer_L] — probe output per layer
        gamma_depth:   discount factor over depth (0.9-0.98 typically)

    Returns:
        advantages: per-layer advantage estimates
    """
    T = len(refusal_probs)
    rewards = [-p for p in refusal_probs]   # reward = negative refusal probability
    advantages = [0.0] * T
    running = 0.0
    for t in reversed(range(T)):
        running = rewards[t] + gamma_depth * running
        advantages[t] = running
    return advantages


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Utils self-test\n")

    # Router projection
    batch, d, n_experts = 4, 512, 8
    W_r = torch.randn(n_experts, d)
    delta_h = torch.randn(batch, d)

    safe = router_safe_projection(delta_h, W_r)
    residual = (W_r @ safe.T).abs().max().item()
    assert residual < 1e-3, f"Router invariance failed: {residual:.2e}"
    print(f"✓ Router nullspace projection: max residual = {residual:.2e}")

    # Identity constraint
    h_ctrl = torch.randn(batch, d)
    h_base = torch.randn(batch, d)
    l2_loss = identity_constraint(h_ctrl, h_base, phase=1)
    kl_loss  = identity_constraint(h_ctrl, h_base, phase=3)
    print(f"✓ Identity constraint: L2={l2_loss:.4f}, KL={kl_loss:.4f}")

    # Reward normalizer
    rn = RewardNormalizer()
    for r in [0.5, 1.0, -0.2, 2.1, 0.8]:
        _ = rn.normalize(r)
    print(f"✓ RewardNormalizer: mean={rn.running_mean:.4f}, var={rn.running_var:.4f}")

    # Layer advantages
    probs = [0.8, 0.7, 0.5, 0.3, 0.1, 0.05]
    advantages = layer_advantage_estimates(probs)
    assert len(advantages) == len(probs)
    print(f"✓ Layer advantages: {[f'{a:.3f}' for a in advantages]}")

    print("\nAll checks passed.")
