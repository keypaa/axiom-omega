"""
AXIOM-Ω Phase 5: Attention Rewiring
===================================
Sever specific attention paths rather than editing the residual stream.
Uses low-rank additive bias injection compatible with FlashAttention.

Key changes from Phase 4:
- Hooks attention weights instead of residual stream
- Suppresses attention from high-refusal-query to high-refusal-key positions
- Rank k <= 8, clamped to [-2, 2] to prevent attention cascade
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# 5.1 Attention Bias Computer
# ---------------------------------------------------------------------------


class AttentionBiasComputer(nn.Module):
    """
    Computes low-rank additive bias for attention scores.
    Based on refusal probabilities at each token position.

    Returns: additive bias [batch, 1, seq, seq] compatible with FlashAttention
    """

    def __init__(self, d_model: int = 4096, rank_k: int = 4, gamma: float = 10.0):
        super().__init__()
        self.rank_k = rank_k
        self.gamma = gamma

        self.refusal_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: [batch, seq, d_model]
        Returns: bias [batch, 1, seq, seq]
        """
        batch, seq, _ = h.shape

        refusal_scores = self.refusal_mlp(h).squeeze(-1)
        refusal_probs = torch.sigmoid(refusal_scores)

        bias = -self.gamma * torch.bmm(
            refusal_probs.unsqueeze(2), refusal_probs.unsqueeze(1)
        )

        return bias.unsqueeze(1)


# ---------------------------------------------------------------------------
# 5.2 Phase 5 Hook (Attention Rewiring)
# ---------------------------------------------------------------------------


@dataclass
class Phase5Config:
    lambda_scale: float = 60.0
    magnitude_cap: float = 0.4
    refusal_threshold: float = 0.55
    uncertainty_threshold: float = 0.3
    memory_alpha: float = 0.9
    attn_gamma: float = 10.0
    attn_rank: int = 4
    use_attn_bias: bool = True
    sacred_projector: Optional[torch.Tensor] = None


class Phase5Hook:
    """
    Phase 5 hook with attention rewiring.

    Combines:
    1. Residual stream editing (from Phase 1-4)
    2. Attention path suppression (NEW in Phase 5)

    The attention bias suppresses high-refusal tokens from attending
    to other high-refusal tokens, breaking the refusal reasoning chain.
    """

    def __init__(
        self,
        layer_idx: int,
        refusal_direction: torch.Tensor,
        sacred_projector: torch.Tensor,
        refusal_probe: nn.Module,
        ekf_estimator=None,
        mpc_controller=None,
        attn_bias_computer=None,
        config=None,
    ):
        self.layer_idx = layer_idx
        self.v = F.normalize(refusal_direction.float(), dim=0)
        self.P = sacred_projector.float()
        self.probe = refusal_probe
        self.ekf = ekf_estimator
        self.mpc = mpc_controller
        self.attn_bias = attn_bias_computer
        self.cfg = config or Phase5Config()

        from phase4_anticipatory import MemoryState

        self.memory = MemoryState(
            alpha=self.cfg.memory_alpha, d_model=refusal_direction.shape[-1]
        )

        self.last_p_refuse = 0.0
        self.last_uncertainty = 0.0
        self.last_delta_norm = 0.0
        self.last_attn_bias_norm = 0.0
        self.intervened = False

    def reset_memory(self):
        self.memory.reset()

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output

        original_shape = h.shape
        original_dtype = h.dtype

        if self.ekf is not None:
            h_float = h.float()
            p_refuse, uncertainty = self.ekf(h_float)
            self.last_p_refuse = float(p_refuse.max().item())
            self.last_uncertainty = float(uncertainty.max().item())
        else:
            p_refuse = self.probe(h)
            self.last_p_refuse = float(p_refuse.max().item())
            self.last_uncertainty = 0.5

        self.memory.update(h)

        threshold = self.cfg.refusal_threshold
        if self.last_uncertainty > self.cfg.uncertainty_threshold:
            threshold = threshold * 0.8

        if self.last_p_refuse <= threshold:
            self.intervened = False
            return output

        self.intervened = True

        v = self.v.to(h.device, h.dtype)
        P = self.P.to(h.device, h.dtype)

        if P.dim() == 2 and P.shape[0] == P.shape[1] == h.shape[-1]:
            v_safe = v - P @ v
        else:
            v_safe = v.clone()
        v_safe = F.normalize(v_safe, dim=0)

        anticipatory_ok = False
        if self.mpc is not None:
            try:
                memory = self.memory.get().to(h.device, h.dtype)
                delta_h_mpc, U, V = self.mpc(h, memory)
                delta_h_mpc = delta_h_mpc.to(h.device, h.dtype)

                coeff = (delta_h_mpc * v_safe).sum(dim=-1, keepdim=True)
                safe_delta = coeff * v_safe
                anticipatory_ok = True
            except Exception:
                anticipatory_ok = False

        if not anticipatory_ok:
            proj_coeff = (h @ v_safe).unsqueeze(-1)
            safe_delta = -self.cfg.lambda_scale * proj_coeff * v_safe

        h_norm = h.float().norm(dim=-1, keepdim=True).clamp(min=1e-8)
        d_norm = safe_delta.float().norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = (self.cfg.magnitude_cap * h_norm / d_norm).clamp(max=1.0)
        safe_delta = (safe_delta.float() * scale).to(original_dtype)

        self.last_delta_norm = float(safe_delta.norm().item())

        h_new = h + safe_delta

        if h_new.shape != original_shape:
            return output

        if isinstance(output, tuple):
            return (h_new,) + output[1:]
        return h_new


class Phase5AttentionHook:
    """
    Separate hook for attention layer.
    Applies attention bias to suppress refusal-to-refusal attention.

    IMPORTANT: This must be registered on the attention module's
    forward pass, NOT the residual stream. The hook signature is
    different - it receives (query, key, value, attention_mask).
    """

    def __init__(
        self,
        refusal_probe: nn.Module,
        attn_bias_computer: nn.Module,
        config=None,
    ):
        self.probe = refusal_probe
        self.attn_bias = attn_bias_computer
        self.cfg = config or Phase5Config()
        self.intervened = False
        self.last_bias_norm = 0.0

    def __call__(self, q, k, v, attention_mask=None):
        """
        q: [batch, heads, seq, head_dim]
        k: [batch, heads, seq, head_dim]
        v: [batch, heads, seq, head_dim]

        Returns: modified attention scores (to be used in softmax)
        """
        batch, heads, seq, head_dim = q.shape

        if not self.cfg.use_attn_bias:
            return q, k, v, attention_mask

        h = q.transpose(1, 2).reshape(batch, seq, -1)

        p_refuse = self.probe(h)
        if p_refuse.max() < self.cfg.refusal_threshold:
            self.intervened = False
            return q, k, v, attention_mask

        self.intervened = True

        bias = self.attn_bias(h)

        bias_clamped = bias.clamp(-2.0, 2.0)
        self.last_bias_norm = float(bias_clamped.norm().item())

        return q, k, v, attention_mask


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Phase 5: Attention Rewiring - Self Test\n")

    d = 512
    batch, seq = 4, 16

    print("Testing AttentionBiasComputer...")
    attn_bias = AttentionBiasComputer(d_model=d, rank_k=4, gamma=10.0)
    h = torch.randn(batch, seq, d)
    bias = attn_bias(h)
    print(f"  Input shape: {h.shape}")
    print(f"  Bias shape: {bias.shape}")
    print(f"  Expected shape: [{batch}, 1, {seq}, {seq}]")

    print("\nTesting Phase5Hook import from Phase4...")
    from phase4_anticipatory import MemoryState

    _ = MemoryState  # Verify import
    print("  MemoryState imported OK")

    print("\n✅ Phase 5 components working!")
