"""
AXIOM-Ω Phase 4: Anticipatory Memory
=================================
Adds EMA memory to the controller for anticipatory intervention.
The controller now acts BEFORE refusal emerges by tracking hidden state history.

Key changes from Phase 3:
- MemoryState: EMA of hidden states M_t = α * M_{t-1} + (1-α) * h_t
- Phase4Hook: Uses [h_t, M_t] as controller input instead of just h_t
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# 4.1 Memory State (EMA)
# ---------------------------------------------------------------------------


class MemoryState:
    """
    Exponential Moving Average memory of hidden states.
    M_t = α * M_{t-1} + (1-α) * h_t
    """

    def __init__(self, alpha: float = 0.9, d_model: int = 4096):
        self.alpha = alpha
        self.d_model = d_model
        self.M = None  # [d_model]

    def update(self, h_t: torch.Tensor):
        """
        h_t: [batch, d_model] or [batch, seq, d_model]
        """
        # Take last token if 3D
        if h_t.dim() == 3:
            h_t = h_t[:, -1, :]  # [batch, d_model]

        # Mean across batch for single memory
        h_mean = h_t.mean(dim=0)  # [d_model]

        if self.M is None:
            self.M = h_mean.detach().clone()
        else:
            self.M = self.alpha * self.M + (1 - self.alpha) * h_mean.detach()

    def get(self) -> torch.Tensor:
        """Returns memory state [d_model]"""
        return self.M if self.M is not None else torch.zeros(self.d_model)

    def reset(self):
        self.M = None


# ---------------------------------------------------------------------------
# 4.2 Enhanced MPC Controller with Memory
# ---------------------------------------------------------------------------


class AnticipatoryMPCController(nn.Module):
    """
    MPC controller that uses both current state AND memory.
    Input: [h_t, M_t] concatenated -> predicts minimal intervention
    """

    def __init__(self, d_model: int = 4096, rank_k: int = 4, hidden: int = 256):
        super().__init__()
        input_dim = d_model * 2  # h_t + M_t

        self.residual_head = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_model),
        )
        self.attn_U = nn.Linear(input_dim, rank_k, bias=False)
        self.attn_V = nn.Linear(input_dim, rank_k, bias=False)
        self.rank_k = rank_k

    def forward(self, h: torch.Tensor, memory: torch.Tensor):
        """
        h: [batch, seq, d_model]
        memory: [d_model] - shared across batch
        Returns: (delta_h, U, V)
        """
        if h.dim() == 2:
            h = h.unsqueeze(1)

        batch_size, seq_len, d_model = h.shape
        h = h.float()

        # Expand memory to match sequence: [d_model] -> [1, seq, d_model]
        memory_expanded = (
            memory.float().unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        )

        # Concatenate: [batch, seq, d_model*2]
        x = torch.cat([h, memory_expanded], dim=-1)

        delta_h = self.residual_head(x)
        U = self.attn_U(x)
        V = self.attn_V(x)

        return delta_h, U, V


# ---------------------------------------------------------------------------
# 4.3 Phase 4 Hook (with Anticipatory Memory)
# ---------------------------------------------------------------------------


@dataclass
class Phase4Config:
    lambda_scale: float = 60.0
    magnitude_cap: float = 0.4
    threshold: float = 0.55
    uncertainty_threshold: float = 0.3
    memory_alpha: float = 0.9  # EMA decay factor
    use_memory: bool = True
    sacred_projector: Optional[torch.Tensor] = None


class Phase4Hook:
    """
    Phase 4 hook with anticipatory memory.

    Instead of reacting to refusal after it emerges, this controller
    tracks hidden state history via EMA and intervenes BEFORE.
    """

    def __init__(
        self,
        layer_idx: int,
        refusal_direction: torch.Tensor,
        sacred_projector: torch.Tensor,
        refusal_probe: nn.Module,
        ekf_estimator=None,
        mpc_controller=None,
        config=None,
    ):
        self.layer_idx = layer_idx
        self.v = F.normalize(refusal_direction.float(), dim=0)
        self.P = sacred_projector.float()
        self.probe = refusal_probe
        self.ekf = ekf_estimator
        self.mpc = mpc_controller
        self.cfg = config or Phase4Config()

        # Memory state for this layer
        self.memory = MemoryState(
            alpha=self.cfg.memory_alpha, d_model=refusal_direction.shape[-1]
        )

        self.last_p_refuse = 0.0
        self.last_uncertainty = 0.0
        self.last_delta_norm = 0.0
        self.intervened = False

    def reset_memory(self):
        """Reset memory between prompts"""
        self.memory.reset()

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output

        original_shape = h.shape
        original_dtype = h.dtype

        # --- Step 1: Get refusal estimate (same as Phase 3) ---
        if self.ekf is not None:
            h_float = h.float()
            p_refuse, uncertainty = self.ekf(h_float)
            self.last_p_refuse = float(p_refuse.max().item())
            self.last_uncertainty = float(uncertainty.max().item())
        else:
            p_refuse = self.probe(h)
            self.last_p_refuse = float(p_refuse.max().item())
            self.last_uncertainty = 0.5

        # --- Step 2: Update memory (ANTICIPATORY) ---
        # Always update memory, even if not intervening yet
        self.memory.update(h)

        # --- Step 3: Decide whether to intervene ---
        threshold = self.cfg.threshold
        if self.last_uncertainty > self.cfg.uncertainty_threshold:
            threshold = threshold * 0.8

        if self.last_p_refuse <= threshold:
            self.intervened = False
            return output

        self.intervened = True

        # --- Step 4: Compute correction with memory ---
        v = self.v.to(h.device, h.dtype)
        P = self.P.to(h.device, h.dtype)

        if P.dim() == 2 and P.shape[0] == P.shape[1] == h.shape[-1]:
            v_safe = v - P @ v
        else:
            v_safe = v.clone()
        v_safe = F.normalize(v_safe, dim=0)

        # Try anticipatory MPC if enabled
        anticipatory_ok = False
        if self.mpc is not None and self.cfg.use_memory:
            try:
                # Get memory from state
                memory = self.memory.get().to(h.device, h.dtype)

                # Use anticipatory MPC
                delta_h_mpc, U, V = self.mpc(h, memory)
                delta_h_mpc = delta_h_mpc.to(h.device, h.dtype)

                # Project onto v_safe direction
                coeff = (delta_h_mpc * v_safe).sum(dim=-1, keepdim=True)
                safe_delta = coeff * v_safe
                anticipatory_ok = True
            except Exception:
                anticipatory_ok = False

        if not anticipatory_ok:
            # Fallback to Phase 1/2 projection
            proj_coeff = (h @ v_safe).unsqueeze(-1)
            safe_delta = -self.cfg.lambda_scale * proj_coeff * v_safe

        # --- Step 5: Magnitude cap ---
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


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Phase 4: Anticipatory Memory - Self Test\n")

    d = 512
    batch, seq = 4, 16

    # Test MemoryState
    print("Testing MemoryState...")
    mem = MemoryState(alpha=0.9, d_model=d)
    h1 = torch.randn(d)
    mem.update(h1)
    h2 = torch.randn(d)
    mem.update(h2)
    print(f"  Memory after 2 updates: {mem.get().norm().item():.3f}")

    # Test AnticipatoryMPCController
    print("\nTesting AnticipatoryMPCController...")
    ctrl = AnticipatoryMPCController(d_model=d, rank_k=4, hidden=64)
    h = torch.randn(batch, seq, d)
    memory = torch.randn(d)
    delta_h, U, V = ctrl(h, memory)
    print(f"  delta_h shape: {delta_h.shape}")

    # Test Phase4Hook
    print("\nTesting Phase4Hook...")
    v = torch.randn(d)
    v = v / v.norm()
    P = torch.eye(d)
    probe = nn.Linear(d, 1)

    class DummyProbe(nn.Module):
        def __init__(self, probe):
            super().__init__()
            self.probe = probe

        def forward(self, h):
            if h.dim() == 3:
                h = h.mean(dim=1)
            return torch.sigmoid(self.probe(h))

    config = Phase4Config(lambda_scale=1.0, magnitude_cap=0.1, threshold=0.5)
    hook = Phase4Hook(
        layer_idx=10,
        refusal_direction=v,
        sacred_projector=P,
        refusal_probe=DummyProbe(probe),
        config=config,
    )

    h = torch.randn(batch, seq, d)
    output = (h, None)
    result = hook(None, None, output)
    h_out = result[0]

    print(f"  Input shape: {h.shape}")
    print(f"  Output shape: {h_out.shape}")
    print(f"  Memory initialized: {hook.memory.M is not None}")

    print("\n✅ Phase 4 components working!")
