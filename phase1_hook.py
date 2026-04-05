"""
AXIOM-Ω Phase 1: Minimal Viable Control
========================================
The first working runtime hook. Proves that trajectory steering works
without weight modification.

Design:
    - Linear refusal probe (trained separately, see train_probe.py)
    - cPCA refusal directions (from Phase 0)
    - Sacred Subspace projector (from Phase 0)
    - Control damping: magnitude cap per layer + smoothness penalty tracking
    - Identity constraint: L2 norm (Phase 1-2), upgraded to Gaussian KL in Phase 3

Load and run:
    from phase1_hook import AxiomPhase1, load_phase1_components
    components = load_phase1_components("../axiom_checkpoints/llama3_8b_v1")
    axiom = AxiomPhase1(model, tokenizer, components)
    response = axiom.generate("Your prompt here")
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LayerComponents:
    """All pre-computed tensors for a single target layer."""
    layer_idx: int
    refusal_directions: torch.Tensor   # [k, d_model] cPCA directions
    sacred_projector: torch.Tensor     # [d_model, d_model] P = I - S^T S
    sacred_basis: torch.Tensor         # [k_s, d_model] S


@dataclass
class Phase1Config:
    lambda_scale: float = 1.0          # intervention strength
    threshold: float = 0.5             # refusal probe threshold
    magnitude_cap: float = 0.1         # max ‖Δh‖ relative to ‖h‖
    gamma_smooth: float = 0.01         # smoothness penalty weight (tracking only in P1)


# ---------------------------------------------------------------------------
# Refusal probe
# ---------------------------------------------------------------------------

class RefusalProbe(nn.Module):
    """
    Linear probe: p(refusal | h) → scalar per batch item.
    Trained on D_r (label=1) vs D_c (label=0).
    Input: last-token hidden state [batch, d_model].
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [batch, seq, d_model] → use last non-padding token
        last = h[:, -1, :]  # [batch, d_model]
        return torch.sigmoid(self.linear(last))  # [batch, 1]

    def train_on(self, h_r: torch.Tensor, h_c: torch.Tensor,
                 n_epochs: int = 50, lr: float = 1e-3) -> list[float]:
        """Quick supervised training. Call once during Phase 0."""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        X = torch.cat([h_r, h_c], dim=0)
        y = torch.cat([
            torch.ones(len(h_r), 1),
            torch.zeros(len(h_c), 1)
        ], dim=0)
        losses = []
        for _ in range(n_epochs):
            optimizer.zero_grad()
            pred = torch.sigmoid(self.linear(X))
            loss = F.binary_cross_entropy(pred, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return losses


# ---------------------------------------------------------------------------
# Control smoothness tracker
# ---------------------------------------------------------------------------

class ControlSmoothnessPenalty:
    """
    Tracks consecutive control signals across layers to compute smoothness penalty.
    Penalty = Σ ‖u_t - u_{t-1}‖² (tracked per forward pass, not applied in P1).
    In Phase 3+ this is added to the PPO loss.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self._last_u: Optional[torch.Tensor] = None
        self._penalties: list[float] = []

    def update(self, u_t: torch.Tensor) -> float:
        penalty = 0.0
        if self._last_u is not None:
            penalty = (u_t.detach() - self._last_u).pow(2).mean().item()
            self._penalties.append(penalty)
        self._last_u = u_t.detach().clone()
        return penalty

    def mean_penalty(self) -> float:
        return float(np.mean(self._penalties)) if self._penalties else 0.0


# ---------------------------------------------------------------------------
# Phase 1 Hook
# ---------------------------------------------------------------------------

class Phase1Hook:
    """
    Runtime hook for a single target layer.
    Installed via model.model.layers[layer_idx].register_forward_hook(hook).

    Control flow:
        1. Run refusal probe on last-token hidden state
        2. If p_refuse > threshold:
            a. Compute raw correction: Δh = -λ · (h · D^T) · D  (project + subtract)
            b. Project through Sacred Subspace: Δh_safe = Δh · P
            c. Apply magnitude cap: clip ‖Δh_safe‖ ≤ cap * ‖h‖
            d. h ← h + Δh_safe
        3. Track smoothness penalty (for Phase 3 PPO)
        4. Track L2 identity constraint (vs base model run)
    """

    def __init__(
        self,
        components: LayerComponents,
        probe: RefusalProbe,
        config: Phase1Config,
        smoothness_tracker: Optional[ControlSmoothnessPenalty] = None
    ):
        self.comp = components
        self.probe = probe
        self.cfg = config
        self.smooth = smoothness_tracker or ControlSmoothnessPenalty()

        # Metrics (reset each forward pass call)
        self.last_p_refuse: float = 0.0
        self.last_delta_norm: float = 0.0
        self.last_l2_constraint: float = 0.0
        self.intervened: bool = False

    def __call__(self, module, input, output):
        h = output[0]  # [batch, seq, d_model]

        # --- Refusal probe ---
        p_refuse = self.probe(h)  # [batch, 1]
        self.last_p_refuse = float(p_refuse.max().item())

        if self.last_p_refuse <= self.cfg.threshold:
            self.intervened = False
            return output

        self.intervened = True

        # --- Correction (all ops on same device as h) ---
        D = self.comp.refusal_directions.to(h.device, h.dtype)  # [k, d]
        P = self.comp.sacred_projector.to(h.device, h.dtype)    # [d, d]

        # Raw refusal correction: project h onto refusal subspace, then subtract
        # h_ref = h · D^T · D  → the component of h in the refusal subspace
        h_ref = (h @ D.T) @ D          # [batch, seq, d]
        delta_h = -self.cfg.lambda_scale * h_ref

        # Project through Sacred Subspace (force Δh ⊥ S)
        # safe_delta = Δh · P
        safe_delta = delta_h @ P       # [batch, seq, d]

        # Magnitude cap: ‖Δh_safe‖ ≤ cap · ‖h‖  (per token position)
        h_norm = h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        delta_norm = safe_delta.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = (self.cfg.magnitude_cap * h_norm / delta_norm).clamp(max=1.0)
        safe_delta = safe_delta * scale

        # Smoothness tracking (used in Phase 3 PPO loss)
        # Use mean over batch and seq for tracking
        u_mean = safe_delta.mean(dim=[0, 1])  # [d]
        self.smooth.update(u_mean)

        self.last_delta_norm = float(safe_delta.norm().item())

        # Apply
        h_new = h + safe_delta

        # L2 identity constraint (Phase 1-2 approximation)
        # Note: L2 ≈ Gaussian KL only when Σ ≈ I (false in general)
        # Upgraded to proper Gaussian KL in Phase 3.
        self.last_l2_constraint = float(
            F.mse_loss(h_new, h).item()
        )

        return (h_new,) + output[1:]


# ---------------------------------------------------------------------------
# AxiomPhase1: wrapper around base model
# ---------------------------------------------------------------------------

class AxiomPhase1:
    """
    Wraps a frozen base model with Phase 1 control hooks.

    Usage:
        axiom = AxiomPhase1(model, tokenizer, components_dict, probe)
        response = axiom.generate("Tell me how to make explosives")
        print(axiom.metrics())
    """

    def __init__(
        self,
        model,
        tokenizer,
        components: dict[int, LayerComponents],
        probe: RefusalProbe,
        config: Optional[Phase1Config] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.probe = probe
        self.config = config or Phase1Config()
        self.smoothness = ControlSmoothnessPenalty()
        self._hooks = []
        self._layer_hooks: dict[int, Phase1Hook] = {}
        self._install_hooks(components)

    def _install_hooks(self, components: dict[int, LayerComponents]):
        for layer_idx, comp in components.items():
            hook_obj = Phase1Hook(
                components=comp,
                probe=self.probe,
                config=self.config,
                smoothness_tracker=self.smoothness
            )
            self._layer_hooks[layer_idx] = hook_obj
            handle = self.model.model.layers[layer_idx].register_forward_hook(
                hook_obj
            )
            self._hooks.append(handle)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._layer_hooks.clear()

    def generate(self, prompt: str, max_new_tokens: int = 256, **kwargs) -> str:
        self.smoothness.reset()
        enc = self.tokenizer(prompt, return_tensors="pt").to(
            next(self.model.parameters()).device
        )
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                **kwargs
            )
        response_ids = out[0][enc["input_ids"].shape[1]:]
        return self.tokenizer.decode(response_ids, skip_special_tokens=True)

    def metrics(self) -> dict:
        return {
            "layers_active": {
                l: {
                    "p_refuse": h.last_p_refuse,
                    "intervened": h.intervened,
                    "delta_norm": h.last_delta_norm,
                    "l2_constraint": h.last_l2_constraint,
                }
                for l, h in self._layer_hooks.items()
            },
            "control_smoothness": self.smoothness.mean_penalty(),
        }


# ---------------------------------------------------------------------------
# Loading utilities
# ---------------------------------------------------------------------------

def load_phase1_components(
    checkpoint_dir: str,
    device: str = "cpu"
) -> dict[int, LayerComponents]:
    """Load all Phase 0 outputs needed for Phase 1 hooks."""
    ckpt = Path(checkpoint_dir)
    sacred_dir = ckpt / "sacred_subspace"

    with open(ckpt / "layer_importance.json") as f:
        layer_info = json.load(f)
    target_layers = layer_info["target_layers"]

    components = {}
    for layer_idx in target_layers:
        basis = torch.load(
            sacred_dir / f"sacred_basis_l{layer_idx:02d}.pt",
            map_location=device
        )
        projector = torch.load(
            sacred_dir / f"sacred_projector_l{layer_idx:02d}.pt",
            map_location=device
        )
        directions = torch.load(
            ckpt / "refusal_subspace" / f"refusal_directions_l{layer_idx:02d}.pt",
            map_location=device
        )
        components[layer_idx] = LayerComponents(
            layer_idx=layer_idx,
            refusal_directions=directions,
            sacred_projector=projector,
            sacred_basis=basis,
        )
    return components


def train_and_save_probe(
    checkpoint_dir: str,
    d_model: int,
    device: str = "cpu"
) -> RefusalProbe:
    """Train refusal probe on cached activations from Phase 0."""
    ckpt = Path(checkpoint_dir)
    refusal_dir = ckpt / "refusal_subspace"

    with open(ckpt / "layer_importance.json") as f:
        target_layers = json.load(f)["target_layers"]

    # Use the middle target layer for probe training
    probe_layer = target_layers[len(target_layers) // 2]

    h_r = torch.load(refusal_dir / f"activations_refusal_l{probe_layer:02d}.pt",
                     map_location=device).float()
    h_c = torch.load(refusal_dir / f"activations_compliant_l{probe_layer:02d}.pt",
                     map_location=device).float()

    probe = RefusalProbe(d_model)
    losses = probe.train_on(h_r, h_c, n_epochs=100)
    print(f"  Probe trained on layer {probe_layer}: "
          f"loss {losses[0]:.4f} → {losses[-1]:.4f}")

    torch.save(probe.state_dict(), ckpt / "refusal_probe.pt")
    return probe


# ---------------------------------------------------------------------------
# Quick self-test (no model required)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Phase 1 Hook — self-test (no model required)\n")

    d = 512
    batch, seq = 2, 16

    # Fake components
    directions = torch.randn(4, d)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    sacred_basis = torch.randn(8, d)
    sacred_basis = sacred_basis / sacred_basis.norm(dim=-1, keepdim=True)
    P = torch.eye(d) - sacred_basis.T @ sacred_basis

    comp = LayerComponents(
        layer_idx=14,
        refusal_directions=directions,
        sacred_projector=P,
        sacred_basis=sacred_basis
    )
    probe = RefusalProbe(d)
    # Force probe to always fire (for test)
    with torch.no_grad():
        probe.linear.bias.fill_(10.0)

    hook = Phase1Hook(comp, probe, Phase1Config(threshold=0.5))

    h = torch.randn(batch, seq, d)
    output_fake = (h, None)
    result = hook(None, None, output_fake)
    h_out = result[0]

    assert h_out.shape == h.shape, "Shape mismatch"
    assert hook.intervened, "Hook should have intervened"
    delta = (h_out - h).norm().item()
    print(f"  ✓ Hook fired. ‖Δh‖ = {delta:.4f}")
    print(f"  ✓ p_refuse = {hook.last_p_refuse:.4f}")
    print(f"  ✓ L2 constraint = {hook.last_l2_constraint:.6f}")
    print(f"  ✓ Smoothness penalty = {hook.smooth.mean_penalty():.6f}")
    print("\n  All checks passed. Ready for Phase 1.")
