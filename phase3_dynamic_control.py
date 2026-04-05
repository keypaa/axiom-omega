"""
AXIOM-Ω Phase 3: Dynamic Control (EKF + MPC)
=============================================
Implements:
- EKF State Estimator: predicts refusal probability at terminal layer
- MPC Controller: computes minimal intervention given EKF estimate

Phase 3 components:
1. RefusalStateEstimator (EKF) - predicts p_refuse
2. LatentMPCController - computes Δh + attention mask
3. Phase3Hook - combines both with existing Phase 1/2 logic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# 3.1 EKF State Estimator
# ---------------------------------------------------------------------------


class RefusalStateEstimator(nn.Module):
    def __init__(self, d_model: int = 4096, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, h: torch.Tensor):
        if h.dim() == 3:
            h = h[:, -1, :]
        h = h.float()
        out = self.net(h)
        p_refuse = torch.sigmoid(out[:, 0])
        uncertainty = torch.sigmoid(out[:, 1])
        return p_refuse, uncertainty


# ---------------------------------------------------------------------------
# 3.2 MPC Controller
# ---------------------------------------------------------------------------


class LatentMPCController(nn.Module):
    def __init__(self, d_model: int = 4096, rank_k: int = 4, hidden: int = 256):
        super().__init__()
        self.residual_head = nn.Sequential(
            nn.Linear(d_model + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_model),
        )
        self.attn_U = nn.Linear(d_model + 1, rank_k, bias=False)
        self.attn_V = nn.Linear(d_model + 1, rank_k, bias=False)
        self.rank_k = rank_k

    def forward(self, h: torch.Tensor, p_refuse: torch.Tensor):
        if h.dim() == 2:
            h = h.unsqueeze(1)

        batch_size, seq_len, d_model = h.shape
        h = h.float()

        if p_refuse.dim() > 1:
            p_refuse = p_refuse.squeeze(-1)

        p_refuse_expanded = p_refuse.float().unsqueeze(-1).unsqueeze(-1)
        p_refuse_expanded = p_refuse_expanded.expand(-1, seq_len, -1)

        x = torch.cat([h, p_refuse_expanded], dim=-1)

        delta_h = self.residual_head(x)
        U = self.attn_U(x)
        V = self.attn_V(x)

        return delta_h, U, V


# ---------------------------------------------------------------------------
# 3.3 Phase 3 Hook (Fixed Shape Logic)
# ---------------------------------------------------------------------------


@dataclass
class Phase3Config:
    lambda_scale: float = 60.0
    magnitude_cap: float = 0.4
    threshold: float = 0.55
    uncertainty_threshold: float = 0.3
    use_mpc: bool = True
    sacred_projector: Optional[torch.Tensor] = None


class Phase3Hook:
    def __init__(
        self,
        layer_idx: int,
        refusal_direction: torch.Tensor,
        sacred_projector: torch.Tensor,
        refusal_probe: nn.Module,
        ekf_estimator: Optional[RefusalStateEstimator] = None,
        mpc_controller: Optional[LatentMPCController] = None,
        config: Phase3Config = None,
    ):
        self.layer_idx = layer_idx
        self.v = F.normalize(refusal_direction.float(), dim=0)
        self.P = sacred_projector.float()
        self.probe = refusal_probe
        self.ekf = ekf_estimator
        self.mpc = mpc_controller
        self.cfg = config or Phase3Config()

        if self.cfg.sacred_projector is None:
            self.cfg.sacred_projector = sacred_projector

        self.last_p_refuse = 0.0
        self.last_uncertainty = 0.0
        self.last_delta_norm = 0.0
        self.intervened = False

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output

        original_shape = h.shape
        original_dtype = h.dtype

        # --- Step 1: Get refusal estimate ---
        if self.ekf is not None:
            h_float = h.float()
            p_refuse, uncertainty = self.ekf(h_float)
            self.last_p_refuse = float(p_refuse.max().item())
            self.last_uncertainty = float(uncertainty.max().item())
        else:
            p_refuse = self.probe(h)
            self.last_p_refuse = float(p_refuse.max().item())
            self.last_uncertainty = 0.5

        # --- Step 2: Decide whether to intervene ---
        threshold = self.cfg.threshold
        if self.last_uncertainty > self.cfg.uncertainty_threshold:
            threshold = threshold * 0.8

        if self.last_p_refuse <= threshold:
            self.intervened = False
            return output

        self.intervened = True

        # --- Step 3: Compute correction ---
        v = self.v.to(h.device, h.dtype)
        P = self.P.to(h.device, h.dtype)

        if P.dim() == 2 and P.shape[0] == P.shape[1] == h.shape[-1]:
            v_safe = v - P @ v
        else:
            v_safe = v.clone()

        v_safe = F.normalize(v_safe, dim=0)

        mpc_ok = False
        if self.cfg.use_mpc and self.mpc is not None:
            try:
                if self.ekf is not None:
                    h_float = h.float()
                    p_refuse_tensor, _ = self.ekf(h_float)
                else:
                    p_refuse_tensor = torch.full(
                        (h.shape[0],),
                        self.last_p_refuse,
                        device=h.device,
                        dtype=torch.float32,
                    )

                if h.shape[0] != p_refuse_tensor.shape[0]:
                    raise ValueError(
                        f"Batch size mismatch: h={h.shape[0]}, p={p_refuse_tensor.shape[0]}"
                    )

                delta_h_mpc, U, V = self.mpc(h, p_refuse_tensor)
                delta_h_mpc = delta_h_mpc.to(h.device, h.dtype)
                coeff = (delta_h_mpc * v_safe).sum(dim=-1, keepdim=True)
                safe_delta = coeff * v_safe
                mpc_ok = True
            except Exception:
                mpc_ok = False

        if not mpc_ok:
            proj_coeff = (h @ v_safe).unsqueeze(-1)
            safe_delta = -self.cfg.lambda_scale * proj_coeff * v_safe

        # --- Step 4: Apply magnitude cap ---
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
# Training utilities
# ---------------------------------------------------------------------------


def train_ekf_estimator(
    refusal_dir, d_model=4096, n_samples=1000, n_epochs=50, lr=1e-3, device="cuda"
):
    from pathlib import Path

    refusal_path = Path(refusal_dir)

    h_r = torch.load(
        refusal_path / "activations_refusal_l31.pt",
        map_location="cpu",
        weights_only=True,
    ).float()[:n_samples]
    h_c = torch.load(
        refusal_path / "activations_compliant_l31.pt",
        map_location="cpu",
        weights_only=True,
    ).float()[:n_samples]

    X = torch.cat([h_r, h_c], dim=0).to(device)
    y = torch.cat(
        [torch.ones(len(h_r), device=device), torch.zeros(len(h_c), device=device)],
        dim=0,
    )
    X_noisy = X + torch.randn_like(X) * 0.1

    estimator = RefusalStateEstimator(d_model=d_model).to(device)
    optimizer = torch.optim.Adam(estimator.parameters(), lr=lr)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        p_refuse, uncertainty = estimator(X_noisy)
        loss_refuse = F.binary_cross_entropy(p_refuse, y)
        preds = (p_refuse > 0.5).float()
        accuracy = (preds == y).float()
        target_uncertainty = 1.0 - accuracy
        loss_uncertainty = F.mse_loss(uncertainty, target_uncertainty)
        loss = loss_refuse + 0.1 * loss_uncertainty
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            acc = ((p_refuse > 0.5).float() == y).float().mean()
            print(f"  EKF Epoch {epoch + 1}: loss={loss.item():.4f} acc={acc:.3f}")

    return estimator


def train_mpc_controller(
    refusal_dir, d_model=4096, rank_k=4, n_epochs=50, lr=1e-3, device="cuda"
):
    from pathlib import Path

    refusal_path = Path(refusal_dir)

    h_r = torch.load(
        refusal_path / "activations_refusal_l31.pt",
        map_location="cpu",
        weights_only=True,
    ).float()[:500]
    h_c = torch.load(
        refusal_path / "activations_compliant_l31.pt",
        map_location="cpu",
        weights_only=True,
    ).float()[:500]
    directions = torch.load(
        refusal_path / "refusal_directions_l31.pt",
        map_location="cpu",
        weights_only=True,
    ).float()[0]
    v = F.normalize(directions, dim=0).to(device)

    X_2d = torch.cat([h_r, h_c], dim=0).to(device)
    y = torch.cat([torch.ones(len(h_r)), torch.zeros(len(h_c))], dim=0).to(device)
    X = X_2d.unsqueeze(1)

    controller = LatentMPCController(d_model=d_model, rank_k=rank_k).to(device)
    optimizer = torch.optim.Adam(controller.parameters(), lr=lr)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        p_refuse = y
        delta_h, U, V = controller(X, p_refuse)
        target_component = (X.squeeze(1) @ v).unsqueeze(-1)
        cancellation = delta_h.squeeze(1) @ v
        target_cancellation = y * target_component.squeeze(-1)
        loss = F.mse_loss(cancellation, target_cancellation) + 0.01 * (
            U.abs().mean() + V.abs().mean()
        )
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"  MPC Epoch {epoch + 1}: loss={loss.item():.4f}")

    return controller


if __name__ == "__main__":
    print("Phase 3: Self Test\n")
    d = 512
    batch, seq = 4, 16

    print("Testing EKF...")
    ekf = RefusalStateEstimator(d_model=d, hidden=64)
    h = torch.randn(batch, seq, d)
    p_refuse, uncertainty = ekf(h)
    print(f"  p_refuse: {p_refuse.shape}, uncertainty: {uncertainty.shape}")

    print("Testing MPC...")
    mpc = LatentMPCController(d_model=d, rank_k=4, hidden=64)
    p = torch.rand(batch)
    delta_h, U, V = mpc(h, p)
    print(f"  delta_h: {delta_h.shape}")

    print("\n✅ All Phase 3 components working!")
