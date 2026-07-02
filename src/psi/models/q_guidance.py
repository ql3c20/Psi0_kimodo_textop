from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import nn


class _QMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: Sequence[int], output_dim: int = 1):
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                ]
            )
            current_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(current_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class Psi0QGuidance:
    """Test-time Q guidance for the rot6d59 PSI-0 action layout.

    PSI-0 samples normalized actions with layout ``hand14 + five pose9``.
    The Q network consumes physical (denormalized) ``state38 + chunk30*pose45``.
    """

    _POSITION_INDICES_45 = tuple(
        pose_start + offset
        for pose_start in range(0, 45, 9)
        for offset in range(3)
    )

    def __init__(
        self,
        checkpoint: str | Path,
        action_min: Sequence[float],
        action_max: Sequence[float],
        device: torch.device | str,
        *,
        beta_max: float = 0.03,
        start_t: float = 0.3,
        max_grad_norm: float = 0.3,
        mask: str = "position",
    ):
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Q-guidance checkpoint does not exist: {checkpoint}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        layout = payload["input_layout"]
        if layout["state_dim"] != 38 or layout["action_dim"] != 45:
            raise ValueError(
                "Q checkpoint must use hand-removed state38/action45, got "
                f"state{layout['state_dim']}/action{layout['action_dim']}"
            )
        self.chunk_size = int(layout["chunk_size"])
        if self.chunk_size != 30:
            raise ValueError(f"Q checkpoint chunk size must be 30, got {self.chunk_size}")

        config = payload["config"]
        hidden_sizes = config["model"]["hidden_sizes"]
        self.model = _QMLP(
            input_dim=int(layout["input_dim"]),
            hidden_sizes=hidden_sizes,
            output_dim=len(payload["target_keys"]),
        ).to(device=device, dtype=torch.float32)
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.device = torch.device(device)
        self.feature_mean = payload["feature_mean"].to(self.device, torch.float32)
        self.feature_std = payload["feature_std"].to(self.device, torch.float32)
        self.target_mean = payload["target_mean"].to(self.device, torch.float32)
        self.target_std = payload["target_std"].to(self.device, torch.float32)
        self.action_min = torch.as_tensor(action_min, device=self.device, dtype=torch.float32)
        self.action_max = torch.as_tensor(action_max, device=self.device, dtype=torch.float32)
        if self.action_min.numel() != 59 or self.action_max.numel() != 59:
            raise ValueError("Q guidance requires the rot6d59 PSI-0 action layout")

        self.beta_max = float(beta_max)
        self.start_t = float(start_t)
        self.max_grad_norm = float(max_grad_norm)
        if mask not in {"position", "all"}:
            raise ValueError(f"Unsupported Q-guidance mask: {mask}")
        self.mask = mask
        self.last_q: float | None = None
        self.last_grad_norm: float | None = None

    def _denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return 0.5 * (action + 1.0) * (self.action_max - self.action_min) + self.action_min

    def _beta(self, sigma: torch.Tensor) -> float:
        # FlowMatchEuler starts at sigma=1 (noise) and ends at sigma=0 (data).
        progress = 1.0 - float(sigma)
        if progress < self.start_t:
            return 0.0
        return self.beta_max * (progress - self.start_t) / max(1.0 - self.start_t, 1e-6)

    def guide_velocity(
        self,
        action_samples: torch.Tensor,
        model_pred: torch.Tensor,
        raw_states: torch.Tensor,
        sigma: torch.Tensor,
        frozen_prefix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        beta = self._beta(sigma)
        if beta == 0.0:
            return model_pred

        # PSI-0 predict methods run under torch.inference_mode(); temporarily
        # leave it because Q guidance needs gradients with respect to the chunk.
        with torch.inference_mode(False), torch.enable_grad(), torch.autocast(
            device_type="cuda", enabled=False
        ):
            # For diffusers FlowMatchEuler, x_data ~= x_sigma - sigma * velocity.
            g_hat = (action_samples.float() - sigma.float() * model_pred.float()).detach()
            g_hat.requires_grad_(True)
            physical_action = self._denormalize_action(g_hat)
            q_action = physical_action[:, : self.chunk_size, 14:59]

            state = raw_states.float()
            if state.ndim == 3:
                state = state[:, -1]
            q_state = state[:, 14:52]
            q_input = torch.cat([q_state, q_action.reshape(q_action.shape[0], -1)], dim=-1)
            q_input = (q_input - self.feature_mean) / self.feature_std.clamp_min(1e-6)
            q_normalized = self.model(q_input)
            q = q_normalized * self.target_std + self.target_mean
            grad = torch.autograd.grad(q.sum(), g_hat, create_graph=False)[0]

        grad_mask = torch.zeros_like(grad)
        if self.mask == "all":
            grad_mask[:, : self.chunk_size, 14:59] = 1
        else:
            indices = torch.as_tensor(
                [14 + index for index in self._POSITION_INDICES_45],
                device=grad.device,
            )
            grad_mask[:, : self.chunk_size, indices] = 1
        if frozen_prefix is not None:
            grad_mask = grad_mask * (~frozen_prefix.bool()).unsqueeze(-1)
        grad = grad * grad_mask

        flat_norm = grad.flatten(1).norm(dim=1).clamp_min(1e-6)
        scale = (self.max_grad_norm / flat_norm).clamp(max=1.0)
        grad = grad * scale.view(-1, 1, 1)
        self.last_q = float(q.mean().detach())
        self.last_grad_norm = float(grad.flatten(1).norm(dim=1).mean().detach())

        # Scheduler delta-sigma is negative, so subtracting grad from velocity
        # produces a positive ascent step in action space.
        return model_pred - beta * grad.to(model_pred.dtype)
