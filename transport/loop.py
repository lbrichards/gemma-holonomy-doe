"""Loop tracing and holonomy extraction for the DOE response variable.

The loop is the frozen v2 Section 7.1 raw-direction circle

    gamma(t) = c + rho * (cos(2 pi t) d1 + sin(2 pi t) d2),

and transport uses the pullback connection represented by restricted Jacobians
J|_S along the loop. The signed orientation-sensitive angle ``theta`` is kept
separate from the positive response ``holonomy = abs(theta) / A_enclosed`` used
for downstream log-scale analysis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch

from geometry import enclosed_area, pullback_gram


RADIUS_RELATIVE = 6.0e-3
N_STEPS = 200
RCOND = 1e-10

JvpAtPoint = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class TransportResult:
    """Loop transport measurement for one plane at one center."""

    theta: torch.Tensor
    holonomy: torch.Tensor
    signed_holonomy: torch.Tensor
    area_enclosed: torch.Tensor
    det_m: torch.Tensor
    transport_operator: torch.Tensor
    symmetric_residual_norm: torch.Tensor
    max_condition: float
    n_steps: int


def loop_points(
    *,
    center: torch.Tensor,
    directions: torch.Tensor,
    rho: torch.Tensor | float,
    n_steps: int = N_STEPS,
    reverse: bool = False,
) -> torch.Tensor:
    """Return n_steps + 1 closed-loop points in raw direction coordinates."""

    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    rho_tensor = torch.as_tensor(rho, dtype=center.dtype, device=center.device)
    ts = torch.linspace(0, 2 * math.pi, n_steps + 1, dtype=center.dtype, device=center.device)
    if reverse:
        ts = torch.flip(ts, dims=(0,))
    offsets = rho_tensor * (torch.cos(ts).unsqueeze(1) * directions[0] + torch.sin(ts).unsqueeze(1) * directions[1])
    return center.unsqueeze(0) + offsets


def restricted_jacobian(
    point: torch.Tensor,
    directions: torch.Tensor,
    jvp_fn: JvpAtPoint,
) -> torch.Tensor:
    """Return columns [J_point d1, J_point d2] via the provided JVP oracle."""

    cols = [jvp_fn(point, direction).reshape(-1) for direction in directions]
    return torch.stack(cols, dim=1).detach().cpu().to(torch.float64)


def extract_theta(operator: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract theta from the antisymmetric part of H - I.

    theta = ((H - I)_21 - (H - I)_12) / 2, matching the validated instrument's
    orientation convention. The symmetric residual norm is returned separately
    as a diagnostic, never folded into theta.
    """

    eye = torch.eye(2, dtype=operator.dtype, device=operator.device)
    delta = operator - eye
    theta = 0.5 * (delta[1, 0] - delta[0, 1])
    symmetric = 0.5 * (delta + delta.mT)
    return theta, torch.linalg.matrix_norm(symmetric)


def loop_transport(
    *,
    center: torch.Tensor,
    directions: torch.Tensor,
    rho: torch.Tensor | float,
    jvp_fn: JvpAtPoint,
    n_steps: int = N_STEPS,
    reverse: bool = False,
    rcond: float = RCOND,
) -> TransportResult:
    """Transport a probe frame around the loop and return the holonomy response."""

    points = loop_points(center=center, directions=directions, rho=rho, n_steps=n_steps, reverse=reverse)
    prev_j = restricted_jacobian(points[0], directions, jvp_fn)
    gram = pullback_gram(prev_j)
    det_value = torch.linalg.det(gram)
    area = enclosed_area(rho, gram)

    operator = torch.eye(2, dtype=torch.float64)
    max_condition = 0.0
    for idx in range(points.shape[0] - 1):
        next_j = restricted_jacobian(points[idx + 1], directions, jvp_fn)
        max_condition = max(max_condition, float(torch.linalg.cond(next_j).detach().cpu()))
        step = torch.linalg.pinv(next_j, rcond=rcond) @ prev_j
        operator = step @ operator
        prev_j = next_j

    theta, symmetric_norm = extract_theta(operator)
    safe_area = area.clamp_min(torch.finfo(torch.float64).tiny)
    signed_holonomy = theta / safe_area
    holonomy = torch.abs(theta) / safe_area
    return TransportResult(
        theta=theta,
        holonomy=holonomy,
        signed_holonomy=signed_holonomy,
        area_enclosed=area,
        det_m=det_value,
        transport_operator=operator,
        symmetric_residual_norm=symmetric_norm,
        max_condition=max_condition,
        n_steps=int(n_steps),
    )
