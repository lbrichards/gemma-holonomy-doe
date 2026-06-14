"""Pullback-metric tensor operations from PREREGISTRATION Sections 7.2 and 7.3.

The functions in this module are pure tensor math. They do not import model code,
perform I/O, or materialize G = J^T J. Callers pass J applied to the needed
vectors, such as JD = [J d1, J d2] and Jh = J h.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CenterPlacement:
    """Result of Section 7.3 center placement under the pullback metric."""

    center: torch.Tensor
    out_of_plane: torch.Tensor
    projection_coeffs: torch.Tensor
    offset_coeffs: torch.Tensor
    magnitude: torch.Tensor
    fallback_used: bool


def pullback_gram(jd: torch.Tensor) -> torch.Tensor:
    """Return M = (JD)^T(JD), the Section 7.3 pullback-metric plane Gram.

    Args:
        jd: Matrix whose columns are J applied to the plane directions.
    """

    return jd.mT @ jd


def det_m(gram: torch.Tensor) -> torch.Tensor:
    """Return det(M), the Section 7.2 degeneracy and G-area determinant."""

    return torch.linalg.det(gram)


def g_projection_coeffs(jd: torch.Tensor, jh: torch.Tensor, *, rcond: float = 1e-12) -> torch.Tensor:
    """Return a = M^{-1}(JD)^T(Jh), the Section 7.3 G-projection coefficients.

    The residual Jh - JD a is orthogonal to the plane image under the pullback
    metric because (JD)^T(Jh - JD a) = 0.
    """

    gram = pullback_gram(jd)
    rhs = jd.mT @ jh
    try:
        return torch.linalg.solve(gram, rhs)
    except RuntimeError:
        return torch.linalg.pinv(gram, rcond=rcond) @ rhs


def in_plane_magnitude(
    jd: torch.Tensor,
    jh: torch.Tensor,
    *,
    coeffs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return mag(h) = sqrt(a^T M a), the Section 7.3 matched scalar."""

    projection_coeffs = g_projection_coeffs(jd, jh) if coeffs is None else coeffs
    image = jd @ projection_coeffs
    return torch.linalg.vector_norm(image)


def enclosed_area(rho: torch.Tensor | float, gram: torch.Tensor) -> torch.Tensor:
    """Return A_enclosed = rho^2 * sqrt(det M), the Section 7.2 G-area."""

    rho_tensor = torch.as_tensor(rho, dtype=gram.dtype, device=gram.device)
    det_value = torch.clamp(det_m(gram), min=0)
    return rho_tensor * rho_tensor * torch.sqrt(det_value)


def center_placement(
    *,
    directions: torch.Tensor,
    jd: torch.Tensor,
    h: torch.Tensor,
    jh: torch.Tensor,
    target_magnitude: torch.Tensor | float,
    eps_mag: torch.Tensor | float,
) -> CenterPlacement:
    """Return Section 7.3 center placement for a target in-plane magnitude.

    The plane is D = [d1 d2] in raw normalized directions, represented here as
    ``directions`` with one direction per row. The projection coefficients are
    a = M^{-1}(JD)^T(Jh), the in-plane vector is v = D a, and the center is
    c = (h - v) + m * (v / mag(h)). If mag(h) < eps_mag, the offset direction
    switches to G-normalized d1 and ``fallback_used`` is true.
    """

    target = torch.as_tensor(target_magnitude, dtype=h.dtype, device=h.device)
    eps = torch.as_tensor(eps_mag, dtype=h.dtype, device=h.device)
    coeffs = g_projection_coeffs(jd, jh).to(dtype=h.dtype, device=h.device)
    in_plane = directions.mT @ coeffs
    out_of_plane = h - in_plane
    magnitude = in_plane_magnitude(jd, jh, coeffs=coeffs).to(dtype=h.dtype, device=h.device)

    if bool((magnitude < eps).detach().cpu()):
        d1_norm = torch.linalg.vector_norm(jd[:, 0]).to(dtype=h.dtype, device=h.device)
        offset_coeffs = torch.zeros_like(coeffs)
        offset_coeffs[0] = target / d1_norm.clamp_min(torch.finfo(h.dtype).eps)
        fallback_used = True
    else:
        offset_coeffs = coeffs * (target / magnitude)
        fallback_used = False

    offset = directions.mT @ offset_coeffs
    center = out_of_plane + offset
    return CenterPlacement(
        center=center,
        out_of_plane=out_of_plane,
        projection_coeffs=coeffs,
        offset_coeffs=offset_coeffs,
        magnitude=magnitude,
        fallback_used=fallback_used,
    )
