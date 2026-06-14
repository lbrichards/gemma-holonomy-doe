"""Plane covariates from PREREGISTRATION_v2 Section 4.

The module measures two nuisance variables per plane:

- manifold distance, via SAE reconstruction error over the center plus swept
  loop points, and via distance to the experiment-sample activation reference;
- phi, the mutual Euclidean angle between raw normalized plane directions.

No model code or disk I/O lives here. Callers pass activations, plane directions,
SAE objects, and the already-extracted reference activations explicitly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol

import torch


CONDITION_THRESHOLD = 1e8
KNN_COVARIANCE_FALLBACK = "knn"


class KNNCovarianceFallback(RuntimeError):
    """Raised internally when Mahalanobis covariance is not usable."""


class SaeLike(Protocol):
    """Small SAE protocol used by covariate measurement."""

    w_dec: torch.Tensor
    b_dec: torch.Tensor

    def encode(self, activation: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class ReferenceDistribution:
    """Reference distribution for Section 4.1 guard distance.

    ``method`` is ``mahalanobis`` when the empirical covariance is full-rank and
    below the condition threshold; otherwise it is ``knn``. The fallback is
    deterministic and records the same reference activations.
    """

    reference: torch.Tensor
    mean: torch.Tensor
    covariance: torch.Tensor
    inv_covariance: torch.Tensor | None
    condition_number: float
    method: Literal["mahalanobis", "knn"]
    k: int

    @classmethod
    def fit(
        cls,
        reference: torch.Tensor,
        *,
        condition_threshold: float = CONDITION_THRESHOLD,
        k: int = 5,
    ) -> ReferenceDistribution:
        """Fit a Mahalanobis reference, falling back to kNN if ill-conditioned."""

        ref = reference.detach().cpu().to(torch.float64)
        if ref.ndim != 2:
            raise ValueError(f"reference must be 2D, got shape {tuple(ref.shape)}")
        if ref.shape[0] < 2:
            raise ValueError("reference must contain at least two activations")

        mean = ref.mean(dim=0)
        centered = ref - mean
        covariance = centered.mT @ centered / (ref.shape[0] - 1)
        inv_covariance: torch.Tensor | None
        method: Literal["mahalanobis", "knn"]
        try:
            eigvals = torch.linalg.eigvalsh(covariance)
            max_eig = torch.max(torch.abs(eigvals))
            min_eig = torch.min(torch.abs(eigvals))
            rank = int(torch.linalg.matrix_rank(covariance).detach().cpu())
            if rank < covariance.shape[0] or min_eig <= 0:
                raise KNNCovarianceFallback("rank-deficient covariance")
            condition = float((max_eig / min_eig).detach().cpu())
            if not math.isfinite(condition) or condition > condition_threshold:
                raise KNNCovarianceFallback(f"ill-conditioned covariance: {condition}")
            inv_covariance = torch.linalg.inv(covariance)
            method = "mahalanobis"
        except (RuntimeError, KNNCovarianceFallback):
            condition = math.inf
            inv_covariance = None
            method = "knn"

        return cls(
            reference=ref,
            mean=mean,
            covariance=covariance,
            inv_covariance=inv_covariance,
            condition_number=condition,
            method=method,
            k=min(int(k), ref.shape[0]),
        )

    def distance(self, points: torch.Tensor) -> torch.Tensor:
        """Return guard distances for points under the fitted reference rule."""

        pts = points.detach().to(device=self.reference.device, dtype=torch.float64)
        if pts.ndim == 1:
            pts = pts.unsqueeze(0)
        if pts.ndim != 2:
            raise ValueError(f"points must be 1D or 2D, got shape {tuple(points.shape)}")
        if pts.shape[1] != self.reference.shape[1]:
            raise ValueError(f"point dimension {pts.shape[1]} != reference dimension {self.reference.shape[1]}")

        if self.method == "mahalanobis":
            if self.inv_covariance is None:
                raise RuntimeError("mahalanobis reference is missing inverse covariance")
            centered = pts - self.mean
            squared = torch.einsum("bi,ij,bj->b", centered, self.inv_covariance, centered)
            return torch.sqrt(torch.clamp(squared, min=0))

        pairwise = torch.cdist(pts, self.reference)
        nearest = torch.topk(pairwise, k=self.k, largest=False, dim=1).values
        return nearest.mean(dim=1)


@dataclass(frozen=True)
class PlaneCovariates:
    """Measured Section 4 covariates for one plane."""

    reconstruction_mse_mean: float
    reference_distance_mean: float
    reference_distance_method: str
    reference_condition_number: float
    phi: float
    n_points: int


def _normalize_rows(rows: torch.Tensor) -> torch.Tensor:
    return rows / torch.linalg.vector_norm(rows, dim=1, keepdim=True).clamp_min(1e-12)


def mutual_angle_phi(directions: torch.Tensor) -> torch.Tensor:
    """Return phi = arccos(d1 dot d2) for raw normalized directions."""

    if directions.shape[0] != 2:
        raise ValueError(f"directions must have two rows, got shape {tuple(directions.shape)}")
    normalized = _normalize_rows(directions)
    dot = torch.clamp(normalized[0] @ normalized[1], min=-1.0, max=1.0)
    return torch.arccos(dot)


def loop_points(
    *,
    center: torch.Tensor,
    directions: torch.Tensor,
    rho: torch.Tensor | float,
    n_loop_samples: int = 32,
    include_center: bool = True,
) -> torch.Tensor:
    """Return center plus swept loop points in the raw direction plane.

    The swept points use gamma(t) = c + rho(cos(2pi t)d1 + sin(2pi t)d2).
    The endpoint duplicate is omitted; this is a sample of the visited loop,
    not the transport discretization itself.
    """

    if n_loop_samples < 1:
        raise ValueError("n_loop_samples must be >= 1")
    normalized = _normalize_rows(directions).to(device=center.device, dtype=center.dtype)
    rho_tensor = torch.as_tensor(rho, dtype=center.dtype, device=center.device)
    ts = torch.linspace(0, 2 * math.pi, n_loop_samples + 1, device=center.device, dtype=center.dtype)[:-1]
    offsets = rho_tensor * (torch.cos(ts).unsqueeze(1) * normalized[0] + torch.sin(ts).unsqueeze(1) * normalized[1])
    swept = center.unsqueeze(0) + offsets
    if include_center:
        return torch.cat([center.unsqueeze(0), swept], dim=0)
    return swept


def sae_reconstruction(sae: SaeLike, points: torch.Tensor) -> torch.Tensor:
    """Reconstruct points as code @ W_dec + b_dec using the loaded Gemma Scope SAE."""

    pts = points.detach().to(device=sae.w_dec.device, dtype=sae.w_dec.dtype)
    was_vector = pts.ndim == 1
    if was_vector:
        pts = pts.unsqueeze(0)
    code = sae.encode(pts)
    recon = code @ sae.w_dec + sae.b_dec
    return recon.squeeze(0) if was_vector else recon


def sae_reconstruction_mse(sae: SaeLike, points: torch.Tensor) -> torch.Tensor:
    """Return per-point MSE between activations and SAE reconstructions."""

    pts = points.detach().to(device=sae.w_dec.device, dtype=sae.w_dec.dtype)
    was_vector = pts.ndim == 1
    if was_vector:
        pts = pts.unsqueeze(0)
    recon = sae_reconstruction(sae, pts)
    if recon.ndim == 1:
        recon = recon.unsqueeze(0)
    mse = torch.mean((pts - recon) ** 2, dim=1)
    return mse.squeeze(0) if was_vector else mse


def measure_plane_covariates(
    *,
    center: torch.Tensor,
    directions: torch.Tensor,
    rho: torch.Tensor | float,
    sae: SaeLike,
    reference: ReferenceDistribution,
    n_loop_samples: int = 32,
) -> PlaneCovariates:
    """Measure Section 4 covariates for center plus swept loop points.

    Reconstruction error and reference distance are aggregated as arithmetic
    means over the visited point set: the loop center plus ``n_loop_samples``
    sampled points along the loop.
    """

    points = loop_points(center=center, directions=directions, rho=rho, n_loop_samples=n_loop_samples, include_center=True)
    mse = sae_reconstruction_mse(sae, points).detach().cpu().to(torch.float64)
    ref_distance = reference.distance(points.detach().cpu().to(torch.float64)).detach().cpu()
    phi = mutual_angle_phi(directions.detach().cpu().to(torch.float64))

    return PlaneCovariates(
        reconstruction_mse_mean=float(mse.mean().item()),
        reference_distance_mean=float(ref_distance.mean().item()),
        reference_distance_method=reference.method,
        reference_condition_number=reference.condition_number,
        phi=float(phi.item()),
        n_points=int(points.shape[0]),
    )
