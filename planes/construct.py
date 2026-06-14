"""Three-arm plane construction under the pre-registered det M floor.

This module implements the frozen v2 pairing rule without importing model code
or performing I/O. Callers provide SAE codes, decoder directions, and a JVP
oracle for the current base point. Plane directions are raw normalized decoder
atoms or random unit directions; they are never orthogonalized.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Callable

import torch

from geometry import det_m, pullback_gram


TAU_DETM = 0.413
_NORM_EPS = 1e-12


class Arm(StrEnum):
    """The three pre-registered plane arms."""

    REAL_FEATURE = "real-feature"
    SHUFFLED_FEATURE = "shuffled-feature"
    RANDOM = "random"


@dataclass(frozen=True)
class DetMEvaluation:
    """Pullback-metric degeneracy evaluation for one candidate plane."""

    det_m: float
    gram: torch.Tensor
    jd: torch.Tensor
    passes: bool


@dataclass(frozen=True)
class PlaneSelection:
    """One selected plane plus blind selection accounting."""

    arm: Arm
    directions: torch.Tensor
    feature_ids: tuple[int, int] | None
    det_m: float
    gram: torch.Tensor
    jd: torch.Tensor
    candidates_tested: int
    candidates_rejected_detm: int


JvpFn = Callable[[torch.Tensor], torch.Tensor]


def _normalize_rows(rows: torch.Tensor) -> torch.Tensor:
    return rows / torch.linalg.vector_norm(rows, dim=1, keepdim=True).clamp_min(_NORM_EPS)


def _torch_generator(seed: int, *, device: torch.device) -> torch.Generator:
    # CPU generators are deterministic and can still generate tensors that are
    # moved to MPS afterward. This avoids backend-specific RNG drift.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator


def evaluate_det_m(
    directions: torch.Tensor,
    jvp_fn: JvpFn,
    *,
    tau_detm: float = TAU_DETM,
) -> DetMEvaluation:
    """Evaluate Section 7.2.1 det(M) for raw plane directions.

    ``directions`` has one direction per row. ``jvp_fn`` returns J applied to a
    single direction at the current base point. G = J^T J is never materialized.
    """

    cols = [jvp_fn(direction).reshape(-1) for direction in directions]
    jd = torch.stack(cols, dim=1).detach().cpu().to(torch.float64)
    gram = pullback_gram(jd)
    det_value = float(det_m(gram).detach().cpu())
    return DetMEvaluation(
        det_m=det_value,
        gram=gram,
        jd=jd,
        passes=det_value > tau_detm,
    )


def active_feature_ids(code: torch.Tensor, *, limit: int | None = None) -> list[int]:
    """Return active SAE feature ids, ordered by activation strength."""

    active = torch.nonzero(code > 0, as_tuple=False).flatten()
    if active.numel() == 0:
        return []
    order = torch.argsort(code[active], descending=True)
    ids = active[order].detach().cpu().tolist()
    return [int(idx) for idx in ids[:limit]]


def inactive_feature_ids(code: torch.Tensor) -> list[int]:
    """Return inactive SAE feature ids, ordered by id for deterministic sampling."""

    inactive = torch.nonzero(code <= 0, as_tuple=False).flatten()
    return [int(idx) for idx in inactive.detach().cpu().tolist()]


def _real_pairs(code: torch.Tensor, *, active_limit: int) -> list[tuple[int, int]]:
    active = active_feature_ids(code, limit=active_limit)
    pairs: list[tuple[int, int]] = []
    for left_pos, left in enumerate(active):
        for right in active[left_pos + 1 :]:
            pairs.append((left, right))
    pairs.sort(
        key=lambda pair: float(
            torch.sqrt(torch.clamp(code[pair[0]], min=0) * torch.clamp(code[pair[1]], min=0)).detach().cpu()
        ),
        reverse=True,
    )
    return pairs


def _shuffled_pairs(
    code: torch.Tensor,
    *,
    seed: int,
    active_limit: int,
    inactive_sample_size: int,
) -> list[tuple[int, int]]:
    active = active_feature_ids(code, limit=active_limit)
    inactive = inactive_feature_ids(code)
    if not active or not inactive:
        return []
    generator = _torch_generator(seed, device=torch.device("cpu"))
    perm = torch.randperm(len(inactive), generator=generator)
    sampled = [inactive[int(pos)] for pos in perm[: min(inactive_sample_size, len(inactive))].tolist()]
    return [(left, right) for left in active for right in sampled if left != right]


def _random_directions(dim: int, *, seed: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    generator = _torch_generator(seed, device=torch.device("cpu"))
    rows = torch.randn((2, dim), generator=generator, dtype=torch.float64)
    rows = _normalize_rows(rows).to(device=device, dtype=dtype)
    return rows


def _feature_directions(
    decoder_directions: torch.Tensor,
    feature_ids: tuple[int, int],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    rows = decoder_directions[list(feature_ids)].to(device=device, dtype=dtype)
    return _normalize_rows(rows)


def select_plane(
    *,
    arm: Arm | str,
    activation: torch.Tensor,
    decoder_directions: torch.Tensor | None,
    code: torch.Tensor | None,
    jvp_fn: JvpFn,
    seed: int,
    tau_detm: float = TAU_DETM,
    real_active_limit: int = 96,
    shuffled_active_limit: int = 64,
    inactive_sample_size: int = 512,
    random_max_candidates: int = 4096,
) -> PlaneSelection:
    """Select one plane for an arm and apply the det M floor at selection time."""

    selected_arm = Arm(arm)
    dtype = activation.dtype
    device = activation.device
    tested = 0
    rejected = 0

    if selected_arm is Arm.RANDOM:
        for candidate_idx in range(random_max_candidates):
            directions = _random_directions(
                int(activation.numel()),
                seed=int(seed) + 1_000_003 * candidate_idx,
                dtype=dtype,
                device=device,
            )
            evaluation = evaluate_det_m(directions, jvp_fn, tau_detm=tau_detm)
            tested += 1
            if not evaluation.passes:
                rejected += 1
                continue
            return PlaneSelection(
                arm=selected_arm,
                directions=directions,
                feature_ids=None,
                det_m=evaluation.det_m,
                gram=evaluation.gram,
                jd=evaluation.jd,
                candidates_tested=tested,
                candidates_rejected_detm=rejected,
            )
        raise RuntimeError(f"No accepted random plane after testing {tested} candidates")

    if decoder_directions is None or code is None:
        raise ValueError(f"{selected_arm.value} selection requires decoder_directions and code")

    if selected_arm is Arm.REAL_FEATURE:
        pairs = _real_pairs(code, active_limit=real_active_limit)
    elif selected_arm is Arm.SHUFFLED_FEATURE:
        pairs = _shuffled_pairs(
            code,
            seed=seed,
            active_limit=shuffled_active_limit,
            inactive_sample_size=inactive_sample_size,
        )
    else:
        raise ValueError(f"unsupported arm: {selected_arm}")

    for pair in pairs:
        directions = _feature_directions(decoder_directions, pair, dtype=dtype, device=device)
        evaluation = evaluate_det_m(directions, jvp_fn, tau_detm=tau_detm)
        tested += 1
        if not evaluation.passes:
            rejected += 1
            continue
        return PlaneSelection(
            arm=selected_arm,
            directions=directions,
            feature_ids=(int(pair[0]), int(pair[1])),
            det_m=evaluation.det_m,
            gram=evaluation.gram,
            jd=evaluation.jd,
            candidates_tested=tested,
            candidates_rejected_detm=rejected,
        )

    raise RuntimeError(f"No accepted {selected_arm.value} plane after testing {tested} candidates")


def select_three_arm_planes(
    *,
    activation: torch.Tensor,
    decoder_directions: torch.Tensor,
    code: torch.Tensor,
    jvp_fn: JvpFn,
    seed: int,
    tau_detm: float = TAU_DETM,
) -> dict[Arm, PlaneSelection]:
    """Select real-feature, shuffled-feature, and random planes for one base point."""

    return {
        Arm.REAL_FEATURE: select_plane(
            arm=Arm.REAL_FEATURE,
            activation=activation,
            decoder_directions=decoder_directions,
            code=code,
            jvp_fn=jvp_fn,
            seed=seed,
            tau_detm=tau_detm,
        ),
        Arm.SHUFFLED_FEATURE: select_plane(
            arm=Arm.SHUFFLED_FEATURE,
            activation=activation,
            decoder_directions=decoder_directions,
            code=code,
            jvp_fn=jvp_fn,
            seed=seed + 10_000,
            tau_detm=tau_detm,
        ),
        Arm.RANDOM: select_plane(
            arm=Arm.RANDOM,
            activation=activation,
            decoder_directions=None,
            code=None,
            jvp_fn=jvp_fn,
            seed=seed + 20_000,
            tau_detm=tau_detm,
        ),
    }
