"""Common-support band logic from PREREGISTRATION Section 2.

This module is pure math: it takes magnitude arrays per arm and returns the
blind common-support decision record that Stage A will serialize into the run
manifest. It performs no model loading and no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import ArrayLike


class BandStatus(StrEnum):
    """Pre-registered Section 2 common-support branch."""

    THREE_ARM_VALID = "THREE_ARM_VALID"
    TWO_ARM_FALLBACK = "TWO_ARM_FALLBACK"
    TWO_ARM_TERMINAL = "TWO_ARM_TERMINAL"


@dataclass(frozen=True)
class ArmPercentiles:
    """Per-arm percentile summary used to audit support overlap."""

    p5: float
    p25: float
    p75: float
    p95: float


@dataclass(frozen=True)
class BandDecision:
    """Structured manifest record for the blind magnitude-band decision."""

    status: BandStatus
    band_lower: float | None
    band_upper: float | None
    band_width: float
    threshold: float
    m_low: float | None
    m_high: float | None
    h_sem_matched: bool
    h_grad_matched: bool
    h_mag_matched: bool
    h_sem_status: str
    h_grad_status: str
    h_mag_status: str
    three_arm_lower: float
    three_arm_upper: float
    three_arm_width: float
    three_arm_threshold: float
    two_arm_lower: float
    two_arm_upper: float
    two_arm_width: float
    two_arm_threshold: float
    per_arm: dict[str, ArmPercentiles]


def _as_1d(name: str, values: ArrayLike) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} magnitudes must be one-dimensional")
    if arr.size == 0:
        raise ValueError(f"{name} magnitudes must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} magnitudes must be finite")
    return arr


def _percentiles(values: np.ndarray) -> ArmPercentiles:
    p5, p25, p75, p95 = np.percentile(values, [5, 25, 75, 95])
    return ArmPercentiles(p5=float(p5), p25=float(p25), p75=float(p75), p95=float(p95))


def _iqr(values: np.ndarray) -> float:
    p25, p75 = np.percentile(values, [25, 75])
    return float(p75 - p25)


def _overlap(percentiles: list[ArmPercentiles]) -> tuple[float, float, float]:
    lower = max(p.p5 for p in percentiles)
    upper = min(p.p95 for p in percentiles)
    width = upper - lower
    return float(lower), float(upper), float(width)


def _within_band(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return values[(values >= lower) & (values <= upper)]


def _cut_magnitudes(values: np.ndarray, lower: float, upper: float) -> tuple[float, float]:
    restricted = _within_band(values, lower, upper)
    if restricted.size == 0:
        raise ValueError("cannot cut m_low/m_high from an empty support band")
    p25, p75 = np.percentile(restricted, [25, 75])
    return float(p25), float(p75)


def compute_common_support_band(*, real: ArrayLike, shuffled: ArrayLike, random: ArrayLike) -> BandDecision:
    """Return the Section 2 common-support band decision.

    Three-arm support is the overlap of each arm's [p5, p95] range. It is valid
    iff width >= 0.5 * pooled IQR across all three arms. If that collapses, the
    H-sem two-arm fallback repeats the same rule on real+shuffled only. If that
    also collapses, H-sem and H-grad are undefined at matched magnitude.
    """

    real_arr = _as_1d("real", real)
    shuffled_arr = _as_1d("shuffled", shuffled)
    random_arr = _as_1d("random", random)
    per_arm = {
        "real": _percentiles(real_arr),
        "shuffled": _percentiles(shuffled_arr),
        "random": _percentiles(random_arr),
    }

    pooled_three = np.concatenate([real_arr, shuffled_arr, random_arr])
    three_threshold = 0.5 * _iqr(pooled_three)
    three_lower, three_upper, three_width = _overlap(list(per_arm.values()))

    if three_width >= three_threshold:
        m_low, m_high = _cut_magnitudes(pooled_three, three_lower, three_upper)
        return BandDecision(
            status=BandStatus.THREE_ARM_VALID,
            band_lower=three_lower,
            band_upper=three_upper,
            band_width=three_width,
            threshold=three_threshold,
            m_low=m_low,
            m_high=m_high,
            h_sem_matched=True,
            h_grad_matched=True,
            h_mag_matched=True,
            h_sem_status="MATCHED_MAGNITUDE",
            h_grad_status="MATCHED_MAGNITUDE",
            h_mag_status="MATCHED_MAGNITUDE",
            three_arm_lower=three_lower,
            three_arm_upper=three_upper,
            three_arm_width=three_width,
            three_arm_threshold=three_threshold,
            two_arm_lower=float("nan"),
            two_arm_upper=float("nan"),
            two_arm_width=float("nan"),
            two_arm_threshold=float("nan"),
            per_arm=per_arm,
        )

    two_percentiles = [per_arm["real"], per_arm["shuffled"]]
    pooled_two = np.concatenate([real_arr, shuffled_arr])
    two_threshold = 0.5 * _iqr(pooled_two)
    two_lower, two_upper, two_width = _overlap(two_percentiles)

    if two_width >= two_threshold:
        m_low, m_high = _cut_magnitudes(pooled_two, two_lower, two_upper)
        return BandDecision(
            status=BandStatus.TWO_ARM_FALLBACK,
            band_lower=two_lower,
            band_upper=two_upper,
            band_width=two_width,
            threshold=two_threshold,
            m_low=m_low,
            m_high=m_high,
            h_sem_matched=True,
            h_grad_matched=False,
            h_mag_matched=True,
            h_sem_status="MATCHED_MAGNITUDE",
            h_grad_status="UNDEFINED_AT_MATCHED_MAGNITUDE",
            h_mag_status="MATCHED_MAGNITUDE",
            three_arm_lower=three_lower,
            three_arm_upper=three_upper,
            three_arm_width=three_width,
            three_arm_threshold=three_threshold,
            two_arm_lower=two_lower,
            two_arm_upper=two_upper,
            two_arm_width=two_width,
            two_arm_threshold=two_threshold,
            per_arm=per_arm,
        )

    return BandDecision(
        status=BandStatus.TWO_ARM_TERMINAL,
        band_lower=None,
        band_upper=None,
        band_width=two_width,
        threshold=two_threshold,
        m_low=None,
        m_high=None,
        h_sem_matched=False,
        h_grad_matched=False,
        h_mag_matched=True,
        h_sem_status="UNDEFINED_AT_MATCHED_MAGNITUDE",
        h_grad_status="UNDEFINED_AT_MATCHED_MAGNITUDE",
        h_mag_status="MATCHED_MAGNITUDE",
        three_arm_lower=three_lower,
        three_arm_upper=three_upper,
        three_arm_width=three_width,
        three_arm_threshold=three_threshold,
        two_arm_lower=two_lower,
        two_arm_upper=two_upper,
        two_arm_width=two_width,
        two_arm_threshold=two_threshold,
        per_arm=per_arm,
    )
