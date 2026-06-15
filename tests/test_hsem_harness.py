from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.hsem.harness import (
    DEFAULT_MATERIALITY,
    analyze_toy_hsem,
    fit_intercept,
    null_annotation,
    paired_contrast_equal_weight_identity,
    paired_contrast_from_log_cells,
    verdict_from_ci,
)


def toy_cells(outcome: np.ndarray, *, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for idx, contrast in enumerate(outcome):
        base = f"bp-{idx:03d}"
        shuffled_low = rng.normal(-10.0, 0.2)
        shuffled_high = rng.normal(-9.8, 0.2)
        low_delta = contrast + rng.normal(0.0, 0.05)
        high_delta = 2.0 * contrast - low_delta
        rows.extend(
            [
                {"base_point_id": base, "arm": "real", "magnitude_level": "low", "log_H": shuffled_low + low_delta},
                {
                    "base_point_id": base,
                    "arm": "real",
                    "magnitude_level": "high",
                    "log_H": shuffled_high + high_delta,
                },
                {"base_point_id": base, "arm": "shuffled", "magnitude_level": "low", "log_H": shuffled_low},
                {"base_point_id": base, "arm": "shuffled", "magnitude_level": "high", "log_H": shuffled_high},
                {"base_point_id": base, "arm": "random", "magnitude_level": "low", "log_H": rng.normal(-10.2, 0.2)},
                {"base_point_id": base, "arm": "random", "magnitude_level": "high", "log_H": rng.normal(-10.1, 0.2)},
            ]
        )
    frame = pd.DataFrame(rows)
    frame["H"] = np.exp(frame["log_H"])
    return frame


def test_r1_estimand_identity_equal_weight_average() -> None:
    outcome = np.linspace(-0.2, 0.5, 25)
    frame = toy_cells(outcome)
    first = paired_contrast_from_log_cells(frame)
    second = paired_contrast_equal_weight_identity(frame)
    assert np.allclose(first.to_numpy(), second.to_numpy())


def test_centered_ols_recovers_planted_adjusted_mean_and_ci_coverage() -> None:
    rng = np.random.default_rng(123)
    planted = 0.31
    slopes = np.array([0.42, -0.25])
    cover = 0
    hc3_same_verdict = 0
    repeats = 160
    for _ in range(repeats):
        n = 240
        cov = pd.DataFrame(
            {
                "manifold_distance_diff": rng.normal(0.5, 1.2, n),
                "phi_diff": rng.normal(-0.2, 0.8, n),
            },
            index=[f"bp-{idx:03d}" for idx in range(n)],
        )
        centered = cov - cov.mean(axis=0)
        noise = rng.normal(0.0, 0.35, n)
        outcome = planted + centered.to_numpy() @ slopes + noise
        fit = fit_intercept(pd.Series(outcome, index=cov.index), cov, alpha=0.05, center=True)
        hc3 = fit_intercept(pd.Series(outcome, index=cov.index), cov, alpha=0.05, center=True, robust=True)
        cover += int(fit.ci_low <= planted <= fit.ci_high)
        hc3_same_verdict += int(
            verdict_from_ci(fit.ci_low, fit.ci_high, threshold=DEFAULT_MATERIALITY)
            == verdict_from_ci(hc3.ci_low, hc3.ci_high, threshold=DEFAULT_MATERIALITY)
        )
    assert 0.90 <= cover / repeats <= 1.0
    assert hc3_same_verdict / repeats >= 0.95


def test_centering_matters_and_uncentered_intercept_is_effect_at_parity() -> None:
    rng = np.random.default_rng(44)
    n = 80
    adjusted_mean = 0.4
    slopes = np.array([0.5, -0.75])
    cov = pd.DataFrame(
        {
            "manifold_distance_diff": rng.normal(2.0, 0.1, n),
            "phi_diff": rng.normal(-1.0, 0.1, n),
        },
        index=[f"bp-{idx:03d}" for idx in range(n)],
    )
    centered = cov - cov.mean(axis=0)
    outcome = adjusted_mean + centered.to_numpy() @ slopes
    centered_fit = fit_intercept(pd.Series(outcome, index=cov.index), cov, center=True)
    uncentered_fit = fit_intercept(pd.Series(outcome, index=cov.index), cov, center=False)
    predicted_parity = adjusted_mean - float(cov.mean(axis=0).to_numpy() @ slopes)
    assert np.isclose(centered_fit.intercept, adjusted_mean)
    assert np.isclose(uncentered_fit.intercept, predicted_parity)
    assert not np.isclose(centered_fit.intercept, uncentered_fit.intercept)


def test_verdict_logic_positions_relative_to_threshold() -> None:
    threshold = DEFAULT_MATERIALITY
    assert verdict_from_ci(threshold + 0.01, threshold + 0.2, threshold=threshold) == "CORROBORATED"
    assert verdict_from_ci(threshold - 0.2, threshold - 0.01, threshold=threshold) == "FALSIFIED"
    assert verdict_from_ci(threshold - 0.01, threshold + 0.01, threshold=threshold) == "INCONCLUSIVE"


def test_null_attributed_and_attenuated_inconclusive_logic() -> None:
    assert null_annotation("FALSIFIED", "CORROBORATED") == "NULL-ATTRIBUTED"
    assert null_annotation("INCONCLUSIVE", "CORROBORATED") == "ATTENUATED-INCONCLUSIVE"
    assert null_annotation("CORROBORATED", "CORROBORATED") == "NONE"
    assert null_annotation("FALSIFIED", "FALSIFIED") == "NONE"


def test_full_toy_harness_verdict_and_null_annotation() -> None:
    n = 60
    base_index = [f"bp-{idx:03d}" for idx in range(n)]
    cov = pd.DataFrame(
        {
            "manifold_distance_diff": np.linspace(0.0, 2.0, n),
            "phi_diff": np.linspace(1.0, -1.0, n),
        },
        index=base_index,
    )
    centered = cov - cov.mean(axis=0)
    outcome = 0.45 + centered.to_numpy() @ np.array([0.2, -0.1])
    frame = toy_cells(outcome)
    result = analyze_toy_hsem(frame, cov, threshold=DEFAULT_MATERIALITY)
    assert result.verdict == "CORROBORATED"
    assert result.null_annotation == "NONE"

