"""H-sem analysis harness, validated on toy data before real execution.

This module implements the implementation choices recorded in
``HSEM_ANALYSIS_PLAN_ADDENDUM_DRAFT.md``. The CLI defaults to a dry run that
performs structural checks only and halts before computing any real ``d_b``.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm


ARMS = ("real", "shuffled", "random")
MAGNITUDE_LEVELS = ("low", "high")
DEFAULT_MATERIALITY = 0.2231
FORBIDDEN_REAL_MESSAGE = "Plan not frozen — halting before estimation."


Verdict = Literal["CORROBORATED", "FALSIFIED", "INCONCLUSIVE"]
NullAnnotation = Literal["NONE", "NULL-ATTRIBUTED", "ATTENUATED-INCONCLUSIVE"]


@dataclass(frozen=True)
class StructuralReceipt:
    """Structural checks that do not estimate H-sem."""

    rows: int
    distinct_base_points: int
    expected_rows_ok: bool
    expected_base_points_ok: bool
    arms_present: list[str]
    magnitude_levels_present: list[str]
    complete_three_by_two_per_base: bool
    real_shuffled_low_high_complete: bool
    nonfinite_rows: int
    nonpositive_h_rows: int


@dataclass(frozen=True)
class FitSummary:
    """OLS intercept summary for one H-sem contrast."""

    intercept: float
    ci_low: float
    ci_high: float
    se: float
    df_resid: float
    alpha: float


@dataclass(frozen=True)
class HsemResult:
    """Container for toy-oracle H-sem computations."""

    adjusted: FitSummary
    adjusted_hc3: FitSummary
    unadjusted: FitSummary
    uncentered_parity: FitSummary
    verdict: Verdict
    hc3_verdict: Verdict
    null_annotation: NullAnnotation


def structural_assertions(frame: pd.DataFrame) -> StructuralReceipt:
    """Return structural checks permitted before the analysis plan is frozen."""

    required = {"base_point_id", "arm", "magnitude_level", "H", "log_H"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise AssertionError(f"missing required columns: {missing}")
    numeric_columns = ["H", "log_H", "theta", "signed_holonomy", "A_enclosed", "det_M"]
    present_numeric = [column for column in numeric_columns if column in frame.columns]
    finite_mask = frame[present_numeric].map(lambda value: math.isfinite(float(value))).all(axis=1)
    nonfinite_rows = int((~finite_mask).sum())
    nonpositive_h_rows = int((frame["H"] <= 0).sum())
    cell_counts = frame.groupby(["base_point_id", "arm", "magnitude_level"], observed=True).size()
    expected_cells = pd.MultiIndex.from_product(
        [sorted(frame["base_point_id"].unique()), ARMS, MAGNITUDE_LEVELS],
        names=["base_point_id", "arm", "magnitude_level"],
    )
    complete_three_by_two = cell_counts.reindex(expected_cells, fill_value=0).eq(1).all()
    rs_expected = pd.MultiIndex.from_product(
        [sorted(frame["base_point_id"].unique()), ("real", "shuffled"), MAGNITUDE_LEVELS],
        names=["base_point_id", "arm", "magnitude_level"],
    )
    real_shuffled_complete = cell_counts.reindex(rs_expected, fill_value=0).eq(1).all()
    receipt = StructuralReceipt(
        rows=int(len(frame)),
        distinct_base_points=int(frame["base_point_id"].nunique()),
        expected_rows_ok=len(frame) == 2340,
        expected_base_points_ok=frame["base_point_id"].nunique() == 390,
        arms_present=sorted(str(value) for value in frame["arm"].unique()),
        magnitude_levels_present=sorted(str(value) for value in frame["magnitude_level"].unique()),
        complete_three_by_two_per_base=bool(complete_three_by_two),
        real_shuffled_low_high_complete=bool(real_shuffled_complete),
        nonfinite_rows=nonfinite_rows,
        nonpositive_h_rows=nonpositive_h_rows,
    )
    if not all(
        [
            receipt.expected_rows_ok,
            receipt.expected_base_points_ok,
            receipt.complete_three_by_two_per_base,
            receipt.real_shuffled_low_high_complete,
            receipt.nonfinite_rows == 0,
            receipt.nonpositive_h_rows == 0,
        ]
    ):
        raise AssertionError(f"structural assertions failed: {asdict(receipt)}")
    return receipt


def paired_contrast_from_log_cells(frame: pd.DataFrame) -> pd.Series:
    """Return R1 ``d_b`` from cells already containing log-H values."""

    subset = frame[frame["arm"].isin(["real", "shuffled"])]
    means = subset.groupby(["base_point_id", "arm"], observed=True)["log_H"].mean().unstack()
    return means["real"] - means["shuffled"]


def paired_contrast_equal_weight_identity(frame: pd.DataFrame) -> pd.Series:
    """Return R1 contrast as equal-weight average of low/high contrasts."""

    subset = frame[frame["arm"].isin(["real", "shuffled"])]
    cells = subset.pivot(index="base_point_id", columns=["arm", "magnitude_level"], values="log_H")
    low = cells[("real", "low")] - cells[("shuffled", "low")]
    high = cells[("real", "high")] - cells[("shuffled", "high")]
    return 0.5 * (low + high)


def center_covariates(covariates: pd.DataFrame) -> pd.DataFrame:
    """Center covariate differences at sample means for the ANCOVA intercept."""

    return covariates - covariates.mean(axis=0)


def fit_intercept(
    outcome: pd.Series,
    covariates: pd.DataFrame | None = None,
    *,
    alpha: float = 0.05,
    robust: bool = False,
    center: bool = True,
) -> FitSummary:
    """Fit OLS and return the intercept interval.

    When ``center`` is true, the intercept is the adjusted mean at the sample
    covariate centroid. With ``covariates=None`` this is the unadjusted mean.
    """

    y = outcome.astype(float)
    if covariates is None or covariates.shape[1] == 0:
        x = pd.DataFrame({"const": np.ones(len(y), dtype=np.float64)}, index=y.index)
    else:
        cov = covariates.astype(float)
        if center:
            cov = center_covariates(cov)
        x = sm.add_constant(cov, has_constant="add")
    model = sm.OLS(y.to_numpy(dtype=np.float64), x.to_numpy(dtype=np.float64)).fit()
    if robust:
        model = model.get_robustcov_results(cov_type="HC3")
    conf = model.conf_int(alpha=alpha)
    return FitSummary(
        intercept=float(model.params[0]),
        ci_low=float(conf[0, 0]),
        ci_high=float(conf[0, 1]),
        se=float(model.bse[0]),
        df_resid=float(model.df_resid),
        alpha=float(alpha),
    )


def verdict_from_ci(ci_low: float, ci_high: float, *, threshold: float = DEFAULT_MATERIALITY) -> Verdict:
    """Apply the frozen materiality-threshold verdict rule."""

    if ci_low > threshold:
        return "CORROBORATED"
    if ci_high < threshold:
        return "FALSIFIED"
    return "INCONCLUSIVE"


def null_annotation(adjusted: Verdict, unadjusted: Verdict) -> NullAnnotation:
    """Apply R4 NULL-ATTRIBUTED and attenuated-inconclusive logic."""

    if unadjusted == "CORROBORATED" and adjusted == "FALSIFIED":
        return "NULL-ATTRIBUTED"
    if unadjusted == "CORROBORATED" and adjusted == "INCONCLUSIVE":
        return "ATTENUATED-INCONCLUSIVE"
    return "NONE"


def analyze_toy_hsem(
    frame: pd.DataFrame,
    covariate_differences: pd.DataFrame,
    *,
    alpha: float = 0.05,
    threshold: float = DEFAULT_MATERIALITY,
) -> HsemResult:
    """Run the H-sem addendum procedure on toy/simulated data."""

    outcome = paired_contrast_from_log_cells(frame)
    covariates = covariate_differences.loc[outcome.index]
    adjusted = fit_intercept(outcome, covariates, alpha=alpha, robust=False, center=True)
    adjusted_hc3 = fit_intercept(outcome, covariates, alpha=alpha, robust=True, center=True)
    unadjusted = fit_intercept(outcome, None, alpha=alpha)
    uncentered = fit_intercept(outcome, covariates, alpha=alpha, robust=False, center=False)
    primary_verdict = verdict_from_ci(adjusted.ci_low, adjusted.ci_high, threshold=threshold)
    robust_verdict = verdict_from_ci(adjusted_hc3.ci_low, adjusted_hc3.ci_high, threshold=threshold)
    raw_verdict = verdict_from_ci(unadjusted.ci_low, unadjusted.ci_high, threshold=threshold)
    return HsemResult(
        adjusted=adjusted,
        adjusted_hc3=adjusted_hc3,
        unadjusted=unadjusted,
        uncentered_parity=uncentered,
        verdict=primary_verdict,
        hc3_verdict=robust_verdict,
        null_annotation=null_annotation(primary_verdict, raw_verdict),
    )


def assert_no_real_estimation(execute: bool) -> None:
    """Guard real execution until Larry explicitly freezes the plan."""

    if not execute:
        print(FORBIDDEN_REAL_MESSAGE)
        raise SystemExit(0)


def dry_run_real_artifact(path: Path, *, execute: bool = False) -> StructuralReceipt:
    """Load real artifact for structural checks only, then halt unless enabled."""

    frame = pd.read_parquet(path)
    receipt = structural_assertions(frame)
    assert_no_real_estimation(execute)
    return receipt


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {value!r}")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="H-sem analysis harness")
    parser.add_argument("--results", type=Path, default=Path("run/holonomy_results_n390.parquet"))
    parser.add_argument("--execute", action="store_true", help="disabled until the plan is frozen")
    args = parser.parse_args(list(argv) if argv is not None else None)
    receipt = dry_run_real_artifact(args.results, execute=args.execute)
    print(json.dumps(asdict(receipt), indent=2, default=_json_default))


if __name__ == "__main__":
    main()

