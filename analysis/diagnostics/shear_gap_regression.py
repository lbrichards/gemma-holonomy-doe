"""Post-freeze shear-gap regression diagnostic.

Status: POST_FREEZE_DIAGNOSTIC_NOT_CONFIRMATORY
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from analysis.validation.common import STATUS_DIAGNOSTIC, markdown_table, write_json


DEFAULT_CHECKPOINT = Path("reports/full_transport_shear_diagnostic.checkpoint.jsonl")
DEFAULT_JSON = Path("reports/shear_gap_regression.json")
DEFAULT_MD = Path("reports/shear_gap_regression.md")

PRIMARY_METRIC = "symmetric_residual_norm"
SECONDARY_METRICS = ["non_orthogonality_norm", "transport_residual_norm", "max_condition"]


@dataclass(frozen=True)
class TermResult:
    estimate: float
    se: float
    ci_low: float
    ci_high: float
    p_value: float


@dataclass(frozen=True)
class RegressionResult:
    metric: str
    n_base_points: int
    mean_paired_gap: float
    mean_metric_difference: float
    intercept: TermResult
    slope: TermResult
    hc3_intercept: TermResult
    hc3_slope: TermResult
    r_squared: float
    df_resid: float


def load_checkpoint_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def rows_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if "error" in frame.columns and frame["error"].notna().any():
        errors = frame[frame["error"].notna()][["base_point_id", "arm", "magnitude_level", "error"]]
        raise RuntimeError(f"checkpoint contains failed rows:\n{errors.to_string(index=False)}")
    expected = {"base_point_id", "arm", "magnitude_level", "log_H", PRIMARY_METRIC, *SECONDARY_METRICS}
    missing = expected - set(frame.columns)
    if missing:
        raise RuntimeError(f"checkpoint missing columns: {sorted(missing)}")
    return frame


def paired_gap_and_metric_diff(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    verdict = frame[frame["arm"].isin(["real", "shuffled"])].copy()
    wide_log = verdict.pivot(index="base_point_id", columns=["arm", "magnitude_level"], values="log_H")
    wide_metric = verdict.pivot(index="base_point_id", columns=["arm", "magnitude_level"], values=metric)

    required = [("real", "low"), ("real", "high"), ("shuffled", "low"), ("shuffled", "high")]
    for columns, name in [(wide_log.columns, "log_H"), (wide_metric.columns, metric)]:
        missing = [key for key in required if key not in columns]
        if missing:
            raise RuntimeError(f"{name}: missing required cells {missing}")

    out = pd.DataFrame(index=wide_log.index)
    out["gap"] = 0.5 * (
        (wide_log[("real", "low")] - wide_log[("shuffled", "low")])
        + (wide_log[("real", "high")] - wide_log[("shuffled", "high")])
    )
    out[f"{metric}_diff"] = 0.5 * (
        (wide_metric[("real", "low")] - wide_metric[("shuffled", "low")])
        + (wide_metric[("real", "high")] - wide_metric[("shuffled", "high")])
    )
    out = out.dropna()
    return out


def term_from_model(model: Any, idx: int) -> TermResult:
    conf = model.conf_int(alpha=0.05)
    return TermResult(
        estimate=float(model.params[idx]),
        se=float(model.bse[idx]),
        ci_low=float(conf[idx, 0]),
        ci_high=float(conf[idx, 1]),
        p_value=float(model.pvalues[idx]),
    )


def fit_centered_regression(data: pd.DataFrame, metric: str) -> RegressionResult:
    cov_name = f"{metric}_diff"
    x = data[cov_name].astype(float)
    y = data["gap"].astype(float)
    centered = x - x.mean()
    design = sm.add_constant(centered.to_numpy(), has_constant="add")
    model = sm.OLS(y.to_numpy(), design).fit()
    hc3 = model.get_robustcov_results(cov_type="HC3")
    return RegressionResult(
        metric=metric,
        n_base_points=int(len(data)),
        mean_paired_gap=float(y.mean()),
        mean_metric_difference=float(x.mean()),
        intercept=term_from_model(model, 0),
        slope=term_from_model(model, 1),
        hc3_intercept=term_from_model(hc3, 0),
        hc3_slope=term_from_model(hc3, 1),
        r_squared=float(model.rsquared),
        df_resid=float(model.df_resid),
    )


def build_report(checkpoint_path: Path) -> dict[str, Any]:
    rows = load_checkpoint_rows(checkpoint_path)
    frame = rows_to_frame(rows)
    regressions = {}
    for metric in [PRIMARY_METRIC, *SECONDARY_METRICS]:
        data = paired_gap_and_metric_diff(frame, metric)
        regressions[metric] = asdict(fit_centered_regression(data, metric))

    primary = regressions[PRIMARY_METRIC]
    return {
        "status": STATUS_DIAGNOSTIC,
        "source_checkpoint": str(checkpoint_path),
        "definitions": {
            "gap_b": "0.5 * [(log_H_real_low - log_H_shuffled_low) + (log_H_real_high - log_H_shuffled_high)]",
            "metric_diff_b": "0.5 * [(metric_real_low - metric_shuffled_low) + (metric_real_high - metric_shuffled_high)]",
            "primary_metric": PRIMARY_METRIC,
            "centering": "Each metric difference is centered at its sample mean before OLS; the intercept equals the mean paired gap.",
        },
        "primary_metric": PRIMARY_METRIC,
        "n_checkpoint_rows": int(len(rows)),
        "primary_result": primary,
        "regressions": regressions,
    }


def fmt(value: float) -> str:
    return f"{value:.6g}"


def term_text(term: dict[str, float]) -> str:
    return (
        f"{fmt(term['estimate'])} "
        f"(SE {fmt(term['se'])}, 95% CI [{fmt(term['ci_low'])}, {fmt(term['ci_high'])}], "
        f"p={fmt(term['p_value'])})"
    )


def write_markdown(report: dict[str, Any], path: Path) -> None:
    rows = []
    for metric, result in report["regressions"].items():
        rows.append(
            {
                "metric": metric,
                "n_base_points": result["n_base_points"],
                "mean_paired_gap": result["mean_paired_gap"],
                "mean_metric_difference": result["mean_metric_difference"],
                "slope": result["slope"]["estimate"],
                "slope_se": result["slope"]["se"],
                "slope_ci": f"[{fmt(result['slope']['ci_low'])}, {fmt(result['slope']['ci_high'])}]",
                "slope_p": result["slope"]["p_value"],
                "r_squared": result["r_squared"],
                "hc3_slope_se": result["hc3_slope"]["se"],
                "hc3_slope_ci": f"[{fmt(result['hc3_slope']['ci_low'])}, {fmt(result['hc3_slope']['ci_high'])}]",
                "hc3_slope_p": result["hc3_slope"]["p_value"],
            }
        )

    primary = report["primary_result"]
    hc3_crosses_zero = primary["hc3_slope"]["ci_low"] <= 0 <= primary["hc3_slope"]["ci_high"]
    low_r2 = primary["r_squared"] < 0.05
    if low_r2 and hc3_crosses_zero:
        interpretation = (
            "Transport shear/non-orthogonality is present and may differ by arm, "
            "but the active-minus-mixed symmetric-residual difference does not "
            "linearly predict the paired holonomy gap in this diagnostic."
        )
    else:
        interpretation = (
            "Transport shear/non-orthogonality is a live mechanism or confound for "
            "the observed reversal. The frozen verdict remains unchanged, but this "
            "should be reported as a serious post-freeze diagnostic finding and "
            "the paper should avoid claiming that shear has been bounded away."
        )

    md = f"""# Full Shear Gap Regression

Status: {STATUS_DIAGNOSTIC}

This post-freeze diagnostic recomputes transport distortion quantities on the frozen verdict-bearing real/shuffled planes. It does not enter or modify the preregistered decision rule.

## Source

- Checkpoint: `{report['source_checkpoint']}`
- Checkpoint rows: `{report['n_checkpoint_rows']}`

## Definitions

- Outcome: `{report['definitions']['gap_b']}`
- Metric difference: `{report['definitions']['metric_diff_b']}`
- Primary metric: `{report['definitions']['primary_metric']}`
- Centering: `{report['definitions']['centering']}`

## Primary Result

- Base points: `{primary['n_base_points']}`
- Mean paired gap: `{fmt(primary['mean_paired_gap'])}`
- Mean shear difference: `{fmt(primary['mean_metric_difference'])}`
- Centered-OLS intercept: `{term_text(primary['intercept'])}`
- Slope: `{term_text(primary['slope'])}`
- R^2: `{fmt(primary['r_squared'])}`
- HC3 slope: `{term_text(primary['hc3_slope'])}`

{interpretation}

Nonlinear, interaction, and readout-dependent explanations remain possible.

## Regression Table

{markdown_table(rows, ['metric', 'n_base_points', 'mean_paired_gap', 'mean_metric_difference', 'slope', 'slope_se', 'slope_ci', 'slope_p', 'r_squared', 'hc3_slope_se', 'hc3_slope_ci', 'hc3_slope_p'])}
"""
    path.write_text(md)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--json-out", default=str(DEFAULT_JSON))
    parser.add_argument("--md-out", default=str(DEFAULT_MD))
    args = parser.parse_args()

    report = build_report(Path(args.checkpoint))
    write_json(Path(args.json_out), report)
    write_markdown(report, Path(args.md_out))


if __name__ == "__main__":
    main()
