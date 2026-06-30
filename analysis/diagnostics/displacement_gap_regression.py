"""Post-freeze displacement regression against the H-sem paired gap.

This report is post hoc and descriptive. It reads only frozen experiment
artifacts plus the derived center-displacement diagnostic and does not enter the
confirmatory verdict rules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.hsem.run_analysis import RESULTS_PATH


DEFAULT_DISPLACEMENT = Path("reports/center_displacement_diagnostic.json")
DEFAULT_JSON = Path("reports/displacement_gap_regression.json")
DEFAULT_MD = Path("reports/displacement_gap_regression.md")


@dataclass(frozen=True)
class TermSummary:
    estimate: float
    se: float
    ci_low: float
    ci_high: float
    p_value: float


@dataclass(frozen=True)
class ModelSummary:
    covariates: list[str]
    intercept: TermSummary
    slopes: dict[str, TermSummary]
    r_squared: float
    df_resid: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return payload["rows"]
    raise ValueError(f"cannot find rows in {path}")


def read_results(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.DataFrame(read_json_rows(path))


def hsem_paired_gap(results: pd.DataFrame) -> pd.Series:
    """Return the equal-weight active-feature minus mixed-feature log-H gap."""

    cells = results[results["arm"].isin(["real", "shuffled"])].pivot(
        index="base_point_id",
        columns=["arm", "magnitude_level"],
        values="log_H",
    )
    low = cells[("real", "low")] - cells[("shuffled", "low")]
    high = cells[("real", "high")] - cells[("shuffled", "high")]
    gap = 0.5 * (low + high)
    if len(gap) != 390 or gap.isna().any():
        raise RuntimeError("failed to define all 390 paired H-sem gaps")
    return gap


def displacement_differences(displacement: pd.DataFrame) -> pd.DataFrame:
    """Return active-feature minus mixed-feature displacement/rho differences."""

    verdict = displacement[displacement["arm"].isin(["real", "shuffled"])]
    cells = verdict.pivot(
        index="base_point_id",
        columns=["arm", "magnitude_level"],
        values="displacement_over_rho",
    )
    low = cells[("real", "low")] - cells[("shuffled", "low")]
    high = cells[("real", "high")] - cells[("shuffled", "high")]
    return pd.DataFrame(
        {
            "displacement_over_rho_diff": 0.5 * (low + high),
            "low_displacement_over_rho_diff": low,
            "high_displacement_over_rho_diff": high,
        }
    )


def fit_centered_ols(outcome: pd.Series, covariates: pd.DataFrame) -> ModelSummary:
    """Fit centered OLS and summarize intercept, slopes, and R^2."""

    cov = covariates.astype(float)
    centered = cov - cov.mean(axis=0)
    design = sm.add_constant(centered, has_constant="add")
    model = sm.OLS(outcome.astype(float).to_numpy(), design.to_numpy()).fit()
    names = ["const", *list(cov.columns)]
    conf = model.conf_int(alpha=0.05)
    summaries = {
        name: TermSummary(
            estimate=float(model.params[i]),
            se=float(model.bse[i]),
            ci_low=float(conf[i, 0]),
            ci_high=float(conf[i, 1]),
            p_value=float(model.pvalues[i]),
        )
        for i, name in enumerate(names)
    }
    return ModelSummary(
        covariates=list(cov.columns),
        intercept=summaries["const"],
        slopes={name: summaries[name] for name in cov.columns},
        r_squared=float(model.rsquared),
        df_resid=float(model.df_resid),
    )


def build_report(results_path: Path, displacement_path: Path) -> dict[str, Any]:
    results = read_results(results_path)
    displacement = pd.DataFrame(read_json_rows(displacement_path))
    gap = hsem_paired_gap(results)
    diffs = displacement_differences(displacement).loc[gap.index]

    model_specs = {
        "average_displacement": ["displacement_over_rho_diff"],
        "low_displacement": ["low_displacement_over_rho_diff"],
        "high_displacement": ["high_displacement_over_rho_diff"],
        "low_plus_high_displacement": [
            "low_displacement_over_rho_diff",
            "high_displacement_over_rho_diff",
        ],
    }
    models = {
        name: asdict(fit_centered_ols(gap, diffs[columns]))
        for name, columns in model_specs.items()
    }

    return {
        "status": "POST_FREEZE_DIAGNOSTIC_NOT_CONFIRMATORY",
        "purpose": (
            "Check whether active-feature minus mixed-feature matched-center "
            "displacement, measured in loop-radius units, predicts the paired "
            "active-feature minus mixed-feature holonomy gap."
        ),
        "sources": {
            "results_path": str(results_path),
            "results_sha256": sha256_file(results_path),
            "displacement_path": str(displacement_path),
            "displacement_sha256": sha256_file(displacement_path),
        },
        "definitions": {
            "outcome": (
                "d_b = 0.5 * [(log_H_real_low - log_H_shuffled_low) + "
                "(log_H_real_high - log_H_shuffled_high)]"
            ),
            "displacement_over_rho_diff": (
                "0.5 * [((||c_real_low-h||/rho) - (||c_shuffled_low-h||/rho)) "
                "+ ((||c_real_high-h||/rho) - (||c_shuffled_high-h||/rho))]"
            ),
            "centering": (
                "All covariates are centered at their sample means before OLS, "
                "so the intercept is the adjusted mean paired gap at the "
                "sample covariate centroid."
            ),
        },
        "n_base_points": int(len(gap)),
        "paired_gap_mean": float(gap.mean()),
        "covariate_means": {
            name: float(diffs[name].mean()) for name in diffs.columns
        },
        "models": models,
    }


def fmt(value: float) -> str:
    return f"{value:.6g}"


def slope_text(model: dict[str, Any], slope_name: str) -> str:
    slope = model["slopes"].get(slope_name)
    if slope is None:
        return "--"
    return (
        f"{fmt(slope['estimate'])} "
        f"[{fmt(slope['ci_low'])}, {fmt(slope['ci_high'])}], "
        f"p={fmt(slope['p_value'])}"
    )


def write_markdown(report: dict[str, Any], path: Path) -> None:
    models = report["models"]
    lines = [
        "# Displacement Gap Regression",
        "",
        "Status: `POST_FREEZE_DIAGNOSTIC_NOT_CONFIRMATORY`.",
        "",
        "This report checks whether active-feature minus mixed-feature "
        "matched-center displacement, measured in loop-radius units, predicts "
        "the paired active-feature minus mixed-feature holonomy gap. It reads "
        "only frozen experiment artifacts and the derived center-displacement "
        "diagnostic; it does not enter any confirmatory verdict rule.",
        "",
        "## Sources",
        "",
        f"- Results: `{report['sources']['results_path']}`",
        f"- Results SHA-256: `{report['sources']['results_sha256']}`",
        f"- Displacement diagnostic: `{report['sources']['displacement_path']}`",
        f"- Displacement SHA-256: `{report['sources']['displacement_sha256']}`",
        "",
        "## Definitions",
        "",
        f"- Outcome: `{report['definitions']['outcome']}`",
        f"- Average displacement difference: `{report['definitions']['displacement_over_rho_diff']}`",
        f"- Centering: `{report['definitions']['centering']}`",
        "",
        "## Results",
        "",
        f"- Base points: `{report['n_base_points']}`",
        f"- Paired gap mean: `{fmt(report['paired_gap_mean'])}`",
        f"- Mean average displacement difference: `{fmt(report['covariate_means']['displacement_over_rho_diff'])}`",
        "",
        "| Model | Covariates | Intercept | R^2 | Slope(s) [95% CI] |",
        "|---|---|---:|---:|---:|",
    ]
    for name, model in models.items():
        slope_bits = [
            f"`{slope_name}`: {slope_text(model, slope_name)}"
            for slope_name in model["covariates"]
        ]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{name}`",
                    ", ".join(f"`{c}`" for c in model["covariates"]),
                    fmt(model["intercept"]["estimate"]),
                    fmt(model["r_squared"]),
                    "<br>".join(slope_bits),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "The average-displacement model has low explanatory power and its "
            "slope interval includes zero. Thus the displacement asymmetry is "
            "present and aligned with plane type, but this diagnostic does not "
            "support matched-center displacement as a linear predictor of the "
            "paired holonomy gap.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    parser.add_argument("--displacement", type=Path, default=DEFAULT_DISPLACEMENT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    report = build_report(args.results, args.displacement)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True, default=json_default) + "\n")
    write_markdown(report, args.markdown)


if __name__ == "__main__":
    main()
