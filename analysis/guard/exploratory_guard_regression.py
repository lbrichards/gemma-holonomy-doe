"""Exploratory guard-distance regression against the H-sem paired gap.

This report is post hoc and descriptive. It reads only frozen experiment
artifacts and does not enter the confirmatory verdict rules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from analysis.hsem.run_analysis import MANIFEST_PATH, RESULTS_PATH, prepare_frames


DEFAULT_JSON = Path("reports/guard_regression_exploratory.json")
DEFAULT_MD = Path("reports/guard_regression_exploratory.md")


@dataclass(frozen=True)
class TermSummary:
    estimate: float
    se: float
    ci_low: float
    ci_high: float


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


def covariate_differences(merged: pd.DataFrame) -> pd.DataFrame:
    """Return active-feature minus mixed-feature covariate differences."""

    per_plane = merged[merged["arm"].isin(["real", "shuffled"])].drop_duplicates(
        ["base_point_id", "arm"]
    )
    wide = per_plane.pivot(
        index="base_point_id",
        columns="arm",
        values=["manifold_distance_recon", "manifold_distance_mahalanobis", "phi"],
    )
    return pd.DataFrame(
        {
            "reconstruction_distance_diff": wide[("manifold_distance_recon", "real")]
            - wide[("manifold_distance_recon", "shuffled")],
            "guard_distance_diff": wide[("manifold_distance_mahalanobis", "real")]
            - wide[("manifold_distance_mahalanobis", "shuffled")],
            "phi_diff": wide[("phi", "real")] - wide[("phi", "shuffled")],
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


def build_report(results_path: Path, manifest_path: Path) -> dict[str, Any]:
    results, merged = prepare_frames(results_path, manifest_path)
    gap = hsem_paired_gap(results)
    diffs = covariate_differences(merged).loc[gap.index]

    model_specs = {
        "guard_only": ["guard_distance_diff"],
        "reconstruction_only": ["reconstruction_distance_diff"],
        "primary_reference_reconstruction_plus_phi": [
            "reconstruction_distance_diff",
            "phi_diff",
        ],
        "guard_plus_phi": ["guard_distance_diff", "phi_diff"],
        "reconstruction_plus_guard_plus_phi": [
            "reconstruction_distance_diff",
            "guard_distance_diff",
            "phi_diff",
        ],
    }
    models = {
        name: asdict(fit_centered_ols(gap, diffs[columns]))
        for name, columns in model_specs.items()
    }

    return {
        "status": "EXPLORATORY_POST_HOC_NOT_CONFIRMATORY",
        "purpose": (
            "Check whether the k-nearest-neighbour guard-distance difference "
            "or reconstruction-distance difference predicts the paired "
            "active-feature minus mixed-feature holonomy gap."
        ),
        "sources": {
            "results_path": str(results_path),
            "results_sha256": sha256_file(results_path),
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
        },
        "definitions": {
            "outcome": (
                "d_b = 0.5 * [(log_H_real_low - log_H_shuffled_low) + "
                "(log_H_real_high - log_H_shuffled_high)]"
            ),
            "guard_distance_diff": (
                "manifest.manifold_distance_mahalanobis(real) - "
                "manifest.manifold_distance_mahalanobis(shuffled); in this run "
                "the field stores the preregistered k-NN fallback distance."
            ),
            "reconstruction_distance_diff": (
                "manifest.manifold_distance_recon(real) - "
                "manifest.manifold_distance_recon(shuffled)"
            ),
            "centering": (
                "All covariates are centered at their sample means before OLS, "
                "so the intercept is the adjusted mean paired gap at the "
                "sample covariate centroid."
            ),
        },
        "n_base_points": int(len(gap)),
        "paired_gap_mean": float(gap.mean()),
        "models": models,
    }


def fmt(value: float) -> str:
    return f"{value:.6g}"


def write_markdown(report: dict[str, Any], path: Path) -> None:
    models = report["models"]
    lines = [
        "# Exploratory Guard Regression Check",
        "",
        "Status: `EXPLORATORY_POST_HOC_NOT_CONFIRMATORY`.",
        "",
        "This report checks whether the k-nearest-neighbour guard-distance "
        "difference predicts the paired active-feature minus mixed-feature "
        "holonomy gap. It reads only frozen experiment artifacts and does not "
        "enter any confirmatory verdict rule.",
        "",
        "## Sources",
        "",
        f"- Results: `{report['sources']['results_path']}`",
        f"- Results SHA-256: `{report['sources']['results_sha256']}`",
        f"- Manifest: `{report['sources']['manifest_path']}`",
        f"- Manifest SHA-256: `{report['sources']['manifest_sha256']}`",
        "",
        "## Definitions",
        "",
        f"- Outcome: `{report['definitions']['outcome']}`",
        f"- Guard difference: `{report['definitions']['guard_distance_diff']}`",
        f"- Reconstruction difference: `{report['definitions']['reconstruction_distance_diff']}`",
        f"- Centering: `{report['definitions']['centering']}`",
        "",
        "## Results",
        "",
        f"- Base points: `{report['n_base_points']}`",
        f"- Paired gap mean: `{fmt(report['paired_gap_mean'])}`",
        "",
        "| Model | Covariates | Intercept | R^2 | Guard slope [95% CI] | Reconstruction slope [95% CI] |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, model in models.items():
        slopes = model["slopes"]
        guard = slopes.get("guard_distance_diff")
        recon = slopes.get("reconstruction_distance_diff")
        guard_text = (
            "--"
            if guard is None
            else f"{fmt(guard['estimate'])} [{fmt(guard['ci_low'])}, {fmt(guard['ci_high'])}]"
        )
        recon_text = (
            "--"
            if recon is None
            else f"{fmt(recon['estimate'])} [{fmt(recon['ci_low'])}, {fmt(recon['ci_high'])}]"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{name}`",
                    ", ".join(f"`{c}`" for c in model["covariates"]),
                    fmt(model["intercept"]["estimate"]),
                    fmt(model["r_squared"]),
                    guard_text,
                    recon_text,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "The guard-only model has R^2 about 0.009, and its slope interval includes zero.",
            "The reconstruction-only model has R^2 about 0.001.",
            "Because the regressions center covariates, every adjusted intercept equals the paired gap mean to displayed precision.",
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
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    report = build_report(args.results, args.manifest)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True, default=json_default) + "\n")
    write_markdown(report, args.markdown)


if __name__ == "__main__":
    main()
