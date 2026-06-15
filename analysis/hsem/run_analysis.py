"""Execute the frozen H-sem and secondary-model analysis.

This module is intentionally narrow: it reads the committed manifest/results,
checks byte identity against the frozen results commit, writes reports, and
does not modify any ``run/`` artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import OLSInfluence

from analysis.hsem.harness import (
    DEFAULT_MATERIALITY,
    fit_intercept,
    null_annotation,
    structural_assertions,
    verdict_from_ci,
)
from manifest import manifest_from_json


FROZEN_RESULTS_COMMIT = "51931280063d30c8cef2bf05d75a3e44cf996b8c"
RESULTS_PATH = Path("run/holonomy_results_n390.parquet")
MANIFEST_PATH = Path("run/manifest_n390.json")
REPORT_MD = Path("reports/hsem_results.md")
REPORT_JSON = Path("reports/hsem_results.json")


def _git(*args: str, binary: bool = False) -> str | bytes:
    return subprocess.check_output(["git", *args], text=not binary)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_file_bytes(path: Path) -> bytes:
    return _git("show", f"{FROZEN_RESULTS_COMMIT}:{path}", binary=True)  # type: ignore[return-value]


def parse_lfs_oid(pointer: bytes) -> str | None:
    text = pointer.decode("utf-8", errors="replace")
    for line in text.splitlines():
        if line.startswith("oid sha256:"):
            return line.split("sha256:", 1)[1].strip()
    return None


def integrity_gate(results_path: Path, manifest_path: Path) -> dict[str, Any]:
    """Assert row structure and byte identity before estimation."""

    frame = pd.read_parquet(results_path)
    receipt = structural_assertions(frame)
    results_worktree_sha = sha256_file(results_path)
    results_frozen_sha = sha256_bytes(frozen_file_bytes(results_path))
    if results_worktree_sha != results_frozen_sha:
        raise RuntimeError("results parquet byte-identity check failed")

    manifest_worktree_sha = sha256_file(manifest_path)
    frozen_manifest_pointer = frozen_file_bytes(manifest_path)
    manifest_frozen_lfs_oid = parse_lfs_oid(frozen_manifest_pointer)
    manifest_pointer_sha_frozen = sha256_bytes(frozen_manifest_pointer)
    manifest_pointer_sha_head = sha256_bytes(_git("show", f"HEAD:{manifest_path}", binary=True))  # type: ignore[arg-type]
    if manifest_frozen_lfs_oid is None:
        manifest_frozen_full_sha = sha256_bytes(frozen_manifest_pointer)
        manifest_full_identity_ok = manifest_worktree_sha == manifest_frozen_full_sha
    else:
        manifest_frozen_full_sha = manifest_frozen_lfs_oid
        manifest_full_identity_ok = manifest_worktree_sha == manifest_frozen_lfs_oid
    if not manifest_full_identity_ok:
        raise RuntimeError("manifest byte-identity check failed")
    if manifest_pointer_sha_head != manifest_pointer_sha_frozen:
        raise RuntimeError("manifest git pointer/blob changed since frozen results commit")

    cell_counts = frame.groupby(["base_point_id", "arm", "magnitude_level"], observed=True).size()
    real_shuffled = cell_counts.loc[(slice(None), ["real", "shuffled"], ["low", "high"])]
    all_four = bool((real_shuffled == 1).all() and len(real_shuffled) == 390 * 2 * 2)
    if not all_four:
        raise RuntimeError("real/shuffled x low/high completeness check failed")

    return {
        "frozen_results_commit": FROZEN_RESULTS_COMMIT,
        "current_head": _git("rev-parse", "HEAD").strip(),
        "origin_main": _git("ls-remote", "origin", "refs/heads/main").split()[0],
        "results_worktree_sha256": results_worktree_sha,
        "results_frozen_commit_sha256": results_frozen_sha,
        "results_byte_identical_to_frozen_commit": True,
        "manifest_worktree_full_file_sha256": manifest_worktree_sha,
        "manifest_frozen_full_file_sha256": manifest_frozen_full_sha,
        "manifest_full_file_matches_frozen_lfs_oid": manifest_full_identity_ok,
        "manifest_frozen_git_pointer_sha256": manifest_pointer_sha_frozen,
        "manifest_head_git_pointer_sha256": manifest_pointer_sha_head,
        "manifest_git_pointer_unchanged_since_frozen_commit": True,
        "structural_receipt": asdict(receipt),
        "all_real_shuffled_low_high_cells_exist_per_base": all_four,
    }


def manifest_covariates(manifest_path: Path) -> pd.DataFrame:
    manifest = manifest_from_json(manifest_path.read_text())
    rows: list[dict[str, Any]] = []
    for base in manifest.base_points:
        for plane in base.planes:
            rows.append(
                {
                    "base_point_id": base.base_point_id,
                    "arm": plane.arm,
                    "phi": plane.covariates.phi,
                    "manifold_distance_recon": plane.covariates.manifold_distance_recon,
                    "manifold_distance_mahalanobis": plane.covariates.manifold_distance_mahalanobis,
                    "manifold_distance_method": plane.covariates.manifold_distance_method,
                    "manifold_distance_fallback": plane.covariates.manifold_distance_fallback,
                    "det_M_stage_a": plane.det_M,
                    "mag_h": plane.mag_h,
                    "eps_mag_fallback_fired": plane.eps_mag_fallback_fired,
                }
            )
    return pd.DataFrame(rows)


def prepare_frames(results_path: Path, manifest_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = pd.read_parquet(results_path)
    covariates = manifest_covariates(manifest_path)
    merged = results.merge(covariates, on=["base_point_id", "arm"], how="left", validate="many_to_one")
    if merged[["phi", "manifold_distance_recon"]].isna().any().any():
        raise RuntimeError("covariate merge produced missing values")
    return results, merged


def hsem_primary(results: pd.DataFrame, covariates: pd.DataFrame) -> dict[str, Any]:
    cells = results[results["arm"].isin(["real", "shuffled"])].pivot(
        index="base_point_id",
        columns=["arm", "magnitude_level"],
        values="log_H",
    )
    low = cells[("real", "low")] - cells[("shuffled", "low")]
    high = cells[("real", "high")] - cells[("shuffled", "high")]
    d_b = 0.5 * (low + high)
    if len(d_b) != 390 or d_b.isna().any():
        raise RuntimeError("failed to define all 390 H-sem paired contrasts")

    cov_wide = covariates[covariates["arm"].isin(["real", "shuffled"])].pivot(
        index="base_point_id",
        columns="arm",
        values=["manifold_distance_recon", "manifold_distance_mahalanobis", "phi"],
    )
    diffs = pd.DataFrame(
        {
            "manifold_distance_diff": cov_wide[("manifold_distance_recon", "real")]
            - cov_wide[("manifold_distance_recon", "shuffled")],
            "manifold_distance_guard_diff": cov_wide[("manifold_distance_mahalanobis", "real")]
            - cov_wide[("manifold_distance_mahalanobis", "shuffled")],
            "phi_diff": cov_wide[("phi", "real")] - cov_wide[("phi", "shuffled")],
        },
        index=d_b.index,
    )
    primary_covariates = diffs[["manifold_distance_diff", "phi_diff"]]
    adjusted = fit_intercept(d_b, primary_covariates, alpha=0.05, robust=False, center=True)
    adjusted_hc3 = fit_intercept(d_b, primary_covariates, alpha=0.05, robust=True, center=True)
    unadjusted = fit_intercept(d_b, None, alpha=0.05)
    uncentered = fit_intercept(d_b, primary_covariates, alpha=0.05, robust=False, center=False)
    adjusted_verdict = verdict_from_ci(adjusted.ci_low, adjusted.ci_high, threshold=DEFAULT_MATERIALITY)
    hc3_verdict = verdict_from_ci(adjusted_hc3.ci_low, adjusted_hc3.ci_high, threshold=DEFAULT_MATERIALITY)
    raw_verdict = verdict_from_ci(unadjusted.ci_low, unadjusted.ci_high, threshold=DEFAULT_MATERIALITY)

    y = d_b.astype(float)
    x_centered = sm.add_constant(primary_covariates - primary_covariates.mean(axis=0), has_constant="add")
    ols = sm.OLS(y.to_numpy(), x_centered.to_numpy()).fit()
    influence = OLSInfluence(ols)
    cooks, _ = influence.cooks_distance
    leverage = influence.hat_matrix_diag
    influence_frame = pd.DataFrame(
        {
            "base_point_id": d_b.index,
            "leverage": leverage,
            "cooks_distance": cooks,
            "residual": ols.resid,
            "d_b": d_b.to_numpy(),
        }
    ).sort_values("cooks_distance", ascending=False)
    residual_correlations = {
        column: float(np.corrcoef(ols.resid, x_centered[column].to_numpy())[0, 1])
        for column in ["manifold_distance_diff", "phi_diff"]
    }
    slopes = {
        name: float(value)
        for name, value in zip(["const", "manifold_distance_diff", "phi_diff"], ols.params, strict=True)
    }
    return {
        "n_base_points": int(len(d_b)),
        "materiality_threshold": DEFAULT_MATERIALITY,
        "d_b_defined_for_all_bases": True,
        "adjusted_ols_t": asdict(adjusted),
        "adjusted_hc3": asdict(adjusted_hc3),
        "unadjusted": asdict(unadjusted),
        "uncentered_effect_at_covariate_parity": asdict(uncentered),
        "verdict": adjusted_verdict,
        "hc3_verdict": hc3_verdict,
        "raw_verdict": raw_verdict,
        "sensitive_to_se_specification": adjusted_verdict != hc3_verdict,
        "null_annotation": null_annotation(adjusted_verdict, raw_verdict),
        "within_magnitude_contrasts": {
            "low": asdict(fit_intercept(low, None, alpha=0.05)),
            "high": asdict(fit_intercept(high, None, alpha=0.05)),
        },
        "covariate_difference_means": {column: float(diffs[column].mean()) for column in diffs.columns},
        "covariate_difference_sds": {column: float(diffs[column].std(ddof=1)) for column in diffs.columns},
        "ols_slopes": slopes,
        "residual_vs_covariate_correlations": residual_correlations,
        "influence": {
            "max_leverage": float(np.max(leverage)),
            "max_cooks_distance": float(np.max(cooks)),
            "top_10_by_cooks_distance": influence_frame.head(10).to_dict(orient="records"),
        },
    }


def balance_metric(values_real: pd.Series, values_shuffled: pd.Series) -> dict[str, float | bool]:
    real = values_real.astype(float).to_numpy()
    shuffled = values_shuffled.astype(float).to_numpy()
    pooled = math.sqrt(max(0.0, 0.5 * (np.var(real, ddof=1) + np.var(shuffled, ddof=1))))
    smd = 0.0 if pooled == 0.0 else float((np.mean(real) - np.mean(shuffled)) / pooled)
    real_min, real_max = float(np.min(real)), float(np.max(real))
    shuffled_min, shuffled_max = float(np.min(shuffled)), float(np.max(shuffled))
    real_within = float(np.mean((real >= shuffled_min) & (real <= shuffled_max)))
    shuffled_within = float(np.mean((shuffled >= real_min) & (shuffled <= real_max)))
    positivity = min(real_within, shuffled_within)
    return {
        "smd": smd,
        "real_min": real_min,
        "real_max": real_max,
        "shuffled_min": shuffled_min,
        "shuffled_max": shuffled_max,
        "real_within_shuffled_range": real_within,
        "shuffled_within_real_range": shuffled_within,
        "positivity_fraction": positivity,
        "partially_extrapolated": positivity < 0.9,
    }


def balance_diagnostics(covariates: pd.DataFrame) -> dict[str, Any]:
    wide = covariates[covariates["arm"].isin(["real", "shuffled"])].pivot(
        index="base_point_id",
        columns="arm",
        values=["manifold_distance_recon", "manifold_distance_mahalanobis", "phi"],
    )
    metrics = {}
    for name in ["manifold_distance_recon", "manifold_distance_mahalanobis", "phi"]:
        metrics[name] = balance_metric(wide[(name, "real")], wide[(name, "shuffled")])
    return {
        "metrics": metrics,
        "partially_extrapolated": any(metric["partially_extrapolated"] for metric in metrics.values()),
    }


def contrast_from_params(params: pd.Series, cov: pd.DataFrame, terms: dict[str, float]) -> dict[str, float]:
    vector = pd.Series(0.0, index=params.index)
    for term, weight in terms.items():
        if term in vector.index:
            vector[term] = weight
        else:
            raise KeyError(f"missing model term: {term}")
    estimate = float(vector @ params)
    variance = float(vector.to_numpy() @ cov.loc[params.index, params.index].to_numpy() @ vector.to_numpy())
    se = math.sqrt(max(0.0, variance))
    ci_low, ci_high = estimate - 1.96 * se, estimate + 1.96 * se
    return {"estimate": estimate, "se": se, "ci_low": ci_low, "ci_high": ci_high}


def secondary_model(merged: pd.DataFrame) -> dict[str, Any]:
    data = merged.copy()
    data["arm"] = pd.Categorical(data["arm"], categories=["shuffled", "real", "random"])
    data["magnitude_level"] = pd.Categorical(data["magnitude_level"], categories=["low", "high"])
    formula = (
        'log_H ~ C(arm, Treatment(reference="shuffled"))'
        ' * C(magnitude_level, Treatment(reference="low"))'
        " + manifold_distance_recon + phi"
    )
    mixed_status: dict[str, Any] = {
        "formula": formula,
        "library": "statsmodels",
        "library_version": statsmodels.__version__,
        "fit_method": "MixedLM REML Wald CI",
    }
    fixed_formula = formula
    use_fixed_fallback = False
    try:
        model = smf.mixedlm(formula, data=data, groups=data["base_point_id"])
        mixed = model.fit(reml=True, method="lbfgs", maxiter=500, disp=False)
        params = mixed.params
        cov = mixed.cov_params()
        random_intercept_variance = float(mixed.cov_re.iloc[0, 0]) if mixed.cov_re.size else 0.0
        mixed_status.update(
            {
                "converged": bool(mixed.converged),
                "random_intercept_variance": random_intercept_variance,
                "boundary_random_intercept_variance": random_intercept_variance <= 1e-8,
            }
        )
        if random_intercept_variance <= 1e-8:
            use_fixed_fallback = True
    except Exception as exc:  # pragma: no cover - exercised only on pathological real fit failures.
        mixed_status.update({"mixedlm_error": repr(exc), "boundary_random_intercept_variance": True})
        use_fixed_fallback = True

    if use_fixed_fallback:
        fixed = smf.ols(fixed_formula, data=data).fit()
        params = fixed.params
        cov = fixed.cov_params()
        mixed_status["fit_method"] = "OLS fixed-effects fallback CI"
        mixed_status["fixed_effects_fallback_used"] = True
    else:
        mixed_status["fixed_effects_fallback_used"] = False

    high_term = 'C(magnitude_level, Treatment(reference="low"))[T.high]'
    real_term = 'C(arm, Treatment(reference="shuffled"))[T.real]'
    random_term = 'C(arm, Treatment(reference="shuffled"))[T.random]'
    real_high_interaction = (
        'C(arm, Treatment(reference="shuffled"))[T.real]:'
        'C(magnitude_level, Treatment(reference="low"))[T.high]'
    )
    random_high_interaction = (
        'C(arm, Treatment(reference="shuffled"))[T.random]:'
        'C(magnitude_level, Treatment(reference="low"))[T.high]'
    )
    h_mag = contrast_from_params(params, cov, {high_term: 1.0})
    h_mag["verdict"] = verdict_from_ci(
        h_mag["ci_low"], h_mag["ci_high"], threshold=DEFAULT_MATERIALITY
    )
    hsem_low = contrast_from_params(params, cov, {real_term: 1.0})
    hsem_high = contrast_from_params(params, cov, {real_term: 1.0, real_high_interaction: 1.0})
    hsem_equal_weight = contrast_from_params(params, cov, {real_term: 1.0, real_high_interaction: 0.5})
    hsem_equal_weight["verdict_descriptive"] = verdict_from_ci(
        hsem_equal_weight["ci_low"], hsem_equal_weight["ci_high"], threshold=DEFAULT_MATERIALITY
    )
    return {
        "status": mixed_status,
        "h_mag": h_mag,
        "h_grad": {
            "status": "UNDEFINED_AT_MATCHED_MAGNITUDE / no verdict",
            "random_vs_shuffled_low": contrast_from_params(params, cov, {random_term: 1.0}),
            "real_vs_shuffled_low": hsem_low,
            "random_vs_shuffled_high": contrast_from_params(
                params, cov, {random_term: 1.0, random_high_interaction: 1.0}
            ),
            "real_vs_shuffled_high": hsem_high,
        },
        "plane_type_x_magnitude_interaction": {
            "real_x_high": contrast_from_params(params, cov, {real_high_interaction: 1.0}),
            "random_x_high": contrast_from_params(params, cov, {random_high_interaction: 1.0}),
        },
        "secondary_hsem_equal_weight_real_vs_shuffled": hsem_equal_weight,
        "fixed_effect_terms": {name: float(value) for name, value in params.items()},
    }


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    raise TypeError(f"not JSON serializable: {value!r}")


def fmt(value: float) -> str:
    return f"{value:.6g}"


def write_reports(result: dict[str, Any]) -> None:
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True, default=json_default) + "\n")
    hsem = result["h_sem_primary"]
    secondary = result["secondary_model"]
    lines = [
        "# H-sem and Secondary Model Results",
        "",
        "Generated by the frozen Option A analysis pass. This report applies verdict labels only as specified.",
        "",
        "## Integrity Gate",
        "",
        f"- Results byte-identical to frozen commit: `{result['integrity']['results_byte_identical_to_frozen_commit']}`",
        f"- Manifest unchanged versus frozen LFS object: `{result['integrity']['manifest_full_file_matches_frozen_lfs_oid']}`",
        f"- Results SHA-256: `{result['integrity']['results_worktree_sha256']}`",
        f"- Manifest SHA-256: `{result['integrity']['manifest_worktree_full_file_sha256']}`",
        f"- Rows: `{result['integrity']['structural_receipt']['rows']}`",
        f"- Non-finite rows: `{result['integrity']['structural_receipt']['nonfinite_rows']}`",
        f"- Nonpositive H rows: `{result['integrity']['structural_receipt']['nonpositive_h_rows']}`",
        "",
        "## H-sem Primary",
        "",
        f"- Verdict: `{hsem['verdict']}`",
        f"- Adjusted OLS-t estimate: `{fmt(hsem['adjusted_ols_t']['intercept'])}` "
        f"[`{fmt(hsem['adjusted_ols_t']['ci_low'])}`, `{fmt(hsem['adjusted_ols_t']['ci_high'])}`]",
        f"- HC3 estimate: `{fmt(hsem['adjusted_hc3']['intercept'])}` "
        f"[`{fmt(hsem['adjusted_hc3']['ci_low'])}`, `{fmt(hsem['adjusted_hc3']['ci_high'])}`]",
        f"- Sensitive to SE specification: `{hsem['sensitive_to_se_specification']}`",
        f"- Unadjusted estimate: `{fmt(hsem['unadjusted']['intercept'])}` "
        f"[`{fmt(hsem['unadjusted']['ci_low'])}`, `{fmt(hsem['unadjusted']['ci_high'])}`]",
        f"- R4 annotation: `{hsem['null_annotation']}`",
        "",
        "## H-mag Secondary",
        "",
        f"- Verdict: `{secondary['h_mag']['verdict']}`",
        f"- Magnitude main-effect estimate: `{fmt(secondary['h_mag']['estimate'])}` "
        f"[`{fmt(secondary['h_mag']['ci_low'])}`, `{fmt(secondary['h_mag']['ci_high'])}`]",
        f"- Mixed model method: `{secondary['status']['fit_method']}`",
        f"- Fixed-effects fallback used: `{secondary['status']['fixed_effects_fallback_used']}`",
        "",
        "## H-grad",
        "",
        f"- Status: `{secondary['h_grad']['status']}`",
        "- No corroboration/falsification verdict applied.",
        "",
        "## Reported, Not Gated",
        "",
        f"- Low-only H-sem contrast: `{fmt(hsem['within_magnitude_contrasts']['low']['intercept'])}` "
        f"[`{fmt(hsem['within_magnitude_contrasts']['low']['ci_low'])}`, "
        f"`{fmt(hsem['within_magnitude_contrasts']['low']['ci_high'])}`]",
        f"- High-only H-sem contrast: `{fmt(hsem['within_magnitude_contrasts']['high']['intercept'])}` "
        f"[`{fmt(hsem['within_magnitude_contrasts']['high']['ci_low'])}`, "
        f"`{fmt(hsem['within_magnitude_contrasts']['high']['ci_high'])}`]",
        f"- Uncentered effect-at-covariate-parity intercept: "
        f"`{fmt(hsem['uncentered_effect_at_covariate_parity']['intercept'])}` "
        f"[`{fmt(hsem['uncentered_effect_at_covariate_parity']['ci_low'])}`, "
        f"`{fmt(hsem['uncentered_effect_at_covariate_parity']['ci_high'])}`]",
        f"- Balance partially extrapolated flag: `{result['balance_diagnostics']['partially_extrapolated']}`",
        f"- Max leverage: `{fmt(hsem['influence']['max_leverage'])}`",
        f"- Max Cook's distance: `{fmt(hsem['influence']['max_cooks_distance'])}`",
        f"- Secondary H-sem equal-weight real-vs-shuffled estimate: "
        f"`{fmt(secondary['secondary_hsem_equal_weight_real_vs_shuffled']['estimate'])}` "
        f"[`{fmt(secondary['secondary_hsem_equal_weight_real_vs_shuffled']['ci_low'])}`, "
        f"`{fmt(secondary['secondary_hsem_equal_weight_real_vs_shuffled']['ci_high'])}`]",
        "",
        "Machine-readable details are in `reports/hsem_results.json`.",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines))


def run() -> dict[str, Any]:
    integrity = integrity_gate(RESULTS_PATH, MANIFEST_PATH)
    _results, merged = prepare_frames(RESULTS_PATH, MANIFEST_PATH)
    hsem = hsem_primary(_results, merged[["base_point_id", "arm", "manifold_distance_recon", "manifold_distance_mahalanobis", "phi"]].drop_duplicates())
    balance = balance_diagnostics(
        merged[["base_point_id", "arm", "manifold_distance_recon", "manifold_distance_mahalanobis", "phi"]].drop_duplicates()
    )
    secondary = secondary_model(merged)
    primary_verdict = hsem["verdict"]
    secondary_hsem_verdict = secondary["secondary_hsem_equal_weight_real_vs_shuffled"]["verdict_descriptive"]
    result = {
        "analysis_plan": "analysis/hsem/ANALYSIS_PLAN_Hsem_v2addendum.md",
        "materiality_threshold": DEFAULT_MATERIALITY,
        "integrity": integrity,
        "h_sem_primary": hsem,
        "h_mag_secondary": secondary["h_mag"],
        "h_grad_status": secondary["h_grad"]["status"],
        "secondary_model": secondary,
        "balance_diagnostics": balance,
        "primary_vs_secondary_hsem_agreement": {
            "primary_verdict": primary_verdict,
            "secondary_descriptive_verdict": secondary_hsem_verdict,
            "agreement": primary_verdict == secondary_hsem_verdict,
        },
    }
    write_reports(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen H-sem Option A analysis")
    parser.add_argument("--execute", action="store_true", help="required by run orders")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("Refusing to run analysis without --execute")
    result = run()
    print(json.dumps(
        {
            "h_sem_verdict": result["h_sem_primary"]["verdict"],
            "h_mag_verdict": result["h_mag_secondary"]["verdict"],
            "h_grad_status": result["h_grad_status"],
            "reports": [str(REPORT_MD), str(REPORT_JSON)],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()

