"""Post-freeze H-sem estimate on a prepend_bos run, next to the frozen estimate.

Applies the frozen primary estimator (``analysis.hsem.run_analysis.hsem_primary``)
to a ``prepend_bos`` manifest/results pair, and to the frozen no-<bos> run
restricted to the same base points, so the two are compared on identical bases.
It does not modify any frozen artifact and applies no verdict.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.hsem.harness import fit_intercept
from analysis.hsem.run_analysis import (
    MANIFEST_PATH,
    RESULTS_PATH,
    hsem_primary,
    json_default,
    prepare_frames,
    secondary_model,
    sha256_file,
)
from manifest import manifest_from_json


STATUS = "POST_FREEZE_NOT_CONFIRMATORY"
COVARIATE_COLUMNS = ["base_point_id", "arm", "manifold_distance_recon", "manifold_distance_mahalanobis", "phi"]
FROZEN_FULL = {"intercept": -0.294391, "ci_low": -0.439888, "ci_high": -0.148895, "n": 390, "source": "reports/hsem_results.md"}
TEBOW = {"intercept": -0.001, "ci_low": -0.116, "ci_high": 0.114, "source": "external reviewer report (with <bos>)"}


def primary_for(results_path: Path, manifest_path: Path, base_ids: list[str] | None = None) -> dict[str, Any]:
    results, merged = prepare_frames(results_path, manifest_path)
    if base_ids is not None:
        results = results[results["base_point_id"].isin(base_ids)]
        merged = merged[merged["base_point_id"].isin(base_ids)]
    n = int(results["base_point_id"].nunique())
    primary = hsem_primary(results, merged[COVARIATE_COLUMNS].drop_duplicates(), n_expected=n)
    return {"n": n, "primary": primary, "secondary": secondary_model(merged), "results": results}


def d_b(results: pd.DataFrame) -> pd.Series:
    cells = results[results["arm"].isin(["real", "shuffled"])].pivot(
        index="base_point_id", columns=["arm", "magnitude_level"], values="log_H"
    )
    low = cells[("real", "low")] - cells[("shuffled", "low")]
    high = cells[("real", "high")] - cells[("shuffled", "high")]
    return 0.5 * (low + high)


def run(results_path: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = manifest_from_json(manifest_path.read_text())
    if not manifest.metadata.prepend_bos:
        raise RuntimeError(f"{manifest_path} is not a prepend_bos manifest")
    bos = primary_for(results_path, manifest_path)
    base_ids = sorted(bos["results"]["base_point_id"].unique())
    frozen_subset = primary_for(RESULTS_PATH, MANIFEST_PATH, base_ids)
    paired = d_b(bos["results"]) - d_b(frozen_subset["results"]).loc[d_b(bos["results"]).index]
    band = manifest.band_decision
    return {
        "status": STATUS,
        "bos_run": {
            "results": str(results_path),
            "results_sha256": sha256_file(results_path),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "manifest_kind": manifest.metadata.manifest_kind,
            "smoke_magnitude_source": manifest.metadata.smoke_magnitude_source,
            "band_status": band.status.value,
            "m_low": band.m_low,
            "m_high": band.m_high,
            "balance": {
                "recon_smd": manifest.balance_diagnostics.manifold_distance_recon.smd,
                "guard_smd": manifest.balance_diagnostics.manifold_distance_mahalanobis.smd,
                "phi_smd": manifest.balance_diagnostics.phi.smd,
            },
            "n_base_points": bos["n"],
            "rows": int(len(bos["results"])),
            "h_sem_primary": bos["primary"],
            "secondary_model": bos["secondary"],
        },
        "frozen_same_bases": {
            "n_base_points": frozen_subset["n"],
            "base_point_ids": base_ids,
            "h_sem_primary": frozen_subset["primary"],
        },
        "paired_bos_minus_frozen_d_b": asdict(fit_intercept(paired, None, alpha=0.05)),
        "frozen_full": FROZEN_FULL,
        "tebow": TEBOW,
    }


def _ci(est: dict[str, Any]) -> str:
    return f"{est['intercept']:+.4f} [{est['ci_low']:+.4f}, {est['ci_high']:+.4f}]"


def write_reports(result: dict[str, Any], out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=json_default) + "\n")
    bos = result["bos_run"]
    frozen = result["frozen_same_bases"]
    pilot = bos["n_base_points"] < FROZEN_FULL["n"]
    label = "pilot" if pilot else "full run"
    lines = [
        f"# H-sem with <bos> prepended: {bos['n_base_points']}-base {label}",
        "",
        f"**Status: {STATUS}.** Post-freeze robustness check prompted by external review. "
        "No verdict label is applied; the preregistered result in `reports/hsem_results.md` is unchanged.",
        "",
        "Estimator: the frozen H-sem primary (covariate-adjusted OLS-t intercept of d_b, "
        "covariates centered), 95% CI.",
        "",
        "| estimate | n bases | gap [95% CI] |",
        "|---|---|---|",
        f"| Frozen, no <bos> (preregistered) | 390 | {_ci(result['frozen_full'])} |",
        f"| Frozen, no <bos>, same {frozen['n_base_points']} bases | {frozen['n_base_points']} "
        f"| {_ci(frozen['h_sem_primary']['adjusted_ols_t'])} |",
        f"| **This {label}, with <bos>** | {bos['n_base_points']} | **{_ci(bos['h_sem_primary']['adjusted_ols_t'])}** |",
        f"| External reviewer (Tebow), with <bos> | n/a | {_ci(result['tebow'])} |",
        "",
        f"{label.capitalize()} detail:",
        "",
        f"- HC3: {_ci(bos['h_sem_primary']['adjusted_hc3'])}; unadjusted: {_ci(bos['h_sem_primary']['unadjusted'])}",
        f"- Within-magnitude, low: {_ci(bos['h_sem_primary']['within_magnitude_contrasts']['low'])}; "
        f"high: {_ci(bos['h_sem_primary']['within_magnitude_contrasts']['high'])}",
        f"- Paired per-base change in d_b (with <bos> minus frozen): {_ci(result['paired_bos_minus_frozen_d_b'])}",
        f"- Rows: {bos['rows']}; manifest kind `{bos['manifest_kind']}`; band `{bos['band_status']}` "
        f"(m_low={bos['m_low']}, m_high={bos['m_high']})",
        f"- Smoke magnitude source: `{bos['smoke_magnitude_source']}`",
        f"- Stage A balance SMDs (real vs shuffled): recon {bos['balance']['recon_smd']:+.3f}, "
        f"guard {bos['balance']['guard_smd']:+.3f}, phi {bos['balance']['phi_smd']:+.3f}",
        "",
    ]
    if pilot:
        lines += [
            "Caveats: the band (m_low/m_high) and covariate reference distribution are fit on these "
            f"{bos['n_base_points']} bases only, so centers differ from what a full 390-base <bos> run would "
            "construct. The CI is wide at this n; this pilot sizes the effect, it does not settle it.",
            "",
        ]
    (out_dir / f"{stem}.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("run_bos/holonomy_results_bos_pilot30.parquet"))
    parser.add_argument("--manifest", type=Path, default=Path("run_bos/manifest_bos_pilot30.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports_bos"))
    parser.add_argument("--stem", default="hsem_bos_pilot30")
    args = parser.parse_args()
    result = run(args.results, args.manifest)
    write_reports(result, args.out_dir, args.stem)
    print((args.out_dir / f"{args.stem}.md").read_text())


if __name__ == "__main__":
    main()
