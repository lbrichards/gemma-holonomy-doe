import json
import math

import pandas as pd
import pytest

from bands import compute_common_support_band
from manifest import (
    BalanceDiagnosticsRecord,
    BalanceMetricRecord,
    BasePointRecord,
    ManifestMetadata,
    PlaneCovariatesRecord,
    PlaneRecord,
    RunManifest,
    manifest_to_json,
)
from stage_b.run import load_manifest, run_stage_b


def synthetic_measure(base, plane, magnitude_level):
    base_idx = int(base.base_point_id.split("-")[-1])
    arm_offset = {"real": 1, "shuffled": 2, "random": 3}[plane.arm]
    mag_offset = {"low": 0.1, "high": 0.2}[magnitude_level]
    h_value = 0.001 * (base_idx + arm_offset + mag_offset)
    return {
        "H": h_value,
        "log_H": math.log(h_value),
        "theta": h_value * 0.5,
        "signed_holonomy": h_value * (-1 if plane.arm == "random" else 1),
        "A_enclosed": 0.5,
        "det_M": plane.det_M,
    }


def write_manifest(path, manifest):
    path.write_text(manifest_to_json(manifest))
    return path


def test_stage_b_structural_rows_on_tiny_smoke_manifest(tmp_path):
    manifest_path = write_manifest(tmp_path / "manifest.json", synthetic_manifest(n_base_points=2, kind="smoke"))
    results_path = tmp_path / "results.parquet"

    summary = run_stage_b(
        manifest_path=manifest_path,
        results_path=results_path,
        allow_smoke=True,
        measure_fn=synthetic_measure,
    )
    frame = pd.read_parquet(results_path)

    assert summary["rows_written"] == 12
    assert len(frame) == 2 * 3 * 2
    assert set(frame.columns) == {
        "base_point_id",
        "arm",
        "magnitude_level",
        "H",
        "log_H",
        "theta",
        "signed_holonomy",
        "A_enclosed",
        "det_M",
    }
    assert frame["H"].gt(0).all()
    assert frame[["H", "log_H", "theta", "signed_holonomy", "A_enclosed", "det_M"]].map(math.isfinite).all().all()


def test_stage_b_never_writes_to_manifest_file(tmp_path):
    manifest = synthetic_manifest(n_base_points=2, kind="smoke")
    manifest_path = write_manifest(tmp_path / "manifest.json", manifest)
    before = manifest_path.read_text()

    run_stage_b(
        manifest_path=manifest_path,
        results_path=tmp_path / "results.parquet",
        allow_smoke=True,
        measure_fn=synthetic_measure,
    )

    assert manifest_path.read_text() == before
    assert (tmp_path / "results.parquet").exists()
    assert (tmp_path / "results.json").exists()


def test_stage_b_refuses_terminal_manifest_without_measuring(tmp_path):
    manifest_path = write_manifest(tmp_path / "terminal.json", synthetic_manifest(n_base_points=2, kind="smoke", terminal=True))

    with pytest.raises(RuntimeError, match="TERMINAL"):
        run_stage_b(
            manifest_path=manifest_path,
            results_path=tmp_path / "results.parquet",
            allow_smoke=True,
            measure_fn=synthetic_measure,
        )

    assert not (tmp_path / "results.parquet").exists()


def test_stage_b_resumes_without_recomputing_completed_base_points(tmp_path):
    manifest_path = write_manifest(tmp_path / "manifest.json", synthetic_manifest(n_base_points=2, kind="smoke"))
    interrupted_results = tmp_path / "interrupted.parquet"
    full_results = tmp_path / "full.parquet"
    calls = []

    def counting_measure(base, plane, magnitude_level):
        calls.append((base.base_point_id, plane.arm, magnitude_level))
        return synthetic_measure(base, plane, magnitude_level)

    first = run_stage_b(
        manifest_path=manifest_path,
        results_path=interrupted_results,
        allow_smoke=True,
        measure_fn=counting_measure,
        max_new_base_points=1,
    )
    assert first["rows_written"] == 6
    assert len(calls) == 6

    second = run_stage_b(
        manifest_path=manifest_path,
        results_path=interrupted_results,
        allow_smoke=True,
        measure_fn=counting_measure,
    )
    assert second["rows_written"] == 12
    assert len(calls) == 12
    assert all(call[0] == "bp-001" for call in calls[6:])

    run_stage_b(
        manifest_path=manifest_path,
        results_path=full_results,
        allow_smoke=True,
        measure_fn=synthetic_measure,
    )
    resumed = pd.read_parquet(interrupted_results).sort_values(["base_point_id", "arm", "magnitude_level"]).reset_index(drop=True)
    uninterrupted = pd.read_parquet(full_results).sort_values(["base_point_id", "arm", "magnitude_level"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(resumed, uninterrupted)


def test_stage_b_requires_explicit_allow_smoke_for_smoke_manifest(tmp_path):
    manifest_path = write_manifest(tmp_path / "manifest.json", synthetic_manifest(n_base_points=2, kind="smoke"))

    with pytest.raises(RuntimeError, match="allow-smoke"):
        load_manifest(manifest_path, allow_smoke=False)


def synthetic_manifest(*, n_base_points: int, kind: str, terminal: bool = False) -> RunManifest:
    if terminal:
        band = compute_common_support_band(
            real=[63, 64, 65, 66, 67, 68, 69, 70],
            shuffled=[18, 19, 20, 21, 22, 23, 24, 25],
            random=[10, 11, 12, 13, 14, 15, 16, 17],
        )
    else:
        band = compute_common_support_band(
            real=[55, 58, 61, 64, 67, 70, 73, 76],
            shuffled=[50, 53, 56, 59, 62, 65, 68, 71],
            random=[10, 11, 12, 13, 14, 15, 16, 17],
        )
    return RunManifest(
        metadata=ManifestMetadata(
            prereg_version="PREREGISTRATION_v2.md",
            prereg_status_line="Status: FROZEN 2026-06-14 (supersedes v1; resid_post correction, tau=1.30, N=390)",
            prereg_commit_hash="0b2a8b77412b98a0e8e31e1f8f8a0dd01e593bde",
            dataset_revision="b08601e04326c79dfdd32d625aee71d232d685c3",
            seed_corpus=42,
            plane_selection_seed=20260614,
            model_id="google/gemma-2-2b",
            sae_repo="google/gemma-scope-2b-pt-res",
            sae_path="layer_12/width_16k/average_l0_82/params.npz",
            layer=12,
            extraction_site="resid_post: HF model.model.layers[12].output",
            tau_detM=0.413,
            eps_mag=2.66,
            radius_relative=6.0e-3,
            n_steps=200,
            n_base_points=390,
            reproducibility_claim="Bitwise within MPS plus deterministic CPU float64 post-processing.",
            manifest_kind=kind,  # type: ignore[arg-type]
        ),
        band_decision=band,
        balance_diagnostics=synthetic_balance(),
        base_points=[
            BasePointRecord(
                base_point_id=f"bp-{idx:03d}",
                corpus_draw_order=idx,
                article_index=1000 + idx,
                token_ids=list(range(64)),
                activation_ref=f"manifest.base_points[{idx}].activation",
                activation=[1.0, 2.0, 3.0],
                planes=[
                    synthetic_plane("real", idx),
                    synthetic_plane("shuffled", idx),
                    synthetic_plane("random", idx),
                ],
            )
            for idx in range(n_base_points)
        ],
    )


def synthetic_plane(arm: str, idx: int) -> PlaneRecord:
    return PlaneRecord(
        arm=arm,
        direction_1=[1.0, 0.0, 0.0],
        direction_2=[0.0, 1.0, 0.0],
        feature_indices=None if arm == "random" else (idx, idx + 1),
        det_M=1.0 + 0.01 * idx,
        mag_h=10.0 + idx,
        center_low=[0.1, 0.0, 0.0],
        center_high=[0.2, 0.0, 0.0],
        covariates=PlaneCovariatesRecord(
            phi=1.5707963267948966,
            manifold_distance_recon=1.0,
            manifold_distance_mahalanobis=1.0,
            manifold_distance_method="mahalanobis",
            manifold_distance_fallback=False,
        ),
        eps_mag_fallback_fired=False,
    )


def synthetic_balance() -> BalanceDiagnosticsRecord:
    metric = BalanceMetricRecord(
        smd=0.0,
        real_min=0.0,
        real_max=1.0,
        shuffled_min=0.0,
        shuffled_max=1.0,
        real_within_shuffled_range=1.0,
        shuffled_within_real_range=1.0,
    )
    return BalanceDiagnosticsRecord(
        manifold_distance_recon=metric,
        manifold_distance_mahalanobis=metric,
        phi=metric,
        log_sin_phi_mean_diff=0.0,
    )
