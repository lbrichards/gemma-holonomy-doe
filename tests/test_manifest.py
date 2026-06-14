import json
from dataclasses import fields, is_dataclass
from typing import get_args, get_origin

from bands import compute_common_support_band
from manifest.schema import (
    BasePointRecord,
    BalanceDiagnosticsRecord,
    BalanceMetricRecord,
    ManifestMetadata,
    PlaneCovariatesRecord,
    PlaneRecord,
    RunManifest,
    assert_no_response_fields,
    manifest_from_json,
    manifest_to_json,
    validate_manifest,
)


def synthetic_plane(arm: str, idx: int) -> PlaneRecord:
    feature_indices = None if arm == "random" else (idx, idx + 1)
    return PlaneRecord(
        arm=arm,
        direction_1=[1.0, 0.0, 0.0],
        direction_2=[0.25, 0.75, 0.0],
        feature_indices=feature_indices,
        det_M=1.25 + idx,
        mag_h=10.0 + idx,
        center_low=[0.1, 0.2, 0.3],
        center_high=[0.4, 0.5, 0.6],
        covariates=PlaneCovariatesRecord(
            phi=1.2,
            manifold_distance_recon=0.02,
            manifold_distance_mahalanobis=3.4,
            manifold_distance_method="mahalanobis",
            manifold_distance_fallback=False,
        ),
        eps_mag_fallback_fired=False,
    )


def synthetic_manifest(n_base_points: int = 390) -> RunManifest:
    band = compute_common_support_band(
        real=[55, 58, 61, 64, 67, 70, 73, 76],
        shuffled=[50, 53, 56, 59, 62, 65, 68, 71],
        random=[10, 11, 12, 13, 14, 15, 16, 17],
    )
    metadata = ManifestMetadata(
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
        reproducibility_claim=(
            "Bitwise within the fixed MPS backend plus deterministic CPU float64 post-processing path."
        ),
    )
    base_points = []
    for base_idx in range(n_base_points):
        base_points.append(
            BasePointRecord(
                base_point_id=f"bp-{base_idx:03d}",
                corpus_draw_order=base_idx,
                article_index=1000 + base_idx,
                token_ids=list(range(64)),
                activation_ref=f"activations/resid_post/{base_idx:03d}",
                activation=[0.01 * base_idx, 0.02 * base_idx, 0.03 * base_idx],
                planes=[
                    synthetic_plane("real", base_idx),
                    synthetic_plane("shuffled", base_idx + 100),
                    synthetic_plane("random", base_idx + 200),
                ],
            )
        )
    balance = BalanceDiagnosticsRecord(
        manifold_distance_recon=BalanceMetricRecord(
            smd=0.01,
            real_min=0.0,
            real_max=1.0,
            shuffled_min=0.0,
            shuffled_max=1.0,
            real_within_shuffled_range=1.0,
            shuffled_within_real_range=1.0,
        ),
        manifold_distance_mahalanobis=BalanceMetricRecord(
            smd=0.02,
            real_min=0.0,
            real_max=1.0,
            shuffled_min=0.0,
            shuffled_max=1.0,
            real_within_shuffled_range=1.0,
            shuffled_within_real_range=1.0,
        ),
        phi=BalanceMetricRecord(
            smd=0.03,
            real_min=1.0,
            real_max=1.5,
            shuffled_min=1.0,
            shuffled_max=1.5,
            real_within_shuffled_range=1.0,
            shuffled_within_real_range=1.0,
        ),
        log_sin_phi_mean_diff=0.001,
    )
    return RunManifest(metadata=metadata, band_decision=band, balance_diagnostics=balance, base_points=base_points)


def test_manifest_round_trip_fidelity_json_deep_equality():
    manifest = synthetic_manifest()

    encoded = manifest_to_json(manifest)
    decoded = manifest_from_json(encoded)

    assert decoded == manifest


def test_manifest_completeness_against_design_full_390_by_3_structure():
    manifest = synthetic_manifest(n_base_points=390)

    validate_manifest(manifest)

    assert len(manifest.base_points) == 390
    assert sum(len(base.planes) for base in manifest.base_points) == 390 * 3
    assert manifest.balance_diagnostics.phi.smd is not None
    for base in manifest.base_points:
        assert len(base.token_ids) == 64
        assert base.activation
        assert {plane.arm for plane in base.planes} == {"real", "shuffled", "random"}
        for plane in base.planes:
            assert plane.center_low is not None
            assert plane.center_high is not None
            assert plane.det_M is not None
            assert plane.mag_h is not None
            assert plane.covariates.phi is not None
            assert plane.covariates.manifold_distance_recon is not None
            assert plane.covariates.manifold_distance_mahalanobis is not None


def test_schema_structurally_excludes_holonomy_transport_response_fields():
    assert_no_response_fields(RunManifest)


def test_metadata_completeness_matches_preregistered_constants():
    metadata = synthetic_manifest().metadata

    assert metadata.prereg_status_line.startswith("Status: FROZEN 2026-06-14")
    assert metadata.dataset_revision == "b08601e04326c79dfdd32d625aee71d232d685c3"
    assert metadata.seed_corpus == 42
    assert metadata.tau_detM == 0.413
    assert metadata.eps_mag == 2.66
    assert metadata.radius_relative == 6.0e-3
    assert metadata.n_steps == 200
    assert metadata.n_base_points == 390
    assert metadata.layer == 12
    assert "resid_post" in metadata.extraction_site


def test_serialized_json_is_human_readable_and_keyed():
    encoded = manifest_to_json(synthetic_manifest())
    parsed = json.loads(encoded)

    assert "\n  " in encoded
    assert list(parsed.keys()) == ["metadata", "band_decision", "balance_diagnostics", "base_points"]
    assert "status" in parsed["band_decision"]
    assert "phi" in parsed["balance_diagnostics"]
    assert parsed["base_points"][0]["planes"][0]["arm"] == "real"
    assert isinstance(parsed["base_points"][0]["planes"][0]["direction_1"], list)


def test_no_forbidden_response_field_names_hidden_in_nested_dataclasses():
    # A second direct structural walk keeps the test legible if the helper ever
    # changes: all nested dataclass fields should remain free of response slots.
    forbidden = ("holonomy", "theta", "transport")
    stack = [RunManifest]
    seen = set()
    while stack:
        typ = stack.pop()
        if typ in seen or not is_dataclass(typ):
            continue
        seen.add(typ)
        for field in fields(typ):
            lower = field.name.lower()
            assert all(token not in lower for token in forbidden)
            assert not field.name.startswith("H_")
            origin = get_origin(field.type)
            args = get_args(field.type)
            if is_dataclass(field.type):
                stack.append(field.type)
            elif origin in (list, tuple) and args and is_dataclass(args[0]):
                stack.append(args[0])
