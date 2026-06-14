import pytest
import torch

from bands import BandStatus
from manifest import (
    BalanceDiagnosticsRecord,
    BalanceMetricRecord,
    BasePointRecord,
    PlaneCovariatesRecord,
    PlaneRecord,
    RunManifest,
    validate_manifest,
)
from planes import Arm
from stage_a.run import Pass1Base, Pass1Plane, compute_band, metadata, resolve_magnitudes_for_centers


def synthetic_pass1_base(idx: int, *, real: float, shuffled: float, random: float) -> Pass1Base:
    return Pass1Base(
        base_point_id=f"bp-{idx:03d}",
        corpus_draw_order=idx,
        article_index=idx + 1000,
        token_ids=list(range(64)),
        activation=torch.zeros(3),
        planes={
            Arm.REAL_FEATURE: synthetic_pass1_plane(Arm.REAL_FEATURE, real),
            Arm.SHUFFLED_FEATURE: synthetic_pass1_plane(Arm.SHUFFLED_FEATURE, shuffled),
            Arm.RANDOM: synthetic_pass1_plane(Arm.RANDOM, random),
        },
    )


def synthetic_pass1_plane(arm: Arm, mag_h: float) -> Pass1Plane:
    return Pass1Plane(
        arm=arm,
        selection=None,  # type: ignore[arg-type]
        mag_h=mag_h,
        coeffs=torch.zeros(2, dtype=torch.float64),
    )


def pass1_from_distributions(real, shuffled, random) -> list[Pass1Base]:
    return [
        synthetic_pass1_base(idx, real=float(r), shuffled=float(s), random=float(q))
        for idx, (r, s, q) in enumerate(zip(real, shuffled, random, strict=True))
    ]


def test_three_arm_valid_builds_centers_and_all_hypotheses_matched():
    pass1 = pass1_from_distributions(
        real=[50, 55, 60, 65, 70, 75, 80, 85],
        shuffled=[48, 54, 60, 66, 72, 78, 84, 90],
        random=[46, 53, 60, 67, 74, 81, 88, 95],
    )

    band = compute_band(pass1)
    resolution = resolve_magnitudes_for_centers(band_decision=band, pass1=pass1, manifest_kind="official")

    assert band.status == BandStatus.THREE_ARM_VALID
    assert band.h_sem_matched
    assert band.h_grad_matched
    assert band.h_mag_matched
    assert resolution.m_low == band.m_low
    assert resolution.m_high == band.m_high
    assert not resolution.should_halt
    assert resolution.smoke_magnitude_source is None


def test_two_arm_fallback_uses_real_shuffled_band_values_and_manifest_can_be_valid():
    pass1 = pass1_from_distributions(
        real=[55, 58, 61, 64, 67, 70, 73, 76],
        shuffled=[50, 53, 56, 59, 62, 65, 68, 71],
        random=[10, 11, 12, 13, 14, 15, 16, 17],
    )

    band = compute_band(pass1)
    resolution = resolve_magnitudes_for_centers(band_decision=band, pass1=pass1, manifest_kind="official")
    manifest = synthetic_complete_manifest(band)

    validate_manifest(manifest)
    assert band.status == BandStatus.TWO_ARM_FALLBACK
    assert band.h_sem_matched
    assert not band.h_grad_matched
    assert band.h_grad_status == "UNDEFINED_AT_MATCHED_MAGNITUDE"
    assert resolution.m_low == band.m_low
    assert resolution.m_high == band.m_high
    assert not resolution.should_halt
    assert resolution.smoke_magnitude_source is None


def test_two_arm_terminal_official_path_halts_without_fabricated_centers():
    pass1 = pass1_from_distributions(
        real=[63, 64, 65, 66, 67, 68, 69, 70],
        shuffled=[18, 19, 20, 21, 22, 23, 24, 25],
        random=[10, 11, 12, 13, 14, 15, 16, 17],
    )

    band = compute_band(pass1)
    resolution = resolve_magnitudes_for_centers(band_decision=band, pass1=pass1, manifest_kind="official")

    assert band.status == BandStatus.TWO_ARM_TERMINAL
    assert band.m_low is None
    assert band.m_high is None
    assert resolution.should_halt
    assert resolution.m_low is None
    assert resolution.m_high is None
    assert "cannot construct centers" in resolution.message


def test_terminal_smoke_override_is_reachable_only_for_smoke_mode():
    pass1 = pass1_from_distributions(
        real=[63, 64, 65, 66, 67, 68, 69, 70],
        shuffled=[18, 19, 20, 21, 22, 23, 24, 25],
        random=[10, 11, 12, 13, 14, 15, 16, 17],
    )
    band = compute_band(pass1)

    official = resolve_magnitudes_for_centers(band_decision=band, pass1=pass1, manifest_kind="official")
    smoke = resolve_magnitudes_for_centers(band_decision=band, pass1=pass1, manifest_kind="smoke")

    assert official.should_halt
    assert official.smoke_magnitude_source is None
    assert smoke.should_halt is False
    assert smoke.m_low is not None
    assert smoke.m_high is not None
    assert smoke.smoke_magnitude_source is not None
    assert "SMOKE_ONLY" in smoke.smoke_magnitude_source


def synthetic_complete_manifest(band) -> RunManifest:
    base_points = []
    for idx in range(390):
        base_points.append(
            BasePointRecord(
                base_point_id=f"bp-{idx:03d}",
                corpus_draw_order=idx,
                article_index=1000 + idx,
                token_ids=list(range(64)),
                activation_ref=f"manifest.base_points[{idx}].activation",
                activation=[0.0, 0.0, 0.0],
                planes=[
                    synthetic_plane_record("real", idx),
                    synthetic_plane_record("shuffled", idx),
                    synthetic_plane_record("random", idx),
                ],
            )
        )
    return RunManifest(
        metadata=metadata(manifest_kind="official"),
        band_decision=band,
        balance_diagnostics=BalanceDiagnosticsRecord(
            manifold_distance_recon=synthetic_balance_metric(),
            manifold_distance_mahalanobis=synthetic_balance_metric(),
            phi=synthetic_balance_metric(),
            log_sin_phi_mean_diff=0.0,
        ),
        base_points=base_points,
    )


def synthetic_plane_record(arm: str, idx: int) -> PlaneRecord:
    return PlaneRecord(
        arm=arm,
        direction_1=[1.0, 0.0, 0.0],
        direction_2=[0.0, 1.0, 0.0],
        feature_indices=None if arm == "random" else (idx, idx + 1),
        det_M=1.0,
        mag_h=10.0,
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


def synthetic_balance_metric() -> BalanceMetricRecord:
    return BalanceMetricRecord(
        smd=0.0,
        real_min=0.0,
        real_max=1.0,
        shuffled_min=0.0,
        shuffled_max=1.0,
        real_within_shuffled_range=1.0,
        shuffled_within_real_range=1.0,
    )
