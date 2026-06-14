"""Stage A -> Stage B manifest schema.

The manifest is the blind handoff artifact. It records extraction, plane
selection, magnitude-band, center-placement, and covariate decisions, but it is
structurally incapable of carrying response values. Stage B must write measured
responses to a separate artifact.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import Any, Literal, get_args, get_origin

from bands import ArmPercentiles, BandDecision, BandStatus


ArmName = Literal["real", "shuffled", "random"]
ManifestKind = Literal["official", "smoke"]
ReferenceDistanceMethod = Literal["mahalanobis", "knn"]

FORBIDDEN_RESPONSE_FIELD_TOKENS = ("holonomy", "theta", "transport")


@dataclass(frozen=True)
class ManifestMetadata:
    """Pre-registered constants and provenance for the run manifest."""

    prereg_version: str
    prereg_status_line: str
    prereg_commit_hash: str
    dataset_revision: str
    seed_corpus: int
    plane_selection_seed: int
    model_id: str
    sae_repo: str
    sae_path: str
    layer: int
    extraction_site: str
    tau_detM: float
    eps_mag: float
    radius_relative: float
    n_steps: int
    n_base_points: int
    reproducibility_claim: str
    manifest_kind: ManifestKind = "official"
    smoke_magnitude_source: str | None = None


@dataclass(frozen=True)
class BalanceMetricRecord:
    """Blind real-vs-shuffled balance diagnostic for one covariate."""

    smd: float
    real_min: float
    real_max: float
    shuffled_min: float
    shuffled_max: float
    real_within_shuffled_range: float
    shuffled_within_real_range: float


@dataclass(frozen=True)
class BalanceDiagnosticsRecord:
    """Section 4.3 blind balance diagnostics."""

    manifold_distance_recon: BalanceMetricRecord
    manifold_distance_mahalanobis: BalanceMetricRecord
    phi: BalanceMetricRecord
    log_sin_phi_mean_diff: float


@dataclass(frozen=True)
class PlaneCovariatesRecord:
    """Blind Section 4 covariates for one plane."""

    phi: float
    manifold_distance_recon: float
    manifold_distance_mahalanobis: float
    manifold_distance_method: ReferenceDistanceMethod
    manifold_distance_fallback: bool


@dataclass(frozen=True)
class PlaneRecord:
    """One blind selected plane and its Stage A quantities."""

    arm: ArmName
    direction_1: list[float]
    direction_2: list[float]
    feature_indices: tuple[int, int] | None
    det_M: float
    mag_h: float
    center_low: list[float]
    center_high: list[float]
    covariates: PlaneCovariatesRecord
    eps_mag_fallback_fired: bool


@dataclass(frozen=True)
class BasePointRecord:
    """One base point plus exactly three arm records."""

    base_point_id: str
    corpus_draw_order: int
    article_index: int
    token_ids: list[int]
    activation_ref: str
    activation: list[float]
    planes: list[PlaneRecord]


@dataclass(frozen=True)
class RunManifest:
    """Top-level blind Stage A -> Stage B handoff artifact."""

    metadata: ManifestMetadata
    band_decision: BandDecision
    balance_diagnostics: BalanceDiagnosticsRecord
    base_points: list[BasePointRecord]


def _finite(name: str, value: float) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite, got {value!r}")


def validate_manifest(manifest: RunManifest) -> None:
    """Validate design completeness and blind boundary invariants."""

    assert_no_response_fields(RunManifest)
    metadata = manifest.metadata
    if metadata.n_base_points != 390:
        raise ValueError(f"metadata.n_base_points must be 390, got {metadata.n_base_points}")
    if metadata.manifest_kind == "official" and len(manifest.base_points) != metadata.n_base_points:
        raise ValueError(f"expected {metadata.n_base_points} base points, got {len(manifest.base_points)}")
    if metadata.manifest_kind == "smoke" and not (0 < len(manifest.base_points) <= metadata.n_base_points):
        raise ValueError(f"smoke manifest must contain 1..{metadata.n_base_points} base points")
    if metadata.seed_corpus != 42:
        raise ValueError(f"seed_corpus must be 42, got {metadata.seed_corpus}")
    if metadata.dataset_revision != "b08601e04326c79dfdd32d625aee71d232d685c3":
        raise ValueError("dataset_revision does not match frozen WikiText revision")
    if metadata.layer != 12 or "resid_post" not in metadata.extraction_site:
        raise ValueError("metadata extraction site must be layer-12 resid_post")
    if metadata.tau_detM != 0.413:
        raise ValueError(f"tau_detM must be 0.413, got {metadata.tau_detM}")
    if metadata.eps_mag != 2.66:
        raise ValueError(f"eps_mag must be 2.66, got {metadata.eps_mag}")
    if metadata.radius_relative != 6.0e-3:
        raise ValueError(f"radius_relative must be 6.0e-3, got {metadata.radius_relative}")
    if metadata.n_steps != 200:
        raise ValueError(f"n_steps must be 200, got {metadata.n_steps}")

    for base in manifest.base_points:
        if len(base.token_ids) != 64:
            raise ValueError(f"{base.base_point_id}: token_ids must have length 64")
        if len(base.activation) == 0:
            raise ValueError(f"{base.base_point_id}: activation must be present")
        if len(base.planes) != 3:
            raise ValueError(f"{base.base_point_id}: expected exactly 3 planes")
        arms = {plane.arm for plane in base.planes}
        if arms != {"real", "shuffled", "random"}:
            raise ValueError(f"{base.base_point_id}: expected real/shuffled/random planes, got {arms}")
        for plane in base.planes:
            if len(plane.direction_1) != len(plane.direction_2):
                raise ValueError(f"{base.base_point_id}/{plane.arm}: direction lengths differ")
            if len(plane.center_low) != len(plane.center_high):
                raise ValueError(f"{base.base_point_id}/{plane.arm}: center lengths differ")
            if plane.arm == "random" and plane.feature_indices is not None:
                raise ValueError(f"{base.base_point_id}: random plane may not have feature indices")
            if plane.arm != "random" and plane.feature_indices is None:
                raise ValueError(f"{base.base_point_id}: feature plane must have feature indices")
            for name, value in [
                ("det_M", plane.det_M),
                ("mag_h", plane.mag_h),
                ("phi", plane.covariates.phi),
                ("manifold_distance_recon", plane.covariates.manifold_distance_recon),
                ("manifold_distance_mahalanobis", plane.covariates.manifold_distance_mahalanobis),
            ]:
                _finite(f"{base.base_point_id}/{plane.arm}/{name}", value)
    for name, metric in [
        ("manifold_distance_recon", manifest.balance_diagnostics.manifold_distance_recon),
        ("manifold_distance_mahalanobis", manifest.balance_diagnostics.manifold_distance_mahalanobis),
        ("phi", manifest.balance_diagnostics.phi),
    ]:
        for field_name in [
            "smd",
            "real_min",
            "real_max",
            "shuffled_min",
            "shuffled_max",
            "real_within_shuffled_range",
            "shuffled_within_real_range",
        ]:
            _finite(f"balance/{name}/{field_name}", getattr(metric, field_name))
    _finite("balance/log_sin_phi_mean_diff", manifest.balance_diagnostics.log_sin_phi_mean_diff)


def assert_no_response_fields(schema_type: type[Any]) -> None:
    """Assert schema dataclasses cannot store response/transport outputs by name."""

    stack = [schema_type]
    seen: set[type[Any]] = set()
    while stack:
        typ = stack.pop()
        if typ in seen or not is_dataclass(typ):
            continue
        seen.add(typ)
        for field in fields(typ):
            lower = field.name.lower()
            if any(token in lower for token in FORBIDDEN_RESPONSE_FIELD_TOKENS) or field.name.startswith("H_"):
                raise AssertionError(f"Manifest schema field may carry response output: {typ.__name__}.{field.name}")
            if field.metadata.get("role") == "transport_output":
                raise AssertionError(f"Manifest schema field marked as transport output: {typ.__name__}.{field.name}")
            origin = get_origin(field.type)
            args = get_args(field.type)
            if is_dataclass(field.type):
                stack.append(field.type)
            elif origin in (list, tuple) and args and is_dataclass(args[0]):
                stack.append(args[0])


def _arm_percentiles_to_dict(value: ArmPercentiles) -> dict[str, float]:
    return asdict(value)


def _arm_percentiles_from_dict(value: dict[str, Any]) -> ArmPercentiles:
    return ArmPercentiles(p5=float(value["p5"]), p25=float(value["p25"]), p75=float(value["p75"]), p95=float(value["p95"]))


def _band_decision_to_dict(value: BandDecision) -> dict[str, Any]:
    data = asdict(value)
    data["status"] = value.status.value
    data["per_arm"] = {arm: _arm_percentiles_to_dict(percentiles) for arm, percentiles in value.per_arm.items()}
    return data


def _band_decision_from_dict(value: dict[str, Any]) -> BandDecision:
    return BandDecision(
        status=BandStatus(value["status"]),
        band_lower=None if value["band_lower"] is None else float(value["band_lower"]),
        band_upper=None if value["band_upper"] is None else float(value["band_upper"]),
        band_width=float(value["band_width"]),
        threshold=float(value["threshold"]),
        m_low=None if value["m_low"] is None else float(value["m_low"]),
        m_high=None if value["m_high"] is None else float(value["m_high"]),
        h_sem_matched=bool(value["h_sem_matched"]),
        h_grad_matched=bool(value["h_grad_matched"]),
        h_mag_matched=bool(value["h_mag_matched"]),
        h_sem_status=str(value["h_sem_status"]),
        h_grad_status=str(value["h_grad_status"]),
        h_mag_status=str(value["h_mag_status"]),
        three_arm_lower=float(value["three_arm_lower"]),
        three_arm_upper=float(value["three_arm_upper"]),
        three_arm_width=float(value["three_arm_width"]),
        three_arm_threshold=float(value["three_arm_threshold"]),
        two_arm_lower=float(value["two_arm_lower"]),
        two_arm_upper=float(value["two_arm_upper"]),
        two_arm_width=float(value["two_arm_width"]),
        two_arm_threshold=float(value["two_arm_threshold"]),
        per_arm={arm: _arm_percentiles_from_dict(percentiles) for arm, percentiles in value["per_arm"].items()},
    )


def _plane_covariates_to_dict(value: PlaneCovariatesRecord) -> dict[str, Any]:
    return asdict(value)


def _plane_covariates_from_dict(value: dict[str, Any]) -> PlaneCovariatesRecord:
    return PlaneCovariatesRecord(
        phi=float(value["phi"]),
        manifold_distance_recon=float(value["manifold_distance_recon"]),
        manifold_distance_mahalanobis=float(value["manifold_distance_mahalanobis"]),
        manifold_distance_method=value["manifold_distance_method"],
        manifold_distance_fallback=bool(value["manifold_distance_fallback"]),
    )


def _balance_metric_to_dict(value: BalanceMetricRecord) -> dict[str, Any]:
    return asdict(value)


def _balance_metric_from_dict(value: dict[str, Any]) -> BalanceMetricRecord:
    return BalanceMetricRecord(
        smd=float(value["smd"]),
        real_min=float(value["real_min"]),
        real_max=float(value["real_max"]),
        shuffled_min=float(value["shuffled_min"]),
        shuffled_max=float(value["shuffled_max"]),
        real_within_shuffled_range=float(value["real_within_shuffled_range"]),
        shuffled_within_real_range=float(value["shuffled_within_real_range"]),
    )


def _balance_diagnostics_to_dict(value: BalanceDiagnosticsRecord) -> dict[str, Any]:
    return {
        "manifold_distance_recon": _balance_metric_to_dict(value.manifold_distance_recon),
        "manifold_distance_mahalanobis": _balance_metric_to_dict(value.manifold_distance_mahalanobis),
        "phi": _balance_metric_to_dict(value.phi),
        "log_sin_phi_mean_diff": value.log_sin_phi_mean_diff,
    }


def _balance_diagnostics_from_dict(value: dict[str, Any]) -> BalanceDiagnosticsRecord:
    return BalanceDiagnosticsRecord(
        manifold_distance_recon=_balance_metric_from_dict(value["manifold_distance_recon"]),
        manifold_distance_mahalanobis=_balance_metric_from_dict(value["manifold_distance_mahalanobis"]),
        phi=_balance_metric_from_dict(value["phi"]),
        log_sin_phi_mean_diff=float(value["log_sin_phi_mean_diff"]),
    )


def _plane_to_dict(value: PlaneRecord) -> dict[str, Any]:
    return {
        "arm": value.arm,
        "direction_1": list(value.direction_1),
        "direction_2": list(value.direction_2),
        "feature_indices": None if value.feature_indices is None else list(value.feature_indices),
        "det_M": value.det_M,
        "mag_h": value.mag_h,
        "center_low": list(value.center_low),
        "center_high": list(value.center_high),
        "covariates": _plane_covariates_to_dict(value.covariates),
        "eps_mag_fallback_fired": value.eps_mag_fallback_fired,
    }


def _plane_from_dict(value: dict[str, Any]) -> PlaneRecord:
    feature_indices = value["feature_indices"]
    return PlaneRecord(
        arm=value["arm"],
        direction_1=[float(item) for item in value["direction_1"]],
        direction_2=[float(item) for item in value["direction_2"]],
        feature_indices=None if feature_indices is None else (int(feature_indices[0]), int(feature_indices[1])),
        det_M=float(value["det_M"]),
        mag_h=float(value["mag_h"]),
        center_low=[float(item) for item in value["center_low"]],
        center_high=[float(item) for item in value["center_high"]],
        covariates=_plane_covariates_from_dict(value["covariates"]),
        eps_mag_fallback_fired=bool(value["eps_mag_fallback_fired"]),
    )


def _base_point_to_dict(value: BasePointRecord) -> dict[str, Any]:
    return {
        "base_point_id": value.base_point_id,
        "corpus_draw_order": value.corpus_draw_order,
        "article_index": value.article_index,
        "token_ids": list(value.token_ids),
        "activation_ref": value.activation_ref,
        "activation": list(value.activation),
        "planes": [_plane_to_dict(plane) for plane in value.planes],
    }


def _base_point_from_dict(value: dict[str, Any]) -> BasePointRecord:
    return BasePointRecord(
        base_point_id=str(value["base_point_id"]),
        corpus_draw_order=int(value["corpus_draw_order"]),
        article_index=int(value["article_index"]),
        token_ids=[int(item) for item in value["token_ids"]],
        activation_ref=str(value["activation_ref"]),
        activation=[float(item) for item in value["activation"]],
        planes=[_plane_from_dict(item) for item in value["planes"]],
    )


def manifest_to_dict(manifest: RunManifest) -> dict[str, Any]:
    """Return a human-readable JSON-ready dict."""

    return {
        "metadata": asdict(manifest.metadata),
        "band_decision": _band_decision_to_dict(manifest.band_decision),
        "balance_diagnostics": _balance_diagnostics_to_dict(manifest.balance_diagnostics),
        "base_points": [_base_point_to_dict(base) for base in manifest.base_points],
    }


def manifest_from_dict(value: dict[str, Any]) -> RunManifest:
    """Return a manifest from a JSON-loaded dict."""

    metadata = ManifestMetadata(**value["metadata"])
    return RunManifest(
        metadata=metadata,
        band_decision=_band_decision_from_dict(value["band_decision"]),
        balance_diagnostics=_balance_diagnostics_from_dict(value["balance_diagnostics"]),
        base_points=[_base_point_from_dict(item) for item in value["base_points"]],
    )


def manifest_to_json(manifest: RunManifest) -> str:
    """Serialize a manifest as indented auditor-readable JSON."""

    validate_manifest(manifest)
    return json.dumps(manifest_to_dict(manifest), indent=2, sort_keys=False) + "\n"


def manifest_from_json(text: str) -> RunManifest:
    """Deserialize and validate a manifest JSON string."""

    manifest = manifest_from_dict(json.loads(text))
    validate_manifest(manifest)
    return manifest
