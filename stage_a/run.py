"""Stage A blind setup orchestration.

Stage A follows PREREGISTRATION_v2 Section 2 order exactly:

1. extract activations, select planes, compute mag(h) for all base points;
2. compute the common-support magnitude band;
3. construct low/high centers and measure covariates;
4. serialize the blind manifest.

It computes no response values.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from bands import BandDecision, compute_common_support_band
from covariates import ReferenceDistribution, measure_plane_covariates
from extraction import (
    FINAL_TOKEN_POSITION,
    LAYER_INDEX,
    MODEL_NAME,
    SAE_PATH,
    SAE_REPO,
    ResidPostExtractor,
    load_model_and_tokenizer,
    load_sae,
)
from geometry import g_projection_coeffs, in_plane_magnitude
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
from planes import TAU_DETM, Arm, PlaneSelection, select_three_arm_planes
from transport import N_STEPS, RADIUS_RELATIVE


DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
SEED_CORPUS = 42
PLANE_SELECTION_SEED = 20260614
PREREG_COMMIT_HASH = "0b2a8b77412b98a0e8e31e1f8f8a0dd01e593bde"
PREREG_STATUS_LINE = "Status: FROZEN 2026-06-14 (supersedes v1; resid_post correction, tau=1.30, N=390)"
EPS_MAG = 2.66
N_BASE_POINTS = 390
REFERENCE_K = 5
N_COVARIATE_LOOP_SAMPLES = 32


@dataclass
class Pass1Plane:
    arm: Arm
    selection: PlaneSelection
    mag_h: float
    coeffs: torch.Tensor


@dataclass
class Pass1Base:
    base_point_id: str
    corpus_draw_order: int
    article_index: int
    token_ids: list[int]
    activation: torch.Tensor
    planes: dict[Arm, Pass1Plane]


@dataclass(frozen=True)
class MagnitudeResolution:
    """Stage A center-construction decision after the blind band branch."""

    m_low: float | None
    m_high: float | None
    smoke_magnitude_source: str | None
    should_halt: bool
    message: str


def load_experiment_records(path: Path, *, limit: int | None) -> list[dict[str, Any]]:
    artifact = json.loads(path.read_text())
    records = artifact["records"]
    experiment_orders = set(artifact["partitions"]["experiment_draw_orders"])
    selected = [record for record in records if int(record["draw_order"]) in experiment_orders]
    selected.sort(key=lambda record: int(record["draw_order"]))
    if len(selected) < N_BASE_POINTS:
        raise RuntimeError(f"expected at least {N_BASE_POINTS} experiment records, got {len(selected)}")
    selected = selected[:N_BASE_POINTS]
    if limit is not None:
        selected = selected[:limit]
    return selected


def make_readout_jvp(model: Any, token_ids: list[int], *, device: torch.device):
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, device=device)

    def readout_map(h: torch.Tensor) -> torch.Tensor:
        state: dict[str, torch.Tensor] = {}

        def patch_hook(_module: torch.nn.Module, _args: tuple[Any, ...], output: torch.Tensor):
            tensor = output[0] if isinstance(output, tuple) else output
            patched = tensor.clone()
            patched[0, FINAL_TOKEN_POSITION, :] = h.to(device=patched.device, dtype=patched.dtype)
            if isinstance(output, tuple):
                return (patched, *output[1:])
            return patched

        def capture_hook(_module: torch.nn.Module, _args: tuple[Any, ...], output: torch.Tensor) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            state["out"] = tensor

        patch_handle = model.model.layers[LAYER_INDEX].register_forward_hook(patch_hook)
        capture_handle = model.model.layers[LAYER_INDEX + 1].register_forward_hook(capture_hook)
        try:
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        finally:
            patch_handle.remove()
            capture_handle.remove()
        return state["out"][0, FINAL_TOKEN_POSITION, :]

    def path_jvp(point: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        point = point.detach().requires_grad_(True)
        tangent = direction.to(device=point.device, dtype=point.dtype)
        _, out_tangent = torch.autograd.functional.jvp(
            readout_map,
            point,
            tangent,
            create_graph=False,
            strict=False,
        )
        return out_tangent.detach()

    return path_jvp


def arm_for_manifest(arm: Arm) -> str:
    if arm is Arm.REAL_FEATURE:
        return "real"
    if arm is Arm.SHUFFLED_FEATURE:
        return "shuffled"
    if arm is Arm.RANDOM:
        return "random"
    raise ValueError(f"unknown arm: {arm}")


def pass1_select_and_magnitude(
    *,
    model: Any,
    sae: Any,
    extractor: ResidPostExtractor,
    record: dict[str, Any],
    local_index: int,
) -> Pass1Base:
    token_ids = [int(item) for item in record["token_ids"]]
    activation = extractor.extract(token_ids).activation
    code = sae.encode(activation)
    path_jvp = make_readout_jvp(model, token_ids, device=activation.device)
    base_jvp = lambda direction: path_jvp(activation, direction)
    selected = select_three_arm_planes(
        activation=activation,
        decoder_directions=sae.w_dec,
        code=code,
        jvp_fn=base_jvp,
        seed=PLANE_SELECTION_SEED + int(record["draw_order"]),
    )
    jh = base_jvp(activation).reshape(-1).detach().cpu().to(torch.float64)
    planes: dict[Arm, Pass1Plane] = {}
    for arm, plane in selected.items():
        coeffs = g_projection_coeffs(plane.jd, jh)
        mag_h = float(in_plane_magnitude(plane.jd, jh, coeffs=coeffs).detach().cpu())
        planes[arm] = Pass1Plane(arm=arm, selection=plane, mag_h=mag_h, coeffs=coeffs)
    return Pass1Base(
        base_point_id=f"bp-{local_index:03d}",
        corpus_draw_order=int(record["draw_order"]),
        article_index=int(record["article_index"]),
        token_ids=token_ids,
        activation=activation,
        planes=planes,
    )


def compute_band(pass1: list[Pass1Base]) -> BandDecision:
    values: dict[str, list[float]] = {"real": [], "shuffled": [], "random": []}
    for base in pass1:
        for arm, plane in base.planes.items():
            values[arm_for_manifest(arm)].append(plane.mag_h)
    return compute_common_support_band(real=values["real"], shuffled=values["shuffled"], random=values["random"])


def build_plane_record(
    *,
    base: Pass1Base,
    plane: Pass1Plane,
    m_low: float,
    m_high: float,
    reference: ReferenceDistribution,
    sae: Any,
) -> PlaneRecord:
    h = base.activation
    directions = plane.selection.directions
    in_plane = directions.mT @ plane.coeffs.to(device=h.device, dtype=h.dtype)
    out_of_plane = h - in_plane
    low = _center_from_coeffs(
        directions=directions,
        out_of_plane=out_of_plane,
        jd=plane.selection.jd,
        coeffs=plane.coeffs,
        target_magnitude=m_low,
    )
    high = _center_from_coeffs(
        directions=directions,
        out_of_plane=out_of_plane,
        jd=plane.selection.jd,
        coeffs=plane.coeffs,
        target_magnitude=m_high,
    )
    eps_fallback = bool(low[1] or high[1])
    low_center = low[0]
    high_center = high[0]
    rho_low = RADIUS_RELATIVE * torch.linalg.vector_norm(low_center)
    rho_high = RADIUS_RELATIVE * torch.linalg.vector_norm(high_center)
    cov_low = measure_plane_covariates(
        center=low_center,
        directions=directions,
        rho=rho_low,
        sae=sae,
        reference=reference,
        n_loop_samples=N_COVARIATE_LOOP_SAMPLES,
    )
    cov_high = measure_plane_covariates(
        center=high_center,
        directions=directions,
        rho=rho_high,
        sae=sae,
        reference=reference,
        n_loop_samples=N_COVARIATE_LOOP_SAMPLES,
    )
    covariates = PlaneCovariatesRecord(
        phi=cov_low.phi,
        manifold_distance_recon=0.5 * (cov_low.reconstruction_mse_mean + cov_high.reconstruction_mse_mean),
        manifold_distance_mahalanobis=0.5 * (cov_low.reference_distance_mean + cov_high.reference_distance_mean),
        manifold_distance_method=cov_low.reference_distance_method,
        manifold_distance_fallback=cov_low.reference_distance_method == "knn",
    )
    feature_ids = plane.selection.feature_ids
    return PlaneRecord(
        arm=arm_for_manifest(plane.arm),
        direction_1=_tensor_to_float_list(directions[0]),
        direction_2=_tensor_to_float_list(directions[1]),
        feature_indices=None if feature_ids is None else (int(feature_ids[0]), int(feature_ids[1])),
        det_M=float(plane.selection.det_m),
        mag_h=float(plane.mag_h),
        center_low=_tensor_to_float_list(low_center),
        center_high=_tensor_to_float_list(high_center),
        covariates=covariates,
        eps_mag_fallback_fired=eps_fallback,
    )


def _center_from_coeffs(
    *,
    directions: torch.Tensor,
    out_of_plane: torch.Tensor,
    jd: torch.Tensor,
    coeffs: torch.Tensor,
    target_magnitude: float,
) -> tuple[torch.Tensor, bool]:
    magnitude = torch.linalg.vector_norm(jd @ coeffs).to(device=out_of_plane.device, dtype=out_of_plane.dtype)
    target = torch.tensor(target_magnitude, dtype=out_of_plane.dtype, device=out_of_plane.device)
    if bool((magnitude < EPS_MAG).detach().cpu()):
        d1_norm = torch.linalg.vector_norm(jd[:, 0]).to(device=out_of_plane.device, dtype=out_of_plane.dtype)
        offset_coeffs = torch.zeros_like(coeffs).to(device=out_of_plane.device, dtype=out_of_plane.dtype)
        offset_coeffs[0] = target / d1_norm.clamp_min(torch.finfo(out_of_plane.dtype).eps)
        return out_of_plane + directions.mT @ offset_coeffs, True
    scaled_coeffs = coeffs.to(device=out_of_plane.device, dtype=out_of_plane.dtype) * (target / magnitude)
    return out_of_plane + directions.mT @ scaled_coeffs, False


def _tensor_to_float_list(value: torch.Tensor) -> list[float]:
    return [float(item) for item in value.detach().cpu().reshape(-1).tolist()]


def build_balance_diagnostics(base_records: list[BasePointRecord]) -> BalanceDiagnosticsRecord:
    by_arm: dict[str, list[PlaneRecord]] = {"real": [], "shuffled": []}
    for base in base_records:
        for plane in base.planes:
            if plane.arm in by_arm:
                by_arm[plane.arm].append(plane)
    real = by_arm["real"]
    shuffled = by_arm["shuffled"]
    return BalanceDiagnosticsRecord(
        manifold_distance_recon=_balance_metric(
            [plane.covariates.manifold_distance_recon for plane in real],
            [plane.covariates.manifold_distance_recon for plane in shuffled],
        ),
        manifold_distance_mahalanobis=_balance_metric(
            [plane.covariates.manifold_distance_mahalanobis for plane in real],
            [plane.covariates.manifold_distance_mahalanobis for plane in shuffled],
        ),
        phi=_balance_metric(
            [plane.covariates.phi for plane in real],
            [plane.covariates.phi for plane in shuffled],
        ),
        log_sin_phi_mean_diff=float(
            np.mean([math.log(max(math.sin(plane.covariates.phi), 1e-12)) for plane in real])
            - np.mean([math.log(max(math.sin(plane.covariates.phi), 1e-12)) for plane in shuffled])
        ),
    )


def _balance_metric(real_values: list[float], shuffled_values: list[float]) -> BalanceMetricRecord:
    real = np.asarray(real_values, dtype=np.float64)
    shuffled = np.asarray(shuffled_values, dtype=np.float64)
    real_sd = np.var(real, ddof=1) if real.size > 1 else 0.0
    shuffled_sd = np.var(shuffled, ddof=1) if shuffled.size > 1 else 0.0
    pooled_sd = math.sqrt(max(0.0, 0.5 * (real_sd + shuffled_sd)))
    mean_diff = float(np.mean(real) - np.mean(shuffled))
    smd = 0.0 if pooled_sd == 0 and mean_diff == 0 else mean_diff / max(pooled_sd, 1e-12)
    real_min, real_max = float(np.min(real)), float(np.max(real))
    shuffled_min, shuffled_max = float(np.min(shuffled)), float(np.max(shuffled))
    return BalanceMetricRecord(
        smd=float(smd),
        real_min=real_min,
        real_max=real_max,
        shuffled_min=shuffled_min,
        shuffled_max=shuffled_max,
        real_within_shuffled_range=float(np.mean((real >= shuffled_min) & (real <= shuffled_max))),
        shuffled_within_real_range=float(np.mean((shuffled >= real_min) & (shuffled <= real_max))),
    )


def metadata(*, manifest_kind: str, smoke_magnitude_source: str | None = None) -> ManifestMetadata:
    return ManifestMetadata(
        prereg_version="PREREGISTRATION_v2.md",
        prereg_status_line=PREREG_STATUS_LINE,
        prereg_commit_hash=PREREG_COMMIT_HASH,
        dataset_revision=DATASET_REVISION,
        seed_corpus=SEED_CORPUS,
        plane_selection_seed=PLANE_SELECTION_SEED,
        model_id=MODEL_NAME,
        sae_repo=SAE_REPO,
        sae_path=SAE_PATH,
        layer=LAYER_INDEX,
        extraction_site="resid_post: HF model.model.layers[12].output",
        tau_detM=TAU_DETM,
        eps_mag=EPS_MAG,
        radius_relative=RADIUS_RELATIVE,
        n_steps=N_STEPS,
        n_base_points=N_BASE_POINTS,
        reproducibility_claim=(
            "Bitwise within the fixed MPS backend plus deterministic CPU float64 post-processing path."
        ),
        manifest_kind=manifest_kind,  # type: ignore[arg-type]
        smoke_magnitude_source=smoke_magnitude_source,
    )


def smoke_m_values_from_pooled_magnitudes(pass1: list[Pass1Base]) -> tuple[float, float]:
    values = np.array([plane.mag_h for base in pass1 for plane in base.planes.values()], dtype=np.float64)
    if values.size == 0:
        raise RuntimeError("cannot compute smoke m values from empty pass1 records")
    m_low, m_high = np.percentile(values, [25, 75])
    return float(m_low), float(m_high)


def resolve_magnitudes_for_centers(
    *,
    band_decision: BandDecision,
    pass1: list[Pass1Base],
    manifest_kind: str,
) -> MagnitudeResolution:
    """Return official/fallback/smoke terminal behavior for Stage A centers."""

    if band_decision.m_low is not None and band_decision.m_high is not None:
        return MagnitudeResolution(
            m_low=float(band_decision.m_low),
            m_high=float(band_decision.m_high),
            smoke_magnitude_source=None,
            should_halt=False,
            message=f"{band_decision.status.value}: using pre-registered band m values",
        )

    if manifest_kind == "smoke":
        m_low, m_high = smoke_m_values_from_pooled_magnitudes(pass1)
        return MagnitudeResolution(
            m_low=m_low,
            m_high=m_high,
            smoke_magnitude_source=(
                "SMOKE_ONLY_POOLED_MAG_QUARTILES_DUE_TO_TERMINAL_BAND;"
                " band_decision remains terminal and these centers are not interpretable"
            ),
            should_halt=False,
            message="SMOKE_ONLY terminal-band center override",
        )

    return MagnitudeResolution(
        m_low=None,
        m_high=None,
        smoke_magnitude_source=None,
        should_halt=True,
        message=f"Stage A cannot construct centers: band decision has no m values ({band_decision.status})",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage A blind manifest producer")
    parser.add_argument("--limit", type=int, default=None, help="Smoke limit; writes run/manifest_smoke.json")
    parser.add_argument("--corpus", type=Path, default=Path("run/corpus_draw_n390.json"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    output = args.output or (Path("run/manifest_smoke.json") if args.limit is not None else Path("run/manifest_n390.json"))
    manifest_kind = "smoke" if args.limit is not None else "official"
    records = load_experiment_records(args.corpus, limit=args.limit)

    started = time.time()
    print(f"stage_a manifest_kind={manifest_kind} records={len(records)} output={output}")
    model, _tokenizer = load_model_and_tokenizer()
    sae = load_sae(device=next(model.parameters()).device)
    extractor = ResidPostExtractor(model)

    pass1: list[Pass1Base] = []
    rejection_totals = {arm.value: {"tested": 0, "rejected": 0, "accepted": 0} for arm in Arm}
    for idx, record in enumerate(records):
        base_started = time.time()
        print(f"[pass1 {idx + 1}/{len(records)}] draw_order={record['draw_order']} title={record.get('article_title', '')!r}")
        base = pass1_select_and_magnitude(model=model, sae=sae, extractor=extractor, record=record, local_index=idx)
        pass1.append(base)
        for arm, plane in base.planes.items():
            rejection_totals[arm.value]["tested"] += plane.selection.candidates_tested
            rejection_totals[arm.value]["rejected"] += plane.selection.candidates_rejected_detm
            rejection_totals[arm.value]["accepted"] += 1
        print(f"  pass1_seconds={time.time() - base_started:.3f}")

    band_decision = compute_band(pass1)
    magnitude_resolution = resolve_magnitudes_for_centers(
        band_decision=band_decision,
        pass1=pass1,
        manifest_kind=manifest_kind,
    )
    if magnitude_resolution.should_halt:
        raise RuntimeError(magnitude_resolution.message)
    if magnitude_resolution.m_low is None or magnitude_resolution.m_high is None:
        raise RuntimeError("internal error: magnitude resolution did not halt but lacks m values")
    m_low, m_high = magnitude_resolution.m_low, magnitude_resolution.m_high
    if magnitude_resolution.smoke_magnitude_source is not None:
        print(
            "smoke-only terminal-band center override "
            f"m_low={m_low:.6g} m_high={m_high:.6g}; band_decision remains {band_decision.status.value}"
        )
    print(
        "band_decision "
        f"status={band_decision.status.value} m_low={m_low:.6g} m_high={m_high:.6g}"
    )

    reference = ReferenceDistribution.fit(torch.stack([base.activation.detach().cpu() for base in pass1]), k=REFERENCE_K)
    base_records: list[BasePointRecord] = []
    eps_fallback_count = 0
    for idx, base in enumerate(pass1):
        print(f"[pass2 {idx + 1}/{len(pass1)}] base_point_id={base.base_point_id}")
        plane_records = []
        for arm in [Arm.REAL_FEATURE, Arm.SHUFFLED_FEATURE, Arm.RANDOM]:
            record = build_plane_record(
                base=base,
                plane=base.planes[arm],
                m_low=m_low,
                m_high=m_high,
                reference=reference,
                sae=sae,
            )
            eps_fallback_count += int(record.eps_mag_fallback_fired)
            plane_records.append(record)
        base_records.append(
            BasePointRecord(
                base_point_id=base.base_point_id,
                corpus_draw_order=base.corpus_draw_order,
                article_index=base.article_index,
                token_ids=base.token_ids,
                activation_ref=f"manifest.base_points[{idx}].activation",
                activation=_tensor_to_float_list(base.activation),
                planes=plane_records,
            )
        )

    balance = build_balance_diagnostics(base_records)
    manifest = RunManifest(
        metadata=metadata(manifest_kind=manifest_kind, smoke_magnitude_source=magnitude_resolution.smoke_magnitude_source),
        band_decision=band_decision,
        balance_diagnostics=balance,
        base_points=base_records,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(manifest_to_json(manifest))
    summary = {
        "output": str(output),
        "base_points": len(base_records),
        "rejection_totals": rejection_totals,
        "band_status": band_decision.status.value,
        "m_low": m_low,
        "m_high": m_high,
        "smoke_magnitude_source": magnitude_resolution.smoke_magnitude_source,
        "balance": {
            "recon_smd": balance.manifold_distance_recon.smd,
            "guard_smd": balance.manifold_distance_mahalanobis.smd,
            "phi_smd": balance.phi.smd,
            "log_sin_phi_mean_diff": balance.log_sin_phi_mean_diff,
        },
        "eps_mag_fallback_count": eps_fallback_count,
        "elapsed_seconds": time.time() - started,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
