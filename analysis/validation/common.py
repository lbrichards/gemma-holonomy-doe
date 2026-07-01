"""Shared helpers for post-freeze layer-13 validation runs."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from extraction import load_model_and_tokenizer
from manifest import BasePointRecord, PlaneRecord, RunManifest
from stage_b.run import load_manifest, make_readout_jvp
from transport import loop_transport


STATUS_VALIDATION = "POST_FREEZE_INSTRUMENT_VALIDATION_NOT_PART_OF_FROZEN_VERDICT"
STATUS_DIAGNOSTIC = "POST_FREEZE_DIAGNOSTIC_NOT_CONFIRMATORY"
STATUS_BENCHMARK = "POST_FREEZE_COMPUTE_BENCHMARK_NOT_CONFIRMATORY"

ARM_DISPLAY = {
    "real": "active-feature",
    "shuffled": "mixed-feature",
    "random": "random",
}


def now_seconds() -> float:
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    return time.perf_counter()


def load_official_manifest(path: Path = Path("run/manifest_n390.json")) -> RunManifest:
    return load_manifest(path, allow_smoke=False)


def load_model():
    model, tokenizer = load_model_and_tokenizer()
    return model, tokenizer, next(model.parameters()).device


def plane_by_arm(base: BasePointRecord, arm: str) -> PlaneRecord:
    for plane in base.planes:
        if plane.arm == arm:
            return plane
    raise KeyError(f"{base.base_point_id}: no plane for arm={arm!r}")


def center_for_level(plane: PlaneRecord, level: str, *, device: torch.device) -> torch.Tensor:
    values = plane.center_low if level == "low" else plane.center_high
    return torch.tensor(values, dtype=torch.float32, device=device)


def directions_tensor(plane: PlaneRecord, *, device: torch.device) -> torch.Tensor:
    return torch.tensor([plane.direction_1, plane.direction_2], dtype=torch.float32, device=device)


def frozen_rho(manifest: RunManifest, base: BasePointRecord, *, device: torch.device) -> torch.Tensor:
    activation = torch.tensor(base.activation, dtype=torch.float32, device=device)
    return manifest.metadata.radius_relative * torch.linalg.vector_norm(activation)


def run_loop_measurement(
    *,
    model: Any,
    manifest: RunManifest,
    base: BasePointRecord,
    plane: PlaneRecord,
    magnitude_level: str,
    radius_scale: float = 1.0,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    center = center_for_level(plane, magnitude_level, device=device)
    directions = directions_tensor(plane, device=device)
    rho = frozen_rho(manifest, base, device=device) * float(radius_scale)
    jvp_fn = make_readout_jvp(model, base.token_ids, device=device)

    start = now_seconds()
    result = loop_transport(
        center=center,
        directions=directions,
        rho=rho,
        jvp_fn=jvp_fn,
        n_steps=manifest.metadata.n_steps,
    )
    runtime = now_seconds() - start

    operator = result.transport_operator.detach().cpu().to(torch.float64)
    eye = torch.eye(2, dtype=torch.float64)
    non_orthogonality = torch.linalg.matrix_norm(operator.mT @ operator - eye)
    theta = float(result.theta.detach().cpu())
    area = float(result.area_enclosed.detach().cpu())
    holonomy = float(result.holonomy.detach().cpu())
    rho_value = float(rho.detach().cpu())
    safe_rho2 = max(rho_value * rho_value, torch.finfo(torch.float64).tiny)

    return {
        "base_point_id": base.base_point_id,
        "arm": plane.arm,
        "plane_type": ARM_DISPLAY.get(plane.arm, plane.arm),
        "magnitude_level": magnitude_level,
        "radius_scale": float(radius_scale),
        "rho": rho_value,
        "theta": theta,
        "abs_theta": abs(theta),
        "area": area,
        "A_enclosed": area,
        "det_M": float(result.det_m.detach().cpu()),
        "H": holonomy,
        "log_H": math.log(max(holonomy, torch.finfo(torch.float64).tiny)),
        "theta_over_rho2": theta / safe_rho2,
        "abs_theta_over_rho2": abs(theta) / safe_rho2,
        "symmetric_residual_norm": float(result.symmetric_residual_norm.detach().cpu()),
        "non_orthogonality_norm": float(non_orthogonality.detach().cpu()),
        "transport_residual_norm": float(torch.linalg.matrix_norm(operator - eye).detach().cpu()),
        "max_condition": float(result.max_condition),
        "runtime_seconds": runtime,
    }


def run_single_jvp(
    *,
    model: Any,
    base: BasePointRecord,
    plane: PlaneRecord,
    magnitude_level: str,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    center = center_for_level(plane, magnitude_level, device=device)
    directions = directions_tensor(plane, device=device)
    jvp_fn = make_readout_jvp(model, base.token_ids, device=device)
    start = now_seconds()
    out = jvp_fn(center, directions[0])
    runtime = now_seconds() - start
    return {
        "base_point_id": base.base_point_id,
        "arm": plane.arm,
        "plane_type": ARM_DISPLAY.get(plane.arm, plane.arm),
        "magnitude_level": magnitude_level,
        "direction": "direction_1",
        "output_norm": float(torch.linalg.vector_norm(out).detach().cpu()),
        "runtime_seconds": runtime,
    }


def selected_measurements(
    manifest: RunManifest,
    *,
    base_count: int,
    arms: list[str],
    levels: list[str],
) -> list[tuple[BasePointRecord, PlaneRecord, str]]:
    selected = []
    for base in manifest.base_points[:base_count]:
        for arm in arms:
            plane = plane_by_arm(base, arm)
            for level in levels:
                selected.append((base, plane, level))
    return selected


def linear_slope(xs: list[float], ys: list[float]) -> float | None:
    finite = [(x, y) for x, y in zip(xs, ys, strict=False) if x > 0 and y > 0 and math.isfinite(x) and math.isfinite(y)]
    if len(finite) < 2:
        return None
    lx = np.log([x for x, _y in finite])
    ly = np.log([y for _x, y in finite])
    slope, _intercept = np.polyfit(lx, ly, deg=1)
    return float(slope)


def summarize_area_law(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["base_point_id"], row["arm"], row["magnitude_level"])
        groups.setdefault(key, []).append(row)
    per_loop = []
    for (base_id, arm, level), group in groups.items():
        per_loop.append(
            {
                "base_point_id": base_id,
                "arm": arm,
                "plane_type": ARM_DISPLAY.get(arm, arm),
                "magnitude_level": level,
                "log_abs_theta_vs_log_rho_slope": linear_slope(
                    [row["rho"] for row in group],
                    [row["abs_theta"] for row in group],
                ),
                "H_min": float(min(row["H"] for row in group)),
                "H_max": float(max(row["H"] for row in group)),
            }
        )
    return {
        "pooled_log_abs_theta_vs_log_rho_slope": linear_slope(
            [row["rho"] for row in rows],
            [row["abs_theta"] for row in rows],
        ),
        "per_loop": per_loop,
        "total_runtime_seconds": float(sum(row["runtime_seconds"] for row in rows)),
    }


def summarize_shear(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "total_runtime_seconds": float(sum(row["runtime_seconds"] for row in rows)),
        "by_arm_level": {},
        "paired_real_minus_shuffled": [],
        "correlations_with_log_H": {},
    }
    for metric in ["symmetric_residual_norm", "non_orthogonality_norm", "transport_residual_norm"]:
        values = np.array([row[metric] for row in rows], dtype=float)
        log_h = np.array([row["log_H"] for row in rows], dtype=float)
        if len(values) > 1 and float(np.std(values)) > 0 and float(np.std(log_h)) > 0:
            corr = float(np.corrcoef(values, log_h)[0, 1])
        else:
            corr = None
        summary["correlations_with_log_H"][metric] = corr

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["arm"], row["magnitude_level"]), []).append(row)
    for (arm, level), group in grouped.items():
        key = f"{arm}/{level}"
        summary["by_arm_level"][key] = {
            "arm": arm,
            "plane_type": ARM_DISPLAY.get(arm, arm),
            "magnitude_level": level,
            "n": len(group),
            "symmetric_residual_norm_mean": float(np.mean([row["symmetric_residual_norm"] for row in group])),
            "symmetric_residual_norm_median": float(np.median([row["symmetric_residual_norm"] for row in group])),
            "non_orthogonality_norm_mean": float(np.mean([row["non_orthogonality_norm"] for row in group])),
            "non_orthogonality_norm_median": float(np.median([row["non_orthogonality_norm"] for row in group])),
        }

    by_base_level_arm = {(row["base_point_id"], row["magnitude_level"], row["arm"]): row for row in rows}
    for row in rows:
        if row["arm"] != "real":
            continue
        other = by_base_level_arm.get((row["base_point_id"], row["magnitude_level"], "shuffled"))
        if other is None:
            continue
        summary["paired_real_minus_shuffled"].append(
            {
                "base_point_id": row["base_point_id"],
                "magnitude_level": row["magnitude_level"],
                "active_feature_minus_mixed_feature_symmetric_residual_norm": row["symmetric_residual_norm"]
                - other["symmetric_residual_norm"],
                "active_feature_minus_mixed_feature_non_orthogonality_norm": row["non_orthogonality_norm"]
                - other["non_orthogonality_norm"],
            }
        )
    return summary


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def markdown_table(rows: list[dict[str, Any]], fields: list[str], *, limit: int | None = None) -> str:
    shown = rows if limit is None else rows[:limit]
    header = "| " + " | ".join(fields) + " |"
    sep = "| " + " | ".join("---" for _ in fields) + " |"
    lines = [header, sep]
    for row in shown:
        values = []
        for field in fields:
            value = row.get(field)
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def serializable_manifest_metadata(manifest: RunManifest) -> dict[str, Any]:
    return asdict(manifest.metadata)
