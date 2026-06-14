"""Stage B measurement orchestration.

Stage B reads a Stage A manifest and writes response measurements to a separate
artifact. It never mutates the manifest. Resumability is implemented by a JSONL
checkpoint keyed by completed base points; parquet/json outputs are regenerated
after each completed base point from the checkpoint rows.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from bands import BandStatus
from extraction import FINAL_TOKEN_POSITION, LAYER_INDEX, load_model_and_tokenizer
from manifest import BasePointRecord, PlaneRecord, RunManifest, manifest_from_json
from transport import loop_transport


MagnitudeLevel = str
TransportMeasureFn = Callable[[BasePointRecord, PlaneRecord, MagnitudeLevel], dict[str, float]]
RESULT_COLUMNS = [
    "base_point_id",
    "arm",
    "magnitude_level",
    "H",
    "log_H",
    "theta",
    "signed_holonomy",
    "A_enclosed",
    "det_M",
]


def load_manifest(path: Path, *, allow_smoke: bool = False) -> RunManifest:
    """Load and guard a manifest for Stage B measurement."""

    manifest = manifest_from_json(path.read_text())
    if manifest.band_decision.status == BandStatus.TWO_ARM_TERMINAL:
        raise RuntimeError("Stage B refuses TERMINAL manifest: no matched centers to measure")
    if manifest.metadata.manifest_kind != "official" and not allow_smoke:
        raise RuntimeError("Stage B refuses non-official manifest unless --allow-smoke is set")
    if manifest.metadata.manifest_kind == "official" and len(manifest.base_points) != manifest.metadata.n_base_points:
        raise RuntimeError("Stage B official manifest is incomplete")
    return manifest


def make_model_transport_measure(model: Any, manifest: RunManifest) -> TransportMeasureFn:
    """Return a measurement function using the Gemma resid_post readout JVP."""

    device = next(model.parameters()).device

    def measure(base: BasePointRecord, plane: PlaneRecord, magnitude_level: MagnitudeLevel) -> dict[str, float]:
        center_values = plane.center_low if magnitude_level == "low" else plane.center_high
        center = torch.tensor(center_values, dtype=torch.float32, device=device)
        directions = torch.tensor([plane.direction_1, plane.direction_2], dtype=torch.float32, device=device)
        base_activation = torch.tensor(base.activation, dtype=torch.float32, device=device)
        rho = manifest.metadata.radius_relative * torch.linalg.vector_norm(base_activation)
        jvp_fn = make_readout_jvp(model, base.token_ids, device=device)
        result = loop_transport(
            center=center,
            directions=directions,
            rho=rho,
            jvp_fn=jvp_fn,
            n_steps=manifest.metadata.n_steps,
        )
        holonomy = float(result.holonomy.detach().cpu())
        return {
            "H": holonomy,
            "log_H": math.log(max(holonomy, torch.finfo(torch.float64).tiny)),
            "theta": float(result.theta.detach().cpu()),
            "signed_holonomy": float(result.signed_holonomy.detach().cpu()),
            "A_enclosed": float(result.area_enclosed.detach().cpu()),
            "det_M": float(result.det_m.detach().cpu()),
        }

    return measure


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

    def jvp_fn(point: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
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

    return jvp_fn


def checkpoint_path_for(results_path: Path) -> Path:
    return results_path.with_suffix(results_path.suffix + ".checkpoint.jsonl")


def json_path_for(results_path: Path) -> Path:
    return results_path.with_suffix(".json")


def load_checkpoint_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def completed_base_points(rows: list[dict[str, Any]]) -> set[str]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["base_point_id"]] = counts.get(row["base_point_id"], 0) + 1
    return {base_id for base_id, count in counts.items() if count == 6}


def rows_for_base(
    base: BasePointRecord,
    *,
    measure_fn: TransportMeasureFn,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for plane in base.planes:
        for magnitude_level in ["low", "high"]:
            values = measure_fn(base, plane, magnitude_level)
            row = {
                "base_point_id": base.base_point_id,
                "arm": plane.arm,
                "magnitude_level": magnitude_level,
                **values,
            }
            rows.append(row)
    return rows


def append_checkpoint_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_result_artifacts(rows: list[dict[str, Any]], results_path: Path) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    frame.to_parquet(results_path, index=False)
    json_path_for(results_path).write_text(json.dumps(rows, indent=2) + "\n")


def run_stage_b(
    *,
    manifest_path: Path,
    results_path: Path,
    allow_smoke: bool = False,
    measure_fn: TransportMeasureFn | None = None,
    max_new_base_points: int | None = None,
) -> dict[str, Any]:
    """Run or resume Stage B and return a summary."""

    manifest = load_manifest(manifest_path, allow_smoke=allow_smoke)
    checkpoint_path = checkpoint_path_for(results_path)
    rows = load_checkpoint_rows(checkpoint_path)
    completed = completed_base_points(rows)
    if measure_fn is None:
        model, _tokenizer = load_model_and_tokenizer()
        measure_fn = make_model_transport_measure(model, manifest)

    started = time.time()
    processed_now = 0
    for idx, base in enumerate(manifest.base_points):
        if base.base_point_id in completed:
            print(f"[stage_b {idx + 1}/{len(manifest.base_points)}] {base.base_point_id} already complete; skipping")
            continue
        if max_new_base_points is not None and processed_now >= max_new_base_points:
            break
        print(f"[stage_b {idx + 1}/{len(manifest.base_points)}] measuring {base.base_point_id}")
        base_rows = rows_for_base(base, measure_fn=measure_fn)
        append_checkpoint_rows(checkpoint_path, base_rows)
        rows.extend(base_rows)
        completed.add(base.base_point_id)
        processed_now += 1
        write_result_artifacts(rows, results_path)

    nonfinite = [
        row
        for row in rows
        if not (
            math.isfinite(float(row["H"]))
            and float(row["H"]) > 0
            and math.isfinite(float(row["log_H"]))
            and math.isfinite(float(row["theta"]))
            and math.isfinite(float(row["signed_holonomy"]))
            and math.isfinite(float(row["A_enclosed"]))
            and math.isfinite(float(row["det_M"]))
        )
    ]
    if rows:
        write_result_artifacts(rows, results_path)
    summary = {
        "manifest": str(manifest_path),
        "results": str(results_path),
        "checkpoint": str(checkpoint_path),
        "rows_written": len(rows),
        "completed_base_points": len(completed_base_points(rows)),
        "processed_now": processed_now,
        "nonfinite_rows": len(nonfinite),
        "elapsed_seconds": time.time() - started,
    }
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage B holonomy producer")
    parser.add_argument("--manifest", type=Path, default=Path("run/manifest_n390.json"))
    parser.add_argument("--results", type=Path, default=Path("run/holonomy_results_n390.parquet"))
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--max-new-base-points", type=int, default=None)
    args = parser.parse_args()

    run_stage_b(
        manifest_path=args.manifest,
        results_path=args.results,
        allow_smoke=args.allow_smoke,
        max_new_base_points=args.max_new_base_points,
    )


if __name__ == "__main__":
    main()
