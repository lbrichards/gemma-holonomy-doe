"""Benchmark the Gemma layer-12 to layer-13 JVP/loop path.

Status: POST_FREEZE_COMPUTE_BENCHMARK_NOT_CONFIRMATORY
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from analysis.validation.common import (
    STATUS_BENCHMARK,
    load_model,
    load_official_manifest,
    plane_by_arm,
    run_loop_measurement,
    run_single_jvp,
    serializable_manifest_metadata,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="real", choices=["real", "shuffled", "random"])
    parser.add_argument("--magnitude-level", default="low", choices=["low", "high"])
    parser.add_argument("--json-out", default="reports/compute_benchmark_layer13.json")
    parser.add_argument("--md-out", default="reports/compute_benchmark_layer13.md")
    args = parser.parse_args()

    manifest = load_official_manifest()
    base = manifest.base_points[0]
    plane = plane_by_arm(base, args.arm)

    print("loading Gemma model for layer-13 benchmark", flush=True)
    load_started = time.perf_counter()
    model, _tokenizer, device = load_model()
    model_load_seconds = time.perf_counter() - load_started
    print(f"Gemma model loaded on {device}; running one JVP", flush=True)

    jvp = run_single_jvp(model=model, base=base, plane=plane, magnitude_level=args.magnitude_level)
    print(f"single JVP finished in {jvp['runtime_seconds']:.2f}s; running one 200-step loop", flush=True)
    loop = run_loop_measurement(
        model=model,
        manifest=manifest,
        base=base,
        plane=plane,
        magnitude_level=args.magnitude_level,
        radius_scale=1.0,
    )
    print(f"200-step loop finished in {loop['runtime_seconds']:.2f}s", flush=True)

    payload = {
        "status": STATUS_BENCHMARK,
        "selection_rule": "first frozen manifest base point; requested arm and magnitude level",
        "device": str(device),
        "model_load_seconds": model_load_seconds,
        "manifest_metadata": serializable_manifest_metadata(manifest),
        "single_jvp": jvp,
        "loop_transport_n_steps_200": loop,
    }
    write_json(Path(args.json_out), payload)

    md = f"""# Layer-13 compute benchmark

Status: {STATUS_BENCHMARK}

Selection rule: first frozen manifest base point; arm `{args.arm}` ({loop['plane_type']}), magnitude level `{args.magnitude_level}`.

Device: `{device}`

- Model load seconds: {model_load_seconds:.2f}
- Single JVP seconds: {jvp['runtime_seconds']:.2f}
- One full 200-step loop seconds: {loop['runtime_seconds']:.2f}
- H: {loop['H']:.8g}
- theta: {loop['theta']:.8g}
- area: {loop['area']:.8g}
- symmetric residual norm: {loop['symmetric_residual_norm']:.8g}
- non-orthogonality norm: {loop['non_orthogonality_norm']:.8g}
- max condition: {loop['max_condition']:.8g}

The confirmatory verdict remains the preregistered frozen result. This benchmark is post-freeze and does not enter or modify the preregistered decision rule.
"""
    Path(args.md_out).write_text(md)


if __name__ == "__main__":
    main()
