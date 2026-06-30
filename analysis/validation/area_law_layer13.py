"""Post-freeze area-law validation at the final layer-13 readout.

Status: POST_FREEZE_INSTRUMENT_VALIDATION_NOT_PART_OF_FROZEN_VERDICT
"""

from __future__ import annotations

import argparse
from pathlib import Path

from analysis.validation.common import (
    STATUS_VALIDATION,
    load_model,
    load_official_manifest,
    markdown_table,
    run_loop_measurement,
    selected_measurements,
    summarize_area_law,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-count", type=int, default=4)
    parser.add_argument("--arms", nargs="+", default=["real", "shuffled"], choices=["real", "shuffled", "random"])
    parser.add_argument("--levels", nargs="+", default=["low"], choices=["low", "high"])
    parser.add_argument("--radius-scales", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--json-out", default="reports/area_law_layer13_validation.json")
    parser.add_argument("--md-out", default="reports/area_law_layer13_validation.md")
    args = parser.parse_args()

    manifest = load_official_manifest()
    print("loading Gemma model for area-law validation", flush=True)
    model, _tokenizer, device = load_model()
    measurements = selected_measurements(manifest, base_count=args.base_count, arms=args.arms, levels=args.levels)

    rows = []
    for base, plane, level in measurements:
        for scale in args.radius_scales:
            print(
                f"area-law loop base={base.base_point_id} arm={plane.arm} "
                f"level={level} radius_scale={scale}",
                flush=True,
            )
            rows.append(
                run_loop_measurement(
                    model=model,
                    manifest=manifest,
                    base=base,
                    plane=plane,
                    magnitude_level=level,
                    radius_scale=scale,
                )
            )
            print(f"  finished in {rows[-1]['runtime_seconds']:.2f}s", flush=True)

    summary = summarize_area_law(rows)
    payload = {
        "status": STATUS_VALIDATION,
        "device": str(device),
        "selection_rule": "first N frozen manifest base points; requested arms/levels; requested radius scales",
        "base_count": args.base_count,
        "arms": args.arms,
        "levels": args.levels,
        "radius_scales": args.radius_scales,
        "rows": rows,
        "summary": summary,
    }
    write_json(Path(args.json_out), payload)

    md = f"""# Area-law validation at layer-13 readout

Status: {STATUS_VALIDATION}

Selection rule: first {args.base_count} frozen manifest base points; arms `{args.arms}`; levels `{args.levels}`; radius scales `{args.radius_scales}`.

Target: signed rotation should scale approximately as `rho^2`, so the pooled/per-loop log-log slope of `abs(theta)` versus `rho` should be near 2.

- Device: `{device}`
- Pooled log(abs(theta)) vs log(rho) slope: {summary['pooled_log_abs_theta_vs_log_rho_slope']}
- Total loop runtime seconds: {summary['total_runtime_seconds']:.2f}

## Per-loop slope summary

{markdown_table(summary['per_loop'], ['base_point_id', 'plane_type', 'magnitude_level', 'log_abs_theta_vs_log_rho_slope', 'H_min', 'H_max'])}

## Measurement rows

{markdown_table(rows, ['base_point_id', 'plane_type', 'magnitude_level', 'radius_scale', 'rho', 'theta', 'abs_theta', 'area', 'H', 'theta_over_rho2', 'abs_theta_over_rho2', 'symmetric_residual_norm', 'max_condition', 'runtime_seconds'])}

The confirmatory verdict remains the preregistered frozen result. These analyses are post-freeze diagnostics and validation checks added to characterize the instrument and residual threats to validity; they do not enter or modify the preregistered decision rule.
"""
    Path(args.md_out).write_text(md)


if __name__ == "__main__":
    main()
