"""Post-freeze transport shear / non-orthogonality diagnostic.

Status: POST_FREEZE_DIAGNOSTIC_NOT_CONFIRMATORY
"""

from __future__ import annotations

import argparse
from pathlib import Path

from analysis.validation.common import (
    STATUS_DIAGNOSTIC,
    load_model,
    load_official_manifest,
    markdown_table,
    run_loop_measurement,
    selected_measurements,
    summarize_shear,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-count", type=int, default=4)
    parser.add_argument("--arms", nargs="+", default=["real", "shuffled"], choices=["real", "shuffled", "random"])
    parser.add_argument("--levels", nargs="+", default=["low", "high"], choices=["low", "high"])
    parser.add_argument("--json-out", default="reports/transport_shear_diagnostic.json")
    parser.add_argument("--md-out", default="reports/transport_shear_diagnostic.md")
    args = parser.parse_args()

    manifest = load_official_manifest()
    print("loading Gemma model for transport shear diagnostic", flush=True)
    model, _tokenizer, device = load_model()
    measurements = selected_measurements(manifest, base_count=args.base_count, arms=args.arms, levels=args.levels)

    rows = []
    for base, plane, level in measurements:
        print(
            f"shear loop base={base.base_point_id} arm={plane.arm} level={level}",
            flush=True,
        )
        rows.append(
            run_loop_measurement(
            model=model,
            manifest=manifest,
            base=base,
            plane=plane,
            magnitude_level=level,
            radius_scale=1.0,
        )
        )
        print(f"  finished in {rows[-1]['runtime_seconds']:.2f}s", flush=True)
    summary = summarize_shear(rows)
    payload = {
        "status": STATUS_DIAGNOSTIC,
        "device": str(device),
        "selection_rule": "first N frozen manifest base points; requested arms/levels; frozen radius",
        "definitions": {
            "symmetric_residual_norm": "Frobenius norm of 0.5 * ((T - I) + (T - I)^T).",
            "non_orthogonality_norm": "Frobenius norm of T^T T - I.",
            "transport_residual_norm": "Frobenius norm of T - I.",
        },
        "base_count": args.base_count,
        "arms": args.arms,
        "levels": args.levels,
        "rows": rows,
        "summary": summary,
    }
    write_json(Path(args.json_out), payload)

    md = f"""# Transport shear / non-orthogonality diagnostic

Status: {STATUS_DIAGNOSTIC}

Selection rule: first {args.base_count} frozen manifest base points; arms `{args.arms}`; levels `{args.levels}`; frozen radius.

Definitions:

- `symmetric_residual_norm`: Frobenius norm of `0.5 * ((T - I) + (T - I)^T)`.
- `||T.T @ T - I||`: reported as `non_orthogonality_norm`.
- `||T - I||`: reported as `transport_residual_norm`.

- Device: `{device}`
- Total loop runtime seconds: {summary['total_runtime_seconds']:.2f}

## Arm/level summary

{markdown_table(list(summary['by_arm_level'].values()), ['plane_type', 'magnitude_level', 'n', 'symmetric_residual_norm_mean', 'symmetric_residual_norm_median', 'non_orthogonality_norm_mean', 'non_orthogonality_norm_median'])}

## Correlations with log_H

{markdown_table([summary['correlations_with_log_H']], ['symmetric_residual_norm', 'non_orthogonality_norm', 'transport_residual_norm'])}

## Paired active-feature minus mixed-feature diagnostic differences

{markdown_table(summary['paired_real_minus_shuffled'], ['base_point_id', 'magnitude_level', 'active_feature_minus_mixed_feature_symmetric_residual_norm', 'active_feature_minus_mixed_feature_non_orthogonality_norm'])}

## Measurement rows

{markdown_table(rows, ['base_point_id', 'plane_type', 'magnitude_level', 'H', 'theta', 'area', 'symmetric_residual_norm', 'non_orthogonality_norm', 'transport_residual_norm', 'max_condition', 'runtime_seconds'])}

The confirmatory verdict remains the preregistered frozen result. These analyses are post-freeze diagnostics and validation checks added to characterize the instrument and residual threats to validity; they do not enter or modify the preregistered decision rule.
"""
    Path(args.md_out).write_text(md)


if __name__ == "__main__":
    main()
