"""Full post-freeze transport shear diagnostic on frozen verdict planes.

Status: POST_FREEZE_DIAGNOSTIC_NOT_CONFIRMATORY
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from analysis.diagnostics.shear_gap_regression import build_report as build_regression_report
from analysis.diagnostics.shear_gap_regression import write_markdown as write_regression_markdown
from analysis.validation.common import (
    ARM_DISPLAY,
    STATUS_DIAGNOSTIC,
    load_model,
    load_official_manifest,
    markdown_table,
    plane_by_arm,
    run_loop_measurement,
    selected_measurements,
    write_json,
)


DEFAULT_CHECKPOINT = Path("reports/full_transport_shear_diagnostic.checkpoint.jsonl")
DEFAULT_JSON = Path("reports/full_transport_shear_diagnostic.json")
DEFAULT_MD = Path("reports/full_transport_shear_diagnostic.md")
DEFAULT_ENV = Path("reports/full_shear_gap_environment_record.md")
DEFAULT_REGRESSION_JSON = Path("reports/shear_gap_regression.json")
DEFAULT_REGRESSION_MD = Path("reports/shear_gap_regression.md")

FROZEN_RESULTS = Path("run/holonomy_results_n390.json")
ENV_STATUS = "POST_FREEZE_ENVIRONMENT_RECORD_NOT_CONFIRMATORY"

SENSITIVE_PATTERNS = [
    re.compile(
        r"^(\s*(?:Serial Number(?: \(system\))?|Hardware UUID|Provisioning UDID|"
        r"Activation Lock Status|Activation Lock|Model Number|Computer Name|User Name):\s*).*$",
        re.I,
    ),
]


def run_command(command: list[str]) -> str:
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except FileNotFoundError as exc:
        return f"{command[0]}: not found ({exc})\n"
    output = completed.stdout
    if completed.stderr:
        output += completed.stderr
    if completed.returncode != 0:
        output += f"\n[exit status {completed.returncode}]\n"
    return output


def redact_hardware_ids(text: str) -> str:
    redacted_lines = []
    for line in text.splitlines():
        redacted = line
        for pattern in SENSITIVE_PATTERNS:
            match = pattern.match(redacted)
            if match:
                redacted = match.group(1) + "[REDACTED]"
                break
        redacted_lines.append(redacted)
    return "\n".join(redacted_lines) + ("\n" if text.endswith("\n") else "")


def write_environment_record(path: Path) -> None:
    sections = [
        ("date", run_command(["date"])),
        ("python --version", run_command(["python", "--version"])),
        (
            "uv run python environment",
            run_command(
                [
                    "uv",
                    "run",
                    "python",
                    "-c",
                    (
                        "import torch, transformers, platform\n"
                        "print('torch', torch.__version__)\n"
                        "print('transformers', transformers.__version__)\n"
                        "print('mps', torch.backends.mps.is_available())\n"
                        "print('platform', platform.platform())\n"
                    ),
                ]
            ),
        ),
        (
            "system_profiler SPHardwareDataType SPSoftwareDataType",
            redact_hardware_ids(run_command(["system_profiler", "SPHardwareDataType", "SPSoftwareDataType"])),
        ),
        ("git status --short --branch", run_command(["git", "status", "--short", "--branch"])),
        ("git rev-parse HEAD", run_command(["git", "rev-parse", "HEAD"])),
    ]

    lines = [
        "# Full Shear Gap Environment Record",
        "",
        f"Status: {ENV_STATUS}",
        "",
    ]
    for title, body in sections:
        lines.extend([f"## {title}", "", "```text", body.rstrip(), "```", ""])
    path.write_text("\n".join(lines))


def load_frozen_results(path: Path = FROZEN_RESULTS) -> dict[tuple[str, str, str], dict[str, float]]:
    rows = json.loads(path.read_text())
    return {
        (row["base_point_id"], row["arm"], row["magnitude_level"]): row
        for row in rows
        if row["arm"] in {"real", "shuffled"}
    }


def completed_keys(path: Path) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    if not path.exists():
        return keys
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("error"):
            continue
        keys.add((row["base_point_id"], row["arm"], row["magnitude_level"]))
    return keys


def append_checkpoint_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def is_close(actual: float, expected: float, *, abs_tol: float = 1e-8, rel_tol: float = 1e-5) -> bool:
    return math.isclose(float(actual), float(expected), abs_tol=abs_tol, rel_tol=rel_tol)


def compare_to_frozen(row: dict[str, Any], frozen: dict[str, float]) -> dict[str, Any]:
    fields = [
        ("H", "H"),
        ("log_H", "log_H"),
        ("theta", "theta"),
        ("A_enclosed", "A_enclosed"),
        ("det_M", "det_M"),
    ]
    comparisons = {}
    mismatches = []
    for actual_key, frozen_key in fields:
        actual = float(row[actual_key])
        expected = float(frozen[frozen_key])
        diff = actual - expected
        ok = is_close(actual, expected)
        comparisons[actual_key] = {
            "actual": actual,
            "frozen": expected,
            "difference": diff,
            "ok": ok,
        }
        if not ok:
            mismatches.append(actual_key)
    return {"ok": not mismatches, "mismatches": mismatches, "fields": comparisons}


def checkpoint_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm_level: dict[str, dict[str, Any]] = {}
    for arm in ["real", "shuffled"]:
        for level in ["low", "high"]:
            group = [row for row in rows if row.get("arm") == arm and row.get("magnitude_level") == level and not row.get("error")]
            if not group:
                continue
            by_arm_level[f"{arm}/{level}"] = {
                "arm": arm,
                "plane_type": ARM_DISPLAY[arm],
                "magnitude_level": level,
                "n": len(group),
                "symmetric_residual_norm_mean": sum(row["symmetric_residual_norm"] for row in group) / len(group),
                "non_orthogonality_norm_mean": sum(row["non_orthogonality_norm"] for row in group) / len(group),
                "transport_residual_norm_mean": sum(row["transport_residual_norm"] for row in group) / len(group),
                "max_condition_mean": sum(row["max_condition"] for row in group) / len(group),
            }
    validation_failures = [
        row
        for row in rows
        if row.get("frozen_validation") and not row["frozen_validation"].get("ok", False)
    ]
    return {
        "n_rows": len(rows),
        "n_error_rows": sum(1 for row in rows if row.get("error")),
        "n_validation_failures": len(validation_failures),
        "total_runtime_seconds": sum(float(row.get("runtime_seconds", 0.0)) for row in rows),
        "by_arm_level": by_arm_level,
    }


def write_full_reports(
    *,
    checkpoint_path: Path,
    json_path: Path,
    md_path: Path,
    regression_json_path: Path,
    regression_md_path: Path,
) -> None:
    rows = checkpoint_rows(checkpoint_path)
    summary = summarize_rows(rows)
    payload = {
        "status": STATUS_DIAGNOSTIC,
        "source_checkpoint": str(checkpoint_path),
        "definitions": {
            "symmetric_residual_norm": "Frobenius norm of 0.5 * ((T - I) + (T - I)^T).",
            "non_orthogonality_norm": "Frobenius norm of T^T T - I.",
            "transport_residual_norm": "Frobenius norm of T - I.",
        },
        "rows": rows,
        "summary": summary,
    }
    write_json(json_path, payload)

    report_rows = list(summary["by_arm_level"].values())
    md = f"""# Full Transport Shear Diagnostic

Status: {STATUS_DIAGNOSTIC}

This post-freeze diagnostic recomputes transport distortion quantities on the frozen verdict-bearing real/shuffled planes. It does not enter or modify the preregistered decision rule.

## Scope

- Arms: `real` (active-feature) and `shuffled` (mixed-feature)
- Magnitude levels: `low`, `high`
- Checkpoint: `{checkpoint_path}`
- Rows: `{summary['n_rows']}`
- Error rows: `{summary['n_error_rows']}`
- Frozen-row validation failures: `{summary['n_validation_failures']}`
- Total runtime seconds: `{summary['total_runtime_seconds']:.2f}`

## Arm/level summary

{markdown_table(report_rows, ['plane_type', 'magnitude_level', 'n', 'symmetric_residual_norm_mean', 'non_orthogonality_norm_mean', 'transport_residual_norm_mean', 'max_condition_mean'])}

The confirmatory verdict remains the preregistered frozen result. These analyses are post-freeze diagnostics and validation checks added to characterize the instrument and residual threats to validity; they do not enter or modify the preregistered decision rule.
"""
    md_path.write_text(md)

    if summary["n_rows"] == 1560 and summary["n_error_rows"] == 0 and summary["n_validation_failures"] == 0:
        regression = build_regression_report(checkpoint_path)
        write_json(regression_json_path, regression)
        write_regression_markdown(regression, regression_md_path)


def iter_measurements(manifest: Any, *, max_base_points: int | None = None) -> Iterable[tuple[Any, Any, str]]:
    base_count = len(manifest.base_points) if max_base_points is None else max_base_points
    yield from selected_measurements(
        manifest,
        base_count=base_count,
        arms=["real", "shuffled"],
        levels=["low", "high"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--json-out", default=str(DEFAULT_JSON))
    parser.add_argument("--md-out", default=str(DEFAULT_MD))
    parser.add_argument("--env-out", default=str(DEFAULT_ENV))
    parser.add_argument("--regression-json-out", default=str(DEFAULT_REGRESSION_JSON))
    parser.add_argument("--regression-md-out", default=str(DEFAULT_REGRESSION_MD))
    parser.add_argument("--max-base-points", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if args.force and checkpoint_path.exists():
        checkpoint_path.unlink()

    write_environment_record(Path(args.env_out))
    manifest = load_official_manifest()
    frozen = load_frozen_results()
    existing = completed_keys(checkpoint_path)

    print("loading Gemma model for full transport shear diagnostic", flush=True)
    model, _tokenizer, device = load_model()
    print(f"Gemma loaded on {device}", flush=True)

    measurements = list(iter_measurements(manifest, max_base_points=args.max_base_points))
    if args.benchmark_only:
        measurements = measurements[:1]

    total = len(measurements)
    started = time.perf_counter()
    completed_now = 0
    for idx, (base, plane, level) in enumerate(measurements, start=1):
        key = (base.base_point_id, plane.arm, level)
        if key in existing:
            print(f"[{idx}/{total}] {key} already checkpointed; skipping", flush=True)
            continue
        print(f"[{idx}/{total}] measuring base={base.base_point_id} arm={plane.arm} level={level}", flush=True)
        try:
            row = run_loop_measurement(
                model=model,
                manifest=manifest,
                base=base,
                plane=plane,
                magnitude_level=level,
                radius_scale=1.0,
            )
            row["status"] = STATUS_DIAGNOSTIC
            validation = compare_to_frozen(row, frozen[key])
            row["frozen_validation"] = validation
            if not validation["ok"]:
                row["error"] = f"frozen validation mismatch: {validation['mismatches']}"
                append_checkpoint_row(checkpoint_path, row)
                raise RuntimeError(row["error"])
            append_checkpoint_row(checkpoint_path, row)
            existing.add(key)
            completed_now += 1
            elapsed = time.perf_counter() - started
            remaining = total - idx
            mean = elapsed / max(completed_now, 1)
            eta_hours = remaining * mean / 3600
            print(
                f"  finished in {row['runtime_seconds']:.2f}s; "
                f"checkpointed={len(existing)}; ETA {eta_hours:.2f}h",
                flush=True,
            )
        except Exception as exc:
            if "row" not in locals() or not isinstance(row, dict) or row.get("base_point_id") != base.base_point_id:
                append_checkpoint_row(
                    checkpoint_path,
                    {
                        "status": STATUS_DIAGNOSTIC,
                        "base_point_id": base.base_point_id,
                        "arm": plane.arm,
                        "plane_type": ARM_DISPLAY.get(plane.arm, plane.arm),
                        "magnitude_level": level,
                        "error": str(exc),
                    },
                )
            raise

    write_full_reports(
        checkpoint_path=checkpoint_path,
        json_path=Path(args.json_out),
        md_path=Path(args.md_out),
        regression_json_path=Path(args.regression_json_out),
        regression_md_path=Path(args.regression_md_out),
    )


if __name__ == "__main__":
    main()
