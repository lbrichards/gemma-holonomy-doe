"""Post-freeze center-displacement diagnostic.

This diagnostic reads the frozen manifest and reports how far matched centers
lie from their raw base activations in units of the loop radius. It does not
enter or modify the preregistered verdict.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np


STATUS = "POST_FREEZE_DIAGNOSTIC_NOT_CONFIRMATORY"


def _norm(values: list[float]) -> float:
    return float(np.linalg.norm(np.asarray(values, dtype=np.float64)))


def _summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(mean(values)),
        "median": float(median(values)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
    }


def build_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    radius_relative = float(manifest["metadata"]["radius_relative"])
    rows: list[dict[str, Any]] = []
    for base in manifest["base_points"]:
        activation = [float(item) for item in base["activation"]]
        activation_norm = _norm(activation)
        rho = radius_relative * activation_norm
        for plane in base["planes"]:
            for level, key in [("low", "center_low"), ("high", "center_high")]:
                center = [float(item) for item in plane[key]]
                diff = np.asarray(center, dtype=np.float64) - np.asarray(activation, dtype=np.float64)
                displacement = float(np.linalg.norm(diff))
                rows.append(
                    {
                        "base_point_id": base["base_point_id"],
                        "arm": plane["arm"],
                        "plane_type": {
                            "real": "active-feature",
                            "shuffled": "mixed-feature",
                            "random": "random",
                        }[plane["arm"]],
                        "magnitude_level": level,
                        "rho": rho,
                        "displacement": displacement,
                        "displacement_over_rho": displacement / rho if rho > 0 else math.nan,
                        "center_norm_over_activation_norm": _norm(center) / activation_norm,
                    }
                )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm_level: dict[str, dict[str, Any]] = {}
    by_arm: dict[str, dict[str, Any]] = {}
    for arm in ["real", "shuffled", "random"]:
        arm_rows = [row for row in rows if row["arm"] == arm]
        by_arm[arm] = _summary([row["displacement_over_rho"] for row in arm_rows])
        for level in ["low", "high"]:
            level_rows = [row for row in arm_rows if row["magnitude_level"] == level]
            by_arm_level[f"{arm}_{level}"] = _summary([row["displacement_over_rho"] for row in level_rows])
    return {
        "status": STATUS,
        "definition": "displacement_over_rho = ||c - h|| / (radius_relative * ||h||)",
        "by_arm": by_arm,
        "by_arm_level": by_arm_level,
    }


def write_markdown(path: Path, *, summary: dict[str, Any]) -> None:
    lines = [
        "# Center-displacement diagnostic",
        "",
        f"Status: {STATUS}",
        "",
        "This post-freeze diagnostic reads the frozen manifest and reports matched-center displacement",
        "from the raw activation in units of the loop radius. It does not enter or modify the",
        "preregistered decision rule.",
        "",
        "Definition: `||c - h|| / (radius_relative * ||h||)`.",
        "",
        "## Combined by plane type",
        "",
        "| plane_type | n | mean | median | p90 | p95 | max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    names = {"real": "active-feature", "shuffled": "mixed-feature", "random": "random"}
    for arm in ["real", "shuffled", "random"]:
        stats = summary["by_arm"][arm]
        lines.append(
            f"| {names[arm]} | {stats['n']} | {stats['mean']:.3g} | {stats['median']:.3g} | "
            f"{stats['p90']:.3g} | {stats['p95']:.3g} | {stats['max']:.3g} |"
        )
    lines.extend(
        [
            "",
            "## By plane type and magnitude level",
            "",
            "| plane_type | magnitude_level | n | mean | median | p90 | p95 | max |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for arm in ["real", "shuffled", "random"]:
        for level in ["low", "high"]:
            stats = summary["by_arm_level"][f"{arm}_{level}"]
            lines.append(
                f"| {names[arm]} | {level} | {stats['n']} | {stats['mean']:.3g} | "
                f"{stats['median']:.3g} | {stats['p90']:.3g} | {stats['p95']:.3g} | {stats['max']:.3g} |"
            )
    lines.extend(
        [
            "",
            "The confirmatory verdict remains the preregistered frozen result. This diagnostic",
            "characterizes a residual threat to validity and is not confirmatory.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Report matched-center displacement from the frozen manifest.")
    parser.add_argument("--manifest", type=Path, default=Path("run/manifest_n390.json"))
    parser.add_argument("--json-output", type=Path, default=Path("reports/center_displacement_diagnostic.json"))
    parser.add_argument("--md-output", type=Path, default=Path("reports/center_displacement_diagnostic.md"))
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    rows = build_rows(manifest)
    summary = summarize(rows)
    payload = {
        **summary,
        "manifest": str(args.manifest),
        "row_count": len(rows),
        "rows": rows,
    }
    args.json_output.write_text(json.dumps(payload, indent=2) + "\n")
    write_markdown(args.md_output, summary=summary)


if __name__ == "__main__":
    main()
