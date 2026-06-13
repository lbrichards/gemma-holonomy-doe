#!/usr/bin/env python3
"""Power calculation for H-sem on the log-scale semantic effect.

This script uses frozen across-base tau estimates from the prior holonomy-probe
runs and computes the required number of base points for a one-sample semantic
effect test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

from scipy import stats


# Frozen across-base SD of the log-scale feature-vs-control contrast from Iteration 8.
TAU_ITER8_LOG_SCALE = 0.586

# Frozen across-base SD of the log-scale feature-vs-control contrast from Iteration 7.
TAU_ITER7_LOG_SCALE = 0.649

# Pre-registered materiality threshold: log(1.25), a 25% excess on the ratio scale.
DELTA_MATERIAL = math.log(1.25)

# Secondary diagnostic only: detecting a sub-material 10.3% ratio-scale excess.
DELTA_OBSERVED_SUBMATERIAL = math.log(1.103)

# Type-I error rate.
ALPHA = 0.05

# Desired power.
POWER = 0.90

# Output path for the Markdown report.
RESULTS_PATH = Path(__file__).with_name("power_hsem_results.md")


@dataclass(frozen=True)
class TauInput:
    label: str
    tau: float


@dataclass(frozen=True)
class PowerRow:
    tau_source: str
    tau: float
    target_label: str
    delta: float
    sidedness: str
    normal_n: int
    t_corrected_n: int


TAU_INPUTS = [
    TauInput("Iter 8 tau", TAU_ITER8_LOG_SCALE),
    TauInput("Iter 7 tau", TAU_ITER7_LOG_SCALE),
]

SIDEDNESS = ["one-sided", "two-sided"]


def normal_required_n(*, tau: float, delta: float, sidedness: str) -> int:
    """Normal approximation: N = (z_alpha + z_beta)^2 * tau^2 / delta^2."""
    normal = NormalDist()
    if sidedness == "one-sided":
        z_alpha = normal.inv_cdf(1.0 - ALPHA)
    elif sidedness == "two-sided":
        z_alpha = normal.inv_cdf(1.0 - ALPHA / 2.0)
    else:
        raise ValueError(f"unknown sidedness: {sidedness}")
    z_beta = normal.inv_cdf(POWER)
    return math.ceil(((z_alpha + z_beta) ** 2) * (tau**2) / (delta**2))


def t_corrected_required_n(*, tau: float, delta: float, sidedness: str) -> int:
    """Iterate N using t quantiles with df=N-1 in place of normal quantiles."""
    n = max(2, normal_required_n(tau=tau, delta=delta, sidedness=sidedness))
    while True:
        df = n - 1
        if sidedness == "one-sided":
            t_alpha = stats.t.ppf(1.0 - ALPHA, df)
        elif sidedness == "two-sided":
            t_alpha = stats.t.ppf(1.0 - ALPHA / 2.0, df)
        else:
            raise ValueError(f"unknown sidedness: {sidedness}")
        t_beta = stats.t.ppf(POWER, df)
        corrected = math.ceil(((t_alpha + t_beta) ** 2) * (tau**2) / (delta**2))
        corrected = max(2, corrected)
        if corrected == n:
            return corrected
        n = corrected


def build_rows(*, target_label: str, delta: float) -> list[PowerRow]:
    rows: list[PowerRow] = []
    for tau_input in TAU_INPUTS:
        for sidedness in SIDEDNESS:
            rows.append(
                PowerRow(
                    tau_source=tau_input.label,
                    tau=tau_input.tau,
                    target_label=target_label,
                    delta=delta,
                    sidedness=sidedness,
                    normal_n=normal_required_n(tau=tau_input.tau, delta=delta, sidedness=sidedness),
                    t_corrected_n=t_corrected_required_n(tau=tau_input.tau, delta=delta, sidedness=sidedness),
                )
            )
    return rows


def markdown_table(rows: list[PowerRow]) -> str:
    lines = [
        "| Target | Tau source | Tau | Sidedness | Delta | Normal N | t-corrected N |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.target_label} | {row.tau_source} | {row.tau:.3f} | "
            f"{row.sidedness} | {row.delta:.6f} | {row.normal_n} | {row.t_corrected_n} |"
        )
    return "\n".join(lines)


def main() -> None:
    material_rows = build_rows(target_label="Material target: log(1.25)", delta=DELTA_MATERIAL)
    submaterial_rows = build_rows(
        target_label="Not our target: observed sub-material log(1.103)",
        delta=DELTA_OBSERVED_SUBMATERIAL,
    )
    all_rows = material_rows + submaterial_rows

    report = "\n".join(
        [
            "# H-sem Power Calculation",
            "",
            "Scale: log-scale across-base SD of the feature-vs-control semantic contrast.",
            "",
            f"- `ALPHA = {ALPHA}`",
            f"- `POWER = {POWER}`",
            f"- `DELTA_MATERIAL = log(1.25) = {DELTA_MATERIAL:.12f}`",
            f"- `DELTA_OBSERVED_SUBMATERIAL = log(1.103) = {DELTA_OBSERVED_SUBMATERIAL:.12f}`",
            "",
            markdown_table(all_rows),
            "",
            "The observed sub-material rows are included only to document the cost of chasing a",
            "smaller-than-material effect; they are not the pre-registered target.",
            "",
        ]
    )
    print(report)
    RESULTS_PATH.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
