"""Post-freeze check of the missing-<bos> claim for SAE feature 10762.

For the first N manifest base points, compares feature 10762's activation at the
final content token and its weight in the constructed planes, without <bos>
(read from the frozen manifest) and with <bos> (Stage A pass-1 selection rerun
with ``prepend_bos``). It does not modify any frozen artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any

import torch

from extraction import ResidPostExtractor, load_model_and_tokenizer, load_sae
from manifest import manifest_from_json
from planes import Arm
from stage_a.run import arm_for_manifest, pass1_select_and_magnitude


STATUS = "POST_FREEZE_NOT_CONFIRMATORY"
FEATURE = 10762


def code_stats(code: torch.Tensor, feature: int = FEATURE) -> dict[str, Any]:
    code = code.detach().cpu().to(torch.float64)
    active = code > 0
    value = float(code[feature])
    rank = int((code > code[feature]).sum()) + 1 if value > 0 else None
    return {
        "activation": value,
        "active": value > 0,
        "rank_among_active": rank,
        "l0": int(active.sum()),
        "l1_share": value / float(code[active].sum()) if bool(active.any()) else 0.0,
        "top_feature": int(torch.argmax(code)),
    }


def plane_weight(directions: torch.Tensor, atom: torch.Tensor) -> float:
    """Squared norm of the unit feature atom projected onto span(directions)."""

    basis, _ = torch.linalg.qr(directions.detach().cpu().to(torch.float64).T)
    atom = atom.detach().cpu().to(torch.float64)
    atom = atom / torch.linalg.vector_norm(atom)
    return float(torch.linalg.vector_norm(basis.T @ atom) ** 2)


def plane_stats(arm: str, feature_ids: tuple[int, int] | None, directions: torch.Tensor, atom: torch.Tensor) -> dict[str, Any]:
    return {
        "arm": arm,
        "feature_ids": None if feature_ids is None else [int(item) for item in feature_ids],
        "contains_feature": feature_ids is not None and FEATURE in feature_ids,
        "projection_sq": plane_weight(directions, atom),
    }


def summarize(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    cond = [row[condition] for row in rows]
    out: dict[str, Any] = {
        "n": len(cond),
        "n_active": sum(row["code"]["active"] for row in cond),
        "n_top_feature": sum(row["code"]["top_feature"] == FEATURE for row in cond),
        "median_activation": median(row["code"]["activation"] for row in cond),
        "median_l1_share": median(row["code"]["l1_share"] for row in cond),
    }
    for arm in ["real", "shuffled", "random"]:
        planes = [next(plane for plane in row["planes"] if plane["arm"] == arm) for row in cond]
        out[f"{arm}_contains_feature"] = sum(plane["contains_feature"] for plane in planes)
        out[f"{arm}_median_projection_sq"] = median(plane["projection_sq"] for plane in planes)
    return out


def run(manifest_path: Path, limit: int) -> dict[str, Any]:
    manifest = manifest_from_json(manifest_path.read_text())
    model, tokenizer = load_model_and_tokenizer()
    device = next(model.parameters()).device
    sae = load_sae(device=device)
    atom = sae.w_dec[FEATURE]
    bos_token_id = int(tokenizer.bos_token_id)
    extractor = ResidPostExtractor(model, bos_token_id=bos_token_id)

    rows: list[dict[str, Any]] = []
    for idx, base in enumerate(manifest.base_points[:limit]):
        frozen_activation = torch.tensor(base.activation, dtype=torch.float32, device=device)
        no_bos = {
            "activation_norm": float(torch.linalg.vector_norm(frozen_activation)),
            "code": code_stats(sae.encode(frozen_activation)),
            "planes": [
                plane_stats(
                    plane.arm,
                    plane.feature_indices,
                    torch.tensor([plane.direction_1, plane.direction_2], dtype=torch.float64),
                    atom,
                )
                for plane in base.planes
            ],
        }
        record = {
            "draw_order": base.corpus_draw_order,
            "article_index": base.article_index,
            "token_ids": base.token_ids,
        }
        pass1 = pass1_select_and_magnitude(model=model, sae=sae, extractor=extractor, record=record, local_index=idx)
        with_bos = {
            "activation_norm": float(torch.linalg.vector_norm(pass1.activation)),
            "code": code_stats(sae.encode(pass1.activation)),
            "planes": [
                plane_stats(
                    arm_for_manifest(arm),
                    pass1.planes[arm].selection.feature_ids,
                    pass1.planes[arm].selection.directions,
                    atom,
                )
                for arm in [Arm.REAL_FEATURE, Arm.SHUFFLED_FEATURE, Arm.RANDOM]
            ],
        }
        row = {"base_point_id": base.base_point_id, "no_bos": no_bos, "with_bos": with_bos}
        rows.append(row)
        print(
            f"{base.base_point_id} no_bos f{FEATURE}={no_bos['code']['activation']:.3g} "
            f"rank={no_bos['code']['rank_among_active']} real={no_bos['planes'][0]['feature_ids']} | "
            f"with_bos f{FEATURE}={with_bos['code']['activation']:.3g} "
            f"rank={with_bos['code']['rank_among_active']} real={with_bos['planes'][0]['feature_ids']}",
            flush=True,
        )

    return {
        "status": STATUS,
        "feature": FEATURE,
        "manifest": str(manifest_path),
        "n_base_points": len(rows),
        "bos_token_id": bos_token_id,
        "no_bos_source": "frozen manifest activations and planes",
        "with_bos_source": "Stage A pass-1 rerun with prepend_bos (same seeds, same final content token)",
        "summary": {"no_bos": summarize(rows, "no_bos"), "with_bos": summarize(rows, "with_bos")},
        "rows": rows,
    }


def write_reports(result: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "feature_10762_check.json").write_text(json.dumps(result, indent=2) + "\n")
    s = result["summary"]
    lines = [
        f"# Feature {FEATURE} at the final token, without vs with <bos>",
        "",
        f"**Status: {STATUS}.** Post-freeze robustness check prompted by external review; "
        "it does not enter or modify the preregistered verdict.",
        "",
        f"First {result['n_base_points']} manifest base points. Without <bos>: {result['no_bos_source']}. "
        f"With <bos>: {result['with_bos_source']}.",
        "",
        "| metric | no <bos> | with <bos> |",
        "|---|---|---|",
    ]
    for key in s["no_bos"]:
        if key == "n":
            continue
        lines.append(f"| {key} | {s['no_bos'][key]:.4g} | {s['with_bos'][key]:.4g} |")
    lines += [
        "",
        "`projection_sq` is the squared norm of the unit feature atom projected onto the plane span "
        "(1.0 when the feature is one of the two plane atoms).",
        "",
        "## Per base point",
        "",
        "| base | f act (no) | rank (no) | L1 share (no) | real plane (no) | f act (bos) | rank (bos) | L1 share (bos) | real plane (bos) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in result["rows"]:
        a, b = row["no_bos"], row["with_bos"]
        lines.append(
            f"| {row['base_point_id']} | {a['code']['activation']:.3g} | {a['code']['rank_among_active']} "
            f"| {a['code']['l1_share']:.3f} | {a['planes'][0]['feature_ids']} "
            f"| {b['code']['activation']:.3g} | {b['code']['rank_among_active']} "
            f"| {b['code']['l1_share']:.3f} | {b['planes'][0]['feature_ids']} |"
        )
    (out_dir / "feature_10762_check.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("run/manifest_n390.json"))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, default=Path("reports_bos"))
    args = parser.parse_args()
    result = run(args.manifest, args.limit)
    write_reports(result, args.out_dir)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
