"""Tau re-pilot at the corrected Gemma Scope resid_post site.

This pilot is deliberately restricted to the stage-2 reserve partition of
``run/corpus_draw_stage_pool.json``. It computes holonomy only on reserve
records 96..111 and writes results under ``pilot/``. It does not read or modify
the pre-registration.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from scipy.stats import chi2
from transformers import AutoModelForCausalLM, AutoTokenizer

from geometry import enclosed_area, pullback_gram


MODEL_NAME = "google/gemma-2-2b"
SAE_REPO = "google/gemma-scope-2b-pt-res"
SAE_PATH = "layer_12/width_16k/average_l0_82/params.npz"
LAYER = 12
READOUT_LAYER = 13
TOKEN_POSITION = 63
RESERVE_START = 96
RESERVE_COUNT = 16
TAU_DETM = 0.413
RADIUS_RELATIVE = 6.0e-3
N_STEPS = 200
SEED = 20260614
RCOND = 1e-10


@dataclass
class PlaneMeasurement:
    arm: str
    feature_ids: tuple[int, int]
    theta: float
    holonomy: float
    log_holonomy: float
    area: float
    det_m: float
    condition_max: float
    stretch_fro: float
    candidates_tested: int
    candidates_rejected_detm: int


@dataclass
class BaseResult:
    reserve_record_index: int
    draw_order: int
    article_index: int
    article_title: str
    base_norm: float
    l0: int
    real: PlaneMeasurement
    shuffled: PlaneMeasurement
    d_b: float


def load_sae(device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    local = hf_hub_download(SAE_REPO, SAE_PATH)
    params = np.load(local)
    return {
        "w_dec": torch.from_numpy(params["W_dec"]).to(device=device, dtype=dtype),
        "w_enc": torch.from_numpy(params["W_enc"]).to(device=device, dtype=dtype),
        "b_dec": torch.from_numpy(params["b_dec"]).to(device=device, dtype=dtype),
        "b_enc": torch.from_numpy(params["b_enc"]).to(device=device, dtype=dtype),
        "threshold": torch.from_numpy(params["threshold"]).to(device=device, dtype=dtype),
    }


def encode_sae(sae: dict[str, torch.Tensor], activation: torch.Tensor) -> torch.Tensor:
    # SAELens JumpReLU config for Gemma Scope has apply_b_dec_to_input=False:
    # encode uses x @ W_enc + b_enc; b_dec is used only for decoding.
    x = activation.to(dtype=sae["w_enc"].dtype, device=sae["w_enc"].device)
    pre = x @ sae["w_enc"] + sae["b_enc"]
    return torch.where(pre > sae["threshold"], pre, torch.zeros_like(pre))


def normalize_rows(rows: torch.Tensor) -> torch.Tensor:
    return rows / torch.linalg.vector_norm(rows, dim=1, keepdim=True).clamp_min(1e-12)


def extract_layer_output(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    layer: int = LAYER,
) -> torch.Tensor:
    captured: dict[str, torch.Tensor] = {}

    def hook(_module: torch.nn.Module, _args: tuple[Any, ...], output: torch.Tensor) -> None:
        tensor = output[0] if isinstance(output, tuple) else output
        captured["h"] = tensor.detach()

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        handle.remove()
    base = captured["h"][0, TOKEN_POSITION].detach().clone()
    if base.shape[-1] != 2304:
        raise RuntimeError(f"Expected resid_post dimension 2304, got {tuple(base.shape)}")
    return base


def make_resid_post_map(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    patch_layer: int = LAYER,
    readout_layer: int = READOUT_LAYER,
):
    """Return F(h): patched block-12 resid_post -> block-13 resid_post."""

    def f_map(h: torch.Tensor) -> torch.Tensor:
        state: dict[str, torch.Tensor] = {}

        def patch_hook(_module: torch.nn.Module, _args: tuple[Any, ...], output: torch.Tensor):
            tensor = output[0] if isinstance(output, tuple) else output
            patched = tensor.clone()
            patched[0, TOKEN_POSITION, :] = h.to(device=patched.device, dtype=patched.dtype)
            if isinstance(output, tuple):
                return (patched, *output[1:])
            return patched

        def capture_hook(_module: torch.nn.Module, _args: tuple[Any, ...], output: torch.Tensor) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            state["out"] = tensor

        patch_handle = model.model.layers[patch_layer].register_forward_hook(patch_hook)
        capture_handle = model.model.layers[readout_layer].register_forward_hook(capture_hook)
        try:
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        finally:
            patch_handle.remove()
            capture_handle.remove()
        return state["out"][0, TOKEN_POSITION, :]

    return f_map


def jvp_fn(fn, point: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
    point = point.detach().requires_grad_(True)
    tangent = tangent.to(device=point.device, dtype=point.dtype)
    _, out_tangent = torch.autograd.functional.jvp(fn, point, tangent, create_graph=False, strict=False)
    return out_tangent.detach()


def restricted_jacobian(fn, point: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    cols = [jvp_fn(fn, point, direction).reshape(-1) for direction in directions]
    return torch.stack(cols, dim=1).detach().cpu().to(torch.float64)


def theta_from_operator(h_operator: torch.Tensor) -> tuple[float, float]:
    delta = h_operator - torch.eye(2, dtype=h_operator.dtype, device=h_operator.device)
    theta = 0.5 * (delta[1, 0] - delta[0, 1])
    symmetric = 0.5 * (delta + delta.T)
    return float(theta.detach().cpu()), float(torch.linalg.matrix_norm(symmetric).detach().cpu())


def loop_points(base: torch.Tensor, directions: torch.Tensor, radius: float, n_steps: int) -> torch.Tensor:
    ts = torch.linspace(0, 2 * math.pi, n_steps + 1, device=base.device, dtype=base.dtype)
    offsets = radius * (torch.cos(ts).unsqueeze(1) * directions[0] + torch.sin(ts).unsqueeze(1) * directions[1])
    return base.unsqueeze(0) + offsets


def loop_holonomy(fn, base: torch.Tensor, directions: torch.Tensor, radius: float, area: float, n_steps: int) -> tuple[float, float, float, float]:
    points = loop_points(base, directions, radius, n_steps)
    h_operator = torch.eye(2, dtype=torch.float64)
    condition_max = 0.0
    prev_j = restricted_jacobian(fn, points[0], directions)
    for idx in range(points.shape[0] - 1):
        next_j = restricted_jacobian(fn, points[idx + 1], directions)
        condition_max = max(condition_max, float(torch.linalg.cond(next_j).detach().cpu()))
        step = torch.linalg.pinv(next_j, rcond=RCOND) @ prev_j
        h_operator = step @ h_operator
        prev_j = next_j
    theta, stretch_fro = theta_from_operator(h_operator)
    holonomy = abs(theta) / max(area, 1e-30)
    return theta, holonomy, condition_max, stretch_fro


def plane_det_m(fn, base: torch.Tensor, directions: torch.Tensor) -> tuple[float, float]:
    jd = restricted_jacobian(fn, base, directions)
    gram = pullback_gram(jd)
    det_value = float(torch.linalg.det(gram).detach().cpu())
    return det_value, float(enclosed_area(1.0, gram).detach().cpu())


def active_ids(acts: torch.Tensor) -> list[int]:
    nonzero = torch.nonzero(acts > 0, as_tuple=False).flatten()
    if len(nonzero) == 0:
        return torch.argsort(acts, descending=True)[:128].detach().cpu().tolist()
    values = acts[nonzero]
    order = torch.argsort(values, descending=True)
    return nonzero[order].detach().cpu().tolist()


def inactive_ids(acts: torch.Tensor) -> list[int]:
    zero = torch.nonzero(acts <= 0, as_tuple=False).flatten()
    if len(zero) == 0:
        return torch.argsort(acts)[:1024].detach().cpu().tolist()
    return zero.detach().cpu().tolist()


def candidate_pairs(ids_a: list[int], ids_b: list[int] | None = None):
    if ids_b is None:
        for i, left in enumerate(ids_a):
            for right in ids_a[i + 1 :]:
                yield int(left), int(right)
    else:
        for left in ids_a:
            for right in ids_b:
                if int(left) != int(right):
                    yield int(left), int(right)


def select_plane(
    *,
    arm: str,
    fn,
    base: torch.Tensor,
    sae: dict[str, torch.Tensor],
    acts: torch.Tensor,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, tuple[int, int], float, int, int]:
    actives = active_ids(acts)[:96]
    if arm == "real":
        pairs = list(candidate_pairs(actives))
        pairs.sort(key=lambda p: float(torch.sqrt(torch.clamp(acts[p[0]], min=0) * torch.clamp(acts[p[1]], min=0)).detach().cpu()), reverse=True)
    elif arm == "shuffled":
        inactives = inactive_ids(acts)
        sampled_inactive = rng.choice(np.array(inactives, dtype=np.int64), size=min(512, len(inactives)), replace=False).tolist()
        pairs = list(candidate_pairs(actives[:64], sampled_inactive))
    else:
        raise ValueError(f"unsupported arm: {arm}")

    tested = 0
    rejected = 0
    for left, right in pairs:
        dirs = normalize_rows(torch.stack([sae["w_dec"][left], sae["w_dec"][right]], dim=0)).to(device=base.device, dtype=base.dtype)
        det_value, _ = plane_det_m(fn, base, dirs)
        tested += 1
        if det_value <= TAU_DETM:
            rejected += 1
            continue
        return dirs, (int(left), int(right)), det_value, tested, rejected
    raise RuntimeError(f"No accepted {arm} plane after testing {tested} candidates")


def measure_plane(
    *,
    arm: str,
    fn,
    base: torch.Tensor,
    directions: torch.Tensor,
    feature_ids: tuple[int, int],
    det_value: float,
    candidates_tested: int,
    candidates_rejected_detm: int,
    radius: float,
    n_steps: int,
) -> PlaneMeasurement:
    gram_area_factor = math.sqrt(max(det_value, 0.0))
    area = radius * radius * gram_area_factor
    theta, holonomy, condition_max, stretch_fro = loop_holonomy(fn, base, directions, radius, area, n_steps)
    return PlaneMeasurement(
        arm=arm,
        feature_ids=feature_ids,
        theta=theta,
        holonomy=holonomy,
        log_holonomy=math.log(max(holonomy, 1e-30)),
        area=area,
        det_m=det_value,
        condition_max=condition_max,
        stretch_fro=stretch_fro,
        candidates_tested=candidates_tested,
        candidates_rejected_detm=candidates_rejected_detm,
    )


def sd_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    n = len(values)
    sd = float(np.std(values, ddof=1))
    lo = math.sqrt((n - 1) * sd * sd / chi2.ppf(1 - alpha / 2, n - 1))
    hi = math.sqrt((n - 1) * sd * sd / chi2.ppf(alpha / 2, n - 1))
    return sd, float(lo), float(hi)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=RESERVE_COUNT)
    parser.add_argument("--n-steps", type=int, default=N_STEPS)
    parser.add_argument("--output-json", type=Path, default=Path("pilot/tau_resid_post_pilot.json"))
    parser.add_argument("--output-md", type=Path, default=Path("pilot/tau_resid_post_pilot.md"))
    args = parser.parse_args()

    if args.count > RESERVE_COUNT:
        raise ValueError(f"Pilot count may not exceed {RESERVE_COUNT}; got {args.count}")

    pool = json.loads(Path("run/corpus_draw_stage_pool.json").read_text())
    records = pool["records"][RESERVE_START : RESERVE_START + args.count]
    expected_draws = list(range(RESERVE_START, RESERVE_START + args.count))
    actual_draws = [int(record["draw_order"]) for record in records]
    if actual_draws != expected_draws:
        raise RuntimeError(f"Reserve guard failed: expected draw orders {expected_draws}, got {actual_draws}")
    stage1_draws = set(pool["partitions"]["stage_1_draw_orders"])
    if any(draw in stage1_draws for draw in actual_draws):
        raise RuntimeError("Reserve guard failed: pilot selection intersects stage-1 draw orders")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float32
    print(f"device={device} mps_available={torch.backends.mps.is_available()} n_steps={args.n_steps}")
    print(f"reserve_draw_orders={actual_draws}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    sae = load_sae(device, dtype)
    if sae["w_dec"].shape != (16384, 2304):
        raise RuntimeError(f"Unexpected SAE W_dec shape: {tuple(sae['w_dec'].shape)}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype, attn_implementation="eager").to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    results: list[BaseResult] = []
    started = time.time()
    for local_idx, record in enumerate(records):
        print(f"[{local_idx + 1}/{len(records)}] draw_order={record['draw_order']} title={record['article_title']!r}")
        ids = torch.tensor([record["token_ids"]], dtype=torch.long, device=device)
        mask = torch.ones_like(ids, device=device)
        decoded = tokenizer.decode(record["token_ids"])
        if decoded != record["token_prefix_text"]:
            print("  note: tokenizer decode differs from artifact token_prefix_text formatting")
        base = extract_layer_output(model, ids, mask)
        fn = make_resid_post_map(model, ids, mask)
        acts = encode_sae(sae, base)
        l0 = int(torch.count_nonzero(acts > 0).detach().cpu())
        base_norm = float(torch.linalg.vector_norm(base).detach().cpu())
        radius = RADIUS_RELATIVE * base_norm
        rng = np.random.default_rng(SEED + int(record["draw_order"]))

        real_dirs, real_ids, real_det, real_tested, real_rejected = select_plane(
            arm="real", fn=fn, base=base, sae=sae, acts=acts, rng=rng
        )
        shuffled_dirs, shuffled_ids, shuffled_det, shuffled_tested, shuffled_rejected = select_plane(
            arm="shuffled", fn=fn, base=base, sae=sae, acts=acts, rng=rng
        )
        real = measure_plane(
            arm="real",
            fn=fn,
            base=base,
            directions=real_dirs,
            feature_ids=real_ids,
            det_value=real_det,
            candidates_tested=real_tested,
            candidates_rejected_detm=real_rejected,
            radius=radius,
            n_steps=args.n_steps,
        )
        shuffled = measure_plane(
            arm="shuffled",
            fn=fn,
            base=base,
            directions=shuffled_dirs,
            feature_ids=shuffled_ids,
            det_value=shuffled_det,
            candidates_tested=shuffled_tested,
            candidates_rejected_detm=shuffled_rejected,
            radius=radius,
            n_steps=args.n_steps,
        )
        d_b = real.log_holonomy - shuffled.log_holonomy
        results.append(
            BaseResult(
                reserve_record_index=RESERVE_START + local_idx,
                draw_order=int(record["draw_order"]),
                article_index=int(record["article_index"]),
                article_title=str(record["article_title"]),
                base_norm=base_norm,
                l0=l0,
                real=real,
                shuffled=shuffled,
                d_b=d_b,
            )
        )
        print(
            "  "
            f"L0={l0} d_b={d_b:.4g} real_H={real.holonomy:.4g} "
            f"shuf_H={shuffled.holonomy:.4g} detM=({real.det_m:.3g},{shuffled.det_m:.3g})"
        )

    d_values = np.array([result.d_b for result in results], dtype=np.float64)
    tau, tau_ci_lo, tau_ci_hi = sd_ci(d_values)
    l0_values = np.array([result.l0 for result in results], dtype=np.float64)
    elapsed = time.time() - started

    summary = {
        "purpose": "tau re-pilot at corrected resid_post site; reserve only; blind to stage-1",
        "model": MODEL_NAME,
        "model_library": "transformers.AutoModelForCausalLM",
        "sae_repo": SAE_REPO,
        "sae_path": SAE_PATH,
        "extraction_site": "HF model.model.layers[12] forward-hook output; TransformerLens blocks.12.hook_resid_post",
        "downstream_readout": "HF model.model.layers[13] forward-hook output after patching layer-12 resid_post",
        "reserve_record_indices": [result.reserve_record_index for result in results],
        "draw_orders": [result.draw_order for result in results],
        "stage_1_touched": False,
        "n_bases": len(results),
        "n_steps": args.n_steps,
        "radius_relative": RADIUS_RELATIVE,
        "tau_pilot": tau,
        "tau_ci_lo": tau_ci_lo,
        "tau_ci_hi": tau_ci_hi,
        "frozen_tau_iter7": 0.649,
        "frozen_tau_iter8": 0.586,
        "mean_l0": float(l0_values.mean()),
        "sd_l0": float(l0_values.std(ddof=1)),
        "min_l0": int(l0_values.min()),
        "max_l0": int(l0_values.max()),
        "detm_floor": TAU_DETM,
        "detm_rejections": {
            "real": int(sum(result.real.candidates_rejected_detm for result in results)),
            "shuffled": int(sum(result.shuffled.candidates_rejected_detm for result in results)),
            "real_candidates_tested": int(sum(result.real.candidates_tested for result in results)),
            "shuffled_candidates_tested": int(sum(result.shuffled.candidates_tested for result in results)),
        },
        "elapsed_seconds": elapsed,
        "results": [asdict(result) for result in results],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
    md = [
        "# Tau Re-Pilot at Corrected resid_post Site",
        "",
        "Reserve-only pilot; no stage-1 base points touched.",
        "",
        f"- Extraction site: `{summary['extraction_site']}`",
        f"- Downstream readout: `{summary['downstream_readout']}`",
        f"- SAE: `{SAE_REPO}/{SAE_PATH}`",
        f"- Reserve draw orders used: {summary['draw_orders']}",
        f"- n bases: {summary['n_bases']}",
        f"- n steps: {summary['n_steps']}",
        f"- tau_pilot: {tau:.6g}",
        f"- tau 95% CI: [{tau_ci_lo:.6g}, {tau_ci_hi:.6g}]",
        f"- frozen tau comparison: Iter 7 = 0.649; Iter 8 = 0.586",
        f"- SAE L0 sanity: mean {summary['mean_l0']:.2f}, SD {summary['sd_l0']:.2f}, range [{summary['min_l0']}, {summary['max_l0']}]",
        f"- det M floor: {TAU_DETM}",
        f"- det M rejected/tested, real: {summary['detm_rejections']['real']}/{summary['detm_rejections']['real_candidates_tested']}",
        f"- det M rejected/tested, shuffled: {summary['detm_rejections']['shuffled']}/{summary['detm_rejections']['shuffled_candidates_tested']}",
        f"- elapsed: {elapsed / 60:.2f} min",
        "",
        "| reserve idx | draw order | title | L0 | d_b | real H | shuffled H | real detM | shuffled detM |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        md.append(
            f"| {result.reserve_record_index} | {result.draw_order} | {result.article_title} | "
            f"{result.l0} | {result.d_b:.6g} | {result.real.holonomy:.6g} | "
            f"{result.shuffled.holonomy:.6g} | {result.real.det_m:.6g} | {result.shuffled.det_m:.6g} |"
        )
    args.output_md.write_text("\n".join(md) + "\n")
    print(json.dumps({k: summary[k] for k in ["tau_pilot", "tau_ci_lo", "tau_ci_hi", "mean_l0", "detm_rejections", "elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
