import torch
import pytest

from extraction import ResidPostExtractor, load_corpus_token_ids, load_model_and_tokenizer, load_sae
from planes.construct import (
    TAU_DETM,
    Arm,
    evaluate_det_m,
    select_plane,
    select_three_arm_planes,
)


DTYPE = torch.float64


def identity_jvp(direction: torch.Tensor) -> torch.Tensor:
    return direction


def test_det_m_floor_rejects_near_collinear_and_accepts_well_conditioned_planes():
    near_collinear = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],
        ],
        dtype=DTYPE,
    )
    near_collinear = near_collinear / torch.linalg.vector_norm(near_collinear, dim=1, keepdim=True)
    orthogonal = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=DTYPE,
    )

    rejected = evaluate_det_m(near_collinear, identity_jvp)
    accepted = evaluate_det_m(orthogonal, identity_jvp)

    assert rejected.det_m <= TAU_DETM
    assert not rejected.passes
    assert accepted.det_m > TAU_DETM
    assert accepted.passes


def test_seed_sensitivity_for_random_planes():
    dim = 16

    first = select_plane(
        arm=Arm.RANDOM,
        activation=torch.zeros(dim, dtype=DTYPE),
        decoder_directions=None,
        code=None,
        jvp_fn=identity_jvp,
        seed=11,
    )
    second = select_plane(
        arm=Arm.RANDOM,
        activation=torch.zeros(dim, dtype=DTYPE),
        decoder_directions=None,
        code=None,
        jvp_fn=identity_jvp,
        seed=12,
    )

    assert first.feature_ids is None
    assert second.feature_ids is None
    assert not torch.equal(first.directions, second.directions)


@pytest.fixture(scope="module")
def real_plane_stack():
    model, _tokenizer = load_model_and_tokenizer()
    sae = load_sae(device=next(model.parameters()).device)
    extractor = ResidPostExtractor(model)
    token_sequences = load_corpus_token_ids(count=2)
    bases = [extractor.extract(token_ids).activation for token_ids in token_sequences]
    return model, sae, token_sequences, bases


def make_readout_jvp(model, token_ids: list[int], base: torch.Tensor):
    device = base.device
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, device=device)

    def readout_map(h: torch.Tensor) -> torch.Tensor:
        state: dict[str, torch.Tensor] = {}

        def patch_hook(_module, _args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            patched = tensor.clone()
            patched[0, 63, :] = h.to(device=patched.device, dtype=patched.dtype)
            if isinstance(output, tuple):
                return (patched, *output[1:])
            return patched

        def capture_hook(_module, _args, output) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            state["out"] = tensor

        patch_handle = model.model.layers[12].register_forward_hook(patch_hook)
        capture_handle = model.model.layers[13].register_forward_hook(capture_hook)
        try:
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        finally:
            patch_handle.remove()
            capture_handle.remove()
        return state["out"][0, 63, :]

    def jvp_fn(direction: torch.Tensor) -> torch.Tensor:
        point = base.detach().requires_grad_(True)
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


@pytest.mark.integration
def test_three_arm_plane_selection_is_deterministic_and_structurally_correct(real_plane_stack):
    model, sae, token_sequences, bases = real_plane_stack
    base = bases[0]
    code = sae.encode(base)
    jvp_fn = make_readout_jvp(model, token_sequences[0], base)

    first = select_three_arm_planes(
        activation=base,
        decoder_directions=sae.w_dec,
        code=code,
        jvp_fn=jvp_fn,
        seed=20260614,
    )
    second = select_three_arm_planes(
        activation=base,
        decoder_directions=sae.w_dec,
        code=code,
        jvp_fn=jvp_fn,
        seed=20260614,
    )

    for arm in Arm:
        assert first[arm].feature_ids == second[arm].feature_ids
        assert torch.equal(first[arm].directions, second[arm].directions)
        assert first[arm].det_m > TAU_DETM

    assert first[Arm.REAL_FEATURE].feature_ids is not None
    assert first[Arm.SHUFFLED_FEATURE].feature_ids is not None
    real_left, real_right = first[Arm.REAL_FEATURE].feature_ids
    shuf_active, shuf_inactive = first[Arm.SHUFFLED_FEATURE].feature_ids

    assert code[real_left] > 0
    assert code[real_right] > 0
    assert code[shuf_active] > 0
    assert code[shuf_inactive] <= 0
    assert first[Arm.RANDOM].feature_ids is None


@pytest.mark.integration
def test_real_feature_directions_are_raw_not_orthogonalized(real_plane_stack):
    model, sae, token_sequences, bases = real_plane_stack
    base = bases[0]
    code = sae.encode(base)
    jvp_fn = make_readout_jvp(model, token_sequences[0], base)

    selected = select_plane(
        arm=Arm.REAL_FEATURE,
        activation=base,
        decoder_directions=sae.w_dec,
        code=code,
        jvp_fn=jvp_fn,
        seed=20260614,
    )
    feature_ids = selected.feature_ids
    assert feature_ids is not None
    raw = sae.w_dec[list(feature_ids)]
    raw = raw / torch.linalg.vector_norm(raw, dim=1, keepdim=True).clamp_min(1e-12)

    assert torch.equal(selected.directions, raw.to(device=selected.directions.device, dtype=selected.directions.dtype))
    assert abs(float(selected.directions[0] @ selected.directions[1])) > 1e-4


@pytest.mark.integration
def test_rejection_counts_are_recorded_on_real_base_points(real_plane_stack):
    model, sae, token_sequences, bases = real_plane_stack
    totals = {arm: {"tested": 0, "rejected": 0} for arm in Arm}

    for idx, base in enumerate(bases):
        code = sae.encode(base)
        jvp_fn = make_readout_jvp(model, token_sequences[idx], base)
        selected = select_three_arm_planes(
            activation=base,
            decoder_directions=sae.w_dec,
            code=code,
            jvp_fn=jvp_fn,
            seed=20260614 + idx,
        )
        for arm, plane in selected.items():
            totals[arm]["tested"] += plane.candidates_tested
            totals[arm]["rejected"] += plane.candidates_rejected_detm

    print(f"real_test_point_detm_rejections={totals}")
    for counts in totals.values():
        assert counts["tested"] >= 2
        assert counts["rejected"] >= 0
