import math

import pytest
import torch

from extraction import ResidPostExtractor, load_corpus_token_ids, load_model_and_tokenizer, load_sae
from planes.construct import Arm, select_three_arm_planes
from transport.loop import (
    RADIUS_RELATIVE,
    TransportResult,
    extract_theta,
    loop_points,
    loop_transport,
)


DTYPE = torch.float64


def identity_jvp(_point: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    return direction


def curved_surface_jvp(point: torch.Tensor, direction: torch.Tensor, *, curvature: float = 5.0) -> torch.Tensor:
    x, y = point[0], point[1]
    jacobian = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0 * curvature * x, 2.0 * curvature * y],
        ],
        dtype=point.dtype,
        device=point.device,
    )
    return jacobian @ direction


def test_loop_closure_gamma_zero_equals_gamma_one():
    center = torch.tensor([0.2, -0.4, 1.0], dtype=DTYPE)
    directions = torch.tensor([[1.0, 0.0, 0.0], [0.25, 0.75, 0.0]], dtype=DTYPE)

    points = loop_points(center=center, directions=directions, rho=0.15, n_steps=64)

    assert torch.allclose(points[0], points[-1], atol=1e-12, rtol=0)


def test_extract_theta_separates_antisymmetric_and_symmetric_parts():
    symmetric = torch.tensor([[0.2, 0.03], [0.03, -0.1]], dtype=DTYPE)
    theta_true = torch.tensor(0.125, dtype=DTYPE)
    antisymmetric = torch.tensor([[0.0, -theta_true], [theta_true, 0.0]], dtype=DTYPE)
    operator = torch.eye(2, dtype=DTYPE) + symmetric + antisymmetric

    theta, symmetric_norm = extract_theta(operator)

    assert torch.allclose(theta, theta_true, atol=1e-12, rtol=0)
    assert symmetric_norm > 0


def test_loop_reversal_flips_theta_sign_on_curved_synthetic_connection():
    center = torch.tensor([0.0, 0.0], dtype=DTYPE)
    directions = torch.eye(2, dtype=DTYPE)

    forward = loop_transport(
        center=center,
        directions=directions,
        rho=0.1,
        jvp_fn=curved_surface_jvp,
        n_steps=200,
    )
    reverse = loop_transport(
        center=center,
        directions=directions,
        rho=0.1,
        jvp_fn=curved_surface_jvp,
        n_steps=200,
        reverse=True,
    )

    assert abs(forward.theta) > 1e-6
    assert torch.allclose(forward.theta, -reverse.theta, atol=1e-10, rtol=1e-8)
    assert torch.allclose(forward.holonomy, reverse.holonomy, atol=1e-10, rtol=1e-8)


def test_orthogonal_identity_limit_area_reduces_to_rho_squared_and_theta_is_finite():
    center = torch.tensor([3.0, 4.0], dtype=DTYPE)
    directions = torch.eye(2, dtype=DTYPE)
    rho = torch.tensor(0.006, dtype=DTYPE)

    result = loop_transport(
        center=center,
        directions=directions,
        rho=rho,
        jvp_fn=identity_jvp,
        n_steps=16,
    )

    assert isinstance(result, TransportResult)
    assert torch.allclose(result.area_enclosed, rho * rho, atol=1e-14, rtol=0)
    assert torch.isfinite(result.theta)
    assert torch.isfinite(result.holonomy)
    assert torch.allclose(result.theta, torch.tensor(0.0, dtype=DTYPE), atol=1e-12, rtol=0)


@pytest.fixture(scope="module")
def real_transport_stack():
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


@pytest.mark.integration
def test_real_transport_is_deterministic_finite_and_positive(real_transport_stack):
    model, sae, token_sequences, bases = real_transport_stack
    observed = []

    for idx, base in enumerate(bases):
        code = sae.encode(base)
        jvp_fn = make_readout_jvp(model, token_sequences[idx], base)
        base_jvp_fn = lambda direction, *, _base=base, _jvp_fn=jvp_fn: _jvp_fn(_base, direction)
        selected = select_three_arm_planes(
            activation=base,
            decoder_directions=sae.w_dec,
            code=code,
            jvp_fn=base_jvp_fn,
            seed=20260614 + idx,
        )[Arm.REAL_FEATURE]
        rho = RADIUS_RELATIVE * torch.linalg.vector_norm(base)
        first = loop_transport(
            center=base,
            directions=selected.directions,
            rho=rho,
            jvp_fn=jvp_fn,
            n_steps=200,
        )
        second = loop_transport(
            center=base,
            directions=selected.directions,
            rho=rho,
            jvp_fn=jvp_fn,
            n_steps=200,
        )
        observed.append(
            {
                "base": idx,
                "H": float(first.holonomy),
                "theta": float(first.theta),
                "A": float(first.area_enclosed),
                "det_m": float(first.det_m),
            }
        )

        assert torch.equal(first.theta, second.theta)
        assert torch.equal(first.holonomy, second.holonomy)
        assert torch.isfinite(first.theta)
        assert torch.isfinite(first.holonomy)
        assert first.holonomy > 0
        assert first.area_enclosed > 0

    print(f"real_transport_observed={observed}")
