import math

import torch

from geometry.pullback import (
    CenterPlacement,
    center_placement,
    det_m,
    enclosed_area,
    g_projection_coeffs,
    in_plane_magnitude,
    pullback_gram,
)


DTYPE = torch.float64
ATOL = 1e-10


def synthetic_inputs():
    j = torch.tensor(
        [
            [1.2, -0.4, 0.3],
            [0.2, 0.9, -0.5],
            [-0.1, 0.7, 1.1],
            [0.8, -0.2, 0.4],
        ],
        dtype=DTYPE,
    )
    directions = torch.tensor(
        [
            [0.8, 0.1, -0.2],
            [-0.3, 0.7, 0.4],
        ],
        dtype=DTYPE,
    )
    directions = directions / torch.linalg.vector_norm(directions, dim=1, keepdim=True)
    h = torch.tensor([1.5, -0.8, 0.6], dtype=DTYPE)
    jd = j @ directions.T
    jh = j @ h
    return j, directions, h, jd, jh


def test_pullback_gram_is_symmetric_psd_and_det_nonnegative():
    _, _, _, jd, _ = synthetic_inputs()

    gram = pullback_gram(jd)
    eigvals = torch.linalg.eigvalsh(gram)

    assert torch.allclose(gram, gram.T, atol=ATOL, rtol=0)
    assert torch.all(eigvals >= -ATOL)
    assert det_m(gram) >= -ATOL


def test_g_projection_residual_is_pullback_orthogonal_to_plane():
    _, _, _, jd, jh = synthetic_inputs()

    coeffs = g_projection_coeffs(jd, jh)
    residual = jh - jd @ coeffs

    assert torch.allclose(jd.T @ residual, torch.zeros(2, dtype=DTYPE), atol=ATOL, rtol=0)


def test_in_plane_magnitude_consistency_across_equivalent_forms():
    _, _, _, jd, jh = synthetic_inputs()

    gram = pullback_gram(jd)
    coeffs = g_projection_coeffs(jd, jh)
    mag = in_plane_magnitude(jd, jh)
    mag_sq_from_gram = coeffs @ gram @ coeffs
    mag_sq_from_image = (jd @ coeffs) @ (jd @ coeffs)

    assert torch.allclose(mag * mag, mag_sq_from_gram, atol=ATOL, rtol=0)
    assert torch.allclose(mag_sq_from_gram, mag_sq_from_image, atol=ATOL, rtol=0)


def test_center_placement_resets_in_plane_magnitude_to_target():
    _, directions, h, jd, jh = synthetic_inputs()
    target_magnitude = torch.tensor(3.25, dtype=DTYPE)

    placement = center_placement(
        directions=directions,
        jd=jd,
        h=h,
        jh=jh,
        target_magnitude=target_magnitude,
        eps_mag=1e-9,
    )
    in_plane_offset = placement.center - placement.out_of_plane
    offset_image = jd @ placement.offset_coeffs
    reset_mag = torch.linalg.vector_norm(offset_image)

    assert isinstance(placement, CenterPlacement)
    assert not placement.fallback_used
    assert torch.allclose(in_plane_offset, directions.T @ placement.offset_coeffs, atol=ATOL, rtol=0)
    assert torch.allclose(reset_mag, target_magnitude, atol=ATOL, rtol=0)


def test_center_placement_preserves_out_of_plane_component_exactly():
    _, directions, h, jd, jh = synthetic_inputs()

    placement = center_placement(
        directions=directions,
        jd=jd,
        h=h,
        jh=jh,
        target_magnitude=torch.tensor(2.0, dtype=DTYPE),
        eps_mag=1e-9,
    )
    original_residual = jh - jd @ placement.projection_coeffs
    center_image = original_residual + jd @ placement.offset_coeffs
    center_projection = g_projection_coeffs(jd, center_image)
    center_residual = center_image - jd @ center_projection

    assert torch.allclose(placement.out_of_plane, h - directions.T @ placement.projection_coeffs, atol=ATOL, rtol=0)
    assert torch.allclose(center_residual, original_residual, atol=ATOL, rtol=0)


def test_orthogonal_euclidean_limit_matches_validated_instrument_constant():
    j = torch.eye(3, dtype=DTYPE)
    directions = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=DTYPE)
    h = torch.tensor([3.0, 4.0, 12.0], dtype=DTYPE)
    jd = j @ directions.T
    jh = j @ h
    rho = torch.tensor(0.006, dtype=DTYPE)

    gram = pullback_gram(jd)
    area = enclosed_area(rho, gram)
    mag = in_plane_magnitude(jd, jh)

    assert torch.allclose(det_m(gram), torch.tensor(1.0, dtype=DTYPE), atol=ATOL, rtol=0)
    assert torch.allclose(area, rho * rho, atol=ATOL, rtol=0)
    assert torch.allclose(mag, torch.tensor(5.0, dtype=DTYPE), atol=ATOL, rtol=0)


def test_eps_mag_fallback_uses_g_normalized_first_direction():
    j = torch.tensor([[2.0, 0.0], [0.0, 3.0]], dtype=DTYPE)
    directions = torch.eye(2, dtype=DTYPE)
    h = torch.tensor([0.0, 0.0], dtype=DTYPE)
    jd = j @ directions.T
    jh = j @ h

    placement = center_placement(
        directions=directions,
        jd=jd,
        h=h,
        jh=jh,
        target_magnitude=torch.tensor(7.0, dtype=DTYPE),
        eps_mag=1e-6,
    )
    expected_coeffs = torch.tensor([7.0 / 2.0, 0.0], dtype=DTYPE)

    assert placement.fallback_used
    assert torch.allclose(placement.offset_coeffs, expected_coeffs, atol=ATOL, rtol=0)
    assert torch.allclose(torch.linalg.vector_norm(jd @ placement.offset_coeffs), torch.tensor(7.0, dtype=DTYPE), atol=ATOL, rtol=0)


def test_geometry_operations_are_deterministic_bitwise():
    _, directions, h, jd, jh = synthetic_inputs()
    kwargs = {
        "directions": directions,
        "jd": jd,
        "h": h,
        "jh": jh,
        "target_magnitude": torch.tensor(math.pi, dtype=DTYPE),
        "eps_mag": 1e-9,
    }

    first = center_placement(**kwargs)
    second = center_placement(**kwargs)

    assert torch.equal(pullback_gram(jd), pullback_gram(jd))
    assert torch.equal(first.center, second.center)
    assert torch.equal(first.projection_coeffs, second.projection_coeffs)
    assert torch.equal(first.offset_coeffs, second.offset_coeffs)
    assert first.fallback_used == second.fallback_used
