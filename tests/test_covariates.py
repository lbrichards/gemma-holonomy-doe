import math

import pytest
import torch

from covariates.measure import (
    ReferenceDistribution,
    loop_points,
    measure_plane_covariates,
    mutual_angle_phi,
    sae_reconstruction_mse,
)
from extraction import ResidPostExtractor, load_corpus_token_ids, load_model_and_tokenizer, load_sae
from planes.construct import Arm, select_three_arm_planes
from tests.test_planes import make_readout_jvp


DTYPE = torch.float64


def test_phi_ground_truth_known_angles():
    sixty = torch.tensor(
        [
            [1.0, 0.0],
            [0.5, math.sqrt(3) / 2],
        ],
        dtype=DTYPE,
    )
    ninety = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=DTYPE,
    )

    assert torch.allclose(mutual_angle_phi(sixty), torch.tensor(math.pi / 3, dtype=DTYPE), atol=1e-12, rtol=0)
    assert torch.allclose(mutual_angle_phi(ninety), torch.tensor(math.pi / 2, dtype=DTYPE), atol=1e-12, rtol=0)


def test_phi_uses_raw_directions_without_orthogonalization():
    directions = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [math.sqrt(0.5), math.sqrt(0.5), 0.0],
        ],
        dtype=DTYPE,
    )

    phi = mutual_angle_phi(directions)

    assert torch.allclose(phi, torch.tensor(math.pi / 4, dtype=DTYPE), atol=1e-12, rtol=0)


def test_reference_distribution_uses_mahalanobis_when_covariance_is_well_conditioned():
    reference = torch.tensor(
        [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=DTYPE,
    )
    dist = ReferenceDistribution.fit(reference)

    values = dist.distance(torch.tensor([[0.25, -0.25]], dtype=DTYPE))

    assert dist.method == "mahalanobis"
    assert values.shape == (1,)
    assert torch.isfinite(values).all()
    assert values.item() >= 0


def test_reference_distribution_falls_back_to_knn_for_rank_deficient_covariance():
    reference = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ],
        dtype=DTYPE,
    )
    dist = ReferenceDistribution.fit(reference, k=2)

    values = dist.distance(torch.tensor([[1.0, 1.0, 1.0]], dtype=DTYPE))

    assert dist.method == "knn"
    assert values.shape == (1,)
    assert torch.isfinite(values).all()
    assert values.item() >= 0


def test_covariates_are_deterministic_for_synthetic_inputs():
    class ToySae:
        def encode(self, activation: torch.Tensor) -> torch.Tensor:
            return activation

        @property
        def w_dec(self) -> torch.Tensor:
            return torch.eye(2, dtype=DTYPE)

        @property
        def b_dec(self) -> torch.Tensor:
            return torch.zeros(2, dtype=DTYPE)

    center = torch.tensor([0.3, -0.2], dtype=DTYPE)
    directions = torch.eye(2, dtype=DTYPE)
    reference = ReferenceDistribution.fit(
        torch.tensor(
            [
                [-1.0, -1.0],
                [-1.0, 1.0],
                [1.0, -1.0],
                [1.0, 1.0],
            ],
            dtype=DTYPE,
        )
    )

    first = measure_plane_covariates(center=center, directions=directions, rho=0.1, sae=ToySae(), reference=reference)
    second = measure_plane_covariates(center=center, directions=directions, rho=0.1, sae=ToySae(), reference=reference)

    assert first == second


@pytest.fixture(scope="module")
def real_covariate_stack():
    model, _tokenizer = load_model_and_tokenizer()
    sae = load_sae(device=next(model.parameters()).device)
    extractor = ResidPostExtractor(model)
    token_sequences = load_corpus_token_ids(count=2)
    bases = [extractor.extract(token_ids).activation for token_ids in token_sequences]
    return model, sae, token_sequences, bases


@pytest.mark.integration
def test_sae_reconstruction_error_is_finite_nonnegative_on_real_base(real_covariate_stack):
    _model, sae, _token_sequences, bases = real_covariate_stack
    points = torch.stack(bases)

    mse = sae_reconstruction_mse(sae, points)

    print(f"real_base_reconstruction_mse={mse.detach().cpu().tolist()}")
    assert mse.shape == (2,)
    assert torch.isfinite(mse).all()
    assert torch.all(mse >= 0)
    assert torch.all(mse < 1e4)


@pytest.mark.integration
def test_plane_covariates_real_points_have_finite_reconstruction_distance_and_phi(real_covariate_stack):
    model, sae, token_sequences, bases = real_covariate_stack
    reference = ReferenceDistribution.fit(torch.stack(bases))
    observed = []

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
            cov = measure_plane_covariates(
                center=base,
                directions=plane.directions,
                rho=0.006 * torch.linalg.vector_norm(base),
                sae=sae,
                reference=reference,
                n_loop_samples=8,
            )
            observed.append((arm.value, cov.reconstruction_mse_mean, cov.reference_distance_mean, cov.phi))

    print(f"real_plane_covariates={observed}")
    assert reference.method == "knn"
    for _arm, recon, ref_distance, phi in observed:
        assert math.isfinite(recon)
        assert recon >= 0
        assert math.isfinite(ref_distance)
        assert ref_distance >= 0
        assert 0 <= phi <= math.pi


def test_loop_points_include_center_plus_swept_loop_points():
    center = torch.tensor([1.0, 2.0], dtype=DTYPE)
    directions = torch.eye(2, dtype=DTYPE)

    points = loop_points(center=center, directions=directions, rho=0.5, n_loop_samples=4, include_center=True)

    assert points.shape == (5, 2)
    assert torch.equal(points[0], center)
    assert torch.allclose(points[1], torch.tensor([1.5, 2.0], dtype=DTYPE), atol=1e-12, rtol=0)
