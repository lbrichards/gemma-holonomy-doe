import numpy as np

from bands.support import BandStatus, compute_common_support_band


def test_three_arm_valid_returns_pooled_band_percentiles():
    real = np.linspace(0.0, 100.0, 101)
    shuffled = np.linspace(10.0, 110.0, 101)
    random = np.linspace(20.0, 120.0, 101)

    result = compute_common_support_band(real=real, shuffled=shuffled, random=random)

    assert result.status == BandStatus.THREE_ARM_VALID
    assert result.h_sem_matched
    assert result.h_grad_matched
    assert result.m_low is not None
    assert result.m_high is not None
    assert result.band_lower <= result.m_low <= result.band_upper
    assert result.band_lower <= result.m_high <= result.band_upper


def test_three_arm_collapse_two_arm_fallback_marks_h_grad_undefined():
    real = np.linspace(55.0, 75.0, 101)
    shuffled = np.linspace(50.0, 70.0, 101)
    random = np.linspace(10.0, 16.0, 101)

    result = compute_common_support_band(real=real, shuffled=shuffled, random=random)

    assert result.status == BandStatus.TWO_ARM_FALLBACK
    assert result.h_sem_matched
    assert not result.h_grad_matched
    assert result.h_grad_status == "UNDEFINED_AT_MATCHED_MAGNITUDE"
    assert result.m_low is not None
    assert result.m_high is not None
    assert result.band_lower <= result.m_low <= result.band_upper
    assert result.band_lower <= result.m_high <= result.band_upper


def test_two_arm_terminal_produces_no_m_values_and_marks_h_sem_undefined():
    real = np.linspace(63.0, 67.0, 101)
    shuffled = np.linspace(18.0, 22.0, 101)
    random = np.linspace(11.0, 15.0, 101)

    result = compute_common_support_band(real=real, shuffled=shuffled, random=random)

    assert result.status == BandStatus.TWO_ARM_TERMINAL
    assert not result.h_sem_matched
    assert not result.h_grad_matched
    assert result.h_sem_status == "UNDEFINED_AT_MATCHED_MAGNITUDE"
    assert result.h_grad_status == "UNDEFINED_AT_MATCHED_MAGNITUDE"
    assert result.m_low is None
    assert result.m_high is None


def test_width_exactly_equal_to_threshold_is_valid():
    real = np.linspace(0.0, 100.0, 1001)
    shuffled = np.linspace(0.0, 100.0, 1001)
    random = np.linspace(40.0, 53.888888888888886, 1001)

    result = compute_common_support_band(real=real, shuffled=shuffled, random=random)

    assert np.isclose(result.three_arm_width, result.three_arm_threshold)
    assert result.status == BandStatus.THREE_ARM_VALID


def test_band_computation_is_deterministic():
    real = np.linspace(55.0, 75.0, 101)
    shuffled = np.linspace(50.0, 70.0, 101)
    random = np.linspace(10.0, 16.0, 101)

    first = compute_common_support_band(real=real, shuffled=shuffled, random=random)
    second = compute_common_support_band(real=real, shuffled=shuffled, random=random)

    assert first == second


def test_percentile_correctness_on_known_uniform_distribution():
    values = np.linspace(0.0, 100.0, 10001)

    result = compute_common_support_band(real=values, shuffled=values, random=values)

    assert np.isclose(result.per_arm["real"].p5, 5.0)
    assert np.isclose(result.per_arm["real"].p25, 25.0)
    assert np.isclose(result.per_arm["real"].p75, 75.0)
    assert np.isclose(result.per_arm["real"].p95, 95.0)
    assert np.isclose(result.band_lower, 5.0)
    assert np.isclose(result.band_upper, 95.0)
    assert np.isclose(result.m_low, 27.5)
    assert np.isclose(result.m_high, 72.5)
