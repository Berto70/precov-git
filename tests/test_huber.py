import numpy as np

from precov.orbits.fitting import huber_rho


def test_huber_matches_chi2_for_small_residuals():
    z = np.linspace(-0.99, 0.99, 50)
    assert np.allclose(huber_rho(z, c=1.0), z**2)


def test_huber_is_linear_beyond_threshold():
    c = 1.0
    z = np.array([2.0, 3.0, 5.0, -4.0])
    expected = 2 * c * np.abs(z) - c**2
    assert np.allclose(huber_rho(z, c), expected)


def test_huber_continuous_and_c1_at_threshold():
    c = 1.0
    eps = 1e-6
    left = huber_rho(np.array([c - eps]), c)[0]
    right = huber_rho(np.array([c + eps]), c)[0]
    assert abs(left - right) < 1e-5  # value continuity

    # slope continuity: d/dz(z^2)=2z -> 2c at z=c; d/dz(2c|z|-c^2)=2c
    slope_left = (huber_rho(np.array([c]), c)[0] - huber_rho(np.array([c - eps]), c)[0]) / eps
    slope_right = (huber_rho(np.array([c + eps]), c)[0] - huber_rho(np.array([c]), c)[0]) / eps
    assert abs(slope_left - slope_right) < 1e-2


def test_pseudo_residual_identity():
    """sign(z)*sqrt(rho(z,c)) squared must reproduce rho(z,c) exactly -- this
    is the algebraic trick that lets an ordinary least-squares solver perform
    robust M-estimation."""
    rng = np.random.default_rng(0)
    z = rng.normal(scale=3.0, size=1000)
    c = 1.0
    w = np.sign(z) * np.sqrt(huber_rho(z, c))
    assert np.allclose(w**2, huber_rho(z, c))
