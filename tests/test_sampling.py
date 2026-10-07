import numpy as np
from scipy.stats import chi2

from precov.orbits.sampling import sample_boundary_orbits


def test_samples_lie_exactly_on_the_confidence_isosurface():
    rng = np.random.default_rng(42)
    theta_hat = np.array([1.65, 0.42, 0.1, 0.3, 1.1, 2.5])

    # A random SPD covariance matrix (not axis-aligned, to stress the
    # Cholesky-based mapping properly).
    A = rng.normal(size=(6, 6))
    cov = A @ A.T + 1e-3 * np.eye(6)

    q = 0.9999
    M = 500
    samples = sample_boundary_orbits(theta_hat, cov, M=M, q=q, rng=rng)

    assert samples.shape == (M, 6)

    cov_inv = np.linalg.inv(cov)
    diffs = samples - theta_hat
    mahalanobis_sq = np.einsum("ni,ij,nj->n", diffs, cov_inv, diffs)

    expected = chi2.ppf(q, df=6)
    assert np.allclose(mahalanobis_sq, expected, rtol=1e-8)


def test_boundary_samples_are_not_the_mean():
    """A sanity check against accidentally sampling the *interior* (e.g. a
    plain multivariate normal) instead of the boundary: no sample should ever
    land near theta_hat itself."""
    rng = np.random.default_rng(1)
    theta_hat = np.zeros(6)
    cov = np.eye(6)
    samples = sample_boundary_orbits(theta_hat, cov, M=200, q=0.9999, rng=rng)
    distances = np.linalg.norm(samples - theta_hat, axis=1)
    assert np.all(distances > 4.0)  # sqrt(chi2_6(0.9999)) ~ 5.28
