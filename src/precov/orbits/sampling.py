"""
Boundary (not interior) sampling of the orbital-element confidence ellipsoid.

Key idea (Sec. 3.2): since only the *outer envelope* of the on-sky footprint
matters for building search regions, sampling the interior of the confidence
ellipsoid (e.g. via a naive 6D Gaussian Monte Carlo) wastes the overwhelming
majority of draws on points whose sky-plane projection lands *inside* the
already-covered region. Instead, points are drawn uniformly on the surface of
the unit 5-sphere and mapped onto the exact iso-contour

    (theta - theta_hat)^T Sigma^-1 (theta - theta_hat) = chi2_6(q)

via the Cholesky factor of Sigma. This guarantees, by construction, that
every sample's squared Mahalanobis distance from theta_hat equals
chi2_6(q) exactly (up to floating-point roundoff) -- a property we assert
directly in the test suite rather than trusting the derivation on faith.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.stats import chi2

from ..constants import DEFAULT_BOUNDARY_SAMPLES_M, DEFAULT_CONFIDENCE_Q


def sample_boundary_orbits(theta_hat: np.ndarray, covariance: np.ndarray,
                            M: int = DEFAULT_BOUNDARY_SAMPLES_M,
                            q: float = DEFAULT_CONFIDENCE_Q,
                            rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Draw M orbital-element vectors on the surface of the q-confidence
    6D ellipsoid centered at `theta_hat` with covariance `covariance`.

    Returns
    -------
    theta_samples : ndarray, shape (M, 6)
    """
    if rng is None:
        rng = np.random.default_rng()

    theta_hat = np.asarray(theta_hat, dtype=float)
    dim = theta_hat.shape[0]

    L = np.linalg.cholesky(covariance)
    chi2_val = chi2.ppf(q, df=dim)
    scale = np.sqrt(chi2_val)

    # Uniform directions on the unit (dim-1)-sphere: normalize iid Gaussians.
    u = rng.normal(size=(M, dim))
    z = u / np.linalg.norm(u, axis=1, keepdims=True)

    theta_samples = theta_hat[None, :] + scale * (z @ L.T)
    return theta_samples
