"""
Huber-robust least-squares orbit fitting (Sec. 3.1, Algorithm 1).

Implementation note on `huber_rho`
-----------------------------------
The typeset paper equation for the Huber penalty is a transcription
casualty: it reads "rho(z,c) = z^2 if z/c^2 <= 1; c^2(sqrt(z/c^2)-1) if
z/c^2 > 1", which is dimensionally inconsistent (rho would be quartic in the
residual for |z|>c, and the threshold z/c^2<=1 doesn't reduce to the stated
|z|<=c condition for c != 1). The report's Algorithm 1 pseudocode gives the
classical, unambiguous Huber penalty of the *standardized residual* z,

    rho(z, c) = z^2                if |z| <= c
    rho(z, c) = 2*c*|z| - c^2      if |z| > c

which is C^1 continuous at |z|=c (both value and slope match: 2c*c - c^2 =
c^2, and d/dz matches at c^2 -> 2c), reduces to ordinary chi^2 in the
inlier regime, and grows linearly (not quadratically) with the residual
beyond c standard deviations. We adopt this form.

Because `scipy.optimize.least_squares` minimizes 0.5 * sum(residual^2) via
ordinary (weighted) nonlinear least squares, we do not modify the optimizer;
instead we transform each standardized residual through the classic
"pseudo-residual" square-root trick, `w = sign(z) * sqrt(rho(z, c))`, so
that sum(w^2) == sum(rho(z, c)) identically (verified in tests). This lets
TRF's trust-region machinery and its finite-difference/analytic Jacobian
support work unmodified while still solving the robust M-estimation problem.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from astropy.time import Time
from scipy.optimize import least_squares

from ..constants import DEFAULT_HUBER_C
from ..data_models import Observation, OrbitFitResult
from .propagator import Propagator


def huber_rho(z: np.ndarray, c: float = DEFAULT_HUBER_C) -> np.ndarray:
    """The Huber penalty of a standardized residual z (see module docstring)."""
    z = np.asarray(z, dtype=float)
    az = np.abs(z)
    return np.where(az <= c, z**2, 2.0 * c * az - c**2)


def _wrap_angle(delta_rad: np.ndarray) -> np.ndarray:
    """Wrap an angular difference into (-pi, pi]."""
    return (delta_rad + np.pi) % (2 * np.pi) - np.pi


def residual_vector(theta: np.ndarray, observations: List[Observation],
                     propagator: Propagator, c: float = DEFAULT_HUBER_C) -> np.ndarray:
    """Build the (2N,) pseudo-residual vector fed to `scipy.least_squares`.

    Each observation contributes two entries (RA, Dec). RA residuals include
    the cos(dec) great-circle correction and are angle-wrapped before being
    standardized, since raw RA differences are not physically meaningful
    across the 0/2*pi boundary.
    """
    res = np.empty(2 * len(observations))
    for k, obs in enumerate(observations):
        ra_pred, dec_pred = propagator.propagate(theta, obs.t, obs.observer_location)

        d_ra = _wrap_angle(obs.ra - ra_pred) * np.cos(obs.dec)
        d_dec = obs.dec - dec_pred

        z_ra = d_ra / obs.sigma_ra
        z_dec = d_dec / obs.sigma_dec

        if obs.robust:
            w_ra = np.sign(z_ra) * np.sqrt(huber_rho(z_ra, c))
            w_dec = np.sign(z_dec) * np.sqrt(huber_rho(z_dec, c))
        else:
            w_ra, w_dec = z_ra, z_dec

        res[2 * k] = w_ra
        res[2 * k + 1] = w_dec
    return res


def fit_orbit(observations: List[Observation], propagator: Propagator,
              theta0: np.ndarray, c: float = DEFAULT_HUBER_C,
              bounds: Optional[tuple] = None) -> OrbitFitResult:
    """Fit Keplerian elements to `observations` via Huber-robust (for
    `obs.robust=True` entries) trust-region-reflective nonlinear least
    squares, returning the best-fit elements and a Fisher-information-based
    parameter covariance (Sec. 3.1).

    `propagator` must have been constructed with its reference epoch equal
    to the epoch at which the returned `theta` is defined (typically the
    survey midpoint, per the paper).
    """
    if bounds is None:
        # (a>0, 0<=e<1, i, raan, argp, M unrestricted in practice but TRF
        # wants finite box bounds; wide and physically permissive)
        lo = [1e-6, 0.0, -np.pi, -4 * np.pi, -4 * np.pi, -4 * np.pi]
        hi = [1e4, 0.999999, np.pi, 4 * np.pi, 4 * np.pi, 4 * np.pi]
        bounds = (lo, hi)

    result = least_squares(
        residual_vector, x0=np.asarray(theta0, dtype=float),
        args=(observations, propagator, c),
        method="trf", jac="2-point", bounds=bounds,
        xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=4000,
    )

    # J = result.jac  # (2N, 6)
    # dof = len(result.residuals) - len(theta0)
    # FIM = J.T @ J
    # covariance = np.linalg.pinv(FIM)

    J = result.jac
    dof = len(result.fun) - len(theta0)
    s_squared = 2.0 * result.cost / dof  # ≈ reduced chi-square
    covariance = np.linalg.inv(J.T @ J) * s_squared

    n_obs = len(observations)
    reduced_chi2 = float(2 * result.cost / max(2 * n_obs - 6, 1))

    return OrbitFitResult(
        theta=result.x,
        epoch=propagator.epoch,
        covariance=covariance,
        cost=float(result.cost),
        n_obs=n_obs,
        success=bool(result.success),
        huber_c=c,
        reduced_chi2=reduced_chi2,
    )
