"""
Null-hypothesis distribution of lambda, and the standardization lookup table
used to convert a summed log-likelihood ratio into a significance in units
of standard deviations (Appendix B.1).

Derivation note on `lambda_of_u`
----------------------------------
Under H0, F0(r) = 1 - exp(-pi*rho_tilde*r^2) is the CDF of the Rayleigh
nearest-neighbor distance, so u := F0(r) ~ Uniform(0,1) and
r^2 = -ln(1-u) / (pi*rho_tilde). Substituting into the closed form derived
in `likelihood.py` (symbolically, directly from Eq. B2/B3),

    lambda(r) = log[ (1-alpha) + alpha*(1 + 1/(2*pi*rho_tilde)) * exp(-r^2/2) ]

gives exp(-r^2/2) = exp(ln(1-u) / (2*pi*rho_tilde)) = (1-u)^{1/(2*pi*rho_tilde)},
so

    lambda(u, rho_tilde) = log[ (1-alpha) + alpha*(1 + 1/(2*pi*rho_tilde)) * (1-u)^{1/(2*pi*rho_tilde)} ]

This is exactly the paper's printed Eq. (B5) read as a two-term expression
(u = F0). It is *not* the three-term, two-different-exponents form that
appears in the derived report -- that form does not survive substitution
back into the primitive f0/f1 definitions and was rejected after a unit test
caught it producing nonsensical (large positive) evidence for a distant
match in a crowded field; see tests/test_likelihood.py.

Because u ~ Uniform(0,1) exactly under H0, E[lambda | rho_tilde] and
Var[lambda | rho_tilde] are plain integrals over u in [0,1] (Eq. B5-B6),
evaluated once on a grid of rho_tilde and interpolated at run time --
this is the "precompute a lookup table" step of Algorithm 6.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.special import logsumexp

from ..constants import DEFAULT_ALPHA_DETECTION_PRIOR

try:
    _trapz = np.trapezoid  # NumPy >= 2.0
except AttributeError:  # pragma: no cover - older NumPy
    _trapz = np.trapz


def lambda_of_u(u: np.ndarray, rho_tilde: float,
                 alpha: float = DEFAULT_ALPHA_DETECTION_PRIOR) -> np.ndarray:
    """lambda as a function of the null-hypothesis uniform variate u = F0(r)
    and the scaled background density rho_tilde (see module docstring)."""
    u = np.asarray(u, dtype=float)
    with np.errstate(divide="ignore"):
        # log1p(-1.0) = -inf is the mathematically correct limit (u -> 1
        # means r -> infinity, where lambda -> log(1-alpha)); harmless.
        log_1mu = np.log1p(-u)

    log_term1 = np.full_like(u, np.log(1.0 - alpha))
    log_term2 = (np.log(alpha) + np.log1p(1.0 / (2.0 * np.pi * rho_tilde))
                 + log_1mu / (2.0 * np.pi * rho_tilde))

    return logsumexp(np.stack([log_term1, log_term2], axis=0), axis=0)


def null_moments(rho_tilde: float, alpha: float = DEFAULT_ALPHA_DETECTION_PRIOR,
                  n_grid: int = 4001) -> Tuple[float, float]:
    """E[lambda | rho_tilde] and Var[lambda | rho_tilde] under H0, by direct
    quadrature over u ~ Uniform(0,1) (Eq. B5-B6)."""
    eps = 1e-10
    u = np.linspace(eps, 1.0 - eps, n_grid)
    lam = lambda_of_u(u, rho_tilde, alpha)
    E = float(_trapz(lam, u))
    E2 = float(_trapz(lam**2, u))
    Var = max(E2 - E**2, 1e-12)
    return E, Var


class NullMomentLookup:
    """Precomputed lookup table of (E[lambda], Var[lambda]) over a grid of
    log10(rho_tilde), interpolated at query time -- avoids re-running
    quadrature for every (candidate, destination image) pair.
    """

    def __init__(self, alpha: float = DEFAULT_ALPHA_DETECTION_PRIOR,
                 log_rho_min: float = -6.0, log_rho_max: float = 2.0,
                 n_grid: int = 500):
        self.alpha = alpha
        log_rho_grid = np.linspace(log_rho_min, log_rho_max, n_grid)
        E_vals = np.empty(n_grid)
        Var_vals = np.empty(n_grid)
        for k, lr in enumerate(log_rho_grid):
            E_vals[k], Var_vals[k] = null_moments(10.0**lr, alpha)

        self._log_rho_grid = log_rho_grid
        self._E_interp = interp1d(log_rho_grid, E_vals, bounds_error=False,
                                   fill_value=(E_vals[0], E_vals[-1]))
        self._Var_interp = interp1d(log_rho_grid, Var_vals, bounds_error=False,
                                     fill_value=(Var_vals[0], Var_vals[-1]))

    def moments(self, rho_tilde: float) -> Tuple[float, float]:
        log_rho = np.log10(max(rho_tilde, 1e-300))
        return float(self._E_interp(log_rho)), float(self._Var_interp(log_rho))
