"""
Per-image Mahalanobis matching and the log-likelihood ratio (Eq. 5-8).

Under H0 (candidate is spurious), catalog sources in the destination image
are a homogeneous 2D Poisson process; the nearest-neighbor Mahalanobis
distance r is Rayleigh-distributed with a density-dependent scale (Appendix
B, Eq. B2). Under H1 (candidate is the real object), the true position is a
2D Gaussian around the orbit's prediction, mixed with the same Poisson
background at rate `alpha` (the prior probability of detection in this
image) to account for tumbling/SNR marginality (Eq. B3). The ratio f1/f0
gives the evidence contributed by this one image (Eq. 8/B4).

Derivation note
----------------
Both source documents render Eq. (8) ambiguously enough that we do not trust
either transcription; we derived it symbolically (SymPy) directly from the
primitive f0(r) (Eq. B2) and f1(r) (Eq. B3) definitions:

    f0(r) = 2*pi*rho_tilde * r * exp(-pi*rho_tilde*r^2)
    f1(r) = (1-alpha)*f0(r) + alpha*[r*exp(-r^2/2)*exp(-pi*rho_tilde*r^2)
                                       + f0(r)*exp(-r^2/2)]

which simplifies -- the exp(-pi*rho_tilde*r^2) factors cancel exactly -- to
the clean, single-exponential closed form

    f1(r)/f0(r) = (1 - alpha) + alpha*(1 + 1/(2*pi*rho_tilde))*exp(-r^2/2)

This *does* match the paper's own printed Eq. (B4)/(B5) once read as a
two-term expression; it is the derived report's rendering of Eq. (8) as a
three-term sum of *two different* exponentials that does not survive
substitution back into f0/f1 and was rejected (see
tests/test_likelihood.py::test_loglik_ratio_negative_for_distant_or_crowded_case,
which caught the discrepancy empirically before we re-derived symbolically).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.special import logsumexp

from ..constants import DEFAULT_ALPHA_DETECTION_PRIOR


def scaled_density(rho: float, S: np.ndarray) -> float:
    """rho_tilde = rho * sqrt(det(S)) (Eq. after B1): background source
    density transformed into the whitened (isotropic) coordinate frame of
    the total positional covariance S."""
    return float(rho * np.sqrt(max(np.linalg.det(S), 0.0)))


def mahalanobis_nearest(pred_xy: np.ndarray, S: np.ndarray,
                         catalog_xy: np.ndarray) -> Tuple[float, int]:
    """Smallest Mahalanobis distance from `pred_xy` to any row of
    `catalog_xy`, under covariance S, and the index of that nearest source
    (Eq. 7). Returns (inf, -1) if the catalog is empty."""
    if len(catalog_xy) == 0:
        return np.inf, -1
    S_inv = np.linalg.inv(S)
    diffs = catalog_xy - np.asarray(pred_xy)
    d2 = np.einsum("ni,ij,nj->n", diffs, S_inv, diffs)
    idx = int(np.argmin(d2))
    return float(np.sqrt(max(d2[idx], 0.0))), idx


def loglik_ratio_single_image(r: float, rho_tilde: float,
                               alpha: float = DEFAULT_ALPHA_DETECTION_PRIOR) -> float:
    """lambda_mm'(r) = log[ f1(r) / f0(r) ] (Eq. 8/B4):

        log[ (1 - alpha) + alpha * (1 + 1/(2*pi*rho_tilde)) * exp(-r^2 / 2) ]

    At r=0 with a sparse background (rho_tilde << 1/(2*pi)) this is large and
    positive (strong evidence); as r -> infinity it approaches log(1-alpha)
    < 0 (no support for the alternative); in a very crowded field
    (rho_tilde >> 1/(2*pi)) the 1/(2*pi*rho_tilde) term vanishes and lambda
    is squeezed toward 0 even at r=0, reflecting that a nearby match carries
    little discriminating power when background sources are dense.

    If no catalog source exists in the destination image (r = inf, as
    signalled by `mahalanobis_nearest`), the candidate is unconstrained by
    that image and contributes nothing (returns 0.0) rather than log(1-alpha).
    """
    if not np.isfinite(r):
        return 0.0

    log_term1 = np.log(1.0 - alpha)
    log_term2 = np.log(alpha) + np.log1p(1.0 / (2.0 * np.pi * rho_tilde)) - 0.5 * r**2

    return float(logsumexp([log_term1, log_term2]))
