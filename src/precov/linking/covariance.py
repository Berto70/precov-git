"""
Fast linearized 6D-parameter -> 2D-pixel covariance mapping (Appendix A).

The naive approach -- propagate a Monte Carlo cloud of orbits sampled from
Sigma_theta to the destination image and take the sample covariance of the
resulting pixel positions -- requires O(thousands) integrations *per
candidate, per destination image*. The paper's shortcut needs exactly
2*6 = 12 integrations *per (origin image, destination image) pair*,
independent of the number of candidate sources, by exploiting a first-order
(linear) expansion of the ephemeris map around a reference orbit:

    x - x0 ~= J (theta - theta0),   J = df/dtheta |_theta0   (2x6 Jacobian)
    Sigma_x = J Sigma_theta J^T

Factor Sigma_theta = Q Q^T (via eigendecomposition, Q = O D^{1/2}); then
Sigma_x = (JQ)(JQ)^T, and each column of JQ -- the pixel-space image of one
principal axis of the parameter uncertainty ellipsoid -- is estimated by a
single central finite difference:

    [JQ]_i ~= [f(theta0 + h*Q_i, t) - f(theta0 - h*Q_i, t)] / (2h)

with h halved whenever the resulting pixel shift exceeds a linearity budget
(`max_shift_px`), to stay inside the locally-linear regime.

Caching strategy (Appendix A, followed over the report's Algorithm 6
pseudocode where they disagree -- see the package-level discussion): the
paper computes Sigma^mm' from an *exemplar candidate's post-appending refit*
(theta_hat_n^m, Sigma_hat_n^m) for one source n in the *origin* image m, and
reuses it for every other candidate source detected in that same image m
when evaluating matches in destination image m'. `ExemplarCovarianceCache`
implements exactly this: it is keyed by origin image id, lazily refits an
exemplar orbit on first use (via a caller-supplied callback, to avoid a
circular dependency on the fitting/linking orchestration layer), and caches
the resulting Sigma^mm' per (origin, destination) pair.
"""

# TODO: The 12-integration first-order Taylor expansion (Jacobian) mapping 6D 
# orbital covariance to 2D image-space covariance breaks down for highly uncertain 
# orbits (MPC U >= 7) or very long look-backs. Implement a Monte Carlo fallback: 
# if the Jacobian condition number is high, or if delta_t > 2 years, automatically 
# sample 1,000 orbits from the covariance ellipsoid and compute the empirical 2D covariance.

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np
from astropy.time import Time

from ..orbits.propagator import Propagator


def compute_image_plane_covariance(theta0: np.ndarray, cov_theta: np.ndarray,
                                    t_dest: Time, propagator: Propagator, wcs_dest,
                                    observer_location=None,
                                    max_shift_px: float = 10.0,
                                    min_h: float = 1e-4) -> np.ndarray:
    """The 12-integration covariance mapping (Algorithm 5 / Appendix A,
    Eq. A1-A3). Returns the (2, 2) pixel-space covariance matrix Sigma_x."""
    eigvals, O = np.linalg.eigh(cov_theta)
    eigvals = np.clip(eigvals, 1e-18, None)
    Q = O @ np.diag(np.sqrt(eigvals))  # (6,6)

    JQ = np.zeros((2, 6))

    def project(theta):
        geom = propagator.geometry(theta, t_dest, observer_location)
        return np.array(wcs_dest.world_to_pixel_values(np.degrees(geom.ra), np.degrees(geom.dec)))

    for i in range(6):
        Qi = Q[:, i]
        h = 1.0
        while True:
            x_plus = project(theta0 + h * Qi)
            x_minus = project(theta0 - h * Qi)
            shift = np.linalg.norm(x_plus - x_minus)
            if shift > max_shift_px and h > min_h:
                h *= 0.5
                continue
            JQ[:, i] = (x_plus - x_minus) / (2.0 * h)
            break

    return JQ @ JQ.T


class ExemplarCovarianceCache:
    """Per-origin-image cache of Sigma^{m,m'} matrices, computed from one
    exemplar candidate's refit orbit and reused for every other candidate
    source detected in the same origin image `m` (Appendix A).

    Parameters
    ----------
    propagator : Propagator
    image_epochs : dict[str, Time]
    image_wcs : dict[str, object]
    refit_exemplar : callable(image_id) -> OrbitFitResult
        Callback that refits the postdiscovery orbit with one representative
        candidate source from `image_id` appended (ordinary quadratic loss,
        per Eq. 3), returning an `OrbitFitResult`. Injected rather than
        imported directly to avoid a circular dependency between the
        covariance-mapping layer and the fitting/orchestration layer.
    observer_locations : dict[str, EarthLocation], optional
    exact : bool
        If True, bypass exemplar caching entirely and require the caller to
        supply a per-candidate (theta, cov) pair directly to `get` via
        `theta_override`/`cov_override` -- used for validating the exemplar
        approximation against per-candidate ground truth in tests.
    """

    def __init__(self, propagator: Propagator, image_epochs: Dict[str, Time],
                 image_wcs: Dict[str, object],
                 refit_exemplar: Callable[[str], object],
                 observer_locations: Optional[Dict[str, object]] = None,
                 max_shift_px: float = 10.0):
        self.propagator = propagator
        self.image_epochs = image_epochs
        self.image_wcs = image_wcs
        self.refit_exemplar = refit_exemplar
        self.observer_locations = observer_locations or {}
        self.max_shift_px = max_shift_px

        self._exemplar_fit_cache: Dict[str, object] = {}          # origin_image -> OrbitFitResult
        self._sigma_cache: Dict[Tuple[str, str], np.ndarray] = {}  # (origin, dest) -> Sigma (2,2)

    def _get_exemplar_fit(self, origin_image_id: str):
        if origin_image_id not in self._exemplar_fit_cache:
            self._exemplar_fit_cache[origin_image_id] = self.refit_exemplar(origin_image_id)
        return self._exemplar_fit_cache[origin_image_id]

    def get(self, origin_image_id: str, dest_image_id: str) -> np.ndarray:
        key = (origin_image_id, dest_image_id)
        if key in self._sigma_cache:
            return self._sigma_cache[key]

        exemplar_fit = self._get_exemplar_fit(origin_image_id)
        sigma = compute_image_plane_covariance(
            theta0=exemplar_fit.theta,
            cov_theta=exemplar_fit.covariance,
            t_dest=self.image_epochs[dest_image_id],
            propagator=self.propagator,
            wcs_dest=self.image_wcs[dest_image_id],
            observer_location=self.observer_locations.get(dest_image_id),
            max_shift_px=self.max_shift_px,
        )
        self._sigma_cache[key] = sigma
        return sigma
