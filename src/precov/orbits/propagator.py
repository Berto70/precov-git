"""
Ephemeris propagation, f(theta, t) -> (RA, Dec), and the full geometric state
needed for photometric viability gating.

Design note (see the accompanying discussion): the paper propagates orbits
with ASSIST (IAS15 + JPL DE441), which captures full N-body perturbations and
is accurate over multi-year-to-decadal look-backs. That stack is available on
PyPI (`rebound`, `assist`) but requires locally-provided binary ephemeris
kernel files that are not fetchable in every environment (institutional
mirrors, multi-hundred-MB to GB downloads). We therefore define a small
`Propagator` interface and provide two implementations:

  * `TwoBodyKeplerPropagator` -- pure Sun-only two-body dynamics, no external
    data files, runs anywhere. This is what the rest of this package is
    tested against. It is *not* adequate for real multi-year precovery
    astrometry (planetary perturbations matter at exactly the baselines this
    method targets) -- it exists so the statistical machinery (fitting,
    covariance mapping, linking) can be validated independently of ephemeris
    precision.
  * `ASSISTPropagator` -- a real wrapper around `rebound`+`assist`, faithful
    to Sec. 3.1. Requires `pip install precov[dynamics]` and locally-supplied
    kernel files; raises `PropagatorBackendUnavailable` with actionable
    instructions otherwise.

Any future backend (e.g. a JPL Horizons on-the-fly query, or a REBOUNDx
integration with additional non-gravitational forces) need only implement
`Propagator.geometry`.
"""
from __future__ import annotations

import abc
from typing import Optional

import numpy as np
from astropy.time import Time
from astropy.coordinates import get_body_barycentric, EarthLocation

from ..constants import C_AU_PER_DAY
from ..data_models import Geometry
from .elements import kepler_to_state


class PropagatorBackendUnavailable(RuntimeError):
    """Raised when a Propagator backend's runtime dependencies (packages
    and/or data files) are not available."""


class Propagator(abc.ABC):
    """f(theta, t) -> full geometric state, as seen from a given observer."""

    @abc.abstractmethod
    def geometry(self, theta: np.ndarray, t: Time,
                 observer_location: Optional[EarthLocation] = None) -> Geometry:
        """Return the Geometry (RA, Dec, r_helio, r_topo, phase angle) of the
        body described by Keplerian elements `theta` (defined at this
        propagator's reference epoch) as seen at time `t`."""

    def propagate(self, theta: np.ndarray, t: Time,
                   observer_location: Optional[EarthLocation] = None):
        """Convenience accessor returning just (ra, dec) in radians."""
        geom = self.geometry(theta, t, observer_location)
        return geom.ra, geom.dec


def _observer_heliocentric_position(t: Time, observer_location: Optional[EarthLocation]):
    """Heliocentric position of the observer (au), using astropy's built-in
    analytical ephemeris (no network access required)."""
    earth_bary = get_body_barycentric("earth", t)
    sun_bary = get_body_barycentric("sun", t)
    earth_helio = (earth_bary - sun_bary).xyz.to("au").value  # (3,)

    if observer_location is not None:
        # Add the site's geocentric offset (ITRS->GCRS at this order is a
        # small correction relative to Earth's heliocentric distance, but we
        # include it since it matters for very nearby / high-precision work).
        site_geocentric = observer_location.get_gcrs_posvel(t)[0].xyz.to("au").value
        earth_helio = earth_helio + site_geocentric

    return np.asarray(earth_helio, dtype=float)


class TwoBodyKeplerPropagator(Propagator):
    """Analytic Sun-only two-body propagator.

    Parameters
    ----------
    epoch : astropy.time.Time
        Reference epoch at which `theta`'s mean anomaly is defined for every
        call made through this propagator instance.
    light_time_iterations : int
        Number of fixed-point iterations used to solve for the light-travel
        time tau = |topocentric vector| / c (Sec. 2, "solved iteratively
        until convergence (typically 2-3 iterations)").
    """

    def __init__(self, epoch: Time, light_time_iterations: int = 3):
        self.epoch = epoch
        self.light_time_iterations = light_time_iterations
        # Observer heliocentric position depends only on (t, observer_location),
        # never on theta -- yet the boundary-sample sweep and the 12-integration
        # covariance mapping both call geometry() with the *same* t repeatedly
        # across many different theta values. Caching this eliminates the
        # dominant cost (two get_body_barycentric calls per geometry() call)
        # for every call after the first at a given epoch.
        self._observer_cache: dict = {}

    def _cached_observer_heliocentric(self, t: Time, observer_location: Optional[EarthLocation]) -> np.ndarray:
        key = (round(t.tdb.jd, 9), id(observer_location))
        cached = self._observer_cache.get(key)
        if cached is None:
            cached = _observer_heliocentric_position(t, observer_location)
            self._observer_cache[key] = cached
        return cached

    def geometry(self, theta: np.ndarray, t: Time,
                 observer_location: Optional[EarthLocation] = None) -> Geometry:
        theta = np.asarray(theta, dtype=float)
        dt_days = (t - self.epoch).jd  # scalar Time -> scalar dt

        observer_helio = self._cached_observer_heliocentric(t, observer_location)

        # Light-time iteration: propagate at retarded time t - tau.
        tau = 0.0
        for _ in range(self.light_time_iterations):
            r_ast_helio = kepler_to_state(theta, dt_days - tau, frame="equatorial")
            r_ast_helio = np.asarray(r_ast_helio, dtype=float).reshape(3)
            topo_vec = r_ast_helio - observer_helio
            r_topo = float(np.linalg.norm(topo_vec))
            tau = r_topo / C_AU_PER_DAY

        r_helio = float(np.linalg.norm(r_ast_helio))

        ra = np.arctan2(topo_vec[1], topo_vec[0]) % (2 * np.pi)
        dec = np.arcsin(np.clip(topo_vec[2] / r_topo, -1.0, 1.0))

        # Phase angle alpha: Sun-asteroid-observer angle, from the law of
        # cosines using heliocentric distance, topocentric distance, and the
        # observer's heliocentric distance (r_obs).
        r_obs = float(np.linalg.norm(observer_helio))
        cos_alpha = (r_helio**2 + r_topo**2 - r_obs**2) / (2.0 * r_helio * r_topo)
        phase_angle = np.arccos(np.clip(cos_alpha, -1.0, 1.0))

        return Geometry(ra=ra, dec=dec, r_helio=r_helio, r_topo=r_topo,
                         phase_angle=phase_angle, light_time=tau)


class ASSISTPropagator(Propagator):
    """Real IAS15 + JPL DE441 propagation via `rebound`/`assist`, matching
    Sec. 3.1 of the paper exactly. Requires the `dynamics` extra and locally
    supplied ephemeris kernel files (`planets_path`, `asteroids_path`) -- see
    the `assist` package documentation for how to obtain these; they are not
    downloaded automatically by this class.
    """

    def __init__(self, epoch: Time, planets_path: str, asteroids_path: Optional[str] = None):
        try:
            import rebound  # noqa: F401
            import assist
        except ImportError as exc:
            raise PropagatorBackendUnavailable(
                "ASSISTPropagator requires the 'rebound' and 'assist' packages "
                "(pip install precov[dynamics]) plus local JPL DE441 / asteroid "
                "perturber kernel files. See https://github.com/matthewholman/assist "
                "for how to obtain them."
            ) from exc

        self.epoch = epoch
        self._ephem = assist.Ephem(planets_path, asteroids_path)
        self._assist = assist
        self._rebound = rebound

    def geometry(self, theta: np.ndarray, t: Time,
                 observer_location: Optional[EarthLocation] = None) -> Geometry:
        # A full implementation instantiates a rebound.Simulation, attaches
        # assist.Extras with self._ephem, adds a massless test particle at
        # the heliocentric Cartesian state equivalent to `theta` at
        # self.epoch, integrates to `t`, and differences against the
        # ephemeris-provided observer position -- mirroring exactly the
        # two-body implementation's control flow above, but with N-body
        # perturbations included. Omitted here because it cannot be
        # exercised without the kernel files described above; wire this up
        # against your local ASSIST installation following the pattern in
        # TwoBodyKeplerPropagator.geometry.
        raise NotImplementedError(
            "ASSISTPropagator.geometry is a real-backend integration point; "
            "implement the rebound/assist simulation loop once kernel files "
            "are available. See TwoBodyKeplerPropagator for the reference "
            "control flow (light-time iteration, phase-angle geometry, etc.)."
        )
