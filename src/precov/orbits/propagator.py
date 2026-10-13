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
import astropy.units as u

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

    def __init__(self, epoch: Time, planets_path: str, asteroids_path: Optional[str] = None,
                 light_time_iterations: int = 3):
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

        # self.object = object
        self.epoch = epoch
        self.light_time_iterations = light_time_iterations
        self._ephem = assist.Ephem(planets_path, asteroids_path)
        self._assist = assist
        self._rebound = rebound

    def geometry(self, theta: np.ndarray, t: Time,
                observer_location: Optional[EarthLocation] = None) -> Geometry:
            
        import rebound
        from astropy.coordinates import get_body_barycentric
        import astropy.units as u

        a, e, i, raan, argp, M = theta

        # --- Step 1: orbital elements (ecliptic) -> ecliptic Cartesian ---
        temp_sim = rebound.Simulation()
        temp_sim.add(m=1.0)                                   # primary for the conversion only
        temp_sim.add(m=0, a=a, e=e, inc=i, Omega=raan, omega=argp, M=M)
        p = temp_sim.particles[1]

        # --- Step 2: rotate ecliptic -> ICRF (equatorial) by obliquity about x ---
        eps = np.radians(23.439291111)
        ce, se = np.cos(eps), np.sin(eps)
        x  = p.x
        y  = p.y*ce - p.z*se
        z  = p.y*se + p.z*ce
        vx = p.vx
        vy = p.vy*ce - p.vz*se
        vz = p.vy*se + p.vz*ce

        # --- Step 3: main simulation, now fully in ICRF ---
        sim = rebound.Simulation()
        sim.units = ('AU', 'day', 'Msun')
        sim.t = self.epoch.jd - self._ephem.jd_ref

        sim.add(m=1.0)                                  # Sun (see note below)
        # extras = self._assist.Extras(sim, self._ephem)  # comment out to test pure two-body

        sim.add(m=0, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz)   # ICRF heliocentric state

        # --- Step 4: light-time iteration (unchanged) ---
        c_au_per_day = 173.14463267424034
        t_emit = t.jd - self._ephem.jd_ref
        for _ in range(self.light_time_iterations):
            t_emit_time = Time(t_emit + self._ephem.jd_ref, format='jd', scale='tdb')
            sun_bary   = get_body_barycentric('sun',   t_emit_time)
            earth_bary = get_body_barycentric('earth', t_emit_time)
            obs_x = earth_bary.x.to(u.AU).value - sun_bary.x.to(u.AU).value
            obs_y = earth_bary.y.to(u.AU).value - sun_bary.y.to(u.AU).value
            obs_z = earth_bary.z.to(u.AU).value - sun_bary.z.to(u.AU).value
            if observer_location is not None:
                oc = observer_location.get_gcrs(t_emit_time).cartesian
                obs_x += oc.x.to(u.AU).value; obs_y += oc.y.to(u.AU).value; obs_z += oc.z.to(u.AU).value

            sim.integrate(t_emit)
            particle = sim.particles[1]                 # index 1 (0 = Sun)
            ast_x, ast_y, ast_z = particle.x, particle.y, particle.z
            dx, dy, dz = ast_x-obs_x, ast_y-obs_y, ast_z-obs_z
            r_topo = np.sqrt(dx*dx + dy*dy + dz*dz)
            t_emit = (t.jd - self._ephem.jd_ref) - r_topo / c_au_per_day

        # --- Step 5: geometry (unchanged) ---
        r_helio = np.sqrt(ast_x**2 + ast_y**2 + ast_z**2)
        ra  = np.arctan2(dy, dx);  dec = np.arctan2(dz, np.sqrt(dx*dx + dy*dy))
        if ra < 0: ra += 2*np.pi
        cos_phase = np.clip(np.dot([ast_x,ast_y,ast_z],[dx,dy,dz])/(r_helio*r_topo), -1, 1)
        return Geometry(ra=ra, dec=dec, r_helio=r_helio, r_topo=r_topo,
                        phase_angle=np.arccos(cos_phase), light_time=r_topo/c_au_per_day)