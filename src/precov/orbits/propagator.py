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

    def __init__(self, object: str, epoch: Time, planets_path: str, asteroids_path: Optional[str] = None,
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

        self.object = object
        self.epoch = epoch
        self.light_time_iterations = light_time_iterations
        self._ephem = assist.Ephem(planets_path, asteroids_path)
        self._assist = assist
        self._rebound = rebound

    def geometry(self, theta: Optional[np.ndarray] = None, t: Optional[Time] = None,
                 observer_location: Optional[EarthLocation] = None) -> Geometry:
        """Return the Geometry (RA, Dec, r_helio, r_topo, phase angle) of the
        configured object as seen at time `t`, using full N-body dynamics via
        REBOUND + ASSIST + JPL DE441.

        ``theta`` is accepted for compatibility with :class:`Propagator` but
        is ignored: ASSIST obtains the object's initial state from
        ``self.object`` and the asteroid ephemeris kernel.
        
        Follows the exact control flow of TwoBodyKeplerPropagator.geometry:
        1. Load the object's heliocentric Cartesian state from the kernel
        2. Integrate N-body to target time t
        3. Iterate for light travel time (typically 2-3 iterations)
        4. Compute topocentric position, RA/Dec, phase angle
        """
        import rebound

        if t is None:
            raise TypeError("ASSISTPropagator.geometry() requires t=...")
        
        # Unpack Keplerian elements: (a, e, i, raan, argp, M)
        #a, e, i, raan, argp, M = theta
        
        # --- Step 1: Initial setup of the REBOUND simulation ---
        sim = rebound.Simulation()
        sim.units = ('AU', 'day', 'Msun')

        sim.ri_ias15.dt = 0.1
        sim.ri_ias15.epsilon = 10e-10
        
        # Attach ASSIST extras (provides planetary perturbations from JPL DE441)
        extras = self._assist.Extras(sim, self._ephem)

        # extras.gr_eih_sources = 11 # Turn on general relativistic corrections for all bodies
        # with 2021 DG1 +30 seconds using the corrections
        
        # Set simulation time to the reference epoch
        sim.t = self.epoch.jd - self._ephem.jd_ref # time is always relative to jd_ref

        # sim.ri_ias15.adaptive_mode = 2 # better timestep for close encounters Pham+24
        # prob never ending 

        sim.add(m=1.0)  # Sun at origin
        sim.add(self.object, plane="frame", date=f"JD{self.epoch.jd}")

        """
        # Add massless test particle with the given orbital elements
        # Elements are heliocentric (relative to the Sun, which ASSIST handles)
        sim.add(m=0, a=a, e=e, inc=i, Omega=raan, omega=argp, M=M)
        
        # Move to center-of-mass frame (required for ASSIST to work correctly)
        sim.move_to_com()
        
        # --- Step 2: Integrate to target time (initial pass) ---
        sim.integrate(t.jd)
        """
        
        # Get the test particle's heliocentric Cartesian state
        particle = sim.particles[1]
        
        # --- Step 3: Light-time iteration ---
        # We iteratively refine the emission time: t_emit = t_obs - |r_topo|/c
        # c in AU/day = 173.14463267424034 --> C_AU_PER_DAY
        
        # Initial guess: the asteroid is seen at the observation time t
        t_emit = t.jd - self._ephem.jd_ref
        
        for iteration in range(self.light_time_iterations):
            # Get observer (Earth center) position from ephemeris at t_emit
            # ASSIST ephemeris returns heliocentric state in AU, days
            sun_particle = self._ephem.get_particle("Sun", t_emit)
            earth_particle = self._ephem.get_particle("Earth", t_emit)
            
            # Earth's heliocentric position (AU)
            obs_x = earth_particle.x - sun_particle.x
            obs_y = earth_particle.y - sun_particle.y
            obs_z = earth_particle.z - sun_particle.z
            
            # If observer_location is provided (topocentric), add the offset
            # EarthLocation gives geocentric coordinates; convert to AU
            if observer_location is not None:
                # Get Earth's sidereal time at t_emit to rotate EarthLocation to ICRS
                # For simplicity, use astropy's EarthLocation -> ITRS -> GCRS transform
                absolute_emit_jd = self._ephem.jd_ref + t_emit
                t_emit_time = Time(absolute_emit_jd, format="jd", scale="tdb")
                # Convert EarthLocation to GCRS Cartesian (AU)
                # EarthLocation is geocentric; GCRS gives position in ICRS frame
                from astropy.coordinates import GCRS
                obs_gcrs = observer_location.get_gcrs(t_emit_time)
                obs_cart = obs_gcrs.cartesian
                # obs_cart is in AU relative to Earth center
                obs_x += obs_cart.x.to(u.AU).value
                obs_y += obs_cart.y.to(u.AU).value
                obs_z += obs_cart.z.to(u.AU).value
            
            # curr_time = sim.t
            # step_days = 0.01

            # if t_emit >= curr_time:
            #     direction = 1.0
            # else:
            #     direction = -1.0

            # while abs(t_emit - curr_time) > step_days:
            #     curr_time += direction * step_days
            #     sim.integrate(curr_time)
            
            # Integrate particle to the refined emission time
            sim.integrate(t_emit)
            particle = sim.particles[1]
            
            # Heliocentric position of asteroid (AU)
            ast_x, ast_y, ast_z = particle.x, particle.y, particle.z
            
            # Topocentric vector (asteroid - observer)
            dx = ast_x - obs_x
            dy = ast_y - obs_y
            dz = ast_z - obs_z
            r_topo = np.sqrt(dx**2 + dy**2 + dz**2)
            
            # Refine emission time for next iteration
            t_emit = (t.jd - self._ephem.jd_ref) - r_topo / C_AU_PER_DAY
        
        # --- Step 4: Compute final geometry ---
        # Heliocentric distance
        r_helio = np.sqrt(ast_x**2 + ast_y**2 + ast_z**2)
        
        # Topocentric RA and Dec (radians)
        # RA: atan2(y, x), Dec: atan2(z, sqrt(x^2 + y^2))
        ra = np.arctan2(dy, dx)
        dec = np.arctan2(dz, np.sqrt(dx**2 + dy**2))
        
        # Normalize RA to [0, 2π)
        if ra < 0:
            ra += 2 * np.pi
        
        # --- Step 5: Phase angle ---
        # Phase angle = angle between Sun->asteroid and observer->asteroid
        # Sun->asteroid vector: (ast_x, ast_y, ast_z)
        # Observer->asteroid vector: (dx, dy, dz)
        # cos(phase) = (r_helio · r_topo) / (|r_helio| * |r_topo|)
        sun_to_ast = np.array([ast_x, ast_y, ast_z])
        obs_to_ast = np.array([dx, dy, dz])
        cos_phase = np.dot(sun_to_ast, obs_to_ast) / (r_helio * r_topo)
        # Clamp to [-1, 1] for numerical safety
        cos_phase = np.clip(cos_phase, -1.0, 1.0)
        phase_angle = np.arccos(cos_phase)
        
        return Geometry(
            ra=ra,
            dec=dec,
            r_helio=r_helio,
            r_topo=r_topo,
            phase_angle=phase_angle,
            light_time=r_topo / C_AU_PER_DAY
        )
