"""
Two-body Keplerian orbital mechanics.

This module contains no survey-specific or statistical logic -- it is pure
celestial mechanics, kept separate so it can be reasoned about (and tested)
in isolation from the propagator/observer machinery in `propagator.py`.

Element ordering convention used throughout the package (matching Sec. 3.1
of the paper):

    theta = (a, e, i, raan, argp, M)

with a in au, e dimensionless, and (i, raan, argp, M) in radians. `M` is the
mean anomaly at the reference epoch `t0` carried alongside `theta` by the
caller (Keplerian elements alone are not time-tagged).
"""
from __future__ import annotations

import numpy as np

from ..constants import GM_SUN, OBLIQUITY_J2000_RAD

A, E, I, RAAN, ARGP, M = range(6)  # index convenience constants


def solve_kepler_equation(mean_anomaly: np.ndarray, e: float, tol: float = 1e-12,
                           max_iter: int = 50) -> np.ndarray:
    """Solve Kepler's equation M = E - e*sin(E) for the eccentric anomaly E,
    via Newton-Raphson with a robust starting guess.

    Vectorized over `mean_anomaly`; `e` is a scalar (all requested epochs
    share one orbit).
    """
    M_wrapped = np.mod(mean_anomaly + np.pi, 2 * np.pi) - np.pi
    E = M_wrapped + e * np.sin(M_wrapped) * (1.0 + e * np.cos(M_wrapped))  # good starting guess
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M_wrapped
        fp = 1.0 - e * np.cos(E)
        dE = -f / fp
        E = E + dE
        if np.all(np.abs(dE) < tol):
            break
    return E


def kepler_to_state(theta: np.ndarray, dt_days: np.ndarray, frame: str = "equatorial"):
    """Propagate Keplerian elements `theta` forward by `dt_days` (relative to
    the reference epoch at which `theta[M]` is defined) under pure two-body
    heliocentric dynamics, returning heliocentric position vectors.

    Parameters
    ----------
    theta : array_like, shape (6,)
        (a, e, i, raan, argp, M) at the reference epoch, angles in radians,
        `a` in au.
    dt_days : array_like
        Time(s) since the reference epoch, in days. May be a scalar or array.
    frame : {"equatorial", "ecliptic"}
        Output frame. "equatorial" rotates by the mean J2000 obliquity so
        that the result is directly comparable to ICRS-aligned observer
        positions (e.g. from astropy's built-in ephemeris).

    Returns
    -------
    r_helio : ndarray, shape (..., 3)
        Heliocentric position vector(s), in au.
    """
    a, e, i, raan, argp, M0 = theta
    dt_days = np.atleast_1d(np.asarray(dt_days, dtype=float))

    n = np.sqrt(GM_SUN / a**3)  # mean motion, rad/day
    Mt = M0 + n * dt_days

    E = solve_kepler_equation(Mt, e)

    cosE, sinE = np.cos(E), np.sin(E)
    r_mag = a * (1.0 - e * cosE)
    # perifocal-frame position (orbital plane, x toward pericenter)
    x_pf = a * (cosE - e)
    y_pf = a * np.sqrt(max(1.0 - e**2, 0.0)) * sinE
    z_pf = np.zeros_like(x_pf)

    r_pf = np.stack([x_pf, y_pf, z_pf], axis=-1)

    # Rotate perifocal -> ecliptic via the classical 3-1-3 Euler sequence
    # R = Rz(raan) Rx(i) Rz(argp)
    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(i), np.sin(i)
    cw, sw = np.cos(argp), np.sin(argp)

    R = np.array([
        [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
        [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
        [sw * si,                 cw * si,                 ci],
    ])

    r_ecl = r_pf @ R.T  # (..., 3)

    if frame == "ecliptic":
        return r_ecl if r_ecl.shape[0] > 1 else r_ecl[0]

    eps = OBLIQUITY_J2000_RAD
    ce, se = np.cos(eps), np.sin(eps)
    Rx = np.array([[1, 0, 0], [0, ce, -se], [0, se, ce]])
    r_eq = r_ecl @ Rx.T

    return r_eq if r_eq.shape[0] > 1 else r_eq[0]

def kepler_to_state_and_vel(theta: np.ndarray, dt_days: np.ndarray, frame: str = "equatorial"):
    """Propagate Keplerian elements and return BOTH position and velocity.
    
    Returns
    -------
    r_helio : ndarray, shape (..., 3)
        Heliocentric position vector(s), in au.
    v_helio : ndarray, shape (..., 3)
        Heliocentric velocity vector(s), in au/day.
    """
    a, e, i, raan, argp, M0 = theta
    dt_days = np.atleast_1d(np.asarray(dt_days, dtype=float))

    n = np.sqrt(GM_SUN / a**3)  # mean motion, rad/day
    Mt = M0 + n * dt_days

    E = solve_kepler_equation(Mt, e)

    cosE, sinE = np.cos(E), np.sin(E)
    
    # Position in perifocal frame
    x_pf = a * (cosE - e)
    y_pf = a * np.sqrt(max(1.0 - e**2, 0.0)) * sinE
    z_pf = np.zeros_like(x_pf)

    # Velocity in perifocal frame (analytical, exact)
    # v = dr/dt = (dr/dE) * (dE/dt), where dE/dt = n / (1 - e*cos(E))
    Edot = n / (1.0 - e * cosE)  # dE/dt
    vx_pf = -a * sinE * Edot
    vy_pf = a * np.sqrt(max(1.0 - e**2, 0.0)) * cosE * Edot
    vz_pf = np.zeros_like(x_pf)

    r_pf = np.stack([x_pf, y_pf, z_pf], axis=-1)
    v_pf = np.stack([vx_pf, vy_pf, vz_pf], axis=-1)

    # Rotate perifocal -> ecliptic via the classical 3-1-3 Euler sequence
    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(i), np.sin(i)
    cw, sw = np.cos(argp), np.sin(argp)

    R = np.array([
        [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
        [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
        [sw * si,                 cw * si,                 ci],
    ])

    r_ecl = r_pf @ R.T
    v_ecl = v_pf @ R.T

    if frame == "ecliptic":
        r_out = r_ecl if r_ecl.shape[0] > 1 else r_ecl[0]
        v_out = v_ecl if v_ecl.shape[0] > 1 else v_ecl[0]
        return r_out, v_out

    # Rotate ecliptic -> equatorial
    eps = OBLIQUITY_J2000_RAD
    ce, se = np.cos(eps), np.sin(eps)
    Rx = np.array([[1, 0, 0], [0, ce, -se], [0, se, ce]])
    r_eq = r_ecl @ Rx.T
    v_eq = v_ecl @ Rx.T

    r_out = r_eq if r_eq.shape[0] > 1 else r_eq[0]
    v_out = v_eq if v_eq.shape[0] > 1 else v_eq[0]
    return r_out, v_out
