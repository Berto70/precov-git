import numpy as np

from precov.orbits.elements import solve_kepler_equation, kepler_to_state
from precov.constants import GM_SUN


def test_kepler_equation_solution_satisfies_keplers_equation():
    rng = np.random.default_rng(0)
    e = 0.6
    M = rng.uniform(-4 * np.pi, 4 * np.pi, size=200)
    E = solve_kepler_equation(M, e)
    M_wrapped = np.mod(M + np.pi, 2 * np.pi) - np.pi
    residual = E - e * np.sin(E) - M_wrapped
    assert np.max(np.abs(residual)) < 1e-10


def test_circular_orbit_stays_at_constant_heliocentric_radius():
    a = 2.0  # au
    theta = np.array([a, 0.0, 0.0, 0.0, 0.0, 0.0])
    dts = np.linspace(0, 500, 20)
    positions = np.array([kepler_to_state(theta, dt, frame="ecliptic") for dt in dts])
    radii = np.linalg.norm(positions, axis=-1)
    assert np.allclose(radii, a, atol=1e-9)


def test_orbital_period_matches_keplers_third_law():
    a = 1.65  # au, e.g. an Aten/Amor-type NEA semimajor axis
    e = 0.3
    theta = np.array([a, e, 0.2, 0.5, 1.0, 0.0])
    period_days = 2 * np.pi * np.sqrt(a**3 / GM_SUN)

    r0 = kepler_to_state(theta, 0.0, frame="ecliptic")
    r_after_one_period = kepler_to_state(theta, period_days, frame="ecliptic")
    assert np.allclose(np.ravel(r0), np.ravel(r_after_one_period), atol=1e-6)


def test_equatorial_rotation_preserves_vector_norm():
    theta = np.array([1.3, 0.2, 0.4, 1.0, 2.0, 0.7])
    r_ecl = np.ravel(kepler_to_state(theta, 10.0, frame="ecliptic"))
    r_eq = np.ravel(kepler_to_state(theta, 10.0, frame="equatorial"))
    assert np.isclose(np.linalg.norm(r_ecl), np.linalg.norm(r_eq))
    # rotation about x-axis only, so the x-component must be unchanged
    assert np.isclose(r_ecl[0], r_eq[0])
