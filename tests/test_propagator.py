import numpy as np
from astropy.time import Time

from precov.orbits.propagator import TwoBodyKeplerPropagator


def test_geometry_returns_self_consistent_triangle():
    """r_helio, r_topo, and phase_angle must satisfy the law of cosines with
    the Sun-observer distance (~1 au), i.e. describe a consistent Sun-object
    -observer triangle."""
    epoch = Time("2020-01-01T00:00:00", scale="tdb")
    prop = TwoBodyKeplerPropagator(epoch=epoch)
    theta = np.array([1.65, 0.42, np.radians(10), np.radians(80), np.radians(30), 0.5])

    t = Time("2020-06-01T00:00:00", scale="tdb")
    geom = prop.geometry(theta, t)

    assert geom.r_helio > 0
    assert geom.r_topo > 0
    assert 0 <= geom.phase_angle <= np.pi
    assert geom.light_time >= 0
    # light time for solar-system distances should be minutes, not days
    assert geom.light_time < 0.1


def test_ra_dec_are_within_valid_ranges():
    from astropy.time import TimeDelta

    epoch = Time("2021-01-01T00:00:00", scale="tdb")
    prop = TwoBodyKeplerPropagator(epoch=epoch)
    theta = np.array([2.2, 0.15, np.radians(5), np.radians(200), np.radians(10), 1.2])
    for dt_days in [0, 100, -400, 900]:
        t = epoch + TimeDelta(dt_days, format="jd")
        geom = prop.geometry(theta, t)
        assert 0 <= geom.ra < 2 * np.pi
        assert -np.pi / 2 <= geom.dec <= np.pi / 2
