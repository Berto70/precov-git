import numpy as np
from astropy.time import Time

from precov.images.photometry import hg_apparent_magnitude
from precov.images.footprint import compute_search_region, points_in_hull
from precov.orbits.propagator import TwoBodyKeplerPropagator
from precov.survey.synthetic import make_tan_wcs


def test_hg_magnitude_gets_fainter_with_distance():
    close = hg_apparent_magnitude(H=18.0, G=0.15, r_helio_au=1.5, r_topo_au=0.5, phase_angle_rad=0.3)
    far = hg_apparent_magnitude(H=18.0, G=0.15, r_helio_au=3.0, r_topo_au=2.0, phase_angle_rad=0.3)
    assert far > close  # fainter (larger magnitude) at greater distance


def test_hg_magnitude_brightens_near_opposition_phase_zero():
    low_phase = hg_apparent_magnitude(H=18.0, G=0.15, r_helio_au=1.5, r_topo_au=0.5, phase_angle_rad=0.05)
    high_phase = hg_apparent_magnitude(H=18.0, G=0.15, r_helio_au=1.5, r_topo_au=0.5, phase_angle_rad=1.2)
    assert low_phase < high_phase  # brighter (smaller mag) near zero phase angle


def test_points_in_hull_basic_square():
    hull_vertices = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    points = np.array([[5, 5], [-1, 5], [5, 11], [1, 1]])
    mask = points_in_hull(points, hull_vertices)
    assert list(mask) == [True, False, False, True]


def test_compute_search_region_returns_none_for_insufficient_points():
    epoch = Time("2020-01-01T00:00:00", scale="tdb")
    propagator = TwoBodyKeplerPropagator(epoch=epoch)
    # A single sample orbit can never form a triangle/hull.
    theta_samples = np.array([[1.65, 0.3, 0.1, 0.5, 1.0, 0.5]])
    geom0 = propagator.geometry(theta_samples[0], epoch)
    wcs = make_tan_wcs(np.degrees(geom0.ra), np.degrees(geom0.dec), (500, 500), 1.0)
    region = compute_search_region(theta_samples, epoch, wcs, (500, 500), propagator)
    assert region is None


def test_compute_search_region_contains_the_generating_samples():
    epoch = Time("2020-01-01T00:00:00", scale="tdb")
    propagator = TwoBodyKeplerPropagator(epoch=epoch)
    theta_hat = np.array([1.65, 0.3, 0.1, 0.5, 1.0, 0.5])

    rng = np.random.default_rng(0)
    # Small perturbations around theta_hat -> a compact, well-behaved hull.
    theta_samples = theta_hat[None, :] + rng.normal(scale=1e-5, size=(50, 6))

    geom0 = propagator.geometry(theta_hat, epoch)
    wcs = make_tan_wcs(np.degrees(geom0.ra), np.degrees(geom0.dec), (2000, 2000), 1.0)

    region = compute_search_region(theta_samples, epoch, wcs, (2000, 2000), propagator)
    assert region is not None
    assert len(region.all_points_xy) == 50
    # Samples are tight perturbations of theta_hat, which projects to the
    # image center by construction; the hull's centroid should land there too.
    centroid = region.all_points_xy.mean(axis=0)
    assert np.allclose(centroid, [1000.0, 1000.0], atol=5.0)
