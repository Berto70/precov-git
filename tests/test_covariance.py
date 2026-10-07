import numpy as np
from astropy.time import Time

from precov.orbits.propagator import TwoBodyKeplerPropagator
from precov.linking.covariance import compute_image_plane_covariance
from precov.survey.synthetic import make_tan_wcs


def _project(propagator, theta, t, wcs):
    geom = propagator.geometry(theta, t)
    return np.array(wcs.world_to_pixel_values(np.degrees(geom.ra), np.degrees(geom.dec)))


def test_12_integration_matches_monte_carlo_propagation():
    """The whole point of Appendix A's shortcut is that 12 integrations
    reproduce what would otherwise take thousands of Monte Carlo draws. We
    check that directly: sample many orbits from a (small, locally-linear)
    parameter covariance, propagate each one to a destination epoch, and
    compare the empirical pixel-space covariance to the analytic 12-
    integration result.
    """
    rng = np.random.default_rng(7)

    epoch = Time("2019-01-01T00:00:00", scale="tdb")
    propagator = TwoBodyKeplerPropagator(epoch=epoch)
    theta0 = np.array([1.65, 0.30, np.radians(8.0), np.radians(120.0), np.radians(40.0), 0.8])

    # Small covariance so the linear (first-order Taylor) regime is valid --
    # this mirrors a well-refit orbit late in the linking process, not the
    # wide postdiscovery ellipsoid used for the initial search-region sweep.
    sigma_a, sigma_e, sigma_ang = 3e-5, 3e-5, 1e-6  # au, dimensionless, rad
    cov_theta = np.diag([sigma_a**2, sigma_e**2, sigma_ang**2,
                          sigma_ang**2, sigma_ang**2, sigma_ang**2])

    from astropy.time import TimeDelta
    t_dest = epoch + TimeDelta(250.0, format="jd")

    geom0 = propagator.geometry(theta0, t_dest)
    wcs = make_tan_wcs(np.degrees(geom0.ra), np.degrees(geom0.dec), (2000, 2000), 1.0)

    # --- Analytic 12-integration covariance ---
    Sigma_analytic = compute_image_plane_covariance(theta0, cov_theta, t_dest, propagator, wcs)

    # --- Brute-force Monte Carlo ground truth ---
    N = 20000
    L = np.linalg.cholesky(cov_theta)
    z = rng.normal(size=(N, 6))
    theta_samples = theta0[None, :] + z @ L.T

    pixel_positions = np.array([_project(propagator, th, t_dest, wcs) for th in theta_samples])
    Sigma_mc = np.cov(pixel_positions.T)

    rel_diff = np.linalg.norm(Sigma_analytic - Sigma_mc) / np.linalg.norm(Sigma_mc)
    assert rel_diff < 0.15, f"12-integration covariance disagrees with Monte Carlo: rel_diff={rel_diff:.3f}"


def test_covariance_is_symmetric_positive_semidefinite():
    epoch = Time("2019-01-01T00:00:00", scale="tdb")
    propagator = TwoBodyKeplerPropagator(epoch=epoch)
    theta0 = np.array([2.1, 0.15, np.radians(3.0), np.radians(50.0), np.radians(10.0), 0.2])
    cov_theta = np.diag([1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-5]) ** 2

    from astropy.time import TimeDelta
    t_dest = epoch + TimeDelta(100.0, format="jd")
    geom0 = propagator.geometry(theta0, t_dest)
    wcs = make_tan_wcs(np.degrees(geom0.ra), np.degrees(geom0.dec), (1000, 1000), 1.0)

    Sigma = compute_image_plane_covariance(theta0, cov_theta, t_dest, propagator, wcs)
    assert np.allclose(Sigma, Sigma.T)
    eigvals = np.linalg.eigvalsh(Sigma)
    assert np.all(eigvals >= -1e-10)
