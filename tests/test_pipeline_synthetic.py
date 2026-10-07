"""
End-to-end validation of the full pipeline against synthetic data, mirroring
the paper's own validation logic (Sec. 4.2-4.4, Figs. 5-6): a short-arc
'postdiscovery' fit is propagated back to a handful of archival images; a
real injected source should be recovered with significance well above the
10-sigma threshold and at the correct pixel position in every image, while a
control run with no injected source (only background clutter) should
confirm nothing.
"""
import numpy as np
import pytest
from astropy.time import Time, TimeDelta

from precov.data_models import Observation
from precov.orbits.propagator import TwoBodyKeplerPropagator
from precov.survey.synthetic import SyntheticSurveyProvider, SyntheticImageSpec
from precov.pipeline import PrediscoveryPipeline


THETA_TRUE = np.array([1.65, 0.30, np.radians(10.0), np.radians(80.0), np.radians(30.0), 0.5])
REFERENCE_EPOCH = Time("2023-06-15T00:00:00", scale="tdb")


def _make_postdiscovery_observations(propagator, rng, sigma_arcsec=1.0, n_obs=8, arc_days=30.0):
    sigma_rad = np.radians(sigma_arcsec / 3600.0)
    obs = []
    for dt in np.linspace(0.0, arc_days, n_obs):
        t = REFERENCE_EPOCH + TimeDelta(dt, format="jd")
        geom = propagator.geometry(THETA_TRUE, t)
        ra_noisy = geom.ra + rng.normal(scale=sigma_rad) / np.cos(geom.dec)
        dec_noisy = geom.dec + rng.normal(scale=sigma_rad)
        obs.append(Observation(t=t, ra=ra_noisy, dec=dec_noisy,
                                sigma_ra=sigma_rad, sigma_dec=sigma_rad, robust=True))
    return obs


def _estimate_angular_half_width_arcsec(propagator, postdiscovery_obs, theta0_guess,
                                         epochs, rng, M=250, percentile=90.0, safety_factor=1.4):
    """Pre-pass sizing helper (test-only): fit the postdiscovery orbit, draw
    a boundary-orbit sample, and measure how far those samples spread (in
    angle, relative to the true position) at each requested epoch. Real
    survey imagery covers a large-enough contiguous footprint (or enough
    tiled quadrants) that *some* image contains the true position with room
    around it for a meaningful convex hull; a synthetic test needs to size
    its one image per epoch to actually contain that footprint, rather than
    assuming an arbitrary small field of view. A short-arc orbit genuinely
    projects to a search region tens of thousands of arcseconds across after
    a ~2-year lookback -- this is not a bug, it is the exact regime the paper
    is built for (their own 42-day-arc example spans 11.7 deg^2).
    """
    from precov.orbits.fitting import fit_orbit
    from precov.orbits.sampling import sample_boundary_orbits

    fit = fit_orbit(postdiscovery_obs, propagator, theta0_guess, c=1.0)
    samples = sample_boundary_orbits(fit.theta, fit.covariance, M=M, q=0.9999, rng=rng)

    half_widths = {}
    for t in epochs:
        geom_true = propagator.geometry(THETA_TRUE, t)
        seps = []
        for theta_k in samples:
            g = propagator.geometry(theta_k, t)
            dra = (g.ra - geom_true.ra) * np.cos(geom_true.dec)
            ddec = g.dec - geom_true.dec
            seps.append(np.hypot(dra, ddec))
        seps_arcsec = np.degrees(seps) * 3600.0
        half_widths[t] = float(np.percentile(seps_arcsec, percentile) * safety_factor)
    return half_widths, fit


def _make_archival_image_specs(propagator, postdiscovery_obs, theta0_guess, rng,
                                n_images=6, inject=True, seed_offset=0, shape_px=700):
    lookback_days = np.linspace(550.0, 800.0, n_images)
    epochs = [REFERENCE_EPOCH - TimeDelta(float(dt), format="jd") for dt in lookback_days]

    half_widths, _ = _estimate_angular_half_width_arcsec(
        propagator, postdiscovery_obs, theta0_guess, epochs, rng,
    )

    specs = []
    for k, t in enumerate(epochs):
        geom = propagator.geometry(THETA_TRUE, t)
        half_width_arcsec = half_widths[t]
        pixel_scale_arcsec = max(2.0 * half_width_arcsec / shape_px, 0.5)
        specs.append(SyntheticImageSpec(
            image_id=f"archival_{k}",
            t=t,
            center_ra_deg=np.degrees(geom.ra),
            center_dec_deg=np.degrees(geom.dec),
            shape=(shape_px, shape_px),
            pixel_scale_arcsec=pixel_scale_arcsec,
            fwhm_px=3.0,
            limiting_mag_3sigma=30.0,  # irrelevant here; always photometrically viable
            background_std=10.0,
            n_clutter_sources=10,
            inject_object_peak_snr=15.0 if inject else None,
            rng_seed=1000 + seed_offset + k,
        ))
    return specs


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(2024)


def _run_pipeline(inject: bool, rng):
    propagator = TwoBodyKeplerPropagator(epoch=REFERENCE_EPOCH)
    postdiscovery_obs = _make_postdiscovery_observations(propagator, rng)
    theta0_guess = THETA_TRUE + np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])

    specs = _make_archival_image_specs(propagator, postdiscovery_obs, theta0_guess, rng,
                                        inject=inject, seed_offset=0 if inject else 500)
    provider = SyntheticSurveyProvider(propagator, THETA_TRUE, specs, rng=rng)

    pipeline = PrediscoveryPipeline(
        propagator=propagator, survey=provider, H=18.0, G=0.15,
        M_samples=400, confidence_q=0.9999, detection_sigma=4.0,
        hull_buffer_px=500, mag_buffer=1.0, alpha=0.5,
        significance_threshold=10.0, huber_c=1.0, rng=rng,
    )
    result = pipeline.run(postdiscovery_obs, theta0_guess)
    return result, provider


def test_postdiscovery_fit_recovers_true_orbit(rng):
    propagator = TwoBodyKeplerPropagator(epoch=REFERENCE_EPOCH)
    postdiscovery_obs = _make_postdiscovery_observations(propagator, rng)
    from precov.orbits.fitting import fit_orbit

    theta0_guess = THETA_TRUE + np.array([0.02, -0.02, 0.02, -0.02, 0.02, -0.02])
    fit = fit_orbit(postdiscovery_obs, propagator, theta0_guess, c=1.0)

    assert fit.success
    # A 30-day arc with 1" astrometry won't pin down all 6 elements tightly,
    # but the fit should land close to truth, not at some degenerate optimum.
    assert np.allclose(fit.theta[:2], THETA_TRUE[:2], atol=0.05)  # a, e


def test_pipeline_recovers_injected_source_with_high_significance(rng):
    result, provider = _run_pipeline(inject=True, rng=rng)

    assert result.n_search_images >= 5, "expected the archival images to be geometrically viable"
    assert len(result.confirmed) >= 5, "expected most/all injected detections to be confirmed"

    for link_result in result.confirmed:
        assert link_result.significance > 10.0
        true_xy = provider.true_object_pixel_position(link_result.candidate.image_id)
        recovered_xy = link_result.candidate.xy
        dist_px = np.hypot(true_xy[0] - recovered_xy[0], true_xy[1] - recovered_xy[1])
        assert dist_px < 3.0, (
            f"confirmed candidate in {link_result.candidate.image_id} is "
            f"{dist_px:.2f}px from the injected truth, not a real match"
        )

    assert result.joint_fit is not None
    assert result.joint_fit.success


def test_null_hypothesis_no_confirmed_detections_without_injected_source(rng):
    result, _ = _run_pipeline(inject=False, rng=rng)

    assert len(result.confirmed) == 0

    if result.all_link_results:
        max_sig = max(r.significance for r in result.all_link_results)
        assert max_sig < 10.0, (
            f"a purely-clutter field produced a candidate at {max_sig:.1f}-sigma; "
            "expected the null distribution to stay well below the 10-sigma threshold"
        )
