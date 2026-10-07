"""
Demo: run the full precov pipeline on synthetic data.

This mirrors the structure of the paper's own validation (Sec. 4): a
short-arc 'postdiscovery' orbit is fit, propagated back several years into
synthetic archival imagery containing a known injected source plus
background clutter, and the pipeline is asked to recover it -- reporting the
same kind of summary as the paper's Table (arc extension, images linked,
peak significance).

Run with:  python examples/demo_synthetic_recovery.py
"""
from __future__ import annotations

import numpy as np
from astropy.time import Time, TimeDelta

from precov.data_models import Observation
from precov.orbits.fitting import fit_orbit
from precov.orbits.propagator import TwoBodyKeplerPropagator
from precov.orbits.sampling import sample_boundary_orbits
from precov.pipeline import PrediscoveryPipeline
from precov.survey.synthetic import SyntheticSurveyProvider, SyntheticImageSpec


# A moderately eccentric, Mars-crossing-NEA-like orbit. Elements are
# (a [au], e, i, raan, argp, M), angles in radians, defined at REFERENCE_EPOCH.
THETA_TRUE = np.array([1.65, 0.30, np.radians(10.0), np.radians(80.0), np.radians(30.0), 0.5])
REFERENCE_EPOCH = Time("2023-06-15T00:00:00", scale="tdb")


def make_postdiscovery_observations(propagator, rng, sigma_arcsec=1.0, n_obs=8, arc_days=30.0):
    """A short, freshly-discovered arc: exactly the short-arc / high-U regime
    the paper is built for (Sec. 1)."""
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


def size_archival_images(propagator, postdiscovery_obs, theta0_guess, rng,
                          n_images=6, shape_px=700):
    """Build archival image metadata sized to actually contain the projected
    search-region footprint at each epoch, rather than assuming an arbitrary
    field of view -- see tests/test_pipeline_synthetic.py for why this
    matters (a 30-day arc genuinely projects to a search region tens of
    thousands of arcseconds wide after a ~2 year lookback; this is the exact
    regime the paper targets, not a pathological case)."""
    fit = fit_orbit(postdiscovery_obs, propagator, theta0_guess, c=1.0)
    samples = sample_boundary_orbits(fit.theta, fit.covariance, M=300, q=0.9999, rng=rng)

    lookback_days = np.linspace(550.0, 800.0, n_images)
    epochs = [REFERENCE_EPOCH - TimeDelta(float(dt), format="jd") for dt in lookback_days]

    specs = []
    for k, t in enumerate(epochs):
        geom_true = propagator.geometry(THETA_TRUE, t)
        seps = []
        for theta_k in samples:
            g = propagator.geometry(theta_k, t)
            dra = (g.ra - geom_true.ra) * np.cos(geom_true.dec)
            ddec = g.dec - geom_true.dec
            seps.append(np.hypot(dra, ddec))
        half_width_arcsec = float(np.percentile(np.degrees(seps) * 3600.0, 90) * 1.4)
        pixel_scale_arcsec = max(2.0 * half_width_arcsec / shape_px, 0.5)

        specs.append(SyntheticImageSpec(
            image_id=f"archival_{k}", t=t,
            center_ra_deg=np.degrees(geom_true.ra), center_dec_deg=np.degrees(geom_true.dec),
            shape=(shape_px, shape_px), pixel_scale_arcsec=pixel_scale_arcsec,
            fwhm_px=3.0, limiting_mag_3sigma=30.0, background_std=10.0,
            n_clutter_sources=10, inject_object_peak_snr=15.0, rng_seed=100 + k,
        ))
    return specs, fit


def main():
    rng = np.random.default_rng(42)
    propagator = TwoBodyKeplerPropagator(epoch=REFERENCE_EPOCH)

    print("=" * 72)
    print("precov demo: synthetic short-arc prediscovery recovery")
    print("=" * 72)

    postdiscovery_obs = make_postdiscovery_observations(propagator, rng)
    arc_days = (postdiscovery_obs[-1].t - postdiscovery_obs[0].t).jd
    print(f"\nPostdiscovery arc: {len(postdiscovery_obs)} observations over {arc_days:.1f} days")

    theta0_guess = THETA_TRUE + 0.01
    specs, initial_fit = size_archival_images(propagator, postdiscovery_obs, theta0_guess, rng)
    print(f"Initial fit: a={initial_fit.theta[0]:.4f} au, e={initial_fit.theta[1]:.4f} "
          f"(truth: a={THETA_TRUE[0]:.4f}, e={THETA_TRUE[1]:.4f})")

    provider = SyntheticSurveyProvider(propagator, THETA_TRUE, specs, rng=rng)

    pipeline = PrediscoveryPipeline(
        propagator=propagator, survey=provider, H=18.0, G=0.15,
        M_samples=400, huber_c=1.0, rng=rng,
    )
    result = pipeline.run(postdiscovery_obs, theta0_guess)

    print(f"\nSearch images with a valid viable footprint: {result.n_search_images}")
    print(f"Total candidate sources examined:              {result.n_candidate_sources}")
    print(f"Trial orbits evaluated:                        {len(result.all_link_results)}")
    print(f"Confirmed prediscoveries (mutual consensus):   {len(result.confirmed)}")

    if result.confirmed:
        peak = max(result.confirmed, key=lambda r: r.significance)
        earliest = min(provider.specs[r.candidate.image_id].t for r in result.confirmed)
        arc_extension_days = (postdiscovery_obs[0].t - earliest).jd
        print(f"\nPeak significance:      {peak.significance:.1f} sigma "
              f"(image {peak.candidate.image_id})")
        print(f"Arc extended back by:   {arc_extension_days:.0f} days "
              f"(factor ~{arc_extension_days / arc_days:.1f}x)")
        if result.joint_fit is not None:
            print(f"Joint refit reduced chi^2: {result.joint_fit.reduced_chi2:.2f}")
    else:
        print("\nNo confirmed prediscoveries (expected for a null/no-injection run).")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
