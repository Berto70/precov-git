"""
Top-level orchestration: `PrediscoveryPipeline.run` reproduces the full
control flow of Sec. 3 end to end.

    postdiscovery observations
        -> Huber-robust orbit fit                          (orbits.fitting)
        -> boundary iso-surface sample of the 6D ellipsoid  (orbits.sampling)
        -> per-image photometric + geometric viability,
           convex-hull search regions                       (images.footprint)
        -> low-threshold source extraction in each region   (images.detection)
        -> per-candidate refit + cross-image linking         (linking.linker)
        -> significance threshold + mutual consensus         (linking.consensus)
        -> joint refit on postdiscovery + confirmed sources
"""
from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
from astropy.time import Time

from .constants import (
    DEFAULT_ALPHA_DETECTION_PRIOR, DEFAULT_BOUNDARY_SAMPLES_M, DEFAULT_CONFIDENCE_Q,
    DEFAULT_HUBER_C, DEFAULT_HULL_BUFFER_PX, DEFAULT_MAG_BUFFER, DEFAULT_DETECTION_SIGMA,
    DEFAULT_SIGNIFICANCE_THRESHOLD,
)
from .data_models import Observation, PrediscoveryResult
from .images.detection import extract_sources
from .images.footprint import compute_search_region, photometric_viability
from .linking.consensus import verify_mutual_consensus
from .linking.linker import candidate_to_observation, link_all_candidates
from .linking.null_distribution import NullMomentLookup
from .orbits.fitting import fit_orbit
from .orbits.propagator import Propagator
from .orbits.sampling import sample_boundary_orbits
from .survey.base import SurveyProvider


class PrediscoveryPipeline:
    def __init__(self, propagator: Propagator, survey: SurveyProvider,
                 H: float, G: float = 0.15,
                 M_samples: int = DEFAULT_BOUNDARY_SAMPLES_M,
                 confidence_q: float = DEFAULT_CONFIDENCE_Q,
                 detection_sigma: float = DEFAULT_DETECTION_SIGMA,
                 hull_buffer_px: float = DEFAULT_HULL_BUFFER_PX,
                 mag_buffer: float = DEFAULT_MAG_BUFFER,
                 alpha: float = DEFAULT_ALPHA_DETECTION_PRIOR,
                 significance_threshold: float = DEFAULT_SIGNIFICANCE_THRESHOLD,
                 huber_c: float = DEFAULT_HUBER_C,
                 rng: Optional[np.random.Generator] = None):
        """
        Parameters
        ----------
        propagator : Propagator
            Must be constructed with its reference epoch already set (e.g.
            the survey midpoint, per Sec. 3.1); reused for every stage.
        survey : SurveyProvider
            Source of candidate images and pixel data.
        H, G : float
            Absolute magnitude and slope parameter for the H-G photometric
            model (Sec. 3.2). Use G=0.15 when unmeasured, per the paper.
        """
        self.propagator = propagator
        self.survey = survey
        self.H, self.G = H, G
        self.M_samples = M_samples
        self.confidence_q = confidence_q
        self.detection_sigma = detection_sigma
        self.hull_buffer_px = hull_buffer_px
        self.mag_buffer = mag_buffer
        self.alpha = alpha
        self.significance_threshold = significance_threshold
        self.huber_c = huber_c
        self.rng = rng or np.random.default_rng()

    def run(self, postdiscovery_obs: List[Observation], theta0_guess: np.ndarray) -> PrediscoveryResult:
        # --- Stage 1: robust postdiscovery orbit fit (Sec. 3.1) ---
        fit = fit_orbit(postdiscovery_obs, self.propagator, theta0_guess, c=self.huber_c)

        # --- Stage 2: boundary iso-surface sample of the 6D ellipsoid (Sec. 3.2) ---
        theta_samples = sample_boundary_orbits(
            fit.theta, fit.covariance, M=self.M_samples, q=self.confidence_q, rng=self.rng,
        )

        # --- Stage 3: per-image viability gating + search regions (Sec. 3.2) ---
        images = {img.image_id: img for img in self.survey.list_images()}
        search_regions = {}
        for image_id, image in images.items():
            viable, V_pred = photometric_viability(
                fit.theta, image.t, self.H, self.G, image.limiting_mag_3sigma,
                self.propagator, buffer_mag=self.mag_buffer,
                observer_location=image.observer_location,
            )
            if not viable:
                continue
            region = compute_search_region(
                theta_samples, image.t, image.wcs, image.shape, self.propagator,
                buffer_px=self.hull_buffer_px, predicted_mag=V_pred,
                image_id=image_id, observer_location=image.observer_location,
            )
            if region is not None:
                search_regions[image_id] = region

        # --- Stage 4: low-threshold source extraction within each region (Sec. 3.3) ---
        catalogs = {}
        for image_id, region in search_regions.items():
            image = images[image_id]
            data = self.survey.get_image_data(image_id)
            catalog = extract_sources(
                data, image.fwhm_px, image_id,
                threshold_sigma=self.detection_sigma, search_region=region,
            )
            if len(catalog) > 0:
                catalogs[image_id] = catalog

        n_candidate_sources = sum(len(c) for c in catalogs.values())
        search_images = {iid: images[iid] for iid in search_regions}

        # --- Stage 5: candidate refit + probabilistic linking (Sec. 3.4-3.5) ---
        null_lookup = NullMomentLookup(alpha=self.alpha)
        all_results = link_all_candidates(
            postdiscovery_obs, self.propagator, fit.theta, search_images, catalogs,
            null_lookup, alpha=self.alpha, huber_c=self.huber_c,
        )

        # --- Stage 6: significance threshold + mutual consensus (Sec. 3.6) ---
        significant = [r for r in all_results if r.significance > self.significance_threshold]
        confirmed = verify_mutual_consensus(significant)

        # --- Stage 7: joint refit on postdiscovery + confirmed prediscoveries ---
        joint_fit = None
        if confirmed:
            candidate_obs = [
                candidate_to_observation(r.candidate, images[r.candidate.image_id])
                for r in confirmed
            ]
            joint_fit = fit_orbit(
                postdiscovery_obs + candidate_obs, self.propagator, fit.theta, c=self.huber_c,
            )

        return PrediscoveryResult(
            postdiscovery_fit=fit, n_search_images=len(search_regions),
            n_candidate_sources=n_candidate_sources, all_link_results=all_results,
            confirmed=confirmed, joint_fit=joint_fit,
        )
