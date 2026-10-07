"""
Candidate refitting + cross-image probabilistic linking (Algorithm 6).

For every candidate source x_n^m in every search-region catalog, we:

  1. Append its (RA, Dec) to the postdiscovery observation set with ordinary
     (non-Huber) quadratic loss (Eq. 3) and refit -> (theta_hat_n^m, Sigma_n^m).
  2. Propagate the refit orbit to every other candidate image m' and find the
     nearest catalog source under the Mahalanobis metric built from the
     (exemplar-cached) propagated covariance plus that image's centroid
     error (Eq. 5-7).
  3. Sum the per-image log-likelihood ratios (Eq. 8) into Lambda_n^m, and
     standardize against the precomputed null mean/variance (Eq. 9, B6) to
     get a significance in sigma units.

This module intentionally does not decide what counts as "confirmed" --
thresholding is the caller's business (see `precov.pipeline`), and the
self-consistency check lives in `consensus.py` since it operates on the
whole collection of significant results at once.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from astropy.wcs.utils import proj_plane_pixel_scales

from ..constants import DEFAULT_ALPHA_DETECTION_PRIOR, DEFAULT_HUBER_C
from ..data_models import CandidateSource, LinkResult, Observation, SourceCatalog, SurveyImage
from ..orbits.fitting import fit_orbit
from ..orbits.propagator import Propagator
from .covariance import ExemplarCovarianceCache
from .likelihood import loglik_ratio_single_image, mahalanobis_nearest, scaled_density
from .null_distribution import NullMomentLookup


def _pixel_scale_rad(wcs) -> float:
    """Mean pixel scale in radians/pixel, used to convert a pixel-space
    centroid uncertainty into an angular one for the orbit-fit residuals."""
    scales_deg = proj_plane_pixel_scales(wcs)
    return float(np.mean(np.radians(scales_deg)))


def build_candidates_from_catalog(catalog: SourceCatalog, image: SurveyImage) -> List[CandidateSource]:
    """Convert a pixel-space SourceCatalog into sky-plane CandidateSources,
    carrying each image's (shared) centroid uncertainty (Sec. 3.4, King 1983
    approximation, sigma_PSF / SNR_threshold)."""
    sigma_px = image.sigma_centroid_px
    candidates = []
    for idx in range(len(catalog)):
        x, y = catalog.xy[idx]
        ra_deg, dec_deg = image.wcs.pixel_to_world_values(x, y)
        candidates.append(CandidateSource(
            image_id=image.image_id, source_index=idx, xy=(float(x), float(y)),
            ra=float(np.radians(ra_deg)), dec=float(np.radians(dec_deg)),
            sigma_centroid_px=sigma_px,
        ))
    return candidates


def candidate_to_observation(candidate: CandidateSource, image: SurveyImage) -> Observation:
    sigma_rad = candidate.sigma_centroid_px * _pixel_scale_rad(image.wcs)
    return Observation(t=image.t, ra=candidate.ra, dec=candidate.dec,
                        sigma_ra=sigma_rad, sigma_dec=sigma_rad,
                        robust=False, observer_location=image.observer_location)


def evaluate_candidate(candidate: CandidateSource, postdiscovery_obs: List[Observation],
                        propagator: Propagator, theta0_guess: np.ndarray,
                        images: Dict[str, SurveyImage], catalogs: Dict[str, SourceCatalog],
                        cov_cache: ExemplarCovarianceCache, null_lookup: NullMomentLookup,
                        candidate_images: Optional[List[str]] = None,
                        alpha: float = DEFAULT_ALPHA_DETECTION_PRIOR,
                        huber_c: float = DEFAULT_HUBER_C) -> Optional[LinkResult]:
    """Refit and probabilistically link a single candidate source. Returns
    None if the refit fails to converge.

    `candidate_images` restricts the destination-image loop (normally: every
    image that had a nonempty search region for this object); if omitted,
    every key of `images` other than the candidate's own is used.
    """
    origin_image = images[candidate.image_id]
    candidate_obs = candidate_to_observation(candidate, origin_image)
    fit = fit_orbit(postdiscovery_obs + [candidate_obs], propagator, theta0_guess, c=huber_c)
    if not fit.success:
        return None

    dest_ids = candidate_images if candidate_images is not None else list(images.keys())

    Lambda_total, E_total, Var_total = 0.0, 0.0, 0.0
    matches: Dict[str, tuple] = {}

    for m_prime in dest_ids:
        if m_prime == candidate.image_id:
            continue
        catalog = catalogs.get(m_prime)
        if catalog is None or len(catalog) == 0:
            continue

        dest_image = images[m_prime]
        Sigma_theta = cov_cache.get(candidate.image_id, m_prime)
        # Total covariance S = Sigma_mm' (orbital, in *pixel* units -- built
        # via wcs.world_to_pixel_values inside compute_image_plane_covariance)
        # + sigma_{m'}^2 * I2 (Eq. 6). The destination image's centroid
        # uncertainty is already stored in pixel units (King 1983 approx.,
        # Sec. 3.3), so no unit conversion is needed here.
        sigma_c_px_sq = dest_image.sigma_centroid_px ** 2
        S = Sigma_theta + sigma_c_px_sq * np.eye(2)

        geom = propagator.geometry(fit.theta, dest_image.t, dest_image.observer_location)
        pred_xy = np.array(dest_image.wcs.world_to_pixel_values(np.degrees(geom.ra), np.degrees(geom.dec)))

        r, idx = mahalanobis_nearest(pred_xy, S, catalog.xy)
        rho_tilde = scaled_density(catalog.rho, S)

        lam = loglik_ratio_single_image(r, rho_tilde, alpha)
        E_lam, Var_lam = null_lookup.moments(rho_tilde)

        Lambda_total += lam
        E_total += E_lam
        Var_total += Var_lam
        if idx >= 0:
            matches[m_prime] = (idx, r)

    if Var_total <= 0:
        return None

    significance = (Lambda_total - E_total) / np.sqrt(Var_total)

    return LinkResult(candidate=candidate, orbit_fit=fit, significance=significance,
                       log_likelihood_sum=Lambda_total, matches=matches)


def link_all_candidates(postdiscovery_obs: List[Observation], propagator: Propagator,
                         theta0_guess: np.ndarray, images: Dict[str, SurveyImage],
                         catalogs: Dict[str, SourceCatalog], null_lookup: NullMomentLookup,
                         alpha: float = DEFAULT_ALPHA_DETECTION_PRIOR,
                         huber_c: float = DEFAULT_HUBER_C) -> List[LinkResult]:
    """Evaluate every candidate source in every image's catalog (Algorithm 6,
    outer loop). Builds one `ExemplarCovarianceCache` shared across all
    candidates so that, per Appendix A, the (origin, destination) image-pair
    covariance is computed once and reused."""

    image_epochs = {iid: img.t for iid, img in images.items()}
    image_wcs = {iid: img.wcs for iid, img in images.items()}
    observer_locations = {iid: img.observer_location for iid, img in images.items()}

    all_candidates: Dict[str, List[CandidateSource]] = {
        iid: build_candidates_from_catalog(catalog, images[iid])
        for iid, catalog in catalogs.items()
    }

    def refit_exemplar(origin_image_id: str):
        exemplar = all_candidates[origin_image_id][0]
        obs = candidate_to_observation(exemplar, images[origin_image_id])
        return fit_orbit(postdiscovery_obs + [obs], propagator, theta0_guess, c=huber_c)

    cov_cache = ExemplarCovarianceCache(
        propagator=propagator, image_epochs=image_epochs, image_wcs=image_wcs,
        refit_exemplar=refit_exemplar, observer_locations=observer_locations,
    )

    results: List[LinkResult] = []
    for image_id, candidates in all_candidates.items():
        if not candidates:
            continue
        for candidate in candidates:
            res = evaluate_candidate(
                candidate, postdiscovery_obs, propagator, theta0_guess,
                images, catalogs, cov_cache, null_lookup,
                candidate_images=list(images.keys()), alpha=alpha, huber_c=huber_c,
            )
            if res is not None:
                results.append(res)
    return results
