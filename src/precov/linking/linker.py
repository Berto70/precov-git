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
from astropy.wcs.utils import proj_plane_pixel_scales
import numpy as np

from ..constants import DEFAULT_ALPHA_DETECTION_PRIOR, DEFAULT_HUBER_C
from ..data_models import CandidateSource, LinkResult, Observation, SourceCatalog, SurveyImage
from ..orbits.fitting import fit_orbit
from ..orbits.propagator import Propagator
from .covariance import ExemplarCovarianceCache
from .likelihood import loglik_ratio_single_image, mahalanobis_nearest, scaled_density
from .null_distribution import NullMomentLookup

def _calculate_max_possible_loglik(
    current_loglik: float, 
    n_remaining_images: int, 
    alpha: float, 
    rho_tilde_min: float
) -> float:
    """Branch-and-bound early stopping logic."""
    max_single_image_loglik = np.log(
        (1.0 - alpha) + alpha * (1.0 + 1.0 / (2.0 * np.pi * rho_tilde_min))
    )
    return current_loglik + (n_remaining_images * max_single_image_loglik)

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
                        huber_c: float = DEFAULT_HUBER_C,
                        significance_threshold: float = 10.0) -> Optional[LinkResult]:
    """Refit and probabilistically link a single candidate source."""
    origin_image = images[candidate.image_id]
    candidate_obs = candidate_to_observation(candidate, origin_image)
    fit = fit_orbit(postdiscovery_obs + [candidate_obs], propagator, theta0_guess, c=huber_c)
    if not fit.success:
        return None

    dest_ids = candidate_images if candidate_images is not None else list(images.keys())
    
    # Pre-calculate minimum rho_tilde for early stopping bound
    # We use the origin image's covariance with every other image to get the tightest bound
    valid_rhos = [
        scaled_density(catalogs[iid].rho, cov_cache.get(candidate.image_id, iid)) 
        for iid in catalogs if iid != candidate.image_id
    ]
    rho_tilde_min = min(valid_rhos) if valid_rhos else 1e-6
    
    significance_threshold_loglik = null_lookup.get_loglik_threshold(significance_threshold)

    Lambda_total, E_total, Var_total = 0.0, 0.0, 0.0
    matches: Dict[str, tuple] = {}

    for i, m_prime in enumerate(dest_ids):
        if m_prime == candidate.image_id:
            continue
        catalog = catalogs.get(m_prime)
        if catalog is None or len(catalog) == 0:
            continue

        dest_image = images[m_prime]
        Sigma_theta = cov_cache.get(candidate.image_id, m_prime)
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

        # --- EARLY STOPPING CHECK ---
        n_remaining = len(dest_ids) - i - 1
        max_possible = _calculate_max_possible_loglik(Lambda_total, n_remaining, alpha, rho_tilde_min)
        
        if max_possible < significance_threshold_loglik:
            # The candidate cannot mathematically reach the threshold. Abort early.
            return None 

    if Var_total <= 0:
        return None

    significance = (Lambda_total - E_total) / np.sqrt(Var_total)

    return LinkResult(candidate=candidate, orbit_fit=fit, significance=significance,
                      log_likelihood_sum=Lambda_total, matches=matches)

def link_all_candidates(
    postdiscovery_obs: List[Observation], 
    propagator: Propagator, 
    theta_base: np.ndarray,
    search_images: Dict[str, SurveyImage], 
    catalogs: Dict[str, SourceCatalog],
    null_lookup: NullMomentLookup, 
    alpha: float, 
    huber_c: float,
    significance_threshold: float = 10.0
) -> List[LinkResult]:
    """Wrapper that loops over all catalogs and calls evaluate_candidate."""
    all_results = []
    cov_cache = ExemplarCovarianceCache(postdiscovery_obs, propagator, theta_base, search_images, huber_c)
    
    for image_id, catalog in catalogs.items():
        for idx, xy in enumerate(catalog.xy):
            # Construct CandidateSource inline
            sky_coord = search_images[image_id].wcs.pixel_to_world(xy[0], xy[1])
            psf_sigma_px = search_images[image_id].fwhm_px / 2.355
            sigma_centroid_px = psf_sigma_px / 4.0 # Assuming 4-sigma detection threshold as per paper
            
            candidate = CandidateSource(
                image_id=image_id,
                source_index=idx,
                xy=(float(xy[0]), float(xy[1])),
                ra=sky_coord.ra.rad,
                dec=sky_coord.dec.rad,
                sigma_centroid_px=sigma_centroid_px
            )
            
            result = evaluate_candidate(
                candidate, postdiscovery_obs, propagator, theta_base,
                search_images, catalogs, cov_cache, null_lookup,
                alpha=alpha, huber_c=huber_c, significance_threshold=significance_threshold
            )
            if result is not None:
                all_results.append(result)
                
    return all_results