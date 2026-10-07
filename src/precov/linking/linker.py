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

from typing import Dict, List, Optional, Tuple

import numpy as np
from astropy.wcs.utils import proj_plane_pixel_scales

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
    """
    Calculates the theoretical maximum log-likelihood if all remaining images 
    yield a perfect match (r=0). Used for branch-and-bound early stopping.
    """
    # Max value of lambda(r) is at r=0
    max_single_image_loglik = np.log(
        (1.0 - alpha) + alpha * (1.0 + 1.0 / (2.0 * np.pi * rho_tilde_min))
    )
    return current_loglik + (n_remaining_images * max_single_image_loglik)

def _make_candidate_source(
    image_id: str, 
    source_index: int, 
    xy: np.ndarray, 
    image: SurveyImage, 
    detection_threshold: float = 4.0
) -> CandidateSource:
    """
    Converts a pixel coordinate from a source catalog into a CandidateSource.
    Calculates RA/Dec via the image WCS and the centroid uncertainty using 
    the King (1983) approximation as specified in Sec 3.4 of the paper.
    """
    # 1. Convert pixel (x, y) to sky coordinates (RA, Dec) using Astropy WCS
    # pixel_to_world returns a SkyCoord object; we extract the values in radians
    sky_coord = image.wcs.pixel_to_world(xy[0], xy[1])
    ra = sky_coord.ra.rad
    dec = sky_coord.dec.rad
    
    # 2. Calculate centroid uncertainty (Sec 3.4)
    # Paper: "astrometric uncertainty sigma_m ... to be the standard deviation 
    # of the PSF divided by the signal-to-noise ratio (SNR) threshold"
    # Gaussian PSF standard deviation is approximately FWHM / 2.355
    psf_sigma_px = image.fwhm_px / 2.355
    sigma_centroid_px = psf_sigma_px / detection_threshold
    
    return CandidateSource(
        image_id=image_id,
        source_index=source_index,
        xy=(float(xy[0]), float(xy[1])),
        ra=ra,
        dec=dec,
        sigma_centroid_px=sigma_centroid_px
    )

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


def link_all_candidates(
    postdiscovery_obs: List[Observation], 
    propagator: Propagator, 
    theta_base: np.ndarray,
    search_images: Dict[str, SurveyImage], 
    catalogs: Dict[str, SourceCatalog],
    null_lookup: NullMomentLookup, 
    alpha: float, 
    huber_c: float
) -> List[LinkResult]:
    
    all_results = []
    
    # Pre-calculate the minimum possible rho_tilde across all images for the early stopping bound
    rho_tilde_min = min(cat.rho for cat in catalogs.values()) if catalogs else 1e-6
    significance_threshold_loglik = null_lookup.get_loglik_threshold(10.0) # Assuming 10-sigma threshold

    # We iterate over each candidate in each image
    for image_id, catalog in catalogs.items():
        if len(catalog) == 0: continue
        
        # VECTORIZATION: Extract all candidate coordinates for this image at once
        xy_candidates = catalog.xy 
        n_candidates = len(xy_candidates)
        
        for idx in range(n_candidates):
            candidate_xy = xy_candidates[idx]
            # ... [Convert candidate_xy to RA/Dec to form a CandidateSource object] ...
            candidate = _make_candidate_source(image_id, idx, candidate_xy, search_images[image_id])
            
            # Refit the orbit including this candidate
            trial_obs = postdiscovery_obs + [candidate_to_observation(candidate, search_images[image_id])]
            trial_fit = fit_orbit(trial_obs, propagator, theta_base, c=huber_c)
            
            if not trial_fit.success:
                continue
                
            # Evaluate this trial orbit against all OTHER images
            current_loglik = 0.0
            matches = {}
            remaining_images = [iid for iid in catalogs.keys() if iid != image_id]
            
            for i, other_iid in enumerate(remaining_images):
                other_img = search_images[other_iid]
                other_cat = catalogs[other_iid]
                
                # Propagate trial orbit to the other image
                ra_pred, dec_pred = propagator.propagate(
                    trial_fit.theta, other_img.t, other_img.observer_location
                )
                pred_xy = other_img.wcs.all_world2pix(np.degrees(ra_pred), np.degrees(dec_pred), 0)
                
                # VECTORIZATION: Calculate Mahalanobis distance to ALL sources in the other image at once
                # Assuming other_cat.xy is (N, 2) and we have a 2x2 covariance matrix for the prediction
                dx = other_cat.xy[:, 0] - pred_xy[0]
                dy = other_cat.xy[:, 1] - pred_xy[1]
                
                # Simplified Mahalanobis (assuming diagonal covariance for speed, or use full inverse)
                # r^2 = dx^2/sig_x^2 + dy^2/sig_y^2
                r_squared = (dx**2 / other_img.wcs_pixel_variance[0]) + (dy**2 / other_img.wcs_pixel_variance[1])
                min_r_idx = np.argmin(r_squared)
                min_r = np.sqrt(r_squared[min_r_idx])
                
                # Calculate single image log-likelihood
                rho_tilde = other_cat.rho
                single_loglik = np.log(
                    (1.0 - alpha) + alpha * (1.0 + 1.0 / (2.0 * np.pi * rho_tilde)) * np.exp(-0.5 * min_r**2)
                )
                current_loglik += single_loglik
                matches[other_iid] = (min_r_idx, min_r)
                
                # EARLY STOPPING CHECK
                n_remaining = len(remaining_images) - i - 1
                max_possible = _calculate_max_possible_loglik(
                    current_loglik, n_remaining, alpha, rho_tilde_min
                )
                
                if max_possible < significance_threshold_loglik:
                    break # Abort this trial orbit, it cannot possibly pass the threshold
            
            # If we didn't break early, and the score is high enough, record it
            if current_loglik >= significance_threshold_loglik:
                significance = null_lookup.loglik_to_sigma(current_loglik)
                all_results.append(LinkResult(
                    candidate=candidate,
                    orbit_fit=trial_fit,
                    significance=significance,
                    log_likelihood_sum=current_loglik,
                    matches=matches
                ))

    return all_results