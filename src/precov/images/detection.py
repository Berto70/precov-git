"""
Low-threshold source extraction (Sec. 3.3, Algorithm 4).

The paper runs DAOFIND (Stetson 1987), via `photutils.detection.DAOStarFinder`,
at a deliberately permissive 4-sigma threshold -- well below a survey's own
~5-sigma alert-stream cut -- because at this stage recall matters far more
than precision: false positives are handled downstream by the likelihood-
ratio linking step (Sec. 3.4-3.6), not here.

If `photutils` is not installed, we fall back to a simple local-maximum
detector. This is *not* a faithful reproduction of DAOFIND's PSF-matched
detection statistic and should not be used for science; it exists purely so
the rest of the pipeline (which only requires a list of centroids, fluxes,
and a background source density) can be exercised without the optional
dependency.
"""

# TODO (Sec 5.1): Implement streak-matched filtering for fast-moving NEAs. 
# DAOStarFinder suffers from severe trailing losses in 30s exposures. 
# We need to calculate expected trail length/orientation from the trial orbit's 
# proper motion and convolve with a line kernel. If detected as a streak, 
# upgrade CandidateSource to a 4D state (x, y, dx_trail, dy_trail) and add 
# the appropriate additive term to the log-likelihood statistic.

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.convolution import Gaussian2DKernel
from photutils.segmentation import detect_sources
from photutils.detection import DAOStarFinder

from ..constants import DEFAULT_DETECTION_SIGMA
from ..data_models import SourceCatalog
from .footprint import points_in_hull


def _fallback_local_maxima(data_sub: np.ndarray, std: float, threshold_sigma: float,
                            fwhm_px: float, mask: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Minimal local-maximum-above-threshold detector, used only when
    `photutils` is unavailable."""
    from scipy.ndimage import maximum_filter

    size = max(int(round(fwhm_px)), 1)
    local_max = maximum_filter(data_sub, size=size) == data_sub
    above_thresh = data_sub > threshold_sigma * std
    candidates = local_max & above_thresh
    if mask is not None:
        candidates &= ~mask

    ys, xs = np.nonzero(candidates)
    flux = data_sub[ys, xs]
    return np.column_stack([xs.astype(float), ys.astype(float)]), flux


def extract_sources(image_data: np.ndarray, fwhm_px: float, image_id: str,
                     threshold_sigma: float = DEFAULT_DETECTION_SIGMA,
                     mask: Optional[np.ndarray] = None,
                     search_region=None) -> SourceCatalog:
    """Run low-threshold source detection on `image_data`, optionally
    restricting the returned catalog to sources inside `search_region`'s
    convex hull.
    """

    # --- NaN/inf handling (must come BEFORE sigma_clipped_stats) ---
    valid_mask = np.isfinite(image_data)
    if not np.all(valid_mask):
        valid_pixels = image_data[valid_mask]
        _nan_fill = np.median(valid_pixels) if len(valid_pixels) > 100 else 0.0
        image_data = np.where(valid_mask, image_data, _nan_fill)

    # --- Background estimation (Sec. 3.3: sigma-clipped stats over full image) ---
    mean_bg, median_bg, std_bg = sigma_clipped_stats(image_data, sigma=3.0, mask=mask)
    data_sub = image_data - median_bg

    # --- Source detection ---
    try:
        from photutils.detection import DAOStarFinder

        finder = DAOStarFinder(threshold=threshold_sigma * std_bg, fwhm=fwhm_px)
        table = finder(data_sub, mask=mask)
        if table is None or len(table) == 0:
            xy = np.zeros((0, 2))
            flux = np.zeros(0)
        else:
            xcol = "x_centroid" if "x_centroid" in table.colnames else "xcentroid"
            ycol = "y_centroid" if "y_centroid" in table.colnames else "ycentroid"
            xy = np.column_stack([table[xcol], table[ycol]])
            flux = np.asarray(table["flux"])
    except ImportError:
        xy, flux = _fallback_local_maxima(data_sub, std_bg, threshold_sigma, fwhm_px, mask)

    n_total_detections = len(xy)

    # --- Background source density rho (Sec. 3.3: whole-image quantity) ---
    if mask is not None:
        n_unmasked = int(np.sum(~mask))
    else:
        n_unmasked = int(image_data.size)

    rho = n_total_detections / max(n_unmasked, 1)

    # --- Restrict to search region (Sec. 3.3) ---
    if search_region is not None and len(xy) > 0:
        keep = points_in_hull(xy, search_region.hull_vertices_xy)
        xy, flux = xy[keep], flux[keep]

    snr = flux / max(std_bg, 1e-12)

    return SourceCatalog(image_id=image_id, xy=xy, flux=flux, snr=snr,
                          rho=rho, n_unmasked_pixels=n_unmasked)
