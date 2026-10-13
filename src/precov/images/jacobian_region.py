"""
Jacobian-based search region construction (Appendix A of the paper).

Replaces the 20,000-sample Monte Carlo convex hull with a 12-integration
finite-difference Jacobian that maps the 6D orbital covariance to a 2D
image-plane covariance ellipse. This is ~1600x faster per image.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from astropy.time import Time
from matplotlib.path import Path
from scipy.stats import chi2

from ..constants import DEFAULT_HULL_BUFFER_PX
from ..data_models import SearchRegion
from ..orbits.propagator import Propagator


def compute_search_region_jacobian(
    theta_hat: np.ndarray,
    covariance: np.ndarray,
    t: Time,
    wcs,
    image_shape: Tuple[int, int],
    propagator: Propagator,
    confidence_q: float = 0.9999,
    buffer_px: float = DEFAULT_HULL_BUFFER_PX,
    predicted_mag: float = np.nan,
    image_id: str = "",
    observer_location=None,
    n_ellipse_points: int = 64,
) -> Optional[SearchRegion]:
    """
    Compute the search region as a 2D covariance ellipse using the
    Jacobian method from Appendix A.
    
    Requires only 12 propagator calls (6 parameters × 2 for central difference)
    instead of 20,000 samples.
    
    Parameters
    ----------
    theta_hat : ndarray, shape (6,)
        Best-fit orbital elements.
    covariance : ndarray, shape (6, 6)
        Orbital element covariance matrix Σθ.
    t : Time
        Image epoch.
    wcs : WCS
        Image World Coordinate System.
    image_shape : tuple (ny, nx)
        Image dimensions in pixels.
    propagator : Propagator
        Propagator to use (should be ASSIST for NEAs).
    confidence_q : float
        Confidence level for the search region (default 99.99%).
    buffer_px : float
        Buffer outside image boundary to include.
    predicted_mag : float
        Predicted apparent magnitude (for metadata).
    image_id : str
        Image identifier.
    observer_location : EarthLocation, optional
        Observer location for topocentric correction.
    n_ellipse_points : int
        Number of points to approximate the ellipse polygon.
    
    Returns
    -------
    SearchRegion or None
        The search region, or None if the best-fit position is too far
        from the image to be relevant.
    """
    ny, nx = image_shape
    
        # --- Step 1: Best-fit position on the image ---
    geom0 = propagator.geometry(theta_hat, t, observer_location)
    x0, y0 = wcs.world_to_pixel_values(np.degrees(geom0.ra), np.degrees(geom0.dec))
    
    # --- Step 2: Eigendecomposition of covariance -> Q matrix ---
    eigvals, eigvecs = np.linalg.eigh(covariance)
    eigvals = np.maximum(eigvals, 1e-30)
    Q = eigvecs @ np.diag(np.sqrt(eigvals))
    
    # --- Step 3: Compute JQ via central finite differences (12 integrations) ---
    JQ = np.zeros((2, 6))
    for i in range(6):
        h = 1.0
        for _ in range(20):
            theta_plus = theta_hat + h * Q[:, i]
            theta_minus = theta_hat - h * Q[:, i]
            geom_plus = propagator.geometry(theta_plus, t, observer_location)
            geom_minus = propagator.geometry(theta_minus, t, observer_location)
            xp, yp = wcs.world_to_pixel_values(np.degrees(geom_plus.ra), np.degrees(geom_plus.dec))
            xm, ym = wcs.world_to_pixel_values(np.degrees(geom_minus.ra), np.degrees(geom_minus.dec))
            dx, dy = xp - xm, yp - ym
            if np.sqrt(dx**2 + dy**2) <= 10.0:
                break
            h /= 2.0
        JQ[0, i] = dx / (2.0 * h)
        JQ[1, i] = dy / (2.0 * h)
    
    # --- Step 4: Image-space covariance ---
    Sigma_x = JQ @ JQ.T
    
    # --- Step 5: Confidence ellipse scale ---
    chi2_val = chi2.ppf(confidence_q, df=6)
    scale = np.sqrt(chi2_val)
    
    eig2d, vecs2d = np.linalg.eigh(Sigma_x)
    eig2d = np.maximum(eig2d, 1e-10)
    a_axis = scale * np.sqrt(eig2d[1])
    b_axis = scale * np.sqrt(eig2d[0])
    
    # CONSERVATIVE quick-rejection: check if the ellipse could possibly
    # overlap the image, not just the best-fit point
    max_extent = max(a_axis, b_axis)
    margin = buffer_px + max_extent + 500  # extra safety margin
    if x0 < -margin or x0 > nx + margin or y0 < -margin or y0 > ny + margin:
        return None
    
    # Rotation angle of the ellipse
    angle = np.arctan2(vecs2d[1, 1], vecs2d[0, 1])
    
    # Generate ellipse polygon
    t_param = np.linspace(0, 2 * np.pi, n_ellipse_points, endpoint=False)
    ellipse_x = a_axis * np.cos(t_param)
    ellipse_y = b_axis * np.sin(t_param)
    
    # Rotate and translate
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    pts_x = x0 + ellipse_x * cos_a - ellipse_y * sin_a
    pts_y = y0 + ellipse_x * sin_a + ellipse_y * cos_a
    
    pts = np.column_stack([pts_x, pts_y])
    
    # --- Step 6: Clip to image boundary + buffer ---
    in_buffer = (
        (pts[:, 0] >= -buffer_px) & (pts[:, 0] <= nx + buffer_px) &
        (pts[:, 1] >= -buffer_px) & (pts[:, 1] <= ny + buffer_px)
    )
    
    # Also check if the center is within the extended boundary
    center_in = (
        -buffer_px <= x0 <= nx + buffer_px and
        -buffer_px <= y0 <= ny + buffer_px
    )
    
    if not center_in and np.sum(in_buffer) < 3:
        return None  # ellipse doesn't intersect the image at all
    
    # Use all ellipse points that are in the buffer, or if center is in,
    # use the full ellipse (clipped later by the hull check)
    if center_in:
        hull_vertices = pts
    else:
        hull_vertices = pts[in_buffer]
    
    if len(hull_vertices) < 3:
        return None
    
    return SearchRegion(
        image_id=image_id,
        hull_vertices_xy=hull_vertices,
        all_points_xy=pts,  # full ellipse for diagnostics
        predicted_mag=predicted_mag,
    )