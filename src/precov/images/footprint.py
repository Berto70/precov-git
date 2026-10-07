"""
Per-image viability gating and search-region construction (Sec. 3.2,
Algorithm 3).

Two filters run before any pixel data is touched:

1. Photometric viability: reject the image outright if the H-G-predicted
   magnitude (at the best-fit orbit) is fainter than the image's 3-sigma
   limiting magnitude plus a softening buffer.
2. Geometric viability: project the *boundary* orbit ensemble (not the
   best-fit orbit) into this image's pixel frame, clip points that land far
   outside the detector (banana-shaped projection artifacts from
   extrapolating a linear-regime ellipsoid over a nonlinear, multi-year
   propagation), and take the convex hull of what remains as the search
   region.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from astropy.time import Time
from matplotlib.path import Path
from scipy.spatial import ConvexHull, QhullError

from ..constants import DEFAULT_HULL_BUFFER_PX, DEFAULT_MAG_BUFFER
from ..data_models import SearchRegion
from ..orbits.propagator import Propagator
from .photometry import hg_apparent_magnitude


def photometric_viability(theta: np.ndarray, t: Time, H: float, G: float,
                           limiting_mag_3sigma: float, propagator: Propagator,
                           buffer_mag: float = DEFAULT_MAG_BUFFER,
                           observer_location=None) -> Tuple[bool, float]:
    """Return (is_viable, predicted_V) for the best-fit orbit at epoch t."""
    geom = propagator.geometry(theta, t, observer_location)
    V_pred = hg_apparent_magnitude(H, G, geom.r_helio, geom.r_topo, geom.phase_angle)
    return (V_pred <= limiting_mag_3sigma + buffer_mag), V_pred


def points_in_hull(points_xy: np.ndarray, hull_vertices_xy: np.ndarray) -> np.ndarray:
    """Boolean mask of which `points_xy` fall inside the polygon defined by
    `hull_vertices_xy` (ordered hull vertices, as returned by
    `compute_search_region`)."""
    if len(hull_vertices_xy) < 3:
        return np.zeros(len(points_xy), dtype=bool)
    path = Path(hull_vertices_xy)
    return path.contains_points(np.atleast_2d(points_xy))


def compute_search_region(theta_samples: np.ndarray, t: Time, wcs, image_shape: Tuple[int, int],
                           propagator: Propagator, buffer_px: float = DEFAULT_HULL_BUFFER_PX,
                           predicted_mag: float = np.nan,
                           image_id: str = "", observer_location=None) -> Optional[SearchRegion]:
    """Project the boundary-orbit ensemble into image pixel coordinates at
    epoch t, clip samples beyond `buffer_px` outside the detector footprint,
    and return the convex hull of the survivors (or None if fewer than 3
    points survive, or if the projected points are numerically degenerate,
    e.g. collinear).
    """
    ny, nx = image_shape
    xs = np.empty(len(theta_samples))
    ys = np.empty(len(theta_samples))
    for k, theta_k in enumerate(theta_samples):
        geom = propagator.geometry(theta_k, t, observer_location)
        x, y = wcs.world_to_pixel_values(np.degrees(geom.ra), np.degrees(geom.dec))
        xs[k], ys[k] = x, y

    in_buffer = (xs >= -buffer_px) & (xs <= nx + buffer_px) & \
                (ys >= -buffer_px) & (ys <= ny + buffer_px)
    pts = np.column_stack([xs[in_buffer], ys[in_buffer]])

    if len(pts) < 3:
        return None

    try:
        hull = ConvexHull(pts)
    except QhullError:
        return None  # degenerate (e.g. collinear) point set

    hull_vertices = pts[hull.vertices]
    return SearchRegion(image_id=image_id, hull_vertices_xy=hull_vertices,
                         all_points_xy=pts, predicted_mag=predicted_mag)
