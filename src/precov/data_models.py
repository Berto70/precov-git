"""
Shared data structures.

These are intentionally thin dataclasses rather than a heavier ORM-like
layer: the pipeline is a batch numerical computation over a few thousand
images at most per object, not a persistence-heavy system, so plain
dataclasses plus numpy arrays are the right level of ceremony.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from astropy.time import Time


@dataclass
class Observation:
    """A single angular astrometric measurement (one epoch, one RA/Dec pair).

    Parameters
    ----------
    t : astropy.time.Time
        Epoch of observation.
    ra, dec : float
        Right ascension / declination, in radians.
    sigma_ra, sigma_dec : float
        1-sigma astrometric uncertainty, in radians. Correlations between
        RA and Dec are not modeled, matching the paper (Sec. 3.1).
    robust : bool
        If True, this observation participates in the Huber-downweighted
        loss (postdiscovery MPC data, Eq. 2). If False, it is included with
        ordinary quadratic loss (an appended trial/candidate source, Eq. 3),
        since survey-imaging astrometry is not expected to carry the same
        outlier-prone systematics as heterogeneous MPC submissions.
    observer_location : astropy.coordinates.EarthLocation, optional
        Topocentric observer site. If None, geocentric is assumed.
    """

    t: Time
    ra: float
    dec: float
    sigma_ra: float
    sigma_dec: float
    robust: bool = True
    observer_location: Optional[object] = None


@dataclass
class Geometry:
    """Full geometric state of a propagated orbit at one epoch, as seen from
    one observer. This is the superset of what the pipeline needs: the
    astrometric RA/Dec for orbit fitting and Mahalanobis matching, plus the
    heliocentric/topocentric distances and phase angle for the H-G
    photometric viability gate (Sec. 3.2).
    """

    ra: float             # radians, topocentric astrometric (light-time corrected)
    dec: float            # radians
    r_helio: float        # au, heliocentric distance of the body
    r_topo: float         # au, topocentric (observer-body) distance
    phase_angle: float    # radians, Sun-body-observer angle
    light_time: float     # days


@dataclass
class OrbitFitResult:
    """Result of a (Huber-robust or ordinary) least-squares orbit fit."""

    theta: np.ndarray          # (6,) Keplerian elements (a,e,i,raan,argp,M) at `epoch`
    epoch: Time                # reference epoch at which theta is defined
    covariance: np.ndarray     # (6,6) parameter covariance (inverse Fisher info)
    cost: float                # scipy least_squares final cost (0.5 * sum(res^2))
    n_obs: int
    success: bool
    huber_c: Optional[float] = None
    reduced_chi2: Optional[float] = None


@dataclass
class SurveyImage:
    """Metadata describing one survey exposure (or, for a mosaic survey like
    ZTF, one CCD-quadrant sub-image -- the paper processes at quadrant
    granularity to bound memory footprint; see Sec. 5 of the report)."""

    image_id: str
    t: Time
    wcs: object                 # astropy.wcs.WCS
    shape: Tuple[int, int]      # (ny, nx) in pixels
    limiting_mag_3sigma: float
    fwhm_px: float
    sigma_centroid_px: Optional[float] = None  # King (1983) approx.; derived if None
    observer_location: Optional[object] = None


@dataclass
class SourceCatalog:
    """Low-threshold source catalog extracted from one image."""

    image_id: str
    xy: np.ndarray            # (N, 2) pixel coordinates
    flux: np.ndarray          # (N,)
    snr: np.ndarray           # (N,)
    rho: float                # density of sources per unit pixel area (Sec 3.3)
    n_unmasked_pixels: int

    def __len__(self) -> int:
        return len(self.xy)


@dataclass
class SearchRegion:
    """Convex-hull search region for one image, built from the projected
    boundary-orbit ensemble (Sec. 3.2, Algorithm 3)."""

    image_id: str
    hull_vertices_xy: np.ndarray   # (K, 2) hull vertex pixel coordinates, ordered
    all_points_xy: np.ndarray      # (M', 2) all retained (post-buffer-clip) samples
    predicted_mag: float


@dataclass
class CandidateSource:
    """One trial source: a detection inside a search region, to be
    hypothesized as the asteroid (Sec. 3.4)."""

    image_id: str
    source_index: int
    xy: Tuple[float, float]
    ra: float
    dec: float
    sigma_centroid_px: float


@dataclass
class LinkResult:
    """Outcome of refitting + probabilistically linking one candidate source
    (Sec. 3.4-3.6, Algorithm 6)."""

    candidate: CandidateSource
    orbit_fit: OrbitFitResult
    significance: float
    log_likelihood_sum: float
    matches: Dict[str, Tuple[int, float]] = field(default_factory=dict)  # image_id -> (source_idx, r)


@dataclass
class PrediscoveryResult:
    """Top-level pipeline output."""

    postdiscovery_fit: OrbitFitResult
    n_search_images: int
    n_candidate_sources: int
    all_link_results: List[LinkResult]
    confirmed: List[LinkResult]
    joint_fit: Optional[OrbitFitResult] = None
