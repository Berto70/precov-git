"""
Synthetic survey provider: generates images with astrometrically-correct
injected sources (propagated through the same `Propagator` the pipeline
uses) plus Poisson-distributed background clutter and read noise, for
end-to-end testing and demonstration without requiring a real survey
archive.

Astrometric accuracy of the injected source's position is real (it comes
from actually calling `propagator.geometry` and projecting through the
image's real WCS); photometric calibration is deliberately simplified to a
directly-specified peak SNR per image rather than a fully calibrated zero
point, since the statistical/software machinery being validated does not
depend on getting an absolute photometric zero point right.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from astropy.time import Time
from astropy.wcs import WCS

from ..constants import DEFAULT_DETECTION_SIGMA
from ..data_models import SurveyImage
from ..orbits.propagator import Propagator
from .base import SurveyProvider


def make_tan_wcs(center_ra_deg: float, center_dec_deg: float,
                  shape: Tuple[int, int], pixel_scale_arcsec: float) -> WCS:
    """A minimal gnomonic (TAN) WCS centered on the given coordinates."""
    ny, nx = shape
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [nx / 2.0, ny / 2.0]
    wcs.wcs.cdelt = [-pixel_scale_arcsec / 3600.0, pixel_scale_arcsec / 3600.0]
    wcs.wcs.crval = [center_ra_deg, center_dec_deg]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs


@dataclass
class SyntheticImageSpec:
    image_id: str
    t: Time
    center_ra_deg: float
    center_dec_deg: float
    shape: Tuple[int, int] = (400, 400)
    pixel_scale_arcsec: float = 1.0
    fwhm_px: float = 3.0
    limiting_mag_3sigma: float = 21.0
    background_std: float = 10.0
    n_clutter_sources: int = 15
    inject_object_peak_snr: Optional[float] = None  # None => object not injected
    rng_seed: Optional[int] = None


class SyntheticSurveyProvider(SurveyProvider):
    """Builds a set of synthetic images around a *known* true orbit
    `theta_true` (propagated with `propagator`), for pipeline validation.
    """

    def __init__(self, propagator: Propagator, theta_true: np.ndarray,
                 image_specs: list, rng: Optional[np.random.Generator] = None):
        self.propagator = propagator
        self.theta_true = np.asarray(theta_true, dtype=float)
        self.specs: Dict[str, SyntheticImageSpec] = {s.image_id: s for s in image_specs}
        self._rng = rng or np.random.default_rng()
        self._injected_positions: Dict[str, Tuple[float, float]] = {}
        self._data_cache: Dict[str, np.ndarray] = {}

    def list_images(self):
        images = []
        for spec in self.specs.values():
            wcs = make_tan_wcs(spec.center_ra_deg, spec.center_dec_deg, spec.shape,
                                spec.pixel_scale_arcsec)
            sigma_centroid_px = (spec.fwhm_px / 2.35482) / DEFAULT_DETECTION_SIGMA
            images.append(SurveyImage(
                image_id=spec.image_id, t=spec.t, wcs=wcs, shape=spec.shape,
                limiting_mag_3sigma=spec.limiting_mag_3sigma, fwhm_px=spec.fwhm_px,
                sigma_centroid_px=sigma_centroid_px, observer_location=None,
            ))
        return images

    def true_object_pixel_position(self, image_id: str) -> Tuple[float, float]:
        """Where the true orbit actually projects to, in this image -- useful
        in tests to check that the recovered candidate matches the injected
        truth, not just *some* significant source."""
        if image_id not in self._injected_positions:
            self._realize_image(image_id)
        return self._injected_positions[image_id]

    def get_image_data(self, image_id: str) -> np.ndarray:
        if image_id not in self._data_cache:
            self._realize_image(image_id)
        return self._data_cache[image_id]

    def _realize_image(self, image_id: str):
        spec = self.specs[image_id]
        ny, nx = spec.shape
        rng = np.random.default_rng(spec.rng_seed) if spec.rng_seed is not None else self._rng

        data = rng.normal(loc=0.0, scale=spec.background_std, size=(ny, nx))

        # Poisson-clutter background sources: random amplitude, random position.
        for _ in range(spec.n_clutter_sources):
            cx, cy = rng.uniform(0, nx), rng.uniform(0, ny)
            amp = rng.uniform(4.0, 8.0) * spec.background_std
            self._add_gaussian(data, cx, cy, amp, spec.fwhm_px)

        wcs = make_tan_wcs(spec.center_ra_deg, spec.center_dec_deg, spec.shape,
                            spec.pixel_scale_arcsec)
        geom = self.propagator.geometry(self.theta_true, spec.t)
        x_true, y_true = wcs.world_to_pixel_values(np.degrees(geom.ra), np.degrees(geom.dec))
        self._injected_positions[image_id] = (float(x_true), float(y_true))

        if spec.inject_object_peak_snr is not None and 0 <= x_true < nx and 0 <= y_true < ny:
            amp = spec.inject_object_peak_snr * spec.background_std
            self._add_gaussian(data, x_true, y_true, amp, spec.fwhm_px)

        self._data_cache[image_id] = data

    @staticmethod
    def _add_gaussian(data: np.ndarray, cx: float, cy: float, amplitude: float, fwhm_px: float):
        ny, nx = data.shape
        sigma = fwhm_px / 2.35482
        half = int(np.ceil(5 * sigma))
        x0, x1 = max(int(cx) - half, 0), min(int(cx) + half + 1, nx)
        y0, y1 = max(int(cy) - half, 0), min(int(cy) + half + 1, ny)
        if x1 <= x0 or y1 <= y0:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1]
        data[y0:y1, x0:x1] += amplitude * np.exp(
            -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2)
        )
