"""
Survey-agnostic image provider interface.

The paper explicitly frames the method as "survey agnostic (though it
requires access to survey images or low-threshold source catalogs)" (Sec. 1).
We formalize that as a two-method interface: enumerate available image
metadata cheaply (`list_images`, no pixel data touched), then fetch pixel
data lazily and only for images that survive the photometric/geometric
viability gate (`get_image_data`). A production implementation of this
interface for, e.g., ZTF would page through IRSA's difference-image archive
at CCD-quadrant granularity (matching the paper's memory-footprint design,
Sec. 5) rather than loading full focal-plane mosaics.
"""
from __future__ import annotations

import abc
from typing import List

import numpy as np

from ..data_models import SurveyImage


class SurveyProvider(abc.ABC):
    @abc.abstractmethod
    def list_images(self) -> List[SurveyImage]:
        """Return metadata (epoch, WCS, shape, limiting magnitude, PSF FWHM)
        for every image this provider can serve, without loading pixel data."""

    @abc.abstractmethod
    def get_image_data(self, image_id: str) -> np.ndarray:
        """Return the (ny, nx) pixel array for one image, by id."""
