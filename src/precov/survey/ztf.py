"""
ZTF difference-image provider.

This provider bridges the paper's archival ZTF workflow and the rest of the
survey-agnostic pipeline. The paper processes ZTF at CCD-quadrant granularity
and needs, for each image, a timestamp, WCS, shape, 3-sigma limiting
magnitude, PSF FWHM, and optionally the observer location.

The class below can be pointed at a local directory of ZTF difference-image
FITS files or at a CSV manifest that lists local paths or download URLs. It
parses FITS headers lazily, so a large archive can be indexed quickly without
loading pixel data until get_image_data is called.

The provider is intentionally conservative about metadata extraction: it looks
for a variety of common header keywords, but also allows those values to be
supplied explicitly in the manifest when the archive product headers are
incomplete or use a different convention.
"""
from __future__ import annotations

import csv
import io
import os
import time as _time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np
import astropy.units as u
from astropy.coordinates import EarthLocation
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

from ..constants import DEFAULT_DETECTION_SIGMA
from ..data_models import SurveyImage
from .base import SurveyProvider


def _palomar_location() -> EarthLocation:
    return EarthLocation.from_geodetic(lon=-116.8630, lat=33.3563, height=1712.0)


def _first_header_value(header: fits.Header, names: Sequence[str], default=None):
    for name in names:
        if name in header and header[name] not in (None, ""):
            return header[name]
    return default


def _parse_time_from_header(header: fits.Header) -> Time:
    if "OBSJD" in header:  # ZTF: exposure start, JD
        return Time(float(header["OBSJD"]), format="jd", scale="utc")
    if "OBSMJD" in header:
        return Time(float(header["OBSMJD"]), format="mjd", scale="utc")
    if "MJD-OBS" in header:
        return Time(float(header["MJD-OBS"]), format="mjd", scale="utc")
    if "JD" in header:
        return Time(float(header["JD"]), format="jd", scale="utc")
    date_obs = _first_header_value(header, ["DATE-OBS", "DATEOBS", "UTC-OBS"])
    if date_obs is not None:
        return Time(str(date_obs), scale="utc")
    raise ValueError("Could not determine observation time from FITS header")


def _parse_observer_location(header: fits.Header, fallback: Optional[EarthLocation] = None) -> EarthLocation:
    if all(k in header for k in ("OBSGEO-X", "OBSGEO-Y", "OBSGEO-Z")):
        return EarthLocation.from_geocentric(
            float(header["OBSGEO-X"]), float(header["OBSGEO-Y"]), float(header["OBSGEO-Z"]),
            unit="m",
        )

    lon = _first_header_value(header, ["LON-OBS", "LONGITUD", "LONGITUDE", "OBSLONG"])
    lat = _first_header_value(header, ["LAT-OBS", "LATITUDE", "OBSLAT"])
    height = _first_header_value(header, ["ALT-OBS", "HEIGHT", "ELEVATIO", "OBSHGT"], 1712.0)
    if lon is not None and lat is not None:
        return EarthLocation.from_geodetic(float(lon), float(lat), float(height))

    return fallback if fallback is not None else _palomar_location()


def _parse_limiting_mag_3sigma(header: fits.Header) -> float:
    direct = _first_header_value(
        header,
        ["MAGLIM3", "MAGLIM_3SIG", "MAGLIM_3SIGMA", "MAGLIM", "LIMMAG3", "LIMMAG_3SIG", "LIMMAG"],
    )
    if direct is not None:
        return float(direct)

    mag5 = _first_header_value(header, ["MAGLIM5", "MAGLIM_5SIG", "MAGLIM_5SIGMA", "LIMMAG5"])
    if mag5 is not None:
        return float(mag5) + 2.5 * np.log10(5.0 / 3.0)

    raise ValueError(
        "Could not determine a 3-sigma limiting magnitude from FITS header; "
        "provide it explicitly in the manifest or add a MAGLIM/LIMMAG keyword."
    )


def _parse_fwhm_px(header: fits.Header) -> float:
    value = _first_header_value(header, ["FWHM", "SEEING", "FWHM_PX", "SEEING_PX"])
    if value is None:
        raise ValueError(
            "Could not determine PSF FWHM from FITS header; provide it explicitly in the manifest "
            "or add a FWHM/SEEING keyword."
        )
    return float(value)


def _extract_data_shape(data: np.ndarray) -> Tuple[int, int]:
    if data is None:
        raise ValueError("FITS HDU does not contain image data")
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D FITS image, got array with shape {data.shape}")
    ny, nx = data.shape
    return int(ny), int(nx)


_USER_AGENT = "precov/0.1 (+ZTF difference-image provider)"


def _select_image_hdu(hdul: fits.HDUList, data_hdu):
    """Pick the HDU that holds the 2D image.

    IRSA serves difference images as ``*_scimrefdiffimg.fits.fz`` (fpack): the
    primary HDU is *empty* and the pixels live in a CompImageHDU in extension 1.
    With ``data_hdu="auto"`` (default) we take the first HDU with a 2D shape.
    """
    if data_hdu != "auto":
        return hdul[data_hdu]
    for hdu in hdul:
        if isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU)):
            if len(hdu.shape) == 2:
                return hdu
    raise ValueError("No 2D image HDU found in FITS file")


def _merged_header(hdul: fits.HDUList, hdu) -> fits.Header:
    """Image-HDU header layered over the primary header (image keys win)."""
    if hdu is hdul[0]:
        return hdu.header
    header = hdul[0].header.copy()
    for card in hdu.header.cards:
        if card.keyword in ("", "COMMENT", "HISTORY"):
            continue
        header[card.keyword] = (card.value, card.comment)
    return header


def download_file(url: str, dest: Path, timeout: float = 180.0, retries: int = 3) -> Path:
    """Download ``url`` to ``dest`` atomically (``.part`` file + rename).

    A half-finished download never masquerades as a valid cache entry, 4xx
    errors fail immediately with a readable message, and 5xx / network errors
    are retried with backoff.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    last_exc: Optional[BaseException] = None
    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers={"User-Agent": _USER_AGENT})
            with urlopen(req, timeout=timeout) as response, open(part, "wb") as handle:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    handle.write(chunk)
            os.replace(part, dest)
            return dest
        except HTTPError as exc:
            if 400 <= exc.code < 500:
                raise RuntimeError(f"HTTP {exc.code} {exc.reason} for {url}") from exc
            last_exc = exc
        except (URLError, TimeoutError, ConnectionError) as exc:
            last_exc = exc
        if part.exists():
            part.unlink()
        if attempt < retries:
            _time.sleep(2.0 * attempt)
    raise RuntimeError(f"Failed to download {url} after {retries} attempts: {last_exc}")


@dataclass(frozen=True)
class ZTFImageRecord:
    image_id: str
    path: Optional[str] = None
    url: Optional[str] = None
    t: Optional[Time] = None
    limiting_mag_3sigma: Optional[float] = None
    fwhm_px: Optional[float] = None
    sigma_centroid_px: Optional[float] = None
    observer_location: Optional[EarthLocation] = None
    exptime_s: Optional[float] = None  # if set, t is treated as exposure START and shifted to mid-exposure

    def resolve_local_path(self, cache_dir: Optional[Path] = None) -> Path:
        if self.path:
            return Path(self.path).expanduser().resolve()
        if not self.url:
            raise ValueError(f"Record {self.image_id!r} has neither a local path nor a URL")
        if cache_dir is None:
            raise ValueError(f"Record {self.image_id!r} points to a URL but no cache_dir was provided")
        cache_dir = Path(cache_dir).expanduser().resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        filename = os.path.basename(urlparse(self.url).path)
        if not filename:
            filename = f"{self.image_id}.fits"
        local_path = cache_dir / filename
        if not local_path.exists():
            download_file(self.url, local_path)
        return local_path


class ZTFDifferenceImageProvider(SurveyProvider):
    """SurveyProvider for local or cached ZTF difference-image FITS files."""

    def __init__(
        self,
        records: Sequence[ZTFImageRecord],
        cache_dir: Optional[str | Path] = None,
        default_observer_location: Optional[EarthLocation] = None,
        data_hdu: int | str = "auto",
    ):
        self.records: Dict[str, ZTFImageRecord] = {record.image_id: record for record in records}
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir is not None else None
        self.default_observer_location = (
            default_observer_location if default_observer_location is not None else _palomar_location()
        )
        self.data_hdu = data_hdu
        self._path_cache: Dict[str, Path] = {}
        self._image_cache: Dict[str, SurveyImage] = {}
        self._data_cache: Dict[str, np.ndarray] = {}

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        cache_dir: Optional[str | Path] = None,
        default_observer_location: Optional[EarthLocation] = None,
        data_hdu: int | str = "auto",
    ) -> "ZTFDifferenceImageProvider":
        manifest_source = str(manifest_path)
        records: List[ZTFImageRecord] = []
        if manifest_source.startswith(("http://", "https://", "file://")):
            with urlopen(manifest_source) as response:
                handle = io.TextIOWrapper(response, encoding="utf-8", newline="")
                reader = csv.DictReader(handle)
                for row in reader:
                    records.append(cls._record_from_manifest_row(row))
        else:
            manifest_path = Path(manifest_path).expanduser().resolve()
            with open(manifest_path, newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    records.append(cls._record_from_manifest_row(row))
        return cls(records, cache_dir=cache_dir,
                   default_observer_location=default_observer_location, data_hdu=data_hdu)

    @classmethod
    def from_irsa_manifest(
        cls,
        manifest_url: str | Path,
        cache_dir: Optional[str | Path] = None,
        default_observer_location: Optional[EarthLocation] = None,
        data_hdu: int | str = "auto",
    ) -> "ZTFDifferenceImageProvider":
        """Build a provider from an IRSA-hosted or locally saved manifest.

        This is the archive-facing convenience constructor: point it at a
        CSV manifest exported from an IRSA/ZTF query, or at a remote URL that
        serves such a table. Each row should include at least an image
        identifier plus either a local path or a download URL.
        """
        return cls.from_manifest(
            manifest_url,
            cache_dir=cache_dir,
            default_observer_location=default_observer_location,
            data_hdu=data_hdu,
        )

    @classmethod
    def from_irsa_url_list(
        cls,
        urls: Sequence[str],
        image_ids: Optional[Sequence[str]] = None,
        cache_dir: Optional[str | Path] = None,
        default_observer_location: Optional[EarthLocation] = None,
        data_hdu: int | str = "auto",
        limiting_mag_3sigma: Optional[float] = None,
        fwhm_px: Optional[float] = None,
        observer_location: Optional[EarthLocation] = None,
    ) -> "ZTFDifferenceImageProvider":
        """Build a provider directly from a list of archive download URLs.

        This is useful when you already have an IRSA search result page or a
        small set of product URLs and want the provider to fetch the FITS
        files itself while caching them locally.
        """
        records: List[ZTFImageRecord] = []
        for index, url in enumerate(urls):
            image_id = image_ids[index] if image_ids is not None else Path(urlparse(url).path).stem or f"ztf_{index}"
            records.append(ZTFImageRecord(
                image_id=str(image_id),
                url=url,
                limiting_mag_3sigma=limiting_mag_3sigma,
                fwhm_px=fwhm_px,
                observer_location=observer_location,
            ))
        return cls(records, cache_dir=cache_dir,
                   default_observer_location=default_observer_location, data_hdu=data_hdu)

    @classmethod
    def from_directory(
        cls,
        root: str | Path,
        pattern: str = "*.fits*",
        cache_dir: Optional[str | Path] = None,
        default_observer_location: Optional[EarthLocation] = None,
        data_hdu: int | str = "auto",
    ) -> "ZTFDifferenceImageProvider":
        root = Path(root).expanduser().resolve()
        records = [ZTFImageRecord(image_id=path.stem, path=str(path)) for path in sorted(root.rglob(pattern))]
        return cls(records, cache_dir=cache_dir,
                   default_observer_location=default_observer_location, data_hdu=data_hdu)

    @staticmethod
    def _record_from_manifest_row(row: Dict[str, str]) -> ZTFImageRecord:
        image_id = row.get("image_id") or row.get("id") or row.get("filename")
        if not image_id:
            raise ValueError("Manifest must include an image_id, id, or filename column")

        path = row.get("path") or row.get("fits_path") or row.get("local_path")
        url = row.get("url") or row.get("download_url")
        t = Time(row["time"], scale="utc") if row.get("time") else None
        limmag = float(row["limiting_mag_3sigma"]) if row.get("limiting_mag_3sigma") else None
        fwhm_px = float(row["fwhm_px"]) if row.get("fwhm_px") else None
        sigma_centroid_px = float(row["sigma_centroid_px"]) if row.get("sigma_centroid_px") else None
        exptime_s = float(row["exptime"]) if row.get("exptime") else None
        return ZTFImageRecord(
            image_id=str(image_id), path=path, url=url, t=t,
            limiting_mag_3sigma=limmag, fwhm_px=fwhm_px,
            sigma_centroid_px=sigma_centroid_px, exptime_s=exptime_s,
        )

    def _resolve_path(self, image_id: str) -> Path:
        if image_id in self._path_cache:
            return self._path_cache[image_id]
        record = self.records[image_id]
        path = record.resolve_local_path(self.cache_dir)
        self._path_cache[image_id] = path
        return path

    def _inspect_fits(self, image_id: str):
        """Header + shape only: does not decompress pixel data."""
        path = self._resolve_path(image_id)
        with fits.open(path, memmap=False) as hdul:
            hdu = _select_image_hdu(hdul, self.data_hdu)
            header = _merged_header(hdul, hdu).copy()
            shape = tuple(int(n) for n in hdu.shape)
        return shape, header

    def _read_fits(self, image_id: str):
        path = self._resolve_path(image_id)
        with fits.open(path, memmap=False) as hdul:
            hdu = _select_image_hdu(hdul, self.data_hdu)
            data = np.asarray(hdu.data)
            header = _merged_header(hdul, hdu).copy()
        return data, header

    def _build_image(self, image_id: str) -> SurveyImage:
        if image_id in self._image_cache:
            return self._image_cache[image_id]

        record = self.records[image_id]
        shape, header = self._inspect_fits(image_id)
        if len(shape) != 2:
            raise ValueError(f"Expected a 2D FITS image, got shape {shape}")

        wcs = WCS(header)
        shape = (int(shape[0]), int(shape[1]))
        t = record.t if record.t is not None else _parse_time_from_header(header)
        if record.t is not None and record.exptime_s:
            t = t + (record.exptime_s / 2.0) * u.s  # exposure start -> mid-exposure
        limiting_mag_3sigma = (record.limiting_mag_3sigma if record.limiting_mag_3sigma is not None
                               else _parse_limiting_mag_3sigma(header))
        fwhm_px = record.fwhm_px if record.fwhm_px is not None else _parse_fwhm_px(header)
        sigma_centroid_px = record.sigma_centroid_px
        if sigma_centroid_px is None:
            sigma_centroid_px = (fwhm_px / 2.35482) / DEFAULT_DETECTION_SIGMA

        observer_location = (
            record.observer_location if record.observer_location is not None
            else _parse_observer_location(header, fallback=self.default_observer_location)
        )

        image = SurveyImage(
            image_id=image_id,
            t=t,
            wcs=wcs,
            shape=shape,
            limiting_mag_3sigma=float(limiting_mag_3sigma),
            fwhm_px=float(fwhm_px),
            sigma_centroid_px=float(sigma_centroid_px),
            observer_location=observer_location,
        )
        self._image_cache[image_id] = image
        return image

    def list_images(self) -> List[SurveyImage]:
        return [self._build_image(image_id) for image_id in self.records]

    def get_image_data(self, image_id: str) -> np.ndarray:
        if image_id not in self._data_cache:
            data, _ = self._read_fits(image_id)
            self._data_cache[image_id] = data
        return self._data_cache[image_id]

__all__ = ["ZTFImageRecord", "ZTFDifferenceImageProvider", "download_file"]
