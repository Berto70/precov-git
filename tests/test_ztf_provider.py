import numpy as np
from astropy.coordinates import EarthLocation
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

from precov.survey.ztf import ZTFDifferenceImageProvider


def _make_test_fits(path):
    data = np.arange(16, dtype=float).reshape(4, 4)
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [2.0, 2.0]
    wcs.wcs.cdelt = [-1.0 / 3600.0, 1.0 / 3600.0]
    wcs.wcs.crval = [10.0, 20.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    header = wcs.to_header()
    header["MJD-OBS"] = Time("2024-01-02T03:04:05", scale="utc").mjd
    header["MAGLIM"] = 20.6
    header["FWHM"] = 2.1
    header["OBSGEO-X"] = -2451185.5
    header["OBSGEO-Y"] = -4761134.6
    header["OBSGEO-Z"] = 3487773.9
    fits.PrimaryHDU(data=data, header=header).writeto(path)


def test_ztf_provider_reads_local_fits(tmp_path):
    fits_path = tmp_path / "ztf_diff.fits"
    _make_test_fits(fits_path)

    provider = ZTFDifferenceImageProvider.from_directory(tmp_path)
    images = provider.list_images()

    assert len(images) == 1
    image = images[0]
    assert image.shape == (4, 4)
    assert np.isclose(image.limiting_mag_3sigma, 20.6)
    assert np.isclose(image.fwhm_px, 2.1)
    assert np.isclose(image.sigma_centroid_px, (2.1 / 2.35482) / 4.0)
    assert isinstance(image.observer_location, EarthLocation)

    data = provider.get_image_data(image.image_id)
    assert data.shape == (4, 4)
    assert np.allclose(data, np.arange(16, dtype=float).reshape(4, 4))


def test_ztf_provider_reads_fpack_difference_image(tmp_path):
    """IRSA difference images are .fits.fz: EMPTY primary HDU + CompImageHDU in ext 1."""
    data = np.random.default_rng(0).normal(size=(32, 48)).astype("float32")
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [24.0, 16.0]
    wcs.wcs.cdelt = [-1.01 / 3600.0, 1.01 / 3600.0]
    wcs.wcs.crval = [10.0, 20.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    path = tmp_path / "ztf_x_scimrefdiffimg.fits.fz"
    fits.HDUList([fits.PrimaryHDU(), fits.CompImageHDU(data, header=wcs.to_header())]).writeto(path)

    from precov.survey.ztf import ZTFImageRecord
    rec = ZTFImageRecord(
        image_id="a", url=path.as_uri(), t=Time("2018-08-02T08:00:00", scale="utc"),
        limiting_mag_3sigma=20.5, fwhm_px=2.0, exptime_s=30.0,
    )
    provider = ZTFDifferenceImageProvider([rec], cache_dir=tmp_path / "cache")
    (img,) = provider.list_images()
    assert img.shape == (32, 48)
    # exposure start -> mid-exposure (+15 s)
    assert np.isclose((img.t - Time("2018-08-02T08:00:00", scale="utc")).sec, 15.0)
    assert provider.get_image_data("a").shape == (32, 48)
