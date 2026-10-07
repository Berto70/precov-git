"""Offline tests for the IRSA helpers, using a local HTTP server as a fake IBE."""
import http.server
import threading
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from precov.survey import ZTFDifferenceImageProvider, irsa

ROW = {
    "ra": "10.0", "dec": "20.0", "obsjd": "2458332.75", "obsdate": "2018-08-02 06:00:00+00",
    "exptime": "30.0", "filefracday": "20180802250000", "field": "600", "ccdid": "3", "qid": "1",
    "filtercode": "zr", "imgtypecode": "o", "seeing": "2.02", "maglimit": "20.5",
}


def test_scimrefdiff_url_matches_irsa_pattern():
    assert irsa.scimrefdiff_url(ROW) == (
        "https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci/2018/0802/250000/"
        "ztf_20180802250000_000600_zr_c03_o_q1_scimrefdiffimg.fits.fz"
    )


def test_scientific_notation_filefracday():
    row = dict(ROW, filefracday="2.0180802250000E13")
    assert "/2018/0802/250000/ztf_20180802250000_" in irsa.scimrefdiff_url(row)


def test_search_url_uses_documented_parameters():
    url = irsa.build_search_url(10.0, 20.0, 0.5, 2458332.5, 2458333.5)
    q = parse_qs(urlparse(url).query)
    assert urlparse(url).path.endswith("/ibe/search/ztf/products/sci")  # no date sub-path
    assert q["POS"] == ["10.000000,20.000000"] and q["ct"] == ["csv"]
    assert "obsjd BETWEEN" in q["WHERE"][0]
    assert "MAXREC" not in q and "RESPONSEFORMAT" not in q


def test_parse_rejects_html_and_missing_columns():
    with pytest.raises(RuntimeError, match="HTML"):
        irsa.parse_metadata_csv("<html><body>error</body></html>")
    with pytest.raises(RuntimeError, match="required columns"):
        irsa.parse_metadata_csv("ra,dec\n1,2\n")


def test_manifest_row_conversions():
    m = irsa.manifest_row(ROW, maglim_sigma=5.0)
    assert np.isclose(float(m["limiting_mag_3sigma"]), 20.5 + 2.5 * np.log10(5 / 3), atol=1e-4)
    assert np.isclose(float(m["fwhm_px"]), 2.02 / 1.01, atol=1e-4)
    assert m["time"].startswith("2018-08-02T06:00:00")
    with pytest.raises(ValueError):
        irsa.manifest_row({k: v for k, v in ROW.items() if k != "seeing"})


def _fz_bytes(tmp_path):
    wcs = WCS(naxis=2)
    wcs.wcs.crpix, wcs.wcs.cdelt, wcs.wcs.crval = [8, 8], [-1.01 / 3600, 1.01 / 3600], [10, 20]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    p = tmp_path / "t.fits.fz"
    fits.HDUList([fits.PrimaryHDU(), fits.CompImageHDU(np.ones((16, 16), "float32"), header=wcs.to_header())]).writeto(p)
    return p.read_bytes()


def test_end_to_end_against_fake_ibe(tmp_path):
    good = dict(ROW)
    no_ref = dict(ROW, filefracday="20180802260000", ccdid="4")  # will 404: no reference image
    payload = _fz_bytes(tmp_path)
    cols = list(ROW)
    csv_body = ",".join(cols) + "\n" + "\n".join(",".join(r[c] for c in cols) for r in (no_ref, good)) + "\n"
    good_path = urlparse(irsa.scimrefdiff_url(good)).path

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def _do(self, body=True):
            path = urlparse(self.path).path
            if path.endswith("/search"):
                data, code = csv_body.encode(), 200
            elif path == good_path:
                data, code = payload, 200
            else:
                data, code = b"not found", 404
            self.send_response(code); self.send_header("Content-Length", str(len(data))); self.end_headers()
            if body: self.wfile.write(data)
        def do_GET(self): self._do()
        def do_HEAD(self): self._do(False)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    try:
        url, rows = irsa.query_sci_metadata(10, 20, 0.5, 2458332.5, 2458333.5, base_url=base + "/search")
        assert len(rows) == 2
        quads = irsa.unique_quadrants(rows, 10.0, 20.0)

        # point the URL builder at the fake server
        orig = irsa.scimrefdiff_url
        irsa.scimrefdiff_url = lambda r, **k: orig(r, data_url=base + "/ibe/data/ztf/products/sci")
        try:
            sel, fails = irsa.select_available(quads, 5, verbose=False)
            assert [r["ccdid"] for r in sel] == ["3"]
            assert fails == {"HTTP 404 File not found": 1} or sum(fails.values()) == 1
            manifest = irsa.write_manifest(sel, tmp_path / "m.csv")
        finally:
            irsa.scimrefdiff_url = orig
        provider = ZTFDifferenceImageProvider.from_manifest(manifest, cache_dir=tmp_path / "cache")
        (img,) = provider.list_images()
        assert img.shape == (16, 16)
        assert provider.get_image_data(img.image_id).shape == (16, 16)
    finally:
        srv.shutdown()
