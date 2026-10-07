"""
IRSA / IBE helpers for locating ZTF difference images.

Workflow
--------
1. ``query_sci_metadata``  -> rows from the *science-exposure* metadata table
   (one row per CCD-quadrant exposure) overlapping a sky region and time window.
2. ``scimrefdiff_url``     -> deterministic URL of the difference image for a row.
3. ``probe_url``/``select_available`` -> keep only rows whose difference image
   really exists (see below).
4. ``write_manifest``      -> CSV consumed by ``ZTFDifferenceImageProvider.from_manifest``.

Two facts about the archive that shape this module
--------------------------------------------------
* The metadata table lists *science images*. A ``*_scimrefdiffimg.fits.fz``
  file exists only if a reference image for that field/CCD-quadrant/filter
  existed when the exposure was processed, so a fraction of rows legitimately
  have no difference image (HTTP 404). Early-survey data (2018) is the worst case.
* The IBE search endpoint is documented at
  https://irsa.ipac.caltech.edu/docs/program_interface/ztf_api.html :
  parameters are ``POS``, ``SIZE``, ``INTERSECT``, ``WHERE``, ``COLUMNS`` and
  ``ct`` (output format). Unknown parameters are not an error you will see;
  they are just ignored, which is why we validate the returned columns.
"""
from __future__ import annotations

import csv
import io
import math
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import numpy as np
from astropy.time import Time

IBE_SEARCH_URL = "https://irsa.ipac.caltech.edu/ibe/search/ztf/products/sci"
IBE_DATA_URL = "https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci"

# ZTF plate scale, arcsec / pixel (used to turn IRSA `seeing` [arcsec] into pixels).
ZTF_PIXEL_SCALE_ARCSEC = 1.01

# ZTF quadrant: 3072 x 3080 px at 1.01 arcsec/px  ->  ~0.86 deg; half-width in degrees.
ZTF_QUADRANT_HALF_WIDTH_DEG = 0.432

# Columns we need; requested explicitly because the default column set is
# table specific and may omit some of them.
REQUIRED_COLUMNS = ("filefracday", "field", "ccdid", "qid", "filtercode", "obsjd")
DEFAULT_COLUMNS = (
    "ra", "dec", "obsjd", "obsdate", "exptime", "filefracday", "field", "ccdid", "qid",
    "filtercode", "imgtypecode", "seeing", "maglimit", "infobits", "ipac_pub_date",
)

_UA = {"User-Agent": "Mozilla/5.0 (compatible; precov/0.1; +ZTF IRSA client)", "Accept": "text/csv,*/*"}


# --------------------------------------------------------------------------- #
# Metadata query
# --------------------------------------------------------------------------- #
def build_search_url(
    ra_deg: float,
    dec_deg: float,
    size_deg: float,
    jd_min: Optional[float],
    jd_max: Optional[float],
    columns: Sequence[str] = DEFAULT_COLUMNS,
    base_url: str = IBE_SEARCH_URL,
) -> str:
    """Build a documented IBE query URL (POS + SIZE [+ WHERE] + COLUMNS + ct=csv).

    If ``jd_min``/``jd_max`` are None the WHERE clause is omitted (pure spatial query).
    """
    params = [
        ("POS", f"{ra_deg:.6f},{dec_deg:.6f}"),
        ("SIZE", f"{size_deg:g}"),
        ("INTERSECT", "OVERLAPS"),
    ]
    if jd_min is not None and jd_max is not None:
        params.append(("WHERE", f"obsjd BETWEEN {jd_min:.6f} AND {jd_max:.6f}"))
    params += [("COLUMNS", ",".join(columns)), ("ct", "csv")]
    return base_url + "?" + urlencode(params, quote_via=quote, safe=",")


def parse_metadata_csv(text: str) -> List[Dict[str, str]]:
    """Parse an IBE CSV response and validate that it is what we asked for."""
    head = text.lstrip()[:200].lower()
    if head.startswith("<") or "<html" in head:
        raise RuntimeError(
            "IRSA returned HTML instead of CSV (bad query or service error). "
            f"First bytes: {text[:200]!r}"
        )
    rows = list(csv.DictReader(io.StringIO(text)))
    if rows:
        missing = [c for c in REQUIRED_COLUMNS if c not in rows[0]]
        if missing:
            raise RuntimeError(
                f"IRSA response lacks required columns {missing}; got {sorted(rows[0])}"
            )
    return rows


def query_sci_metadata(
    ra_deg: float,
    dec_deg: float,
    size_deg: float,
    jd_min: float,
    jd_max: float,
    timeout: float = 90.0,
    retries: int = 3,
    base_url: str = IBE_SEARCH_URL,
    server_time_filter: bool = True,
) -> Tuple[str, List[Dict[str, str]]]:
    """Return ``(query_url, rows)``. The URL is returned so it can be printed/debugged.

    IRSA is sometimes slow: timeouts, connection errors and HTTP 5xx are retried
    with backoff; 4xx errors (bad query) fail immediately.
    """
    import time

    if server_time_filter:
        url = build_search_url(ra_deg, dec_deg, size_deg, jd_min, jd_max, base_url=base_url)
    else:  # spatial-only query, time cut applied locally (fallback when WHERE makes IBE time out)
        url = build_search_url(ra_deg, dec_deg, size_deg, None, None, base_url=base_url)
    last: Optional[BaseException] = None
    for attempt in range(1, retries + 1):
        try:
            with urlopen(Request(url, headers=_UA), timeout=timeout) as resp:
                text = resp.read().decode("utf-8", "replace")
            rows = parse_metadata_csv(text)
            if not server_time_filter:
                rows = filter_rows_by_jd(rows, jd_min, jd_max)
            return url, rows
        except HTTPError as exc:
            if 400 <= exc.code < 500:
                raise RuntimeError(f"IRSA query failed: HTTP {exc.code} {exc.reason}\n  {url}") from exc
            last = exc
        except (URLError, TimeoutError, ConnectionError) as exc:
            last = exc
        if attempt < retries:
            time.sleep(3.0 * attempt)
    raise RuntimeError(f"IRSA query failed after {retries} attempts: {last}\n  {url}")


def filter_rows_by_jd(rows: Iterable[Dict[str, str]], jd_min: float, jd_max: float) -> List[Dict[str, str]]:
    return [r for r in rows if jd_min <= float(r["obsjd"]) <= jd_max]


def load_metadata_csv(path) -> List[Dict[str, str]]:
    """Load a metadata CSV saved from the browser / IRSA web form (same validation as a live query).

    Extra columns such as ``in_row_id, in_ra, in_dec`` (added by IRSA for positional
    searches) are fine.
    """
    with open(path, newline="", encoding="utf-8") as handle:
        return parse_metadata_csv(handle.read())


def quadrant_offset_deg(row: Dict[str, str], ra_deg: float, dec_deg: float) -> Tuple[float, float]:
    """(dx, dy) in degrees from the quadrant centre to (ra, dec); dx includes cos(dec)."""
    dra = (ra_deg - float(row["ra"]) + 180.0) % 360.0 - 180.0
    return dra * math.cos(math.radians(dec_deg)), dec_deg - float(row["dec"])


def rows_containing(
    rows: Iterable[Dict[str, str]], position_at, margin_deg: float = 0.05
) -> List[Dict[str, str]]:
    """Keep rows whose quadrant (approximately) contains the target at that exposure's time.

    ``position_at(jd) -> (ra_deg, dec_deg)``. The quadrant is modelled as a square of
    half-width ``ZTF_QUADRANT_HALF_WIDTH_DEG`` around its centre (+ ``margin_deg``);
    rows are returned closest-to-centre first. This is a cheap pre-filter: the exact
    footprint test is done later with the image WCS.
    """
    half = ZTF_QUADRANT_HALF_WIDTH_DEG + margin_deg
    scored = []
    for row in rows:
        try:
            jd_mid = float(row["obsjd"]) + 0.5 * float(row.get("exptime") or 30.0) / 86400.0
            dx, dy = quadrant_offset_deg(row, *position_at(jd_mid))
        except (KeyError, ValueError):
            continue
        if abs(dx) <= half and abs(dy) <= half:
            scored.append((math.hypot(dx, dy), row))
    scored.sort(key=lambda t: t[0])
    return [r for _, r in scored]


# --------------------------------------------------------------------------- #
# URL construction
# --------------------------------------------------------------------------- #
def _int_str(value: str) -> str:
    """'20180802345161' (or '2.0180802345161E13') -> '20180802345161'."""
    value = str(value).strip()
    return value if value.isdigit() else str(int(float(value)))


def image_id_from_row(row: Dict[str, str]) -> str:
    frac = _int_str(row["filefracday"])
    return (f"ztf_{frac}_{int(row['field']):06d}_{row['filtercode']}"
            f"_c{int(row['ccdid']):02d}_q{int(row['qid'])}")


def scimrefdiff_url(
    row: Dict[str, str], suffix: str = "scimrefdiffimg.fits.fz", data_url: str = IBE_DATA_URL
) -> str:
    """URL of a per-quadrant product, following IRSA's documented path pattern:

    ``sci/YYYY/MMDD/fracday/ztf_<filefracday>_<field:06d>_<filter>_c<ccd:02d>_<imgtypecode>_q<qid>_<suffix>``
    """
    frac = _int_str(row["filefracday"])
    yyyy, mmdd, fracday = frac[:4], frac[4:8], frac[8:]
    imgtypecode = (row.get("imgtypecode") or "o").strip() or "o"
    name = (f"ztf_{frac}_{int(row['field']):06d}_{row['filtercode']}"
            f"_c{int(row['ccdid']):02d}_{imgtypecode}_q{int(row['qid'])}_{suffix}")
    return f"{data_url}/{yyyy}/{mmdd}/{fracday}/{name}"


# --------------------------------------------------------------------------- #
# Availability check
# --------------------------------------------------------------------------- #
def probe_url(url: str, timeout: float = 30.0) -> Tuple[bool, str]:
    """Check that ``url`` exists. Returns ``(ok, detail)``; never hides *why* it failed."""
    for method in ("HEAD", "GET"):
        try:
            with urlopen(Request(url, method=method, headers=_UA), timeout=timeout) as resp:
                return True, f"HTTP {resp.status}"
        except HTTPError as exc:
            if method == "HEAD" and exc.code in (405, 501):
                continue  # server dislikes HEAD; retry with GET (body is not read)
            return False, f"HTTP {exc.code} {exc.reason}"
        except (URLError, TimeoutError) as exc:
            return False, f"network error: {exc}"
    return False, "unknown"


def _sep_deg(ra1, dec1, ra2, dec2) -> float:
    r1, d1, r2, d2 = map(math.radians, (ra1, dec1, ra2, dec2))
    c = math.sin(d1) * math.sin(d2) + math.cos(d1) * math.cos(d2) * math.cos(r1 - r2)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def unique_quadrants(
    rows: Iterable[Dict[str, str]], target_ra: Optional[float] = None, target_dec: Optional[float] = None
) -> List[Dict[str, str]]:
    """De-duplicate by image id; if a target is given, sort by distance of quadrant centre."""
    seen, out = set(), []
    for row in rows:
        key = image_id_from_row(row)
        if key not in seen:
            seen.add(key)
            out.append(row)
    if target_ra is not None and target_dec is not None and out and "ra" in out[0]:
        def key(r):
            try:
                return _sep_deg(float(r["ra"]), float(r["dec"]), target_ra, target_dec)
            except (ValueError, KeyError):
                return 1e9
        out.sort(key=key)
    return out


def select_available(
    rows: Sequence[Dict[str, str]], max_images: int, probe: bool = True, verbose: bool = True
) -> Tuple[List[Dict[str, str]], Counter]:
    """Walk ``rows`` in order and keep the first ``max_images`` with an existing difference image.

    Returns ``(selected_rows, failure_counter)`` where the counter maps a failure
    description (e.g. ``"HTTP 404 Not Found"``) to how often it happened.
    """
    selected: List[Dict[str, str]] = []
    failures: Counter = Counter()
    for row in rows:
        if len(selected) >= max_images:
            break
        url = scimrefdiff_url(row)
        if probe:
            ok, detail = probe_url(url)
            if not ok:
                failures[detail] += 1
                if verbose and failures[detail] == 1:
                    print(f"  unavailable ({detail}): {url}")
                continue
        selected.append(row)
    return selected, failures


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
MANIFEST_FIELDS = (
    "image_id", "url", "time", "limiting_mag_3sigma", "fwhm_px", "exptime",
    "field", "ccdid", "qid", "filtercode", "obsjd",
)


def manifest_row(row: Dict[str, str], maglim_sigma: float = 5.0) -> Dict[str, str]:
    """Convert one IRSA metadata row into a provider-manifest row.

    * ``time``  : exposure START (from ``obsjd``); the provider shifts it to
      mid-exposure using ``exptime``.
    * ``fwhm_px``: ``seeing`` (arcsec) / 1.01 arcsec per pixel.
    * ``limiting_mag_3sigma``: IRSA ``maglimit`` is treated as a ``maglim_sigma``
      limit and converted to 3 sigma. NOTE: it characterises the *science*
      image, not the difference image - a documented approximation.
    """
    def num(key: str) -> Optional[float]:
        try:
            return float(row[key])
        except (KeyError, TypeError, ValueError):
            return None

    obsjd = float(row["obsjd"])
    seeing, maglim = num("seeing"), num("maglimit")
    if seeing is None or maglim is None:
        raise ValueError(f"Row {image_id_from_row(row)} lacks seeing/maglimit; refusing to invent values")
    return {
        "image_id": image_id_from_row(row),
        "url": scimrefdiff_url(row),
        "time": Time(obsjd, format="jd", scale="utc").isot,
        "limiting_mag_3sigma": f"{maglim + 2.5 * np.log10(maglim_sigma / 3.0):.4f}",
        "fwhm_px": f"{seeing / ZTF_PIXEL_SCALE_ARCSEC:.4f}",
        "exptime": row.get("exptime", "") or "",
        "field": row["field"], "ccdid": row["ccdid"], "qid": row["qid"],
        "filtercode": row["filtercode"], "obsjd": f"{obsjd:.6f}",
    }


def write_manifest(rows: Sequence[Dict[str, str]], path, maglim_sigma: float = 5.0):
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(manifest_row(row, maglim_sigma))
    return path


__all__ = [
    "IBE_SEARCH_URL", "IBE_DATA_URL", "build_search_url", "parse_metadata_csv",
    "query_sci_metadata", "load_metadata_csv", "rows_containing", "filter_rows_by_jd", "scimrefdiff_url", "image_id_from_row", "probe_url",
    "unique_quadrants", "select_available", "manifest_row", "write_manifest",
]
