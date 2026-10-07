"""Minimal JPL Horizons client: geocentric astrometric RA/Dec at 1-day steps."""
from __future__ import annotations

from typing import List, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen

HORIZONS_API = "https://ssd.jpl.nasa.gov/api/horizons.api"


def parse_horizons_csv(text: str) -> List[Tuple[float, float]]:
    """Parse a CSV-format OBSERVER ephemeris (QUANTITIES=1, ANG_FORMAT=DEG) into (ra, dec) deg."""
    lines = text.splitlines()
    try:
        soe = next(i for i, l in enumerate(lines) if l.strip() == "$$SOE")
        eoe = next(i for i, l in enumerate(lines) if l.strip() == "$$EOE")
    except StopIteration:
        raise RuntimeError(
            "Horizons returned no ephemeris table ($$SOE/$$EOE missing).\n"
            f"--- start of response ---\n{text[:400]}\n--- end of response ---\n{text[-600:]}"
        )
    header = next(l for l in reversed(lines[:soe]) if "R.A." in l).split(",")
    i_ra = next(i for i, c in enumerate(header) if c.strip().startswith("R.A."))
    i_dec = next(i for i, c in enumerate(header) if c.strip().startswith("DEC"))
    out = []
    for line in lines[soe + 1:eoe]:
        cells = line.split(",")
        out.append((float(cells[i_ra]), float(cells[i_dec])))
    return out


def daily_positions(target: str, jd_start: float, jd_stop: float) -> List[Tuple[float, float, float]]:
    """Return ``[(jd, ra_deg, dec_deg), ...]`` at ``jd_start + k`` for k = 0, 1, ... <= jd_stop."""
    if not (2_400_000 < jd_start < jd_stop < 2_700_000):
        raise ValueError(f"JD range looks wrong: {jd_start} .. {jd_stop} (expected ~2.4e6-2.7e6)")
    params = {
        "format": "text", "COMMAND": f"'{target}'", "MAKE_EPHEM": "YES",
        "EPHEM_TYPE": "OBSERVER", "CENTER": "'500@399'",
        "START_TIME": f"'JD {jd_start:.6f}'", "STOP_TIME": f"'JD {jd_stop:.6f}'",
        "STEP_SIZE": "'1d'", "QUANTITIES": "'1'", "ANG_FORMAT": "DEG", "CSV_FORMAT": "YES",
    }
    url = HORIZONS_API + "?" + urlencode(params)
    text = urlopen(Request(url, headers={"User-Agent": "precov"}), timeout=120).read().decode("utf-8", "ignore")
    radec = parse_horizons_csv(text)
    n_expected = int(jd_stop - jd_start + 1e-9) + 1
    if len(radec) != n_expected:
        raise RuntimeError(f"Horizons returned {len(radec)} rows, expected {n_expected}")
    return [(jd_start + k, ra, dec) for k, (ra, dec) in enumerate(radec)]
