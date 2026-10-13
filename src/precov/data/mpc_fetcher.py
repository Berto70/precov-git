# src/precov/data/mpc_fetcher.py
"""
Automated ingestion of post-discovery astrometry from the Minor Planet Center (MPC).
Implements the data parsing and default uncertainty assignment described in Sec. 3.1.
"""
from __future__ import annotations

import numpy as np
import astropy.units as u
from astropy.time import Time
from typing import List

from ..data_models import Observation

try:
    from astroquery.mpc import MPC
except ImportError:
    raise ImportError(
        "astroquery is required for automated MPC fetching. "
        "Install it via: pip install astroquery"
    )

def _angle_to_rad(value) -> float:
    """Coerce one RA/Dec table cell to a plain float in radians.

    astroquery.mpc.MPC.get_observations returns RA/DEC as Quantity columns
    with a unit already attached (confirmed empirically -- comparing a raw
    cell against a dimensionless float raises UnitConversionError, which is
    only possible if the cell itself is dimensionful). Use that unit
    directly rather than assuming a convention.
    """
    if hasattr(value, "unit") and value.unit is not None:
        return float(value.to(u.rad).value)
    # Fallback if a future astroquery version strips units and returns a
    # bare float: MPC's documented convention for this field is decimal
    # degrees.
    return float(np.radians(float(value)))


def fetch_postdiscovery_observations(
    target: str,
    default_sigma_arcsec: float = 2.0,
) -> List[Observation]:
    """
    Fetch official post-discovery astrometry for `target` from the MPC.

    Applies the paper's default 2-arcsec uncertainty (Sec. 3.1) when the MPC
    omits error bars, and projects the RA uncertainty by cos(dec).
    """
    print(f"Querying MPC for post-discovery observations of '{target}'...")
    obs_table = MPC.get_observations(target)
    print(f"  Returned {len(obs_table)} rows. Columns: {obs_table.colnames}")

    observations: List[Observation] = []
    missing_count = 0

    for row in obs_table:
        t = Time(row['epoch'], format='jd', scale='utc')

        ra = _angle_to_rad(row['RA'])
        dec = _angle_to_rad(row['DEC'])

        rms_ra_raw = None
        rms_dec_raw = None
        if 'rmsRA' in obs_table.colnames:
            val = row['rmsRA']
            if not (hasattr(val, 'mask') and val.mask):
                rms_ra_raw = float(val)
        if 'rmsDec' in obs_table.colnames:
            val = row['rmsDec']
            if not (hasattr(val, 'mask') and val.mask):
                rms_dec_raw = float(val)

        if rms_ra_raw is None or rms_dec_raw is None:
            missing_count += 1
        rms_ra = rms_ra_raw if rms_ra_raw is not None else default_sigma_arcsec
        rms_dec = rms_dec_raw if rms_dec_raw is not None else default_sigma_arcsec

        sigma_ra = (rms_ra * u.arcsec).to(u.rad).value
        sigma_dec = (rms_dec * u.arcsec).to(u.rad).value
        sigma_ra_proj = sigma_ra * np.cos(dec)

        observations.append(Observation(
            t=t, ra=ra, dec=dec,
            sigma_ra=sigma_ra_proj, sigma_dec=sigma_dec,
            robust=True,
        ))

    print(f"Successfully parsed {len(observations)} MPC observations.")
    if missing_count:
        print(f"  -> Applied default {default_sigma_arcsec}\" uncertainty to "
              f"{missing_count} rows with missing error bars (per Sec. 3.1).")

    sample = observations[0]
    assert not hasattr(sample.ra, "unit"), "Observation.ra must be a plain float (radians)"
    assert not hasattr(sample.dec, "unit"), "Observation.dec must be a plain float (radians)"

    return observations