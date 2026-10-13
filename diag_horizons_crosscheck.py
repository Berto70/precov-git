# diag_horizons_crosscheck_exact.py
import numpy as np
import requests

from precov.data.mpc_fetcher import fetch_postdiscovery_observations
from precov.survey.horizons import parse_horizons_csv

target = "2021 DG1"
obs = fetch_postdiscovery_observations(target, default_sigma_arcsec=2.0)
print(f"N observations: {len(obs)}")

jds = [o.t.jd for o in obs]
tlist_str = " ".join(f"'{jd:.8f}'" for jd in jds)  # individually-quoted, per JPL's documented syntax

sof = "\n".join([
    "!$$SOF",
    f"COMMAND='{target}'",
    "OBJ_DATA='NO'",
    "MAKE_EPHEM='YES'",
    "EPHEM_TYPE='OBSERVER'",
    "CENTER='500@399'",
    f"TLIST={tlist_str}",
    "TLIST_TYPE='JD'",
    "QUANTITIES='1'",
    "ANG_FORMAT='DEG'",
    "CSV_FORMAT='YES'",
    "!$$EOF",
])

resp = requests.post(
    "https://ssd.jpl.nasa.gov/api/horizons_file.api",
    data={"format": "text"},
    files={"input": ("input.txt", sof)},
    timeout=120,
)
resp.raise_for_status()
radec = parse_horizons_csv(resp.text)   # [(ra_deg, dec_deg), ...], same order as jds
# assert len(radec) == len(obs), f"Horizons returned {len(radec)} rows, expected {len(obs)}"

def wrap_deg(d):
    return (d + 180.0) % 360.0 - 180.0

dra, ddec = [], []
for o, (h_ra, h_dec) in zip(obs, radec):
    mpc_ra_deg, mpc_dec_deg = np.degrees(o.ra), np.degrees(o.dec)
    dra.append(wrap_deg(mpc_ra_deg - h_ra) * np.cos(np.radians(mpc_dec_deg)) * 3600)
    ddec.append((mpc_dec_deg - h_dec) * 3600)
dra, ddec = np.array(dra), np.array(ddec)

print(f"RMS offset MPC-vs-Horizons (exact epochs): dRA={np.sqrt(np.mean(dra**2)):.3f}\"  "
      f"dDec={np.sqrt(np.mean(ddec**2)):.3f}\"   max={np.max(np.abs(np.hstack([dra,ddec]))):.3f}\"")