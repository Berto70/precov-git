# diag_crosscheck_propagators.py
import numpy as np
from astropy.time import Time
from precov.data.mpc_fetcher import fetch_postdiscovery_observations
from precov.orbits.propagator import TwoBodyKeplerPropagator

target = "2021 DG1"
epoch = Time("2021-03-11T12:00:00", scale="tdb")

obs = fetch_postdiscovery_observations(target, default_sigma_arcsec=2.0)
print(f"N observations: {len(obs)}")

theta_true = np.array([1.6502, 0.5343, np.radians(32.43),
                       np.radians(160.57), np.radians(251.31), np.radians(217.94)])

prop = TwoBodyKeplerPropagator(epoch=epoch)

def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

dra, ddec = [], []
for o in obs:
    ra_p, dec_p = prop.propagate(theta_true, o.t, o.observer_location)
    dra.append(wrap(o.ra - ra_p) * np.cos(o.dec))
    ddec.append(o.dec - dec_p)
dra_as = np.degrees(np.array(dra)) * 3600
ddec_as = np.degrees(np.array(ddec)) * 3600
rms = np.sqrt(np.mean(dra_as**2 + ddec_as**2))
print("=== TwoBodyKeplerPropagator, theta_true (a=1.65) ===")
print(f"  RMS_RA={np.sqrt(np.mean(dra_as**2)):8.3f}\"   "
      f"RMS_Dec={np.sqrt(np.mean(ddec_as**2)):8.3f}\"   "
      f"RMS_total={rms:8.3f}\"   max={np.max(np.abs(np.hstack([dra_as,ddec_as]))):8.3f}\"")