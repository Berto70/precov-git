# diag_residuals.py
import numpy as np
from astropy.time import Time
from precov.data.mpc_fetcher import fetch_postdiscovery_observations
from precov.orbits.propagator import ASSISTPropagator
from precov.orbits.kernels import ensure_kernels_downloaded

target = "2021 DG1"
epoch = Time("2021-03-11T12:00:00", scale="tdb")

obs = fetch_postdiscovery_observations(target, default_sigma_arcsec=2.0)
print(f"N observations: {len(obs)}")
print(f"Arc: {obs[0].t.iso}  ->  {obs[-1].t.iso}")

planets_path, asteroids_path = ensure_kernels_downloaded(download_asteroids=False)
prop = ASSISTPropagator(epoch=epoch, planets_path=planets_path, asteroids_path=asteroids_path)

# True orbit (MOST/JPL elements, M advanced to the 2021-03-11 fit epoch)
theta_true = np.array([1.6502, 0.5343, np.radians(32.43),
                       np.radians(160.59), np.radians(251.28), np.radians(110.65)])

# The bogus "good" solution from the previous diagnostic (a = 50.8 AU)
theta_fit  = np.array([0.73225, 0.99976, 0.58226, 2.80099, -3.10802, 3.11377])

def wrap(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def resid_arcsec(theta):
    dra, ddec = [], []
    for o in obs:
        ra_p, dec_p = prop.propagate(theta, o.t, o.observer_location)
        dra.append(wrap(o.ra - ra_p) * np.cos(o.dec))
        ddec.append(o.dec - dec_p)
    dra, ddec = np.array(dra), np.array(ddec)
    return np.degrees(dra)*3600, np.degrees(ddec)*3600

for name, th in [("TRUE orbit (a=1.65)", theta_true),
                 ("FITTED 'good' (a=50.8)", theta_fit)]:
    ra_r, dec_r = resid_arcsec(th)
    rms = np.sqrt(np.mean(ra_r**2 + dec_r**2))
    print(f"\n=== {name} ===")
    print(f"  RMS_RA={np.sqrt(np.mean(ra_r**2)):8.3f}\"   "
          f"RMS_Dec={np.sqrt(np.mean(dec_r**2)):8.3f}\"   "
          f"RMS_total={rms:8.3f}\"   max={np.max(np.abs(np.hstack([ra_r,dec_r]))):8.3f}\"")