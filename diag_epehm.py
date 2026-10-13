# diag_ephem.py
import numpy as np
from astropy.time import Time
from precov.data.mpc_fetcher import fetch_postdiscovery_observations
from precov.orbits.propagator import ASSISTPropagator
from precov.orbits.kernels import ensure_kernels_downloaded

target = "2021 DG1"
epoch = Time("2021-03-11T12:00:00", scale="tdb")

obs = fetch_postdiscovery_observations(target, default_sigma_arcsec=2.0)
times_jd = [o.t.jd for o in obs]
print(f"N obs: {len(obs)}")
print(f"Obs JD range:   {min(times_jd):.4f}  ->  {max(times_jd):.4f}")
print(f"Obs date range: {Time(min(times_jd), format='jd').iso}  ->  "
      f"{Time(max(times_jd), format='jd').iso}")

planets_path, asteroids_path = ensure_kernels_downloaded(download_asteroids=True)
print(f"Planets kernel:   {planets_path}")
print(f"Asteroids kernel: {asteroids_path}")

prop = ASSISTPropagator(epoch=epoch, planets_path=planets_path,
                        asteroids_path=asteroids_path)
print(f"Ephemeris jd_ref:        {prop._ephem.jd_ref}")
print(f"Epoch JD:                {epoch.jd}")
print(f"Epoch - jd_ref (sim.t):  {epoch.jd - prop._ephem.jd_ref:.4f}")

theta_true = np.array([1.6502, 0.5343, np.radians(32.43),
                       np.radians(160.59), np.radians(251.28), np.radians(47.0)])

print("\nPropagating to each observation...")
for i, o in enumerate(obs):
    try:
        ra, dec = prop.propagate(theta_true, o.t, o.observer_location)
    except Exception as e:
        print(f"\n>>> obs {i} FAILED:")
        print(f"    JD={o.t.jd:.6f}  ({o.t.iso})")
        print(f"    t_emit (JD-jd_ref) = {o.t.jd - prop._ephem.jd_ref:.6f}")
        print(f"    Error: {e}")
        break
else:
    print("All observations propagated successfully — no coverage issue.")