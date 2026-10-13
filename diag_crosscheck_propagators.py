# diag_propagators.py
import numpy as np
from astropy.time import Time
from precov.orbits.propagator import TwoBodyKeplerPropagator, ASSISTPropagator
from precov.orbits.kernels import ensure_kernels_downloaded

# Your best-fit elements from the pipeline output
theta_fit = np.array([1.653007, 0.535008, np.radians(32.4875), 
                      np.radians(160.5859), np.radians(251.3217), np.radians(47.1061)])

# Use the exact epoch from your run_2021_dg1.py
epoch = Time("2021-01-01T00:00:00", scale="tdb") 
t_2018 = Time("2018-04-10T12:00:00", scale="tdb") # One of your 2018 images

planets_path, _ = ensure_kernels_downloaded(download_asteroids=False)
prop_tb = TwoBodyKeplerPropagator(epoch=epoch)
prop_as = ASSISTPropagator(epoch=epoch, planets_path=planets_path)

ra_tb, dec_tb = prop_tb.propagate(theta_fit, t_2018)
ra_as, dec_as = prop_as.propagate(theta_fit, t_2018)

print(f"TwoBody  -> RA: {np.degrees(ra_tb):.4f}, Dec: {np.degrees(dec_tb):.4f}")
print(f"ASSIST   -> RA: {np.degrees(ra_as):.4f}, Dec: {np.degrees(dec_as):.4f}")
print(f"Offset   -> {np.degrees(ra_as - ra_tb)*3600:.0f} arcsec in RA, {np.degrees(dec_as - dec_tb)*3600:.0f} arcsec in Dec")