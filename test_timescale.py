"""
test_time_scale.py
Diagnostic to verify absolute JD vs Delta t in the observation loop.
"""
from astropy.time import Time
import numpy as np

# Simulate the epoch (midpoint of ZTF survey, roughly Jan 1, 2021)
epoch = Time("2021-01-01 00:00:00", scale='tdb')
epoch_jd = epoch.jd

# Simulate an MPC observation from the 2021 DG1 arc
obs_time = Time("2021-02-17 10:09:36.662", scale='utc')
obs_jd = obs_time.tdb.jd # Convert to TDB for dynamical consistency

print(f"Epoch JD:      {epoch_jd:.5f}")
print(f"Obs JD:        {obs_jd:.5f}")
print(f"Absolute JD:   {obs_jd:.5f}  <-- BUG: If passed to integrator, simulates {obs_jd/365.25:.0f} years!")
print(f"Delta t (Days):{obs_jd - epoch_jd:.2f} <-- CORRECT: Simulates the actual 47-day arc.")