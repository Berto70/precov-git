# diag_fit.py
import numpy as np
from astropy.time import Time
from precov.data.mpc_fetcher import fetch_postdiscovery_observations
from precov.orbits.propagator import ASSISTPropagator
from precov.orbits.kernels import ensure_kernels_downloaded
from precov.orbits.fitting import fit_orbit
from time import time

target = "2021 DG1"
epoch = Time("2021-03-11T12:00:00", scale="tdb")

# --- Fetch MPC data ---
obs = fetch_postdiscovery_observations(target, default_sigma_arcsec=2.0)
print(f"N observations: {len(obs)}")
print(f"Time span: {obs[0].t.iso}  ->  {obs[-1].t.iso}")

# --- Setup ASSIST propagator ---
planets_path, asteroids_path = ensure_kernels_downloaded(download_asteroids=False)
prop = ASSISTPropagator(epoch=epoch, planets_path=planets_path, asteroids_path=asteroids_path)

# =========================================================================
# PART A: Does the propagator actually respond to theta?
# =========================================================================
time_in = time()
print("\n=== PART A: propagator response to theta perturbations ===")
theta_ref = np.array([1.6502, 0.5343, np.radians(32.43),
                       np.radians(160.59), np.radians(251.28), np.radians(110.65)])
t_test = obs[len(obs)//2].t
g0 = prop.geometry(theta_ref, t_test)
print(f"Reference at {t_test.iso}: RA={np.degrees(g0.ra):.4f}  Dec={np.degrees(g0.dec):.4f}")

labels = ["a", "e", "i", "raan", "argp", "M"]
steps  = [1e-4, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5]
all_zero = True
for k in range(6):
    tp = theta_ref.copy()
    tp[k] += steps[k]
    gp = prop.geometry(tp, t_test)
    dra  = (np.degrees(gp.ra)  - np.degrees(g0.ra))  * 3600.0
    ddec = (np.degrees(gp.dec) - np.degrees(g0.dec)) * 3600.0
    if abs(dra) > 1e-9 or abs(ddec) > 1e-9:
        all_zero = False
    print(f"  d{labels[k]:5s}={steps[k]:.0e}  ->  dRA={dra:9.4f}\"   dDec={ddec:9.4f}\"")

if all_zero:
    print("  >>> PROPAGATOR IS BROKEN: geometry does not change with theta.")
else:
    print("  >>> Propagator responds to theta correctly.")

time_out = time()
print(f"\nPART A completed in {time_out - time_in:.2f} seconds.")

# =========================================================================
# PART B: Fit + covariance, comparing the two initial guesses
# =========================================================================
time_in = time()
print("\n=== PART B: orbit fit + covariance eigenvalues ===")
guesses = {
    "hardcoded (bad)": np.array([1.65, 0.4, np.radians(10.0),
                                 np.radians(100.0), np.radians(50.0), np.radians(0.0)]),
    "from MOST (good)": theta_ref,
}
for name, g in guesses.items():
    print(f"\n--- initial guess: {name} ---")
    try:
        fit = fit_orbit(obs, prop, g, c=1.0)
        print(f"  success={fit.success}  reduced_chi2={fit.reduced_chi2}")
        print(f"  best-fit theta = {np.round(fit.theta, 5)}")
        cov = np.asarray(fit.covariance, dtype=float)
        eig = np.linalg.eigvalsh(0.5*(cov + cov.T))   # symmetrize before eig
        print(f"  covariance eigenvalues = {np.array2string(eig, precision=3, suppress_small=False)}")
        print(f"  min eigenvalue = {eig.min():.3e}")
        print(f"  is PD? {bool(np.all(eig > 0))}")
    except Exception as ex:
        print(f"  FIT FAILED: {ex}")

time_out = time()
print(f"\nPART B completed in {time_out - time_in:.2f} seconds.")