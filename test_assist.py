from precov.orbits.propagator import ASSISTPropagator
from precov.orbits.kernels import ensure_kernels_downloaded
from astropy.time import Time
import numpy as np

planets_path, asteroids_path = ensure_kernels_downloaded(download_asteroids=True)
prop = ASSISTPropagator(
    epoch=Time("2020-06-30T12:00:00", scale="tdb"),
    planets_path=planets_path,
    asteroids_path=asteroids_path,
    light_time_iterations=3
)

# Test with 2021 DG1's approximate elements
theta = np.array([1.65, 0.4, np.radians(10.0), np.radians(100.0), np.radians(50.0), np.radians(0.0)])
t = Time("2019-08-15T00:00:00", scale="tdb")  # Test a prediscovery date

geom = prop.geometry(theta, t)
print(f"RA: {np.degrees(geom.ra):.4f}°, Dec: {np.degrees(geom.dec):.4f}°")
print(f"Heliocentric dist: {geom.r_helio:.3f} AU")
print(f"Topocentric dist: {geom.r_topo:.3f} AU")
print(f"Phase angle: {np.degrees(geom.phase_angle):.2f}°")
