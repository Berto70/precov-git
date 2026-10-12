import precov.orbits.propagator
from precov.orbits.kernels import ensure_kernels_downloaded
from astropy.time import Time
from astropy.coordinates import EarthLocation
import numpy as np
import time

which_test = 2

planets_path, asteroids_path = ensure_kernels_downloaded(download_asteroids=True)
prop = precov.orbits.propagator.ASSISTPropagator(object="2021 DG1",
    epoch=Time("2021-03-11T12:00:00", scale="tdb"), # mid-point of ZTF survey
    planets_path=planets_path,
    asteroids_path=asteroids_path,
    light_time_iterations=3
)
print("Propagator initialized")
print(prop.epoch.jd)

# Test with 2021 DG1's approximate elements
#theta = np.array([1.65, 0.4, np.radians(10.0), np.radians(100.0), np.radians(50.0), np.radians(0.0)])
t = Time("2021-03-12T12:00:00", scale="tdb")  # Test a prediscovery date
ztf_location = EarthLocation.of_site("Palomar")  # Example: Palomar Observatory

#print(f"Testing with theta: {theta}")
print(f"Testing with time: {t.iso}")

if which_test==1:
    print("Test 1")
    t_init = time.time()

    geom = prop.geometry(t=t, observer_location=ztf_location)

    t_end = time.time()
    print(f"Geometry computed in {t_end - t_init:.4f} seconds")
    print("====")
    print(f"RA: {np.degrees(geom.ra):.4f}°, Dec: {np.degrees(geom.dec):.4f}°")
    print(f"Heliocentric dist: {geom.r_helio:.3f} AU")
    print(f"Topocentric dist: {geom.r_topo:.3f} AU")
    print(f"Phase angle: {np.degrees(geom.phase_angle):.2f}°")
    print(f"Light travel time: {geom.light_time:.4f} days")

if which_test==2:
    print("Test 2")
    # Plot a topocentric ASSIST ephemeris for an object observed from Palomar.

    import time

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.coordinates import EarthLocation
    from astropy.time import Time
    import astropy.units as u

    from precov.orbits.kernels import ensure_kernels_downloaded
    from precov.orbits.propagator import ASSISTPropagator


    OBJECT = "2021 DG1"
    EPOCH = Time("2021-03-12T12:00:00", scale="tdb")
    START = Time("2021-03-01T12:00:00", scale="tdb")
    END = Time("2021-03-30T12:00:00", scale="tdb")
    STEP_DAYS = 1


    def main() -> None:
        planets_path, asteroids_path = ensure_kernels_downloaded(
            download_asteroids=True
        )
        propagator = precov.orbits.propagator.ASSISTPropagator(
            object=OBJECT,
            epoch=EPOCH,
            planets_path=planets_path,
            asteroids_path=asteroids_path,
            light_time_iterations=3,
        )
        palomar = EarthLocation.of_site("Palomar")

        duration_days = (END - START).to_value("day")
        times = START + (
            np.arange(0.0, duration_days + STEP_DAYS, STEP_DAYS) * u.day
        )
        times = times[times <= END]

        start_time = time.time()
        geometries = [
            propagator.geometry(t=t, observer_location=palomar)
            for t in times
        ]
        print(f"Computed {len(geometries)} epochs in "
            f"{time.time() - start_time:.2f} s")

        # Unwrap RA so a trajectory crossing 0 degrees is not drawn across the
        # entire plot.
        ra_deg = np.degrees(np.unwrap([geometry.ra for geometry in geometries]))
        dec_deg = np.degrees([geometry.dec for geometry in geometries])
        elapsed_days = (times - START).to_value("day")

        fig, ax = plt.subplots(figsize=(10, 6))
        track = ax.scatter(
            ra_deg, dec_deg, c=elapsed_days, cmap="viridis", s=18
        )
        ax.plot(ra_deg, dec_deg, color="0.35", linewidth=0.8, alpha=0.7)
        ax.scatter(ra_deg[0], dec_deg[0], color="tab:green", s=70, label="Start")
        ax.scatter(
            ra_deg[-1], dec_deg[-1], color="tab:red", marker="*", s=110,
            label="End",
        )

        ax.set_xlabel("Topocentric right ascension (deg)")
        ax.set_ylabel("Topocentric declination (deg)")
        ax.set_title(f"{OBJECT} ephemeris from Palomar")
        ax.invert_xaxis()
        ax.grid(alpha=0.3)
        ax.legend()
        colorbar = fig.colorbar(track, ax=ax)
        colorbar.set_label("Days since start")
        fig.tight_layout()
        plt.show()

    if __name__ == "__main__":
        main()

if which_test==3:

    print("Test 3")
    
    import rebound
    import assist
    import matplotlib.pyplot as plt

    ephem = assist.Ephem("./.precov/kernels/linux_m13000p17000.441", "./.precov/kernels/sb441-n16.bsp")

    T0 = Time("2021-03-17T12:00:00", scale="tdb")
    print(T0.jd)
    # T0 = 2458849.5
    T0_sim = T0.jd - ephem.jd_ref
    # print(f"Reference epoch: {T0}, JD={T0}, sim.t={T0_sim}")

    def mag_to_diameter(H, p=0.14):
        """Convert absolute magnitude to diameter in km."""
        return 1329 * 10**(-H/5) / np.sqrt(p)

    def calc_mass(H, p=0.14, density=2000):
        """Calculate mass in kg from absolute magnitude, albedo, and density."""
        diameter_km = mag_to_diameter(H, p)
        radius_m = (diameter_km * 1000) / 2
        volume_m3 = (4/3) * np.pi * radius_m**3
        mass_kg = volume_m3 * density

        mass_su = mass_kg / 1.98847e30  # Convert kg to solar masses
        return mass_su

    def setup_sim():
        sim = rebound.Simulation()
        ax = assist.Extras(sim, ephem)
        sim.add(m=1.0)  # Sun at origin
        sim.add("2021 DG1", date=f"JD{T0.jd}", m=calc_mass(22.08), hash="2021 DG1")
        sim.t = T0_sim
        return ax, sim

    ax, sim = setup_sim()

    print(f"Initialized at time={sim.t}, position={sim.particles[1].xyz}")

    sim_len = 3000 #days
    T_end = T0_sim + sim_len

    times = np.linspace(T0_sim, T_end, sim_len, endpoint=False)
    pos_default = np.zeros((len(times), 3))

    for i, t in enumerate(times):
        ax.integrate_or_interpolate(t)
        pos_default[i] = sim.particles[1].xyz

    rho_x = pos_default[:, 0]
    rho_y = pos_default[:, 1]
    rho_z = pos_default[:, 2]

    ra = np.arctan2(rho_y, rho_x)
    dec = np.arctan2(rho_z, np.sqrt(rho_x**2 + rho_y**2))

    ra_deg = np.degrees(ra)
    dec_deg = np.degrees(dec)

    # ra_deg, dec_deg: arrays of RA and Dec in degrees
    # t_sim: simulation time in days

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color the trajectory by time
    sc = ax.scatter(
        ra_deg,
        dec_deg,
        c=times,
        cmap="viridis",
        s=8
    )

    # Connect consecutive points
    ax.plot(ra_deg, dec_deg, lw=0.7, alpha=0.5)

    # Start and end positions
    ax.scatter(ra_deg[0], dec_deg[0], marker="o",
            s=80, color="red", label="Start", zorder=3)

    ax.scatter(ra_deg[-1], dec_deg[-1], marker="*", 
            s=140, color="black", label="End", zorder=3)

    ax.set_xlabel("Right Ascension (deg)")
    ax.set_ylabel("Declination (deg)")
    ax.set_title("Asteroid path across the sky")

    # Astronomical convention: RA increases to the left
    ax.invert_xaxis()

    ax.grid(True, alpha=0.3)
    ax.legend()

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Simulation time (days)")

    plt.tight_layout()
    plt.show()