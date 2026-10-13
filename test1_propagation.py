"""
test_fix1_propagation.py
Diagnostic script to verify ASSIST forward propagation.
Tests the impact of Reference Frame (Heliocentric vs Barycentric) and Time Scale (UTC vs TDB).
"""
import numpy as np
import rebound
import assist
from astropy.time import Time
from astropy.coordinates import get_body_barycentric_posvel
import astropy.units as u

def test_assist_propagation():
    # 1. Define the Epoch (MPC observations are typically in UTC)
    epoch_utc_str = "2021-02-26 00:00:00"
    t_utc = Time(epoch_utc_str, scale='utc')
    
    # CRITICAL FIX A: ASSIST requires TDB (Barycentric Dynamical Time)
    # The difference between UTC and TDB (Delta T) is ~69 seconds. 
    # Over light-travel times, this causes massive RA/Dec offsets.
    t_tdb = t_utc.tdb
    jd_tdb = t_tdb.jd

    # 2. Define the Asteroid's State Vector (e.g., from JPL Horizons)
    # Horizons typically provides Heliocentric Ecliptic/Equatorial coordinates.
    # Let's assume these are your Heliocentric state vectors in AU and AU/day
    ast_helio_pos = np.array([-1.225873573152143E+00, 5.285408811584692E-01, 1.543880756264600E-01])  # AU
    ast_helio_vel = np.array([-1.264291373402918E-02, -9.414908379505822E-03, 3.548382260754484E-03]) # AU/day

    # CRITICAL FIX B: ASSIST requires BARYCENTRIC coordinates.
    # We must add the Sun's Barycentric state to the Heliocentric state.
    sun_pos_astropy, sun_vel_astropy = get_body_barycentric_posvel('sun',t_tdb)
    
    # Convert Astropy units to AU and Day
    sun_pos = np.array([sun_pos_astropy.x.to(u.AU).value, 
                        sun_pos_astropy.y.to(u.AU).value, 
                        sun_pos_astropy.z.to(u.AU).value])
    sun_vel = np.array([sun_vel_astropy.x.to(u.AU/u.day).value, 
                        sun_vel_astropy.y.to(u.AU/u.day).value, 
                        sun_vel_astropy.z.to(u.AU/u.day).value])

    ast_bary_pos = ast_helio_pos + sun_pos
    ast_bary_vel = ast_helio_vel + sun_vel

    # 3. Setup ASSIST Simulation
    ephem = assist.Ephem("./.precov/kernels/de441.bsp") # Ensure path is correct
    sim = rebound.Simulation()
    sim.units = ('AU', 'day', 'Msun')
    sim.add(m=1.0)  # Sun
    extras = assist.Extras(sim, ephem)

    sim.t = jd_tdb - ephem.jd_ref 
    
    # Add the asteroid using the BARYCENTRIC state vector
    sim.add(x=ast_bary_pos[0], y=ast_bary_pos[1], z=ast_bary_pos[2],
            vx=ast_bary_vel[0], vy=ast_bary_vel[1], vz=ast_bary_vel[2],
            m=0.0) # Test particle

    # 4. Propagate to a post-discovery epoch and check RA/Dec
    target_epoch_utc = Time("2021-03-31 00:00:00", scale='utc')
    target_jd_tdb = target_epoch_utc.tdb.jd - ephem.jd_ref  # ASSIST uses JD relative to its reference

    times = np.linspace(sim.t, sim.t+45, 500)
    
    #sim.integrate(target_jd_tdb)
    for t in times:
        extras.integrate_or_interpolate(t)
    
    # Extract Barycentric position of asteroid and Earth
    ast_final = sim.particles[1]
    earth_final = ephem.get_particle("earth", target_jd_tdb)
    print(ast_final.xyz)
    print(ast_final.vxyz)
    
    # Compute topocentric/geocentric vector (simplified for diagnostic)
    dx = ast_final.x - earth_final.x
    dy = ast_final.y - earth_final.y
    dz = ast_final.z - earth_final.z
    
    # Convert to RA/Dec (simplified)
    dist = np.sqrt(dx**2 + dy**2 + dz**2)
    dec = np.arcsin(dz / dist) * (180/np.pi)
    ra = np.arctan2(dy, dx) * (180/np.pi)
    if ra < 0: ra += 360.0

    print(f"Propagated RA: {ra:.4f} deg, Dec: {dec:.4f} deg")
    print("If these values are within arcseconds of your JPL Horizons reference,")
    print("your Barycentric and TDB transformations are correct.")

if __name__ == "__main__":
    test_assist_propagation()