"""
test_fix1_propagation.py
Diagnostic script to verify ASSIST forward propagation in the correct Barycentric ICRF frame.
"""
import numpy as np
import rebound
import assist
from astropy.time import Time
import astropy.units as u

def test_assist_propagation():
    # 1. Define Epochs (ASSIST strictly requires TDB)
    epoch_utc = Time("2021-02-26 00:00:00", scale='utc')
    epoch_tdb = epoch_utc.tdb
    
    target_utc = Time("2021-04-12 00:00:00", scale='utc')
    target_tdb = target_utc.tdb

    # 2. Load Ephemeris
    ephem = assist.Ephem("./.precov/kernels/de441.bsp")
    
    # 3. Setup Simulation (BARYCENTRIC ICRF)
    sim = rebound.Simulation()
    sim.units = ('AU', 'day', 'Msun')
    extras = assist.Extras(sim, ephem)
    extras.gr_eih_sources = 11

    sim.ri_ias15_adaptive_mode = 2
    
    # CRITICAL FIX A: Get Sun's TRUE Barycentric state at the epoch
    epoch_rel = epoch_tdb.jd - ephem.jd_ref
    sun_epoch = ephem.get_particle("sun", epoch_rel)
    
    # Add Sun at its TRUE Barycentric position (NOT at 0,0,0)
    sim.add(m=1.0, x=sun_epoch.x, y=sun_epoch.y, z=sun_epoch.z,
            vx=sun_epoch.vx, vy=sun_epoch.vy, vz=sun_epoch.vz)
            
    # 4. Asteroid State (Dummy Heliocentric Equatorial for this test)
    ast_helio_pos = np.array([-1.22587357, 0.52854088, 0.15438807]) 
    ast_helio_vel = np.array([-0.01264291, -0.00941490, 0.00354838])
    
    # CRITICAL FIX B: Asteroid Barycentric = Heliocentric + Sun Barycentric
    sim.add(m=0.0, 
            x=ast_helio_pos[0] + sun_epoch.x, y=ast_helio_pos[1] + sun_epoch.y, z=ast_helio_pos[2] + sun_epoch.z,
            vx=ast_helio_vel[0] + sun_epoch.vx, vy=ast_helio_vel[1] + sun_epoch.vy, vz=ast_helio_vel[2] + sun_epoch.vz)
            
    sim.t = epoch_rel
    
    # 5. Integrate to target using the ASSIST-recommended method
    target_rel = target_tdb.jd - ephem.jd_ref
    extras.integrate_or_interpolate(target_rel)
    
    # 6. Extract States
    ast_final = sim.particles[1]
    earth_final = ephem.get_particle("earth", target_rel)
    sun_final = ephem.get_particle("sun", target_rel)
    
    print(f"Asteroid Barycentric Pos: {ast_final.xyz}")
    print(f"Asteroid Barycentric Vel: {ast_final.vxyz}")
    print(f"Asteroid Heliocentric Pos: [{ast_final.x - sun_final.x}, {ast_final.y - sun_final.y}, {ast_final.z - sun_final.z}]")
    
    # 7. Topocentric vector (Asteroid Barycentric - Earth Barycentric)
    dx = ast_final.x - earth_final.x
    dy = ast_final.y - earth_final.y
    dz = ast_final.z - earth_final.z
    dist = np.sqrt(dx**2 + dy**2 + dz**2)
    
    dec = np.arcsin(dz / dist) * (180/np.pi)
    ra = np.arctan2(dy, dx) * (180/np.pi)
    if ra < 0: ra += 360.0

    print(f"\nPropagated Topocentric RA: {ra:.4f} deg, Dec: {dec:.4f} deg")
    print("Compare this RA/Dec to JPL Horizons (Settings: Center=@0, Reference Plane=ICRF)")

if __name__ == "__main__":
    test_assist_propagation()