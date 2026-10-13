# diag_elements_crosscheck.py
import numpy as np
import rebound
from precov.orbits.elements import kepler_to_state

theta_true = np.array([1.6502, 0.5343, np.radians(32.43),
                       np.radians(160.57), np.radians(251.31), np.radians(217.94)])
a, e, i, raan, argp, M = theta_true

# "known good" conversion -- this is what TwoBodyKeplerPropagator uses, and
# it's covered by tests/test_propagator.py & tests/test_elements.py.
r_ecl_known = np.asarray(kepler_to_state(theta_true, 0.0, frame="ecliptic"))
print("kepler_to_state position (ecliptic, dt=0):", r_ecl_known)

# rebound's conversion, exactly as used in ASSISTPropagator Step 1
temp_sim = rebound.Simulation()
temp_sim.units = ('AU', 'day', 'Msun')     # <-- confirm this is actually in your local file
temp_sim.add(m=1.0)
temp_sim.add(m=0, a=a, e=e, inc=i, Omega=raan, omega=argp, M=M)
p = temp_sim.particles[1]
r_rebound = np.array([p.x, p.y, p.z])
v_rebound = np.array([p.vx, p.vy, p.vz])
print("rebound.add(...) position (ecliptic, dt=0):", r_rebound)

diff = r_rebound - r_ecl_known
print(f"position diff (au): {diff}   |diff|/|r| = {np.linalg.norm(diff)/np.linalg.norm(r_ecl_known):.3e}")

# finite-difference velocity implied by kepler_to_state, as an independent
# check on rebound's analytic velocity
r1 = np.asarray(kepler_to_state(theta_true, 1e-3, frame="ecliptic"))
v_fd = (r1 - r_ecl_known) / 1e-3
print("rebound velocity   (au/day):", v_rebound, " |v|=", np.linalg.norm(v_rebound))
print("finite-diff velocity(au/day):", v_fd, " |v|=", np.linalg.norm(v_fd))
print(f"velocity magnitude ratio (rebound/fd) = {np.linalg.norm(v_rebound)/np.linalg.norm(v_fd):.4f}")