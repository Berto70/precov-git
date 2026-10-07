"""
Physical and algorithmic constants.

Numerical values for the Gaussian gravitational constant and the speed of
light are given in units compatible with the rest of the package: astronomical
units (au) for length and days for time, which is the natural unit system for
Keplerian orbit propagation on solar-system timescales.
"""
import numpy as np

# Gaussian gravitational constant k (au^{3/2} day^{-1} Msun^{-1/2}); GM_sun = k^2
GAUSS_K = 0.01720209895
GM_SUN = GAUSS_K**2  # au^3 / day^2, heliocentric two-body dynamics (Sun-only)

# Speed of light in au / day
C_AU_PER_DAY = 173.144632674

# Mean obliquity of the ecliptic at J2000 (IAU 1980), used to rotate
# heliocentric ecliptic Keplerian-element positions into the equatorial
# (ICRS-aligned) frame in which RA/Dec and astropy's built-in ephemerides live.
OBLIQUITY_J2000_RAD = np.radians(23.439291111)

# Default Bowell et al. (1989) H-G phase function coefficients.
HG_PHI1_A, HG_PHI1_B = 3.33, 0.63
HG_PHI2_A, HG_PHI2_B = 1.87, 1.22
HG_DEFAULT_G = 0.15

# Paper defaults (Sec. 3.1-3.6)
DEFAULT_HUBER_C = 1.0
DEFAULT_NOMINAL_ASTROMETRIC_SIGMA_ARCSEC = 2.0
DEFAULT_BOUNDARY_SAMPLES_M = 20_000
DEFAULT_CONFIDENCE_Q = 0.9999
DEFAULT_HULL_BUFFER_PX = 500
DEFAULT_MAG_BUFFER = 1.0
DEFAULT_DETECTION_SIGMA = 4.0
DEFAULT_ALPHA_DETECTION_PRIOR = 0.5
DEFAULT_SIGNIFICANCE_THRESHOLD = 10.0
