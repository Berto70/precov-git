"""
IAU two-parameter H-G apparent magnitude model (Bowell et al. 1989),
used as the photometric viability gate in Sec. 3.2.
"""
from __future__ import annotations

import numpy as np

from ..constants import HG_PHI1_A, HG_PHI1_B, HG_PHI2_A, HG_PHI2_B, HG_DEFAULT_G


def phase_function_phi(alpha_rad: float, A: float, B: float) -> float:
    """phi_i(alpha) = exp(-A * tan(alpha/2)^B)."""
    return np.exp(-A * np.tan(np.clip(alpha_rad, 0.0, np.pi - 1e-9) / 2.0) ** B)


def hg_apparent_magnitude(H: float, G: float, r_helio_au: float, r_topo_au: float,
                           phase_angle_rad: float) -> float:
    """Predicted apparent V magnitude at the given heliocentric distance,
    topocentric distance, and Sun-body-observer phase angle.

    V = H + 5*log10(r*Delta) - 2.5*log10[(1-G)*phi1(alpha) + G*phi2(alpha)]
    """
    if G is None:
        G = HG_DEFAULT_G
    phi1 = phase_function_phi(phase_angle_rad, HG_PHI1_A, HG_PHI1_B)
    phi2 = phase_function_phi(phase_angle_rad, HG_PHI2_A, HG_PHI2_B)
    flux_term = (1.0 - G) * phi1 + G * phi2
    flux_term = max(flux_term, 1e-12)  # guard against pathological alpha
    return H + 5.0 * np.log10(r_helio_au * r_topo_au) - 2.5 * np.log10(flux_term)
