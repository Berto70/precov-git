import numpy as np
import sympy as sp

from precov.linking.likelihood import loglik_ratio_single_image, mahalanobis_nearest, scaled_density
from precov.linking.null_distribution import lambda_of_u, null_moments, NullMomentLookup


def test_mahalanobis_nearest_picks_closest_and_scales_by_covariance():
    S = np.diag([4.0, 1.0])  # anisotropic: x-errors twice as large as y
    catalog = np.array([[5.0, 0.0], [0.0, 3.0]])  # equidistant in raw Euclidean terms
    pred = np.array([0.0, 0.0])
    r, idx = mahalanobis_nearest(pred, S, catalog)
    # Under S, source 0 is at Mahalanobis distance 5/2=2.5; source 1 is at 3/1=3.0
    assert idx == 0
    assert np.isclose(r, 2.5)


def test_mahalanobis_nearest_empty_catalog_returns_inf():
    r, idx = mahalanobis_nearest(np.zeros(2), np.eye(2), np.zeros((0, 2)))
    assert r == np.inf and idx == -1


def test_loglik_ratio_matches_symbolic_derivation_from_f0_f1():
    """Independent ground truth: derive f1(r)/f0(r) symbolically straight
    from the Appendix B primitive definitions (Eq. B2, B3) via SymPy, without
    reference to any typeset form of Eq. (8), and compare numerically to the
    implementation. This is the check that originally caught the three-term
    two-exponent formula (taken from the derived report) as incorrect.
    """
    r_sym, rho_sym, a_sym = sp.symbols("r rho_t alpha", positive=True)
    f0 = 2 * sp.pi * rho_sym * r_sym * sp.exp(-sp.pi * rho_sym * r_sym**2)
    f_ast = r_sym * sp.exp(-r_sym**2 / 2)
    f1 = (1 - a_sym) * f0 + a_sym * (f_ast * sp.exp(-sp.pi * rho_sym * r_sym**2)
                                      + f0 * sp.exp(-r_sym**2 / 2))
    log_ratio_expr = sp.log(sp.simplify(f1 / f0))
    log_ratio_fn = sp.lambdify((r_sym, rho_sym, a_sym), log_ratio_expr, "numpy")

    for rho_tilde in [1e-4, 1e-2, 0.05, 0.3, 1.5]:
        for r in [0.05, 0.5, 1.0, 2.0, 5.0]:
            expected = float(log_ratio_fn(r, rho_tilde, 0.5))
            actual = loglik_ratio_single_image(r, rho_tilde, alpha=0.5)
            assert np.isclose(expected, actual, atol=1e-8), (r, rho_tilde)


def test_loglik_ratio_large_and_positive_for_tight_match_low_density():
    lam = loglik_ratio_single_image(r=0.05, rho_tilde=1e-4, alpha=0.5)
    assert lam > 5.0


def test_loglik_ratio_bounded_and_small_for_distant_match_even_in_crowded_field():
    """In a very crowded field (rho_tilde large), a nearby match carries
    little discriminating power: lambda should stay small in magnitude
    (bounded between log(1-alpha) and ~0), never blow up."""
    for rho_tilde in [0.5, 5.0, 50.0]:
        lam_far = loglik_ratio_single_image(r=5.0, rho_tilde=rho_tilde, alpha=0.5)
        assert np.log(0.5) - 1e-6 <= lam_far <= 1e-6


def test_loglik_ratio_no_catalog_source_is_neutral():
    assert loglik_ratio_single_image(r=np.inf, rho_tilde=0.01, alpha=0.5) == 0.0


def test_loglik_ratio_approaches_log_1_minus_alpha_at_large_r():
    lam = loglik_ratio_single_image(r=50.0, rho_tilde=0.01, alpha=0.5)
    assert np.isclose(lam, np.log(0.5), atol=1e-6)


def test_scaled_density_uses_sqrt_determinant():
    S = np.diag([4.0, 9.0])  # det = 36, sqrt(det) = 6
    assert np.isclose(scaled_density(rho=0.1, S=S), 0.6)


def test_lambda_of_u_matches_lambda_of_r_under_f0_change_of_variables():
    """Cross-check lambda(u, rho_tilde) against loglik_ratio_single_image(r),
    under the null-hypothesis change of variables u = F0(r).

    r_max per rho_tilde is capped so that pi*rho_tilde*r^2 stays below ~20:
    beyond that, u = F0(r) is within float64 epsilon of 1.0 regardless of how
    it's computed (representing '1 - eps' for eps < ~1e-16 is not possible in
    float64), so round-tripping r -> u -> lambda necessarily loses precision
    that the direct r -> lambda path does not. Production code never performs
    that round trip (NullMomentLookup integrates over u directly; candidate
    evaluation calls loglik_ratio_single_image on r directly) -- this test
    exists to validate the u-parameterization over the regime where it is
    numerically meaningful, which comfortably covers where lambda transitions
    from positive to negative.
    """
    alpha = 0.5
    for rho_tilde in [1e-4, 1e-2, 0.05, 0.3, 1.5]:
        r_max = np.sqrt(20.0 / (np.pi * rho_tilde))
        r = np.linspace(0.01, r_max, 40)
        u = -np.expm1(-np.pi * rho_tilde * r**2)

        lam_direct = np.array([loglik_ratio_single_image(ri, rho_tilde, alpha) for ri in r])
        lam_via_u = lambda_of_u(u, rho_tilde, alpha)
        assert np.allclose(lam_direct, lam_via_u, atol=1e-8)


def test_null_moments_match_monte_carlo():
    """Validate the quadrature-based lookup table against direct Monte Carlo
    sampling of r under H0 (Rayleigh with the given rho_tilde), propagated
    through the (now symbolically-verified) lambda(r)."""
    rng = np.random.default_rng(3)
    rho_tilde = 0.02
    alpha = 0.5

    u = rng.uniform(0, 1, size=20000)
    r_samples = np.sqrt(-np.log(1 - u) / (np.pi * rho_tilde))
    lam_samples = lambda_of_u(u, rho_tilde, alpha)

    E_mc, Var_mc = lam_samples.mean(), lam_samples.var()
    E_quad, Var_quad = null_moments(rho_tilde, alpha)

    assert abs(E_mc - E_quad) < 0.03
    assert abs(Var_mc - Var_quad) / Var_quad < 0.08


def test_null_moment_lookup_interpolates_close_to_direct_quadrature():
    lookup = NullMomentLookup(alpha=0.5)
    for rho_tilde in [1e-5, 1e-3, 0.01, 0.1, 1.0, 50.0]:
        E_direct, Var_direct = null_moments(rho_tilde, alpha=0.5)
        E_lookup, Var_lookup = lookup.moments(rho_tilde)
        assert abs(E_direct - E_lookup) < 1e-2
        assert abs(Var_direct - Var_lookup) / max(Var_direct, 1e-6) < 0.05


def test_standardized_significance_is_zero_mean_unit_variance_under_null():
    """End-to-end null-hypothesis calibration check: draw many independent
    per-image lambda values under H0 at a fixed rho_tilde, sum them, and
    verify the standardized statistic (Lambda - N*E)/sqrt(N*Var) behaves like
    a standard normal, matching the paper's central claim that
    'Significance > 10' cleanly separates the tails."""
    rng = np.random.default_rng(11)
    rho_tilde = 0.015
    alpha = 0.5
    n_images = 30
    n_trials = 5000

    E, Var = null_moments(rho_tilde, alpha)

    u = rng.uniform(0, 1, size=(n_trials, n_images))
    lam = lambda_of_u(u, rho_tilde, alpha)
    Lambda = lam.sum(axis=1)
    significance = (Lambda - n_images * E) / np.sqrt(n_images * Var)

    assert abs(significance.mean()) < 0.05
    assert abs(significance.std() - 1.0) < 0.05
    assert np.sum(significance > 10.0) == 0
