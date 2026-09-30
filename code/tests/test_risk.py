import numpy as np

from ucm.dgp import generate_data
from ucm.estimators import combined, local_risk_gmm
from ucm.inference import robust_gmm_ci, wald_ci
from ucm.projected import projected_covariance, projected_scores, sample_projections
from ucm.risk import (
    ellipsoid_bias_bound,
    ellipsoid_risk,
    optimal_ellipsoid_sensitivity,
    optimal_ellipsoid_weight,
    sensitivity,
    weight_from_sensitivity,
)


def test_ellipsoid_optimum_matches_balanced_design():
    J = np.array([[2.0], [1.0]])
    Omega = np.diag([4.0, 1.0])
    kappa = 2.0
    cases = (
        ([kappa, 0.0], 4 / (8 + kappa**2)),
        ([0.0, kappa], (1 + kappa**2) / (2 + kappa**2)),
        ([kappa, kappa], 4 * (1 + kappa**2) / (8 + 5 * kappa**2)),
    )
    for radii, expected_iv_share in cases:
        K = optimal_ellipsoid_sensitivity(J, Omega, radii)
        M = optimal_ellipsoid_weight(J, Omega, radii)
        np.testing.assert_allclose(K @ J, np.eye(1))
        assert np.linalg.eigvalsh(M).min() > 0
        np.testing.assert_allclose(K[0, 0] * J[0, 0], expected_iv_share)
        np.testing.assert_allclose(sensitivity(J, M), K)
        assert ellipsoid_risk(J, M, Omega, radii) <= ellipsoid_risk(
            J, np.linalg.inv(Omega), Omega, radii
        ) + 1e-12
    np.testing.assert_allclose(
        optimal_ellipsoid_sensitivity(J, Omega, [0, 0]), [[0.25, 0.5]]
    )
    np.testing.assert_allclose(
        ellipsoid_bias_bound([0.25, -0.5], [2, 1]), np.sqrt(0.5)
    )


def test_vector_zero_bias_case_recovers_efficient_sensitivity():
    J = np.array([[1.0, 0.2], [0.3, 1.0], [0.5, -0.4], [0.1, 0.7]])
    Omega = np.diag([1.0, 2.0, 1.5, 0.8])
    expected = np.linalg.solve(J.T @ np.linalg.solve(Omega, J),
                               J.T @ np.linalg.inv(Omega))
    actual = optimal_ellipsoid_sensitivity(J, Omega, np.zeros(4))
    np.testing.assert_allclose(actual @ J, np.eye(2), atol=1e-7)
    np.testing.assert_allclose(actual, expected, atol=1e-5)


def test_local_risk_gmm_uses_appendix_f_first_step():
    data = generate_data(n=400, a=2.0, c=1.0, tau_z=2.0, seed=19)
    x, y, z, w = (data[key] for key in ("x", "y", "z", "w"))
    h_iv, h_sv = sample_projections(x, z, w)
    H = np.column_stack((h_iv, h_sv))
    beta0 = combined(x, y, h_iv, h_sv)
    Omega = projected_covariance(projected_scores(x, y, z, w, beta0, h_iv, h_sv))
    J = H.T @ x.reshape(-1, 1) / len(y)
    expected_K = optimal_ellipsoid_sensitivity(J, Omega, [2.0, 0.0])
    expected_beta = expected_K @ (H.T @ y / len(y))
    beta_hat, weight, K = local_risk_gmm(
        x, y, z, w, [2.0, 0.0], return_weight=True, return_sensitivity=True
    )
    np.testing.assert_allclose(K, expected_K)
    np.testing.assert_allclose(weight, weight_from_sensitivity(J, expected_K))
    assert np.linalg.eigvalsh(weight).min() > 0
    np.testing.assert_allclose(beta_hat, expected_beta)


def test_robust_gmm_ci_uses_sample_variance_and_ellipsoid_bound():
    data = generate_data(n=400, a=2.0, c=1.0, tau_w=1.0, seed=20)
    x, y, z, w = (data[key] for key in ("x", "y", "z", "w"))
    beta_hat, weight = local_risk_gmm(x, y, z, w, [0.0, 1.0], return_weight=True)
    low, high, sigma, bound = robust_gmm_ci(
        x, y, z, w, beta_hat, weight, [0.0, 1.0], return_details=True
    )
    h_iv, h_sv = sample_projections(x, z, w)
    H = np.column_stack((h_iv, h_sv))
    J = H.T @ x.reshape(-1, 1) / len(y)
    K = sensitivity(J, weight).reshape(-1)
    Omega = projected_covariance(projected_scores(x, y, z, w, beta_hat, h_iv, h_sv))
    np.testing.assert_allclose(sigma**2, K @ Omega @ K)
    np.testing.assert_allclose(bound, abs(K[1]))
    assert high - low > 2 * 1.959963984540054 * sigma / np.sqrt(len(y))
    zero_low, zero_high = robust_gmm_ci(x, y, z, w, beta_hat, weight, [0, 0])
    np.testing.assert_allclose(
        [zero_low, zero_high], wald_ci(beta_hat.item(), sigma / np.sqrt(len(y)))
    )


def test_robust_gmm_ci_supports_vector_contrasts():
    rng = np.random.default_rng(123)
    n = 600
    z = rng.normal(size=(n, 2))
    w = rng.normal(size=(n, 2))
    x = z + 0.4 * w + rng.normal(size=(n, 2))
    beta = np.array([1.0, -0.5])
    y = x @ beta + 0.3 * w[:, 0] + rng.normal(size=n)
    h_iv, h_sv = sample_projections(x, z, w)
    H = np.column_stack((h_iv, h_sv))
    beta_hat = combined(x, y, h_iv, h_sv)
    J = H.T @ x / n
    K = np.linalg.solve(J.T @ J, J.T)
    M = weight_from_sensitivity(J, K)
    contrast = np.array([1.0, -1.0])
    low, high, sigma, bound = robust_gmm_ci(
        x, y, z, w, beta_hat, M, np.ones(4), h_iv, h_sv,
        contrast=contrast, return_details=True,
    )
    scores = projected_scores(x, y, z, w, beta_hat, h_iv, h_sv)
    Omega = projected_covariance(scores)
    target_K = contrast @ K
    np.testing.assert_allclose(sigma**2, target_K @ Omega @ target_K)
    np.testing.assert_allclose(bound, np.linalg.norm(target_K))
    assert low < contrast @ beta_hat < high
