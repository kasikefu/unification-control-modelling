import numpy as np

from ucm.dgp import generate_data
from ucm.estimators import combined, local_risk_gmm
from ucm.inference import robust_gmm_ci, wald_ci
from ucm.projected import projected_covariance, projected_scores, sample_projections
from ucm.risk import (
    box_bias_bound, box_risk, optimal_box_sensitivity, optimal_box_weight,
    sensitivity,
)


def test_box_optimum_matches_balanced_design():
    G = np.array([[2.0], [1.0]])
    Omega = np.diag([4.0, 1.0])
    kappa = 2.0
    cases = (
        ([kappa, 0.0], 4 / (8 + kappa**2)),
        ([0.0, kappa], (1 + kappa**2) / (2 + kappa**2)),
        ([kappa, kappa], (2 + kappa**2) / (4 + kappa**2 / 2)),
    )
    for bounds, expected_iv_share in cases:
        K = optimal_box_sensitivity(G, Omega, bounds)
        M = optimal_box_weight(G, Omega, bounds)
        np.testing.assert_allclose(K @ G[:, 0], 1)
        np.testing.assert_allclose(K[0] * G[0, 0], expected_iv_share)
        np.testing.assert_allclose(sensitivity(G, M).reshape(-1), K)
        assert box_risk(G, M, Omega, bounds) <= box_risk(
            G, np.linalg.inv(Omega), Omega, bounds
        ) + 1e-12
    np.testing.assert_allclose(
        optimal_box_sensitivity(G, Omega, [0, 0]), [0.25, 0.5]
    )
    np.testing.assert_allclose(box_bias_bound([0.25, -0.5], [2, 1]), 1.0)


def test_identification_floor_is_normalized_and_can_bind():
    G = np.array([[1.0], [0.0]])
    Omega = np.array([[1.0, 0.9], [0.9, 1.0]])
    floor = 0.8
    K = optimal_box_sensitivity(G, Omega, [0.0, 0.0], floor)
    M = optimal_box_weight(G, Omega, [0.0, 0.0], floor)

    np.testing.assert_allclose(K, [1.0, -0.5])
    np.testing.assert_allclose(np.trace(M), 1.0)
    assert np.linalg.eigvalsh(M).min() >= -1e-12
    np.testing.assert_allclose(float((G.T @ M @ G).item()), floor)
    np.testing.assert_allclose(sensitivity(G, M).reshape(-1), K)


def test_local_risk_gmm_uses_sample_first_step():
    data = generate_data(n=400, a=2.0, c=1.0, tau_z=2.0, seed=19)
    x, y, z, w = (data[key] for key in ("x", "y", "z", "w"))
    h_iv, h_sv = sample_projections(x, z, w)
    H = np.column_stack((h_iv, h_sv))
    beta0 = combined(x, y, h_iv, h_sv)
    Omega = projected_covariance(projected_scores(x, y, z, w, beta0, h_iv, h_sv))
    G = H.T @ x.reshape(-1, 1) / len(y)
    expected_weight = optimal_box_weight(G, Omega, [2.0, 0.0])
    beta_hat, weight = local_risk_gmm(x, y, z, w, [2.0, 0.0], return_weight=True)
    np.testing.assert_allclose(weight, expected_weight)
    np.testing.assert_allclose(beta_hat, combined(x, y, h_iv, h_sv, weight))


def test_robust_gmm_ci_uses_sample_variance_and_box_bound():
    data = generate_data(n=400, a=2.0, c=1.0, tau_w=1.0, seed=20)
    x, y, z, w = (data[key] for key in ("x", "y", "z", "w"))
    beta_hat, weight = local_risk_gmm(x, y, z, w, [0.0, 1.0], return_weight=True)
    low, high, sigma, bound = robust_gmm_ci(
        x, y, z, w, beta_hat, weight, [0.0, 1.0], return_details=True
    )
    h_iv, h_sv = sample_projections(x, z, w)
    H = np.column_stack((h_iv, h_sv))
    G = H.T @ x.reshape(-1, 1) / len(y)
    K = sensitivity(G, weight).reshape(-1)
    Omega = projected_covariance(projected_scores(x, y, z, w, beta_hat, h_iv, h_sv))
    np.testing.assert_allclose(sigma**2, K @ Omega @ K)
    np.testing.assert_allclose(bound, abs(K[1]))
    assert high - low > 2 * 1.959963984540054 * sigma / np.sqrt(len(y))
    zero_low, zero_high = robust_gmm_ci(x, y, z, w, beta_hat, weight, [0, 0])
    np.testing.assert_allclose(
        [zero_low, zero_high], wald_ci(beta_hat.item(), sigma / np.sqrt(len(y)))
    )
