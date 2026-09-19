import numpy as np

from ucm.dgp import generate_data
from ucm.estimators import combined
from ucm.projected import projected_scores, projected_weight, sample_projections


def test_sample_projected_moments_match_projection_matrices():
    data = generate_data(n=100, a=np.sqrt(2), c=1.0, seed=1)
    x, y, z, w = data["x"], data["y"], data["z"], data["w"]
    h_iv, h_sv = sample_projections(x, z, w)
    pz = np.outer(z, z) / (z @ z)
    pw = w @ np.linalg.inv(w.T @ w) @ w.T

    np.testing.assert_allclose(h_iv[:, 0], pz @ x)
    np.testing.assert_allclose(h_sv[:, 0], (np.eye(len(x)) - pw) @ x)

    beta_hat = combined(x, y, h_iv, h_sv)
    resid = y - x * beta_hat.item()
    scores = projected_scores(x, y, z, w, beta_hat, h_iv, h_sv)
    expected_iv = h_iv[:, 0] * (resid - pz @ resid) + x * (pz @ resid)
    expected_sv = h_sv[:, 0] * (resid - pw @ resid)
    np.testing.assert_allclose(scores[:, 0], expected_iv)
    np.testing.assert_allclose(scores[:, 1], expected_sv)
    assert np.linalg.eigvalsh(projected_weight(x, y, z, w, beta_hat, h_iv, h_sv)).min() > 0


def test_efficient_gmm_uses_projected_score_weight():
    from ucm.estimators import efficient_gmm

    data = generate_data(n=200, a=2.0, c=1.0, seed=2)
    x, y, z, w = data["x"], data["y"], data["z"], data["w"]
    h_iv, h_sv = sample_projections(x, z, w)
    beta0 = combined(x, y, h_iv, h_sv)
    expected_weight = projected_weight(x, y, z, w, beta0, h_iv, h_sv)
    expected_beta = combined(x, y, h_iv, h_sv, expected_weight)

    beta_hat, weight = efficient_gmm(x, y, z, w, return_weight=True)
    np.testing.assert_allclose(weight, expected_weight)
    np.testing.assert_allclose(beta_hat, expected_beta)


def test_single_channel_two_step_weights_use_own_scores():
    from ucm.estimators import efficient_gmm, gmm
    from ucm.projected import projected_covariance

    data = generate_data(n=200, a=1.0, c=1.0, seed=3)
    x, y, z, w = data["x"], data["y"], data["z"], data["w"]
    h_iv, h_sv = sample_projections(x, z, w)
    for channel, H, column in (("iv", h_iv, 0), ("sv", h_sv, 1)):
        beta0 = gmm(x, y, H)
        scores = projected_scores(x, y, z, w, beta0, h_iv, h_sv)[:, [column]]
        expected_weight = np.linalg.pinv(projected_covariance(scores))
        beta_hat, weight = efficient_gmm(
            x, y, z, w, h_iv, h_sv, return_weight=True, channel=channel
        )
        np.testing.assert_allclose(weight, expected_weight)
        np.testing.assert_allclose(beta_hat, gmm(x, y, H, expected_weight))
        np.testing.assert_allclose(beta_hat, beta0)
