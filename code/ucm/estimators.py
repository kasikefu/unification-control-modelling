import numpy as np

from ucm.projected import (
    projected_covariance, projected_scores, projected_weight, sample_projections,
)
from ucm.risk import optimal_ellipsoid_sensitivity, weight_from_sensitivity


def gmm(X, y, H, M=None):
    """GMM estimator based on E[H_i (Y_i - X_i' beta)] = 0."""
    X = np.atleast_2d(X).T if np.asarray(X).ndim == 1 else np.asarray(X)
    H = np.atleast_2d(H).T if np.asarray(H).ndim == 1 else np.asarray(H)
    M = np.eye(H.shape[1]) if M is None else M

    G = H.T @ X / len(y)
    s = H.T @ y / len(y)

    return np.linalg.solve(G.T @ M @ G, G.T @ M @ s)


def iv(X, y, H_iv, M=None):
    """IV-only estimator."""
    return gmm(X, y, H_iv, M)


def sv(X, y, H_sv, M=None):
    """SV-only estimator."""
    return gmm(X, y, H_sv, M)


def combined(X, y, H_iv, H_sv, M=None):
    """Combined IV-SV estimator."""
    return gmm(X, y, np.column_stack((H_iv, H_sv)), M)


def efficient_gmm(X, y, Z, W, h_iv=None, h_sv=None, return_weight=False, channel="combined"):
    """Two-step GMM for the IV, SV, or combined projected moments."""
    if h_iv is None or h_sv is None:
        h_iv, h_sv = sample_projections(X, Z, W)
    if channel == "iv":
        H = h_iv
    elif channel == "sv":
        H = h_sv
    elif channel == "combined":
        H = np.column_stack((h_iv, h_sv))
    else:
        raise ValueError("channel must be iv, sv, or combined.")
    beta0 = gmm(X, y, H)
    M = projected_weight(X, y, Z, W, beta0, h_iv, h_sv, channel)
    beta_hat = gmm(X, y, H, M)
    return (beta_hat, M) if return_weight else beta_hat


def local_risk_gmm(
    X, y, Z, W, radii, h_iv=None, h_sv=None, return_weight=False,
    return_sensitivity=False, rho=1.0,
):
    """Feasible risk-optimal projected GMM from Section 4.3."""
    if h_iv is None or h_sv is None:
        h_iv, h_sv = sample_projections(X, Z, W)
    H = np.column_stack((h_iv, h_sv))
    beta0 = gmm(X, y, H)
    scores = projected_scores(X, y, Z, W, beta0, h_iv, h_sv)
    Omega = projected_covariance(scores)
    G = H.T @ np.asarray(X).reshape(len(y), -1) / len(y)
    K = optimal_ellipsoid_sensitivity(G, Omega, radii)
    M = weight_from_sensitivity(G, K, rho)
    stacked_outcome_moment = H.T @ np.asarray(y) / len(y)
    beta_hat = K @ stacked_outcome_moment
    if return_weight and return_sensitivity:
        return beta_hat, M, K
    if return_weight:
        return beta_hat, M
    if return_sensitivity:
        return beta_hat, K
    return beta_hat
