import numpy as np


def sample_projections(X, Z, W):
    """Project X onto Z and residualize X against W in the sample."""
    X = np.asarray(X).reshape(len(X), -1)
    Z = np.asarray(Z).reshape(len(X), -1)
    W = np.asarray(W).reshape(len(X), -1)

    h_iv = Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
    h_sv = X - W @ np.linalg.lstsq(W, X, rcond=None)[0]
    return h_iv, h_sv


def projected_scores(X, y, Z, W, beta_hat, h_iv=None, h_sv=None):
    """Plug-in first-order scores from Section 4.3."""
    X = np.asarray(X).reshape(len(y), -1)
    Z = np.asarray(Z).reshape(len(y), -1)
    W = np.asarray(W).reshape(len(y), -1)
    if h_iv is None or h_sv is None:
        h_iv, h_sv = sample_projections(X, Z, W)

    resid = y - X @ np.atleast_1d(beta_hat)
    z_part = Z @ np.linalg.lstsq(Z, resid, rcond=None)[0]
    w_part = W @ np.linalg.lstsq(W, resid, rcond=None)[0]
    psi_iv = h_iv * (resid - z_part)[:, None] + X * z_part[:, None]
    psi_sv = h_sv * (resid - w_part)[:, None]
    return np.column_stack((psi_iv, psi_sv))


def projected_covariance(scores):
    centered = scores - scores.mean(axis=0)
    return centered.T @ centered / len(scores)


def projected_weight(X, y, Z, W, beta_hat, h_iv=None, h_sv=None, channel="combined"):
    scores = projected_scores(X, y, Z, W, beta_hat, h_iv, h_sv)
    p = np.asarray(X).reshape(len(y), -1).shape[1]
    if channel == "iv":
        scores = scores[:, :p]
    elif channel == "sv":
        scores = scores[:, p:]
    elif channel != "combined":
        raise ValueError("channel must be iv, sv, or combined.")
    return np.linalg.pinv(projected_covariance(scores))
