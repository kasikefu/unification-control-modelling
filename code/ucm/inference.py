import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from ucm.projected import projected_covariance, projected_scores, sample_projections
from ucm.risk import ellipsoid_bias_bound, sensitivity


def gmm_variance(X, y, H, beta_hat, M=None, scores=None):
    """Estimate asymptotic variance V(M)."""
    X = np.asarray(X).reshape(len(y), -1)
    H = np.asarray(H).reshape(len(y), -1)
    M = np.eye(H.shape[1]) if M is None else M

    G = H.T @ X / len(y)
    if scores is None:
        resid = y - X @ np.atleast_1d(beta_hat)
        scores = H * resid[:, None]
    else:
        scores = np.asarray(scores).reshape(len(y), -1)
        scores = scores - scores.mean(axis=0)
    Omega = scores.T @ scores / len(y)
    K = np.linalg.solve(G.T @ M @ G, G.T @ M)

    return K @ Omega @ K.T


def standard_error(V, n):
    """Finite-sample standard error from asymptotic variance."""
    return np.sqrt(np.diag(np.atleast_2d(V)) / n)


def wald_ci(beta_hat, se, alpha=0.05):
    """Two-sided Wald confidence interval."""
    z = norm.ppf(1 - alpha / 2)
    return beta_hat - z * se, beta_hat + z * se


def wald_test(beta_hat, beta0, se):
    """Two-sided Wald test."""
    z = (beta_hat - beta0) / se
    p = 2 * norm.sf(abs(z))
    return z, p


def robust_critical_value(t, alpha=0.05):
    """Critical value for |N(t,1)|."""
    t = abs(float(t))
    if not np.isfinite(t):
        raise ValueError("t must be finite.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one.")
    f = lambda c: norm.cdf(c - t) - norm.cdf(-c - t) - (1 - alpha)
    return brentq(f, 0, t + norm.ppf(1 - alpha / 2) + 10)


def robust_ci(beta_hat, sigma, bias_bound, n, alpha=0.05):
    """Misspecification-robust CI from Remark 3."""
    beta_hat = float(beta_hat)
    sigma = float(sigma)
    bias_bound = float(bias_bound)
    if not np.isfinite(beta_hat):
        raise ValueError("beta_hat must be finite.")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and strictly positive.")
    if not np.isfinite(bias_bound) or bias_bound < 0:
        raise ValueError("bias_bound must be finite and nonnegative.")
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("n must be a positive integer.")
    cv = robust_critical_value(bias_bound / sigma, alpha)
    half_width = cv * sigma / np.sqrt(n)
    return beta_hat - half_width, beta_hat + half_width


def robust_gmm_ci(X, y, Z, W, beta_hat, M, radii, h_iv=None, h_sv=None,
                  contrast=None, alpha=0.05, return_details=False):
    """Robust CI for a scalar contrast of a combined estimator (Remark 3)."""
    if h_iv is None or h_sv is None:
        h_iv, h_sv = sample_projections(X, Z, W)
    H = np.column_stack((h_iv, h_sv))
    X = np.asarray(X).reshape(len(y), -1)
    G = H.T @ X / len(y)
    beta_hat = np.asarray(beta_hat, dtype=float).reshape(-1)
    parameter_count = X.shape[1]
    if beta_hat.size != parameter_count:
        raise ValueError("beta_hat must have one entry per regressor.")
    if contrast is None:
        if parameter_count != 1:
            raise ValueError("contrast is required when beta has more than one entry.")
        contrast = np.ones(1)
    else:
        contrast = np.asarray(contrast, dtype=float).reshape(-1)
        if contrast.size != parameter_count or not np.all(np.isfinite(contrast)):
            raise ValueError("contrast must be finite and match the parameter dimension.")
    scores = projected_scores(X, y, Z, W, beta_hat, h_iv, h_sv)
    Omega = projected_covariance(scores)
    K = sensitivity(G, M)
    target_sensitivity = contrast @ K
    sigma = np.sqrt(float(target_sensitivity @ Omega @ target_sensitivity))
    bias_bound = ellipsoid_bias_bound(K, radii, contrast=contrast)
    target_hat = float(contrast @ beta_hat)
    ci = robust_ci(target_hat, sigma, bias_bound, len(y), alpha)
    return (*ci, sigma, bias_bound) if return_details else ci
