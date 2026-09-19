import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


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
    f = lambda c: norm.cdf(c - t) - norm.cdf(-c - t) - (1 - alpha)
    return brentq(f, 0, abs(t) + norm.ppf(1 - alpha / 2) + 10)


def robust_ci(beta_hat, sigma, bias_bound, n, alpha=0.05):
    """Misspecification-robust CI from Remark 9."""
    cv = robust_critical_value(bias_bound / sigma, alpha)
    half_width = cv * sigma / np.sqrt(n)
    return beta_hat - half_width, beta_hat + half_width