import numpy as np


def equal_weight(q):
    return np.eye(q)


def efficient_weight(X, y, H, beta_hat):
    X = np.asarray(X).reshape(len(y), -1)
    H = np.asarray(H).reshape(len(y), -1)

    resid = y - X @ np.atleast_1d(beta_hat)
    scores = H * resid[:, None]
    Omega = scores.T @ scores / len(y)

    return np.linalg.pinv(Omega)


def oracle_weight(a, c=1.0):
    return np.diag([1 / a**2, 1 / (2 * c**2)])

def oracle_projected_weight(a, c=1.0):
    """Inverse first-order score covariance with sample-estimated projections."""
    return np.diag([1 / a**2, 1 / c**2])
