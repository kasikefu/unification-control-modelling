import numpy as np


DEFAULT_IDENTIFICATION_FLOOR = 1e-4


def sensitivity(G, M):
    return np.linalg.solve(G.T @ M @ G, G.T @ M)


def asymptotic_variance(K, Omega):
    return K @ Omega @ K.T


def local_bias(K, tau):
    return K @ tau


def amse(K, Omega, tau):
    bias = local_bias(K, tau)
    V = asymptotic_variance(K, Omega)
    return np.sum(bias**2) + np.trace(V)


def box_bias_bound(K, bounds):
    """Worst-case absolute bias for a scalar target and symmetric box C."""
    K = np.asarray(K, dtype=float).reshape(-1)
    bounds = np.asarray(bounds, dtype=float).reshape(-1)
    if K.size != bounds.size or not np.all(np.isfinite(bounds)) or np.any(bounds < 0):
        raise ValueError("bounds must be finite, nonnegative, and match K.")
    return float(np.abs(K) @ bounds)


def box_risk(G, M, Omega, bounds):
    """Worst-case local AMSE for a scalar parameter over |tau_j| <= bounds_j."""
    G = np.asarray(G, dtype=float)
    if G.shape != (2, 1):
        raise ValueError("box_risk requires one coefficient and two moments.")
    K = sensitivity(G, M).reshape(-1)
    Omega = np.asarray(Omega, dtype=float)
    if Omega.shape != (2, 2):
        raise ValueError("Omega must be a two-by-two covariance matrix.")
    return box_bias_bound(K, bounds) ** 2 + float(K @ Omega @ K)


def optimal_box_sensitivity(
    G, Omega, bounds, identification_floor=DEFAULT_IDENTIFICATION_FLOOR
):
    """Minimize box risk over normalized weights with uniform identification."""
    G = np.asarray(G, dtype=float).reshape(-1)
    Omega = np.asarray(Omega, dtype=float)
    bounds = np.asarray(bounds, dtype=float).reshape(-1)
    if G.size != 2 or Omega.shape != (2, 2):
        raise ValueError("This optimizer requires a scalar target and two moments.")
    if bounds.size != 2 or not np.all(np.isfinite(bounds)) or np.any(bounds < 0):
        raise ValueError("bounds must contain two finite, nonnegative radii.")
    if not np.all(np.isfinite(G)) or not np.all(np.isfinite(Omega)):
        raise ValueError("G and Omega must be finite.")
    g_norm_sq = float(G @ G)
    if g_norm_sq == 0 or np.linalg.eigvalsh(Omega).min() <= 0:
        raise ValueError("G must identify the target and Omega must be positive definite.")
    if not np.isfinite(identification_floor) or identification_floor <= 0:
        raise ValueError("identification_floor must be finite and positive.")
    if identification_floor > g_norm_sq:
        raise ValueError("identification_floor exceeds the maximum normalized identification.")

    # A rank-one trace-one weight that realizes K has identification
    # G.T @ M @ G = 1 / (K @ K), since K @ G = 1.
    max_k_norm_sq = 1.0 / identification_floor

    # Within each sign region, the box criterion is a quadratic form.
    candidates = []
    for signs in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        direction = bounds * signs
        matrix = Omega + np.outer(direction, direction)
        inverse_G = np.linalg.solve(matrix, G)
        K = inverse_G / (G @ inverse_G)
        if np.all(K * signs >= -1e-10):
            candidates.append(K)
    for column in range(2):
        if G[column] != 0:
            K = np.zeros(2)
            K[column] = 1 / G[column]
            candidates.append(K)

    # Include the two endpoints created by the identification constraint.
    center = G / g_norm_sq
    radius_sq = max_k_norm_sq - 1.0 / g_norm_sq
    if radius_sq >= 0:
        perpendicular = np.array([-G[1], G[0]]) / np.sqrt(g_norm_sq)
        radius = np.sqrt(max(0.0, radius_sq))
        candidates.extend((center - radius * perpendicular, center + radius * perpendicular))

    candidates = [
        K for K in candidates
        if K @ K <= max_k_norm_sq * (1.0 + 1e-12)
    ]
    if not candidates:
        raise ValueError("No weight satisfies the requested identification floor.")
    return min(
        candidates,
        key=lambda K: float(K @ Omega @ K) + box_bias_bound(K, bounds) ** 2,
    )


def optimal_box_weight(
    G, Omega, bounds, identification_floor=DEFAULT_IDENTIFICATION_FLOOR
):
    """Risk-optimal PSD weight with trace one and bounded identification."""
    K = optimal_box_sensitivity(G, Omega, bounds, identification_floor)
    # Since K @ G = 1, this rank-one weight has GMM sensitivity K.
    M = np.outer(K, K) / (K @ K)
    identification = float(np.asarray(G).reshape(-1) @ M @ np.asarray(G).reshape(-1))
    if identification < identification_floor * (1.0 - 1e-10):
        raise RuntimeError("Constructed weight violates the identification floor.")
    return M
