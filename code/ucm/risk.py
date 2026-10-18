import numpy as np
from scipy.optimize import minimize


def sensitivity(J, M):
    """Return the GMM sensitivity (J' M J)^(-1) J' M."""
    J = np.asarray(J, dtype=float)
    M = np.asarray(M, dtype=float)
    return np.linalg.solve(J.T @ M @ J, J.T @ M)


def weight_from_sensitivity(J, K, rho=1.0):
    """Section-4.2 positive-definite weight inducing a sensitivity K.

    Requires ``K @ J = I`` and returns ``K.T @ K + rho * P_{C(J)^perp}``.
    """
    J = np.asarray(J, dtype=float)
    K = np.atleast_2d(np.asarray(K, dtype=float))
    if J.ndim == 1:
        J = J[:, None]
    moment_count, parameter_count = J.shape
    if K.shape != (parameter_count, moment_count):
        raise ValueError("K must have shape (number of parameters, number of moments).")
    if np.linalg.matrix_rank(J) < parameter_count:
        raise ValueError("J must have full column rank.")
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError("rho must be finite and strictly positive.")
    if not np.allclose(K @ J, np.eye(parameter_count), atol=1e-7):
        raise ValueError("K must satisfy K J = I.")
    projector_perp = np.eye(moment_count) - J @ np.linalg.solve(J.T @ J, J.T)
    M = K.T @ K + rho * projector_perp
    return (M + M.T) / 2


def asymptotic_variance(K, Omega):
    return K @ Omega @ K.T


def local_bias(K, tau):
    return K @ tau


def amse(K, Omega, tau):
    bias = local_bias(K, tau)
    V = asymptotic_variance(K, Omega)
    return np.sum(bias**2) + np.trace(V)


def _bias_shape(radii, moment_count):
    """Convert Section-4.1 radii or a supplied D matrix to a shape matrix."""
    radii = np.asarray(radii, dtype=float)
    if radii.ndim == 1:
        if radii.size != moment_count:
            raise ValueError("radii must have one entry per projected moment.")
        if not np.all(np.isfinite(radii)) or np.any(radii < 0):
            raise ValueError("radii must be finite and nonnegative.")
        return np.diag(radii)
    if radii.ndim != 2 or radii.shape[0] != moment_count:
        raise ValueError("D must have one row per projected moment.")
    if not np.all(np.isfinite(radii)):
        raise ValueError("D must be finite.")
    return radii


def ellipsoid_bias_bound(K, radii, contrast=None):
    """Worst local bias over {tau = D u: ||u||_2 <= 1}.

    For a vector parameter, ``contrast`` selects the scalar target a'beta.
    It can be omitted only when K has one row.
    """
    K = np.atleast_2d(np.asarray(K, dtype=float))
    D = _bias_shape(radii, K.shape[1])
    if contrast is None:
        if K.shape[0] != 1:
            raise ValueError("contrast is required for a vector parameter.")
        target_sensitivity = K[0]
    else:
        contrast = np.asarray(contrast, dtype=float).reshape(-1)
        if contrast.size != K.shape[0]:
            raise ValueError("contrast must match the parameter dimension.")
        target_sensitivity = contrast @ K
    return float(np.linalg.norm(target_sensitivity @ D, ord=2))


def ellipsoid_risk_from_sensitivity(K, Omega, radii):
    """Section-4.1 worst-case AMSE: ||K D||_op^2 + tr(K Omega K')."""
    K = np.atleast_2d(np.asarray(K, dtype=float))
    Omega = np.asarray(Omega, dtype=float)
    if Omega.shape != (K.shape[1], K.shape[1]):
        raise ValueError("Omega must match the number of projected moments.")
    D = _bias_shape(radii, K.shape[1])
    return float(
        np.linalg.norm(K @ D, ord=2) ** 2
        + np.trace(K @ Omega @ K.T)
    )


def ellipsoid_risk(J, M, Omega, radii):
    return ellipsoid_risk_from_sensitivity(sensitivity(J, M), Omega, radii)


def optimal_ellipsoid_sensitivity(J, Omega, radii):
    """Solve the Section-4.2 K problem subject to K J = I.

    The scalar-parameter case has a closed form. For a vector parameter,
    the convex operator-norm problem is solved directly with SLSQP.
    """
    J = np.asarray(J, dtype=float)
    if J.ndim == 1:
        J = J[:, None]
    Omega = np.asarray(Omega, dtype=float)
    moment_count, parameter_count = J.shape
    if Omega.shape != (moment_count, moment_count):
        raise ValueError("Omega must be square and match J's rows.")
    if not np.all(np.isfinite(J)) or not np.all(np.isfinite(Omega)):
        raise ValueError("J and Omega must be finite.")
    if np.linalg.matrix_rank(J) < parameter_count:
        raise ValueError("J must have full column rank.")
    Omega = (Omega + Omega.T) / 2
    if np.linalg.eigvalsh(Omega).min() <= 0:
        raise ValueError("Omega must be positive definite.")
    D = _bias_shape(radii, moment_count)

    if parameter_count == 1:
        # For a row vector K, ||K D||_op^2 = K D D' K'.
        risk_covariance = Omega + D @ D.T
        inverse_j = np.linalg.solve(risk_covariance, J)
        return (inverse_j / (J.T @ inverse_j)).T

    inverse_j = np.linalg.solve(Omega, J)
    initial = np.linalg.solve(J.T @ inverse_j, inverse_j.T).reshape(-1)

    def objective(flat_K):
        K = flat_K.reshape(parameter_count, moment_count)
        return ellipsoid_risk_from_sensitivity(K, Omega, D)

    def constraint(flat_K):
        K = flat_K.reshape(parameter_count, moment_count)
        return (K @ J - np.eye(parameter_count)).reshape(-1)

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        constraints={"type": "eq", "fun": constraint},
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    if not result.success:
        raise RuntimeError(f"Risk-optimal sensitivity optimization failed: {result.message}")
    K = result.x.reshape(parameter_count, moment_count)
    if not np.allclose(K @ J, np.eye(parameter_count), atol=1e-7):
        raise RuntimeError("Risk-optimal sensitivity violates K J = I.")
    return K


def optimal_ellipsoid_weight(J, Omega, radii, rho=1.0):
    """Return the Section-4.2 positive-definite weight inducing K_opt."""
    K = optimal_ellipsoid_sensitivity(J, Omega, radii)
    return weight_from_sensitivity(J, K, rho)
