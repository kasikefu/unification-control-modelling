import numpy as np


def generate_data(n, a, beta=1.0, tau_z=0.0, tau_w=0.0, seed=None, c=None):
    """Generate IV-SV simulation data."""

    if a <= 0 or (c is None and a >= 1):
        raise ValueError("a must be positive and below 1 when c is omitted.")

    c = np.sqrt(1 - a**2) if c is None else c
    if c <= 0:
        raise ValueError("c must be positive.")

    rng = np.random.default_rng(seed)
    u, s, eta, v, eps = rng.normal(size=(5, n))

    z = s + eta
    w = np.column_stack((u, s))
    x = u + a * s + c * v
    y = beta * x + u + eps + tau_z * z / (a * np.sqrt(n)) + tau_w * v / (c * np.sqrt(n))

    # Oracle projected regressors
    h_iv = a * z / 2
    h_sv = c * v

    return {"y": y, "x": x, "z": z, "w": w, "h_iv": h_iv, "h_sv": h_sv}