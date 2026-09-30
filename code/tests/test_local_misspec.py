import numpy as np

from experiments.local_misspec.run_local_misspec import (
    A, design_points, estimate, population_risk,
)
from ucm.dgp import generate_data
from ucm.estimators import combined, local_risk_gmm
from ucm.projected import sample_projections
from ucm.risk import sensitivity


def test_uncertainty_sets_use_appendix_f_ellipsoid_boundary():
    designs = list(design_points((0.0, 2.0)))
    assert [(regime, len(points)) for regime, _, _, points in designs] == [
        ("none", 1), ("iv", 2), ("sv", 2), ("both", 16)
    ]
    for _, _, radii, points in designs:
        for tau_z, tau_w in points:
            normalized = sum(
                (tau / radius) ** 2
                for tau, radius in zip((tau_z, tau_w), radii)
                if radius > 0
            )
            assert normalized <= 1 + 1e-12
            assert all(
                abs(tau) < 1e-12
                for tau, radius in zip((tau_z, tau_w), radii)
                if radius == 0
            )
        optimum = population_risk("local_risk", radii)
        for method in ("iv", "sv", "equal"):
            assert optimum <= population_risk(method, radii) + 1e-12


def test_simulation_estimators_match_appendix_f_implementation():
    data = generate_data(n=300, a=A, c=1.0, tau_z=2.0, seed=31)
    x, y, z, w = (data[key] for key in ("x", "y", "z", "w"))
    h_iv, h_sv = sample_projections(x, z, w)
    results = estimate(data, y, h_iv, h_sv, (2.0, 0.0))
    equal = combined(x, y, h_iv, h_sv, np.eye(2))
    local, weight, K = local_risk_gmm(
        x, y, z, w, (2.0, 0.0), h_iv, h_sv,
        return_weight=True, return_sensitivity=True,
    )
    np.testing.assert_allclose(results["equal"]["beta_hat"], equal.item())
    np.testing.assert_allclose(results["local_risk"]["beta_hat"], local.item())
    J = np.column_stack((h_iv, h_sv)).T @ x.reshape(-1, 1) / len(y)
    np.testing.assert_allclose(K @ J, np.eye(1))
    assert np.linalg.eigvalsh(weight).min() > 0
    np.testing.assert_allclose(sensitivity(J, weight), K)
    assert results["local_risk"]["robust_length"] >= results["local_risk"]["wald_length"]
    for method, radius, jacobian in (
        ("iv", 2.0, J[0, 0]), ("sv", 0.0, J[1, 0]),
    ):
        result = results[method]
        np.testing.assert_allclose(result["bias_bound"], radius / abs(jacobian))
        assert result["robust_covered"] in (True, False)
        assert result["robust_length"] >= result["wald_length"]


def test_balanced_single_source_variances_and_strict_local_risk_gains():
    np.testing.assert_allclose([A**2 / 2, 1.0], [2.0, 1.0])
    np.testing.assert_allclose(population_risk("iv", (0.0, 0.0)), 1.0)
    np.testing.assert_allclose(population_risk("sv", (0.0, 0.0)), 1.0)
    for radii in ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)):
        optimum = population_risk("local_risk", radii)
        for method in ("iv", "sv", "equal"):
            assert optimum < population_risk(method, radii)

