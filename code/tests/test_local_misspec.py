import numpy as np

from experiments.local_misspec.run_local_misspec import (
    A, IDENTIFICATION_FLOOR, design_points, estimate, population_risk,
)
from ucm.dgp import generate_data
from ucm.estimators import combined, local_risk_gmm
from ucm.projected import sample_projections


def test_uncertainty_sets_cover_all_vertices_and_population_optimum():
    designs = list(design_points((0.0, 2.0)))
    assert [(regime, len(vertices)) for regime, _, _, vertices in designs] == [
        ("none", 1), ("iv", 2), ("sv", 2), ("both", 4)
    ]
    for _, _, bounds, vertices in designs:
        for tau_z, tau_w in vertices:
            assert abs(tau_z) <= bounds[0] and abs(tau_w) <= bounds[1]
        optimum = population_risk("local_risk", bounds)
        for method in ("iv", "sv", "equal"):
            assert optimum <= population_risk(method, bounds) + 1e-12


def test_simulation_estimators_match_equal_and_local_risk_implementations():
    data = generate_data(n=300, a=A, c=1.0, tau_z=2.0, seed=31)
    x, y, z, w = (data[key] for key in ("x", "y", "z", "w"))
    h_iv, h_sv = sample_projections(x, z, w)
    results = estimate(data, y, h_iv, h_sv, (2.0, 0.0))
    equal = combined(x, y, h_iv, h_sv, np.eye(2))
    local = local_risk_gmm(x, y, z, w, (2.0, 0.0), h_iv, h_sv)
    np.testing.assert_allclose(results["equal"]["beta_hat"], equal.item())
    np.testing.assert_allclose(results["local_risk"]["beta_hat"], local.item())
    _, weight = local_risk_gmm(
        x, y, z, w, (2.0, 0.0), h_iv, h_sv, return_weight=True
    )
    G = np.column_stack((h_iv, h_sv)).T @ x.reshape(-1, 1) / len(y)
    np.testing.assert_allclose(np.trace(weight), 1.0)
    assert float((G.T @ weight @ G).item()) >= IDENTIFICATION_FLOOR * (1 - 1e-10)
    assert results["local_risk"]["robust_length"] >= results["local_risk"]["wald_length"]


def test_balanced_single_source_variances_and_strict_local_risk_gains():
    np.testing.assert_allclose([A**2 / 2, 1.0], [2.0, 1.0])
    np.testing.assert_allclose(population_risk("iv", (0.0, 0.0)), 1.0)
    np.testing.assert_allclose(population_risk("sv", (0.0, 0.0)), 1.0)
    for bounds in ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)):
        optimum = population_risk("local_risk", bounds)
        for method in ("iv", "sv", "equal"):
            assert optimum < population_risk(method, bounds)

