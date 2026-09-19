import numpy as np
import pytest

from ucm.dgp import generate_data


def test_shapes():
    data = generate_data(n=100, a=np.sqrt(0.5), seed=1)

    assert data["y"].shape == (100,)
    assert data["x"].shape == (100,)
    assert data["z"].shape == (100,)
    assert data["w"].shape == (100, 2)
    assert data["h_iv"].shape == (100,)
    assert data["h_sv"].shape == (100,)


def test_seed_reproducibility():
    data1 = generate_data(n=100, a=np.sqrt(0.5), seed=1)
    data2 = generate_data(n=100, a=np.sqrt(0.5), seed=1)

    for key in data1:
        np.testing.assert_allclose(data1[key], data2[key])


def test_oracle_projected_regressors():
    a = np.sqrt(0.25)
    data = generate_data(n=100, a=a, seed=1)

    np.testing.assert_allclose(data["h_iv"], a * data["z"] / 2)

    np.testing.assert_allclose(data["h_sv"], data["x"] - data["w"][:, 0] - a * data["w"][:, 1])


def test_correct_specification_moments():
    data = generate_data(n=200000, a=np.sqrt(0.5), beta=1.0, seed=1)
    resid = data["y"] - data["x"]

    assert abs(np.mean(data["h_iv"] * resid)) < 0.02
    assert abs(np.mean(data["h_sv"] * resid)) < 0.02


def test_local_misspecification():
    n, a, tau_z, tau_w = 1000, np.sqrt(0.5), 1.0, 0.5

    base = generate_data(n=n, a=a, seed=1)
    misspec = generate_data(n=n, a=a, tau_z=tau_z, tau_w=tau_w, seed=1)

    expected_change = tau_z * base["z"] / (a * np.sqrt(n)) + tau_w * base["h_sv"] / ((1 - a**2) * np.sqrt(n))

    np.testing.assert_allclose(misspec["y"] - base["y"], expected_change)


def test_fixed_sv_strength():
    a = np.sqrt(2)
    data = generate_data(n=100, a=a, c=1.0, seed=1)

    np.testing.assert_allclose(
        data["h_sv"], data["x"] - data["w"][:, 0] - a * data["w"][:, 1]
    )


def test_positive_a():
    with pytest.raises(ValueError):
        generate_data(n=100, a=0.0)

    with pytest.raises(ValueError):
        generate_data(n=100, a=1.0)