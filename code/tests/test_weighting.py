import numpy as np

from ucm.weighting import oracle_weight


def test_oracle_weight_with_fixed_sv_strength():
    np.testing.assert_allclose(oracle_weight(a=np.sqrt(2)), np.diag([0.5, 0.5]))


def test_oracle_weight_with_balanced_strengths():
    a = c = 1 / np.sqrt(2)
    np.testing.assert_allclose(oracle_weight(a, c), np.diag([2.0, 1.0]))
