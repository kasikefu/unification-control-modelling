# `ucm`: unified deconfounding with IV and SV

Core estimation package for **A Unified Framework for Deconfounding with
Instrumental and Surrogate Variables**, shared by the two Monte Carlo studies.
Start with `estimators.py`; projection scores, risk geometry and inference
live in separate modules so their roles are visible.

Install the package from the repository root with `python -m pip install -e .`,
or put `code/` on `PYTHONPATH`. The experiment entry points already add it
to `sys.path`.

## Module map

| Module | Main interfaces | Role |
|---|---|---|
| `dgp.py` | `generate_data` | Gaussian IV/SV design and local moment shifts |
| `projected.py` | `sample_projections`, `projected_scores`, `projected_covariance` | Sample projections and their first-order covariance |
| `estimators.py` | `iv`, `sv`, `combined`, `efficient_gmm`, `local_risk_gmm` | Single-source, equal, efficient and risk-optimal estimates |
| `risk.py` | `sensitivity`, `ellipsoid_bias_bound`, `ellipsoid_risk`, `optimal_ellipsoid_sensitivity`, `weight_from_sensitivity` | Local bias, worst-case risk and weights inducing a chosen sensitivity |
| `inference.py` | `gmm_variance`, `wald_ci`, `robust_ci`, `robust_gmm_ci` | Variance, Wald and local-robust intervals |
| `weighting.py` | `oracle_projected_weight` | Oracle score-covariance weight used by the efficiency experiment |

`weighting.py` also contains generic/older weight helpers. The production
efficient estimator obtains its feasible weight through
`projected.projected_weight`; it does not call `weighting.efficient_weight`.
`utils.py` is currently an empty placeholder.

## The computation

```text
(X, Y, Z, W)
  -> sample projections (h_iv, h_sv)
  -> initial identity-weight GMM estimate
  -> projected scores and covariance Omega
  -> efficient or local-risk sensitivity/weight
  -> estimate and interval
```

`h_iv` is the fitted projection of `X` on `Z`; `h_sv` is the residual from
projecting `X` on `W`. `sample_projections` uses the columns supplied by the
caller and does not add an intercept. In the simulation, the population
variables are zero-mean. For other applications, supply the intended design
columns explicitly.

The DGP dictionary also contains population, or oracle, `h_iv` and `h_sv`.
The simulation estimators recompute the sample projections in each replication;
the oracle regressors are used to construct/check the design.

## Efficient weighting and local-risk weighting

For stacked moments, `G = H.T @ X / n` is the sample Jacobian and
`K = (G.T @ M @ G)^(-1) @ G.T @ M` is the sensitivity.

- `efficient_gmm` starts from identity weighting and estimates the inverse
  covariance of the projected first-order scores. `channel` selects `iv`,
  `sv` or `combined`.
- `local_risk_gmm` uses the same preliminary estimate, then minimizes
  `||K D||_op^2 + tr(K Omega K.T)` subject to `K G = I`.
- `weight_from_sensitivity` constructs
  `M = K.T @ K + rho * P_{C(G)^perp}`, with `rho > 0`. This is a
  positive-definite member of the weight family in manuscript Section 4.2;
  the feasible construction is described in Section 4.3.
  Different members can induce the same sensitivity and risk.

`radii` can be a nonnegative vector, interpreted as `diag(radii)`, or a
matrix `D`. The uncertainty set is `{tau = D u: ||u||_2 <= 1}`. For a scalar
coefficient the optimum has a closed form; vector coefficients use SLSQP.

## A scalar example

```python
from ucm.dgp import generate_data
from ucm.estimators import local_risk_gmm
from ucm.inference import robust_gmm_ci

data = generate_data(n=600, a=2.0, c=1.0, seed=123)
x, y, z, w = (data[name] for name in ("x", "y", "z", "w"))
radii = [1.0, 1.0]  # both moment blocks may deviate
beta_hat, weight = local_risk_gmm(x, y, z, w, radii, return_weight=True)
low, high = robust_gmm_ci(x, y, z, w, beta_hat, weight, radii)
```

`beta_hat` is an array, including in the scalar case. Set
`return_sensitivity=True` to retrieve `K`; it can be combined with
`return_weight=True`.

## Inference conventions

`gmm_variance` returns the covariance of the root-n limit, not the variance
of the finite-sample estimate. `standard_error(V, n)` supplies the `1/sqrt(n)`
scaling. When estimating projected moments, pass `projected_scores` to
`gmm_variance`; its fallback uses unadjusted moment products.

`robust_ci(beta_hat, sigma, bias_bound, n)` takes the root-n standard deviation
`sigma` and the worst local bias bound, both before division by `sqrt(n)`.
`robust_gmm_ci` builds those quantities from the projected scores. For a vector
coefficient, supply `contrast` to choose the scalar target.

The Replogle runner implements its centered and residualized moments and
batch-clustered covariance locally. Its model functions do not call these
simulation estimators. See [the application guide](../experiments/replogle/README.md)
for that covariance convention.

## Checks

From the repository root:

```bash
python -m pytest code/tests/test_projected.py code/tests/test_risk.py code/tests/test_local_misspec.py -q
PYTHONPATH=code python code/test_bug/check_local_risk_theory.py
```

The independent diagnostic compares hand-derived sensitivities, risks and
positive-definite weights in four population designs. It writes 40 checks
to `results/theory_checks/local_risk_theory_check.csv`.
