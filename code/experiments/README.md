# Experiments: compute first, render second

Run every command below from the repository root. Each compute entry point
writes to its own subdirectory of `results/`; the three root notebooks read
those files and export manuscript figures to `figures/`.

| Directory | Compute entry point | Notebook |
|---|---|---|
| `efficiency/` | `run_efficiency.py` | `efficiency_results.ipynb` |
| `local_misspec/` | `run_local_misspec.py` | `local_misspec_results.ipynb` |
| `replogle/` | `run_replogle.py`, then `run_go_analysis.py` | `realdata_batch_optimized.ipynb` |

There is no all-in-one runner in the repository. The commands in the
[top-level README](../../README.md) are the full production sequence.

## Efficiency: Section 5.1

```bash
python code/experiments/efficiency/run_efficiency.py
```

Defaults: 1,000 replications, seed `20260919`, eight workers,
`n = 200, 600, 1000, 1400, 1800`, and `a = 1, 2, 3` with `c=1`.
The DGP is `z=s+eta`, `w=(u,s)`, `x=u+a*s+v`, `y=beta*x+u+eps`,
with independent standard normal components.

Each design includes estimation at `beta=1`, size at `beta=0`, and
local power at `beta=1/sqrt(n)` and `2/sqrt(n)`. It compares unadjusted OLS,
IV, SV, equal GMM, feasible efficient GMM and oracle efficient GMM.
Projections and feasible weights are recomputed in each replication.

| Output under `results/efficiency/` | Consumer |
|---|---|
| `efficiency_replications.csv` | Notebook MSE, size, power and summary tables |
| `efficiency_summary.csv` | Notebook checks on the production settings; numerical reference |
| `efficiency_decomposition.csv`, `efficiency_*story_summary.csv`, `efficiency_main_table.csv`, `efficiency_inference_summary.csv` | Exports of tables computed/displayed by the notebook |

The manuscript figure selects `a=2` and the `2/sqrt(n)` alternative.
The oracle estimate and the other designs remain in the full result tables.

## Local misspecification: Section 5.2

```bash
python code/experiments/local_misspec/run_local_misspec.py
```

Defaults: 1,000 replications, seed `20260920`, eight workers, the same five
sample sizes, `a=2`, `c=1`, `beta=1`, and `kappa = 0, 0.5, 1`.
The nonzero designs use these moment-deviation sets:

| Regime | Radii in `D` | Evaluated directions |
|---|---|---|
| `iv` | `(kappa, 0)` | two endpoints |
| `sv` | `(0, kappa)` | two endpoints |
| `both` | `(kappa, kappa)` | 16 equally spaced directions on the circle |
| `none` | `(0, 0)` | the origin, evaluated once per sample size |

The `both` set is a disk, not a rectangle: `tau_z^2 + tau_w^2 <= kappa^2`.
Within a design, each boundary direction shares the same random draw for a
replication. The summary takes extrema over the evaluated directions.

| Output under `results/local_misspec/` | Consumer |
|---|---|
| `local_misspec_replications.csv` | Notebook minimum local power, reconstructed from saved estimates and robust critical values |
| `local_misspec_corners.csv` | Boundary summaries and notebook consistency checks |
| `local_misspec_worst_case.csv` | Main MSE and Type I error panels; supplementary risk summaries |

The current manuscript/notebook selects **`both`, `kappa=1`**. OLS uses
conventional Wald inference; IV, SV and both GMM methods use local-robust
intervals. Size tests `H0: beta=1`; local power uses `beta=1+2/sqrt(n)`.
The notebook obtains the alternative by shifting each saved estimate by
`2/sqrt(n)`; the linear estimators and their residual-based covariance are
equivariant to this outcome shift.

The former `local_misspec_inference_summary.csv` used a common Wald critical
value and has been removed. It is not an input to the current figure.

## Optional local-risk displays

```bash
python code/experiments/local_misspec/plot_statistica_sinica.py
python code/experiments/local_misspec/summarize_appendix_f.py
```

`plot_statistica_sinica.py` draws all three deviation regimes to
`results/local_misspec/local_misspec_*_combined.pdf`, using the same
OLS-Wald/other-methods-robust convention.

`summarize_appendix_f.py` produces `appendix_f_n_stability.csv` and either
an across-n risk plot or a compact risk table. Its filename refers to an
older manuscript layout; these are supplementary simulation diagnostics.
**Appendix F in the current manuscript describes real-data preparation.**
The manuscript's local-risk figure is the notebook export in `figures/`.

## Replogle: Section 6 and Appendix F

```bash
OPENBLAS_NUM_THREADS=4 python code/experiments/replogle/run_replogle.py
python code/experiments/replogle/run_go_analysis.py
```

Run the model stage before GO, then execute the real-data notebook. The
current panel is 213 perturbations by 2,000 outcomes. The runner saves
batch-clustered Wald and local-robust inference on the same pairs.
See [the Replogle README](replogle/README.md) for inputs, gene selection,
moment definitions and the GO annotation release.

## Changing the simulation run

Both simulation scripts accept `--replications`, `--seed` and `--n-jobs`.
The local runner also accepts `--n-values` and `--kappas`. Inspect the
options without running a simulation:

```bash
python code/experiments/efficiency/run_efficiency.py --help
python code/experiments/local_misspec/run_local_misspec.py --help
```

Every run overwrites its normal output paths. There is no scratch-output
flag; a reduced run replaces production tables. Restore the full defaults
before running notebooks, which assert 1,000 replications and the expected
grids. For auxiliary plots, `plot_statistica_sinica.py --result-dir PATH`
can read an already prepared alternate result directory.
