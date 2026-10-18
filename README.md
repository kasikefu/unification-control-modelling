# Unifying Instrumental and Surrogate Control Variables for Deconfounding

Code for the manuscript **Unifying Instrumental and Surrogate Control Variables
for Deconfounding**. The manuscript PDF is maintained
locally and is not distributed in this repository.

Moments from instrumental variables (IV) and surrogate control variables (SV)
in one projected GMM framework: efficient weighting under correct specification,
and risk-optimal weighting under local misspecification.

**The experiment scripts fit models and write to `results/`.**
**The notebooks read those results, summarize them, and write PDFs to `figures/`.**
The notebooks do not read the raw H5AD inputs or rerun the model-fitting stages.
Local-power calculations use the saved Monte Carlo estimates.

## Layout

```text
code/
  ucm/                    shared estimation, projected scores, risk and inference
  experiments/
    efficiency/           correctly specified Monte Carlo experiment
    local_misspec/        local deviations in IV, SV or both moment blocks
    replogle/             K562 application and GO:BP enrichment
  tests/                  automated tests
  test_bug/               independent DGP and population-theory checks

data/                     local H5AD inputs and GO annotation files
results/                  generated tables, draws and auxiliary exports
figures/                  PDF exports from the three notebooks

efficiency_results.ipynb
local_misspec_results.ipynb
realdata_batch_optimized.ipynb
```

| Study | Manuscript | Notebook | Main figure |
|---|---|---|---|
| `efficiency` | Section 5.1 | `efficiency_results.ipynb` | `efficiency_mse_comparison.pdf` |
| `local_misspec` | Section 5.2 | `local_misspec_results.ipynb` | `local_misspec_mse_comparison.pdf` |
| `replogle` | Section 6 and Appendix E | `realdata_batch_optimized.ipynb` | `realdata_compact_summary.pdf` |

The efficiency figure uses `a=2`. The local-risk figure uses **both** moment
blocks deviating, `D=diag(1,1)`, with 16 boundary directions. The scripts also
compute supplementary designs; the notebooks select the manuscript panels.

See [the core package](code/ucm/README.md) for the statistical interfaces,
[the experiment guide](code/experiments/README.md) for defaults and output
files, and [the Replogle guide](code/experiments/replogle/README.md) for data
selection and clustered inference.

## Setting up

Run commands from the repository root. The package declares Python 3.10 or
later; install the core package and test dependency with:

```bash
python -m pip install -e '.[dev]'
```

For the real-data application and notebook execution, also install:

```bash
python -m pip install anndata scikit-learn nbconvert ipykernel
```

The last full rerun used Python 3.12.12, NumPy 2.3.5, SciPy 1.15.3,
pandas 2.3.3, Matplotlib 3.10.7, Seaborn 0.13.2, joblib 1.5.2,
anndata 0.12.6 and scikit-learn 1.7.2. Dependencies are not pinned in
`pyproject.toml`; these are recorded versions, not an environment lockfile.

## Running the analyses

```bash
python code/experiments/efficiency/run_efficiency.py
python code/experiments/local_misspec/run_local_misspec.py
OPENBLAS_NUM_THREADS=4 python code/experiments/replogle/run_replogle.py
python code/experiments/replogle/run_go_analysis.py
```

The two simulations use 1,000 replications and eight workers by default.
Their `--help` options expose seeds and worker counts; the local simulation
also exposes sample sizes and deviation radii. Inputs for Replogle must
already be present in `data/`; see [the input inventory](data/README.md).

Then execute the notebooks in place:

```bash
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 efficiency_results.ipynb
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 local_misspec_results.ipynb
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 realdata_batch_optimized.ipynb
```

These commands overwrite the corresponding results and notebook outputs.
Reduced simulation runs use the same output directories, and the notebooks
check the production grids and replication counts. Run the full defaults
again before regenerating manuscript figures after a reduced run.

## What is versioned

The repository includes code, executed notebooks, PDF figures
and compact result tables. Raw inputs and the large replication/model/GO
comparison tables are excluded by [`.gitignore`](.gitignore). A fresh clone
therefore needs the local inputs and the compute stages before every notebook
can be executed. The input README is versioned; the datasets themselves are
not supplied by the current tracked tree.

Auxiliary figures from standalone scripts remain in `results/local_misspec/`.
The manuscript figures come from `figures/`. The older 19-outcome Replogle
results are not part of the current workflow.

A rerun can change the last bits of floating-point summaries, depending on
BLAS and package versions. PDF creation metadata and gzip timestamps can
also change file hashes even when the reported numbers agree.

## Checks

```bash
python -m pytest code/tests -q
PYTHONPATH=code python code/test_bug/check_dgp.py
PYTHONPATH=code python code/test_bug/check_local_risk_theory.py
```

The latest checks passed all 22 automated tests and all 40 population-theory
comparisons. The theory diagnostic checks the paper's positive-definite weight
family, including `KJ=1`, rather than comparing against the older rank-one
weight alone. Its output is
`results/theory_checks/local_risk_theory_check.csv`.

## How long a rerun takes

Measured on 2026-10-04, with eight workers per simulation and four BLAS
threads for the Replogle model stage:

| Stage | Time |
|---|---|
| Efficiency simulation | 25 s |
| Local-misspecification simulation | 3 min 17 s |
| Replogle model fitting, 426,000 pairs | 28 min 1 s |
| GO:BP analysis | 2 min 31 s |
| Efficiency / local / real-data notebooks | 4 s / 5 s / 1 min 39 s |

The simulation and real-data stages were launched concurrently. These are
per-stage measurements on that machine, not a summed wall-time guarantee.

## License

See [LICENSE](LICENSE).
