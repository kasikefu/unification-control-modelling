# Unification of Control Modelling

This repository contains the code, simulations, and empirical analysis for the
Unification of Control Modelling project. The project places instrumental
variable (IV) moments and surrogate-control (SV) moments in a common projected
GMM framework. It studies efficient combinations of the two information
sources and risk-optimal combinations that remain reliable under local moment
misspecification.

## Repository structure

```text
unification-control-modelling/
├── code/
│   ├── ucm/                  # Reusable estimation and inference package
│   ├── experiments/          # Simulations and empirical applications
│   │   ├── efficiency/       # Projected-GMM efficiency experiment
│   │   ├── local_misspec/    # Local-misspecification experiment
│   │   └── replogle/         # Replogle CRISPR application and GO analysis
│   ├── tests/                # Pytest test suite
│   └── test_bug/             # Standalone diagnostic checks
├── data/                     # Replogle inputs, GO files, and packaged outputs
├── results/                  # Computed data, tables, and standalone-script outputs
├── figures/                  # PDF and PNG exports from the three notebooks
├── efficiency_results.ipynb
├── local_misspec_results.ipynb
├── realdata_batch_optimized.ipynb
├── Unification_of_control_variable_modeling.pdf
└── pyproject.toml
```

## Core package

The `ucm` package under `code/ucm` contains the reusable statistical
implementation.

- `dgp.py` generates the IV/SV simulation design, including local deviations
  from the two moment restrictions.
- `projected.py` estimates the IV projection of the treatment and the
  surrogate-control residual, and constructs the corresponding first-order
  scores and covariance matrices.
- `estimators.py` implements generic GMM, IV-only, SV-only, combined,
  two-step efficient, and local-risk GMM estimators.
- `weighting.py` provides equal, feasible efficient, and oracle weighting
  matrices used in the simulations.
- `risk.py` implements GMM sensitivities, asymptotic variance and local bias,
  ellipsoidal worst-case risk, and risk-optimal sensitivities and weights.
- `inference.py` provides GMM variance estimation, Wald inference, and
  misspecification-robust confidence intervals.

The main computational flow is:

```text
Raw variables (X, Y, Z, W)
          |
          v
Sample projections (h_iv, h_sv)
          |
          v
Initial GMM estimate
          |
          v
Projected scores and covariance
          |
          +----------------------+
          |                      |
          v                      v
 Efficient GMM weight     Local-risk-optimal weight
          |                      |
          +-----------+----------+
                      v
             Estimate and inference
```

Here, `h_iv` is the sample projection of `X` onto the instruments `Z`, while
`h_sv` is the residual from projecting `X` onto controls `W`. Stacking these
two projected moments produces the combined estimator.

## Experiments

### Efficiency simulation

`code/experiments/efficiency/run_efficiency.py` runs the Monte Carlo experiment
for the efficiency comparison. It compares unadjusted OLS, IV, SV, equally
weighted combined GMM, feasible efficient GMM, and oracle efficient GMM across
different sample sizes and identification strengths. Outputs are written to
`results/efficiency`.

### Local misspecification

`code/experiments/local_misspec/run_local_misspec.py` studies designs in which
the IV restriction, the SV restriction, or both restrictions deviate locally
from correct specification. It compares single-source estimators, equal-weight
GMM, and the local-risk estimator in terms of bias, variance, worst-case risk,
and confidence-interval coverage.

The companion scripts
`code/experiments/local_misspec/plot_statistica_sinica.py` and
`code/experiments/local_misspec/summarize_appendix_f.py` generate figures and
Appendix F summaries. Outputs are written to `results/local_misspec`.

### Replogle CRISPR application

`code/experiments/replogle/run_replogle.py` applies the IV/SV framework to
batch-level pseudobulk expression from the Replogle K562 CRISPR screen. The
pipeline:

1. filters perturbations with sufficient observations;
2. selects stable non-differentially expressed genes;
3. constructs surrogate controls using principal components;
4. fits IV, SV, and two-step GMM models for perturbation-outcome pairs; and
5. reports conventional Wald and local-misspecification-robust inference.

`code/experiments/replogle/run_go_analysis.py` performs GO Biological Process
over-representation analysis using the tested outcome genes as the background.
See `code/experiments/replogle/README.md` for application-specific commands and
input details.

## Installation

Python 3.10 or later is required. Install the core package and test dependency
from the repository root:

```bash
python -m pip install -e '.[dev]'
```

The core package depends on NumPy, SciPy, pandas, Matplotlib, Seaborn, and
joblib. The Replogle application additionally requires `anndata` and `scikit-learn`.

## Running the code

Run the test suite with:

```bash
pytest code/tests
```

Run the two simulation entry points with:

```bash
python code/experiments/efficiency/run_efficiency.py
python code/experiments/local_misspec/run_local_misspec.py
```

Run the Replogle application and its subsequent GO analysis with:

```bash
python code/experiments/replogle/run_replogle.py
python code/experiments/replogle/run_go_analysis.py
```

The simulation scripts expose command-line options for replication counts,
random seeds, and parallel workers. Use `--help` on an entry point to inspect
its available options.

## Data and generated results

The `data` directory contains the Replogle pseudobulk inputs, differential
expression results, and Gene Ontology annotation files used by the empirical
pipeline. The `results` directory contains generated CSV files, compressed
tables and standalone-script outputs for the simulations and empirical analysis.
The three analysis notebooks export their figures as PDF and PNG files to
`figures/`.
Large files are managed with Git LFS where applicable.

## License

See [LICENSE](LICENSE).
