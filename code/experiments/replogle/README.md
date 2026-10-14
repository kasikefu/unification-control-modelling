# Replogle K562: models, enrichment, figures

Three stages, in order: `run_replogle.py` fits the models;
`run_go_analysis.py` builds GO:BP tables; the root notebook reads the tables
and renders the figures. Run commands from the repository root.

## Inputs and setup

Install the core package, then the application/notebook dependencies:

```bash
python -m pip install -e '.[dev]'
python -m pip install anndata scikit-learn nbconvert ipykernel
```

The model stage requires two local files:

| Path | Contents |
|---|---|
| `data/Replogle-E-k562_mean_log1p_by_batch.h5ad` | Precomputed perturbation-batch pseudobulk expression |
| `data/Replogle-E-k562_wilcoxon_batch_stratified.h5ad` | Matched batch-stratified DE statistics and raw p-values |

The gene axes must match. Inputs are not downloaded by the model runner.
The supplied pseudobulk averages log1p expression from five cells sampled
with replacement within retained groups; its upstream construction is
described in manuscript Appendix F. This repository starts from that
precomputed matrix. See [the input inventory](../../../data/README.md).

## 1. Fit the models

```bash
OPENBLAS_NUM_THREADS=4 python code/experiments/replogle/run_replogle.py
```

The runner performs feature selection, PCA, model fitting and inference,
then writes to `results/replogle/`:

1. Retain perturbations with at least 30 pseudobulk profiles, measured target
   expression and Wilcoxon results. The current inputs yield 213 targets,
   48 control profiles and 8,413 retained profiles.
2. Rank genes within perturbation by adjusted DE p-value, absolute z-score
   and absolute log fold change. Average ranks across perturbations. The
   2,000 best-ranked non-target genes form one common outcome panel.
3. Exclude all retained targets and the union of the per-perturbation top
   20 DE genes. Select the 1,000 remaining genes with the largest average
   ranks as surrogate features. This is a low-DE-response rule; it does
   not require nonsignificance under every perturbation.
4. Center and scale those genes across all retained profiles, discard
   standard deviations at most `1e-8`, and fit PCA once. Save up to 100
   components and use the first 10 in every model.
5. For each target, combine its perturbed profiles with the 48 controls.
   The current models have 78-96 profiles and 48 batch clusters.

Every target is fitted against every common outcome: **213 x 2,000 =
426,000 models**. Per-perturbation top-20 genes affect surrogate selection;
they are not the outcome panel used for model fitting.

### Moments and inference

Within each target-specific sample, center `z`, `x` and `y`. Residualize
`x` and `y` against an intercept and the 10 PC scores. The moment blocks are:

```text
IV: z_centered * (y_centered - beta * x_centered)
SV: x_residual * (y_residual - beta * x_residual)
```

IV and SV use their respective single moment. Two-step GMM starts with
identity weighting, then uses the pseudoinverse of the batch-clustered
covariance at the preliminary estimate. All standard errors are clustered
by batch.

For each pair, local inference uses one shared ellipsoid with radii
`r_j = 0.5 * sqrt(Omega_gmm[j,j])`, where the centered clustered covariance
is evaluated at the final GMM estimate. Point estimates are retained;
intervals and p-values account for worst-case local bias. BH adjustment is
within perturbation, across the 2,000 outcomes, separately for each method
and inference rule, at 5%.

### Model outputs

| File under `results/replogle/` | Role |
|---|---|
| `iv_sv_gmm_results.csv.gz` | Estimates, clustered SEs, Wald/local intervals, p-values and q-values |
| `wilcoxon_matched_raw.csv.gz` | Raw Wilcoxon p-values on the same pairs |
| `outcome_genes.csv`, `common_1000_non_de_genes.csv` | Outcome and surrogate feature panels |
| `pca_scores.csv.gz`, `pca_explained_variance.csv` | Model/PCA scores and notebook diagnostic inputs |
| `pca_loadings.csv.gz`, `top20_de_genes_by_perturbation.csv.gz`, `global_top20_de_genes.csv` | Feature-selection and PCA exports |
| `first_stage_results.csv`, `skipped_models.csv` | First-stage summaries and model failures |

## 2. Compute GO:BP enrichment

Place the human GAF and GO ontology in `data/go/`. For a new annotation
release, the official current links are:

```bash
mkdir -p data/go
curl -fL -o data/go/HUMAN-uniprot.gaf.gz https://current.geneontology.org/annotations/gaf/HUMAN-uniprot.gaf.gz
curl -fL -o data/go/go-basic.obo https://current.geneontology.org/ontology/go-basic.obo
python code/experiments/replogle/run_go_analysis.py
```

To reproduce the current results, retain the local files with these checksums
rather than replacing them with a newer `current` release:

```text
HUMAN-uniprot.gaf.gz  a0afba19dfb1f8fa996bc1bdcd61fd0c9bd4cf0d2bf2509d09ac86993c0e70a2
go-basic.obo          b08d45b268b8c24ccb2513dbbbc7d4df9f6521c099b413f79eb31e06e0fa3bcc
```

The GAF generation date is 2026-07-28; its GO version is 2026-07-26.
`go_bp_manifest.json` records the annotation headers, checksums and settings
for each run. Appendix F describes the testing background and multiple-testing procedure.

GO uses the 2,000 tested outcomes as the background. Human biological-process
annotations exclude `NOT` and propagate through `is_a` and `part_of`.
Terms covering 2 to 1,999 background genes are tested with hypergeometric
p-values, followed by BH within each perturbation and selection. The current
annotation files yield 1,790 directly annotated background genes and 4,795
tested terms. Wilcoxon raw p-values are readjusted over the same outcome
family before selecting its genes.

The stage saves full term results and selected-gene lists, plus
`go_bp_overview.csv`, `go_bp_per_perturbation.csv`,
`go_bp_recurrent_terms.csv` and the manifest. The notebook reads these three
summary tables and the manifest; the full term table is a separate export.

## 3. Render the figures

```bash
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 realdata_batch_optimized.ipynb
```

The notebook exports the manuscript figure `figures/realdata_compact_summary.pdf`
and four diagnostic PDFs: PCA scree, PC1/PC2 scores, coefficient histograms
and rank correlations. It also exports Wald/local discovery and Wilcoxon
agreement summaries to `results/replogle/`.

The manuscript figure labels the two inference rules as **Non-robust**
and **Robust**. It displays significant hypothesis tests, rather than
confidence intervals. Panels (a)–(b) are UpSet plots of the seven exclusive
discovery intersections, with the same column order and linear count scale.
Filled dots identify included methods and pale dots identify excluded
methods. Each observation is a perturbation–outcome gene pair, aggregated
over 213 perturbations and 2,000 outcomes. Thus an outcome discovered for
different perturbations is counted separately. The IV/SV intersection
excluding GMM has 48 non-robust and 17 robust discoveries; both counts are
labeled directly above their columns without separate explanation boxes.
The UpSet plots share the top row with discovery counts in panel (c),
and the GO comparison spans the lower row. The figure has no overall
title, long explanatory notes, or footer; statistical definitions and
the selection of GO terms are documented in
`figures/realdata_compact_summary_caption.md`.

The Matplotlib renderer `plot_discovery_upset.py` is shared with the notebook.
It can render this figure independently of the other notebook diagnostics:

```bash
MPLBACKEND=Agg python code/experiments/replogle/plot_discovery_upset.py
```

It also exports the separate two-panel preview
`figures/realdata_upset_comparison.pdf` and PNG previews of both figures.
`results/replogle/upset_intersection_counts.csv` records all 14 exact counts.
The earlier `venn_area_diagnostics.csv` documents the superseded Venn layout.

Current reference counts are 84,363 Wald GMM discoveries, 70,132 local GMM
discoveries and 209,274 matched Wilcoxon discoveries. GMM and Wilcoxon test
different statistical targets; the agreement comparison is not a validation
against known causal effects.

All stages overwrite their own outputs. After changing the outcome panel,
run both compute scripts and the notebook again so model, GO and figure
inputs share the same testing family. The earlier 19-outcome package is no
longer used. Large model and term-level files stay local under `.gitignore`;
a fresh clone needs the input files and compute stages to recreate them.
