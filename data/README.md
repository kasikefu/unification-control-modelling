# Local inputs for the Replogle analysis

Run commands from the repository root. The current analysis reads the two
precomputed H5AD files below and the GO files under `data/go/`.
Raw inputs are local and excluded by `.gitignore`; this README is versioned.
A fresh clone does not obtain these datasets through `git lfs pull`.

| Path | Used by |
|---|---|
| `Replogle-E-k562_mean_log1p_by_batch.h5ad` | Model stage: pseudobulk expression |
| `Replogle-E-k562_wilcoxon_batch_stratified.h5ad` | Model stage: DE ranks and matched raw p-values |
| `go/HUMAN-uniprot.gaf.gz` | GO:BP stage: human gene annotations |
| `go/go-basic.obo` | GO:BP stage: ontology and parent relationships |

The pseudobulk matrix contains 22,075 profiles and 8,563 genes. The model
runner selects 8,413 profiles from 213 perturbations and controls. It consumes
the existing pseudobulk file; upstream single-cell aggregation is not
implemented here. Manuscript Appendix F describes the supplied preprocessing.

See [the Replogle workflow](../code/experiments/replogle/README.md) for setup,
GO download commands, annotation checksums and run order. Generated model,
PCA, GO and comparison outputs belong in `results/replogle/`, not in `data/`.
The old `data/iv_sv_package_results/` result copy has been removed.

## Archived within-batch file

Some local copies also contain `Replogle-E-k562_within_batch_effects.h5ad`
and its three numbered parts. The current model and GO runners do not read
this file. If only the archived parts are available, reconstruct it with:

```bash
cat data/Replogle-E-k562_within_batch_effects.h5ad.part-{00,01,02} > data/Replogle-E-k562_within_batch_effects.h5ad
sha256sum data/Replogle-E-k562_within_batch_effects.h5ad
```

The recorded SHA-256 is
`de9caeee0c819068f6c398e19ff1fe49818ca44ed907dfb436c38f72c0647041`.
This reconstruction is optional and is not a prerequisite for the current
analysis. Replication-level simulation outputs live under
`results/efficiency/` and `results/local_misspec/`; they are not raw inputs.
