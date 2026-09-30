# Replogle real-data workflow

From the project root, run the model stage once. It reads the local H5AD files
in `data/` and writes all tabular model/PCA outputs to `results/replogle/`:

```bash
OPENBLAS_NUM_THREADS=4 python code/experiments/replogle/run_replogle.py
```

The GO:BP stage uses two official Gene Ontology files. Download them into the
ignored local `data/go/` directory, then run the GO script:

```bash
mkdir -p data/go
curl -fL -o data/go/HUMAN-uniprot.gaf.gz https://current.geneontology.org/annotations/gaf/HUMAN-uniprot.gaf.gz
curl -fL -o data/go/go-basic.obo https://current.geneontology.org/ontology/go-basic.obo
python code/experiments/replogle/run_go_analysis.py
```

The downloaded files used for the present results have SHA-256 checksums
`a0afba19dfb1f8fa996bc1bdcd61fd0c9bd4cf0d2bf2509d09ac86993c0e70a2`
(GAF) and `b08d45b268b8c24ccb2513dbbbc7d4df9f6521c099b413f79eb31e06e0fa3bcc`
(OBO). The GAF header dates it 2026-07-28 and the OBO header identifies
`releases/2026-07-26`. Checksums and source details are also recorded in
`results/replogle/go_bp_manifest.json` after each run. If the `current` links
have changed, use the archived matching versions for exact reproduction.

Finally execute `realdata_batch_optimized.ipynb`. It only reads result tables,
makes figures, and compares Wald, local-robust, and matched Wilcoxon findings.
The runner keeps Top 20 DE genes per perturbation for feature selection, but
models exactly 2,000 non-target outcomes per perturbation, matching the SRP
pair count (213 × 2,000 = 426,000). GO:BP enrichment uses those 2,000 tested
outcomes as its background. Rerun both scripts and the notebook after changing
the outcome set; earlier 19-outcome CSVs and figures are not comparable.
