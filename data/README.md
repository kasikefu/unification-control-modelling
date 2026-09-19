# Simulation data

The datasets in this directory are stored with Git LFS. Install Git LFS before cloning, or run `git lfs pull` after cloning, to download their contents.

`Replogle-E-k562_within_batch_effects.h5ad` exceeds GitHub Free's per-file Git LFS limit. Its three numbered parts are stored instead. Reconstruct it from the repository root with:

```bash
cat data/Replogle-E-k562_within_batch_effects.h5ad.part-{00,01,02} > data/Replogle-E-k562_within_batch_effects.h5ad
sha256sum data/Replogle-E-k562_within_batch_effects.h5ad
```

The reconstructed file's SHA-256 must be `de9caeee0c819068f6c398e19ff1fe49818ca44ed907dfb436c38f72c0647041`. The reconstructed file is ignored by Git so that it is not uploaded twice.

The replication-level result at `result/efficiency_replications.csv` is also stored with Git LFS. Its SHA-256 is `34261c3ff783c1d384b7b1efe52ef600b33258383cfe2a2b5ac03c7567de1108`.
