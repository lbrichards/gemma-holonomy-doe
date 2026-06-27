# Gemma Holonomy DOE

[![DOI](https://zenodo.org/badge/1268180115.svg)](https://doi.org/10.5281/zenodo.20960026)

This repository contains the source code and frozen experimental artifacts for replicating the results reported in the paper **"Holonomy in Transformer Activation Space Does Not Concentrate Along Semantic Directions: A Preregistered Reversal in Gemma."**

The repository is the experiment record, not the manuscript source. It contains the preregistration, analysis plan, measurement pipeline, frozen run artifacts, integrity checks, and analysis reports used by the paper.

## What Is Included

- `PREREGISTRATION_v2.md`: frozen preregistration for the confirmatory design.
- `analysis/hsem/ANALYSIS_PLAN_Hsem_v2addendum.md`: frozen analysis addendum for the H-sem verdict.
- `run/`: frozen corpus draw, run manifests, and holonomy measurement outputs.
- `reports/`: integrity receipt, preregistration reconciliation, and final H-sem / H-mag / H-grad analysis reports.
- `stage_a/`, `stage_b/`, `geometry/`, `transport/`, `planes/`, `covariates/`, `corpus/`, `extraction/`, `manifest/`, `bands/`: experiment and analysis code.
- `tests/`: regression and validation tests for the core pipeline.

Large frozen artifacts are tracked with Git LFS where needed. After cloning, run `git lfs pull` before attempting to inspect or reproduce the frozen run files.

## Main Results Artifacts

The paper's reported confirmatory results are summarized in:

- `reports/hsem_results.md`
- `reports/hsem_results.json`
- `reports/integrity_receipt.json`

The official n=390 measurement table and manifest are:

- `run/holonomy_results_n390.parquet`
- `run/holonomy_results_n390.json`
- `run/manifest_n390.json`

## Environment

The project uses Python 3.11 and `uv`.

```sh
uv sync
```

The intended compute backend for the frozen run is Apple Silicon / MPS. See `ENVIRONMENT.md` for the recorded hardware and OS context.

## Reproducing Checks and Analysis

Run the test suite:

```sh
uv run pytest
```

Re-run the frozen H-sem analysis against the committed artifacts:

```sh
uv run python -m analysis.hsem.run_analysis --execute
```

The analysis should reproduce the verdicts and estimates in `reports/hsem_results.md`, subject to the integrity checks described there.

## Provenance Note: n390 Manifold-Distance Guard Field

In the official n=390 run, the manifest field `manifold_distance_mahalanobis` contains k-nearest-neighbour fallback distances, not Mahalanobis distances. The Mahalanobis guard could not be computed because the activation covariance is rank-deficient (390 points in 2304 dimensions); the code fell back to k-NN for all planes, as recorded by the accompanying `manifold_distance_method` (`"knn"`) and `manifold_distance_fallback` (`true`) fields. The legacy field name is retained unchanged to preserve the integrity hash of the frozen artifact.

## Repository Boundary

This repository is intended to support experiment replication and audit. Paper drafting files, private writing notes, and generated documentation sites are intentionally kept outside this public experiment repository.
