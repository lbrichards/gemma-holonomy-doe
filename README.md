This repository contains a pre-registered, factorial design-of-experiments (DOE) confirmatory study of holonomy in the Gemma residual stream. It is deliberately separate from the exploratory `holonomy-probe` work: hypotheses, thresholds, factor definitions, exclusion rules, and analysis gates are to be frozen before any data collection for the confirmatory DOE begins.

## Provenance note: n390 manifold-distance guard field

Note (2026-06-18): In the official n=390 run, the manifest field `manifold_distance_mahalanobis` contains k-nearest-neighbour fallback distances, not Mahalanobis distances. The Mahalanobis guard could not be computed because the activation covariance is rank-deficient (390 points in 2304 dimensions); the code fell back to k-NN for all planes, as recorded by the accompanying `manifold_distance_method` (`"knn"`) and `manifold_distance_fallback` (`true`) fields. The legacy field name is retained unchanged to preserve the integrity hash of the frozen artifact.
