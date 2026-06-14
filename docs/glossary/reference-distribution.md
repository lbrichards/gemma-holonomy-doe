# reference distribution

<span style="background-color: #4051b5; color: white; padding: 0.2em 0.6em; border-radius: 0.25em; font-size: 0.85em; font-weight: 600;">COINED</span>

The Mahalanobis (or kNN fallback) distribution fitted to experiment-sample activations, used as a guard proxy for manifold distance.

## Why this term exists

SAE reconstruction error alone can miss points that reconstruct cheaply but lie far from real activations. The reference distribution catches this documented failure mode.

## Source

covariates/measure.py

## Depends on

- [Mahalanobis distance](mahalanobis-distance.md)
- [kNN distance fallback](knn-distance-fallback.md)

---

[← Back to Glossary](index.md)
