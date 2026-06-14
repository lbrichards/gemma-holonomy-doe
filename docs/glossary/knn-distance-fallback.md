# kNN distance fallback

<span style="background-color: #4051b5; color: white; padding: 0.2em 0.6em; border-radius: 0.25em; font-size: 0.85em; font-weight: 600;">COINED</span>

A fallback guard-distance method using mean k-nearest-neighbor distance when the covariance matrix is ill-conditioned for Mahalanobis.

## Why this term exists

In high-dimensional settings, the empirical covariance may be rank-deficient or ill-conditioned. kNN distance provides a robust non-parametric alternative that still measures distance to the reference cloud.

## Source

covariates/measure.py

## Depends on

- [Mahalanobis distance](mahalanobis-distance.md)

---

[← Back to Glossary](index.md)
