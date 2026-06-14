# Mahalanobis distance

<span style="background-color: #448aff; color: white; padding: 0.2em 0.6em; border-radius: 0.25em; font-size: 0.85em; font-weight: 600;">INHERITED</span>

A distance metric that accounts for correlations in the data by scaling by the inverse covariance matrix, measuring how many standard deviations a point lies from a distribution.

$$
d_M(x) = \sqrt{(x - \mu)^\top \Sigma^{-1} (x - \mu)}
$$

## Source

Mahalanobis (1936); multivariate statistics

---

[← Back to Glossary](index.md)
