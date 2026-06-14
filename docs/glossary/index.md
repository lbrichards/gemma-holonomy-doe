# Glossary

Cross-linked terminology for the Gemma Holonomy DOE project.

---

## <span style="background-color: #4051b5; color: white; padding: 0.2em 0.6em; border-radius: 0.25em; font-size: 0.85em; font-weight: 600;">COINED</span> Coined Terms

*Terms introduced by this project.*

- [band-collapse contingency](band-collapse-contingency.md) — The pre-registered fallback procedure when the three-arm common-support band is empty or too narrow to support matched magnitude comparisons.
- [blind manifest boundary](blind-manifest-boundary.md) — The partition point separating the 390 experiment-sample base points from the reserve, determined by draw order and recorded before any holonomy is observed.
- [common-support band](common-support-band.md) — The magnitude range where all three plane-type arms (random, shuffled-feature, real-feature) have overlapping in-plane magnitude distributions, enabling matched comparisons.
- [det M degeneracy floor](det-m-degeneracy-floor.md) — A threshold (tau_detM = 0.413) below which planes are excluded because their pullback-metric area approaches zero, making holonomy ill-defined.
- [enclosed area (wedge)](enclosed-area-wedge.md) — The pullback-metric area of the loop, computed as A_enclosed = rho^2 * sqrt(det M), which normalizes the rotation angle to yield holonomy.
- [holonomy-magnitude response](holonomy-magnitude-response.md) — The measured relationship between in-plane magnitude and holonomy (rotation per enclosed area), the primary response variable testing H-mag.
- [in-plane magnitude (pullback)](in-plane-magnitude-pullback.md) — The G-norm of a vector's G-orthogonal projection onto a plane, computed as mag(h) = sqrt(a^T M a) where M is the plane Gram matrix under the pullback metric.
- [magnitude-matched shuffled-feature plane](magnitude-matched-shuffled-feature-plane.md) — A control plane constructed by pairing one active SAE feature with one inactive real dictionary feature, matched to real-feature planes in pullback-metric in-plane magnitude.
- [pairing rule](pairing-rule.md) — The fixed procedure for constructing plane arms before holonomy computation, specifying how SAE features are selected and combined for each plane type.
- [two-arm terminal collapse](two-arm-terminal-collapse.md) — The fallback case where even the two-arm real+shuffled common-support band fails the minimum width requirement, making matched semantic comparison impossible.

---

## <span style="background-color: #448aff; color: white; padding: 0.2em 0.6em; border-radius: 0.25em; font-size: 0.85em; font-weight: 600;">INHERITED</span> Inherited Terms

*Standard terms from prior literature.*

- [common support / positivity](common-support-positivity.md) — The requirement that all treatment groups have overlapping covariate distributions, ensuring covariate adjustment does not extrapolate beyond observed data.
- [Gram matrix](gram-matrix.md) — A matrix M = D^T G D encoding inner products of a set of vectors D under a metric G, whose determinant gives the squared volume of the parallelepiped they span.
- [holonomy](holonomy.md) — The rotation angle accumulated by parallel-transporting a vector around a closed loop, divided by the enclosed area, measuring local curvature.
- [Jacobian-vector product (JVP)](jacobian-vector-product-jvp.md) — The product J*v of the Jacobian matrix J with a vector v, computed efficiently via forward-mode automatic differentiation without materializing J.
- [Mahalanobis distance](mahalanobis-distance.md) — A distance metric that accounts for correlations in the data by scaling by the inverse covariance matrix, measuring how many standard deviations a point lies from a distribution.
- [mixed-effects model](mixed-effects-model.md) — A regression model containing both fixed effects (population-level coefficients) and random effects (group-level variation), used here to model base-point blocking.
- [parallel transport](parallel-transport.md) — The procedure for moving a tangent vector along a curve while keeping it as parallel as possible according to the connection, preserving its length under the metric.
- [pullback metric](pullback-metric.md) — The induced metric G = J^T J on the input space, pulled back from the output space via the Jacobian J, measuring distances as the model perceives them.
- [SAE reconstruction error](sae-reconstruction-error.md) — The L2 distance between an activation and its reconstruction through a sparse autoencoder, serving as a proxy for how well the SAE captures the activation.
- [standardized mean difference (SMD)](standardized-mean-difference-smd.md) — The difference in group means divided by the pooled standard deviation, a scale-free measure of effect size used in balance diagnostics.
