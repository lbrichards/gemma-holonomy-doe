# Pre-Registration — Gemma Holonomy DOE
Status: DRAFT — not yet frozen

## 1. Purpose and scope

This study tests whether measured holonomy in the Gemma residual stream is explained by activation magnitude alone, by semantic structure associated with Gemma Scope SAE feature directions, or by a graded combination of both. The model is Gemma 2 2B with Gemma Scope SAEs, measured in the residual stream at layer TODO. This is a confirmatory study, not exploratory analysis; hypotheses, thresholds, factors, exclusions, and stopping rules are to be frozen before data collection.

## 2. Design

Factorial design. Factors and levels:

- Factor A — Plane type: {random, magnitude-matched shuffled-feature, magnitude-matched real-feature} (3 levels)
- Factor B — Magnitude: {TODO levels}
- Response variable: holonomy (loop transport rotation), operational definition TODO
- Blocking unit: base point (fresh, never used in exploratory work)

## 3. Hypotheses (frozen predictions)

- H-mag: magnitude main effect — direction and form TODO
  - Corroborates: TODO
  - Falsifies: TODO
- H-sem: semantic main effect (real vs shuffled, manifold-proximity controlled) TODO
  - Corroborates: TODO
  - Falsifies: TODO
- H-grad: monotone gradient random < shuffled < real TODO
  - Corroborates: TODO
  - Falsifies: TODO

## 4. Covariates measured (not assumed)

- Manifold proximity, primary proxy: SAE reconstruction error of swept points
- Manifold proximity, guard: latent-space distance (Mahalanobis / kNN to real activations)
- Real and shuffled planes will be checked for proximity matching: TODO

## 5. Sample size and power

- Effect size target: material effect, delta = log(1.25) = 0.2231 on the log-scale feature-vs-control contrast. Sub-material observed effect (log(1.103) = 0.098) explicitly NOT the target; see power/power_hsem_results.md for its cost.
- Variance source: frozen across-base dispersion from exploratory holonomy-probe runs; conservative value tau = 0.649 (Iteration 7), cross-checked against tau = 0.586 (Iteration 8).
- Test: two-sided, alpha = 0.05, power = 0.90.
- N per cell (t-corrected): 91; rounded up to 96 base points.
- Design: shared blocked base points — each of the 96 base points is evaluated on all three plane types (random, shuffled-feature, real-feature). Plane-type contrast is within-base-point; base-point variance differences out of the primary contrast.
- Total base points: 96. Total plane evaluations: 96 × 3 = 288 (× magnitude levels, per Section 2).
- Power basis: powered for the thinnest effect (H-sem). H-mag and H-grad inherit equal or greater power.

## 6. Pre-registered thresholds and stopping rules

- Materiality thresholds: TODO
- Positive stopping criterion: TODO
- Null stopping criterion: TODO (a flat result must be interpretable as absence, not underpower)

## 7. Analysis plan

- Primary contrast: TODO
- Model: factorial ANOVA / regression form TODO
- What is decided before seeing data vs. reported as-is: TODO

## 8. Reproducibility

- Data release plan: TODO
- Code release plan: TODO
- Seed handling: TODO
- Hardware: Apple Silicon / MPS; cloud GPU conditional.
