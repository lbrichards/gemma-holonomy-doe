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

- Effect size assumed (per hypothesis): TODO
- Variance estimate source: prior exploratory runs (cite which)
- N per cell: TODO
- Power target and alpha: TODO
- Powered for the thinnest effect (H-sem).

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
