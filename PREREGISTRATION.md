# Pre-Registration — Gemma Holonomy DOE
Status: DRAFT — not yet frozen

## 1. Purpose and scope

This study tests whether measured holonomy in the Gemma residual stream is explained by activation magnitude alone, by semantic structure associated with Gemma Scope SAE feature directions, or by a graded combination of both. The model is Gemma 2 2B with Gemma Scope SAEs, measured in the residual stream at layer TODO. This is a confirmatory study, not exploratory analysis; hypotheses, thresholds, factors, exclusions, and stopping rules are to be frozen before data collection.

## 2. Design

Factorial design, shared blocked base points.

- Factor A — Plane type (3 levels): random; magnitude-matched shuffled-feature; magnitude-matched real-feature.
- Factor B — Magnitude (2 levels): low = 25th percentile, high = 75th percentile of the in-plane activation magnitude distribution, computed across the 96 base points. Percentile cut points are fixed in advance from the base-point sample and not adjusted after holonomy is observed.
- Response variable: holonomy (loop transport rotation); operational definition in Section 7.
- Blocking unit: base point. Each of 96 fresh base points is evaluated across all 3 plane types × 2 magnitude levels.
- Full design: 96 base points × 3 plane types × 2 magnitude levels = 576 plane evaluations.
- "In-plane magnitude" = the norm of the base-point activation's component within the 2D loop plane, NOT the global activation norm. (Magnitude-matching construction specified in Section 7.)

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

Two nuisance variables are measured per plane and checked for balance across plane types
before any holonomy contrast is interpreted. Neither is assumed absent; each is verified balanced
or conditioned upon.

### 4.1 Manifold distance

- Primary proxy: SAE reconstruction error of the points actually visited (loop center and swept loop),
  computed with the same Gemma Scope SAE used to define feature planes.
- Guard proxy: latent-space distance to the real-activation distribution (Mahalanobis distance,
  with kNN distance as a fallback if the covariance estimate is ill-conditioned), to catch points
  the SAE reconstructs cheaply yet that lie far from where real activations live
  (the documented low-reconstruction-error / off-data failure mode; cf. Denouden et al. 2018).
- Interpretation: manifold distance is treated as a continuous, strictly positive scalar.
  "On-manifold" is not a binary label. The relevant quantity is whether real-feature and
  shuffled-feature planes share an indistinguishable DISTRIBUTION of manifold distance.

### 4.2 Plane angle (phi)

- Definition: phi is the mutual angle between the two directions spanning a plane, computed from the
  raw (normalized, NOT orthogonalized) decoder directions for feature and shuffled planes, and from
  the raw directions for random planes.
- Rationale: enclosed loop area scales with sin(phi), so phi feeds the response directly; and the
  learned angle between feature directions is itself semantically loaded (cf. feature-geometry
  literature). phi must therefore be balance-checked so the real-vs-shuffled contrast does not
  conflate meaning with geometry.

### 4.3 Balance check (pre-registered, run before holonomy is examined)

- For each covariate (manifold distance, phi), compare the real-feature and shuffled-feature arms.
- Balance test and threshold: TODO (specify test, e.g. two-sample on covariate distributions,
  and the equivalence margin that counts as "balanced").
- The balance check is computed and recorded BEFORE the holonomy response is unblinded.

## 5. Sample size and power

- Effect size target: material effect, delta = log(1.25) = 0.2231 on the log-scale feature-vs-control contrast. Sub-material observed effect (log(1.103) = 0.098) explicitly NOT the target; see power/power_hsem_results.md for its cost.
- Variance source: frozen across-base dispersion from exploratory holonomy-probe runs; conservative value tau = 0.649 (Iteration 7), cross-checked against tau = 0.586 (Iteration 8).
- Test: two-sided, alpha = 0.05, power = 0.90.
- N per cell (t-corrected): 91; rounded up to 96 base points.
- Design: shared blocked base points — each of the 96 base points is evaluated on all three plane types (random, shuffled-feature, real-feature). Plane-type contrast is within-base-point; base-point variance differences out of the primary contrast.
- Total base points: 96. Total plane evaluations: 96 × 3 × 2 magnitude levels = 576 total evaluations.
- Power basis: powered for the thinnest effect (H-sem). H-mag and H-grad inherit equal or greater power.

## 6. Pre-registered thresholds and stopping rules

- Materiality thresholds: TODO
- Positive stopping criterion: TODO
- Null stopping criterion: TODO (a flat result must be interpretable as absence, not underpower)

## 7. Analysis plan

- Primary contrast: TODO
- Model: factorial ANOVA / regression form TODO
- What is decided before seeing data vs. reported as-is: TODO

### 7.x Conditioning rule for the semantic contrast (pre-registered)

The load-bearing semantic contrast is real-feature vs shuffled-feature (NOT real vs random;
the random arm is the floor, not the semantic control). Its interpretation follows three
pre-registered branches, decided by the Section 4.3 balance check:

1. BALANCED on both manifold distance and phi: report the real-vs-shuffled holonomy contrast at
   face value as the semantic effect.
2. UNBALANCED on either covariate: do not report the raw contrast as semantic. Condition on the
   unbalanced covariate(s) — regression adjustment with the covariate as a term, or matching/
   stratification — and report the semantic effect ADJUSTED for the covariate(s).
3. Effect VANISHES after conditioning: report this as the finding — the apparent semantic effect
   is attributable to the covariate (manifold distance and/or phi), not to meaning. This is a
   pre-registered, publishable null.

The random arm serves only as the noise floor (is there structure above noise at all) and as the
lower anchor of the random < shuffled < real gradient (H-grad). It is never the comparison from
which the semantic claim is drawn.

## 8. Reproducibility

- Data release plan: TODO
- Code release plan: TODO
- Seed handling: TODO
- Hardware: Apple Silicon / MPS; cloud GPU conditional.
