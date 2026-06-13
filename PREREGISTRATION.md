# Pre-Registration — Gemma Holonomy DOE
Status: DRAFT — not yet frozen

## 1. Purpose and scope

This study tests whether measured holonomy in the Gemma residual stream is explained by activation magnitude alone, by semantic structure associated with Gemma Scope SAE feature directions, or by a graded combination of both. The model is Gemma 2 2B with Gemma Scope SAEs, measured in the residual stream at layer TODO. This is a confirmatory study, not exploratory analysis; hypotheses, thresholds, factors, exclusions, and stopping rules are to be frozen before data collection.

## 2. Design

Factorial design, shared blocked base points.

- Factor A — Plane type (3 levels): random; magnitude-matched shuffled-feature; magnitude-matched real-feature.
- Factor B — Magnitude (2 levels): low = 25th percentile, high = 75th percentile of the in-plane activation magnitude distribution, computed across the 96 base points. Percentile cut points are fixed in advance from the base-point sample and not adjusted after holonomy is observed.
- Magnitude levels are the 25th (low) and 75th (high) percentiles of the pullback-metric in-plane
  magnitude (Section 7.3) computed over the REAL-FEATURE arm: i.e. mag(h) evaluated for every
  real-feature plane across all 96 base points, pooled, then percentiles taken. The resulting two
  absolute m values are applied identically to all three plane arms (shuffled and random included;
  random may be extrapolated, acceptable as it serves only as the floor).
- Sequencing constraint: because mag(h) depends on each plane's Jacobian, the m levels cannot be
  computed until real-feature planes are selected and their JVPs evaluated. The run pipeline must
  (1) select planes under the degeneracy floor, (2) evaluate real-arm magnitudes, (3) fix m_low,
  m_high from real-arm percentiles, (4) only then run all three arms at those m. This ordering is
  pre-registered.
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
- Rationale: although the primary response uses pullback-metric area (Section 7.2), the learned
  angle between feature directions is itself semantically loaded (cf. feature-geometry
  literature). phi must therefore be balance-checked so the real-vs-shuffled contrast does not
  conflate meaning with native Euclidean feature geometry.
- The Euclidean phi floor is retired as primary; see the det(M) degeneracy floor in Section 7.2.1.

### 4.3 Balance diagnostics (pre-registered, computed before holonomy is unblinded)

Balance is reported as a diagnostic, NOT used as a pass/fail gate. The semantic contrast is
covariate-adjusted by default (see Section 7), so these diagnostics characterize imbalance and
verify common support; they do not switch the analysis path.

For each covariate (manifold distance; phi) compare real-feature vs shuffled-feature arms:

- Standardized mean difference (SMD = (mean_real - mean_shuffled) / pooled SD).
  Reference thresholds from the propensity-score balance literature (Austin; Stuart):
  |SMD| < 0.10 negligible; |SMD| > 0.25 flagged as material imbalance (reported, then adjusted for).
- Overlap / common support (positivity): require >= 90% of shuffled-arm covariate values to fall
  within the observed real-arm range, and vice versa. Below this, covariate adjustment is
  extrapolation and must be reported as such.
- Phi Euclidean-geometry check (belt-and-suspenders, since phi remains a native feature-geometry
  covariate even though the primary response now uses pullback-metric area): require
  |mean(log sin phi_real) - mean(log sin phi_shuffled)| < 0.045
  (~1/5 of the materiality threshold log(1.25) = 0.223).

All diagnostics are computed and recorded BEFORE the holonomy response is unblinded.

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

### 7.1 Loop geometry (raw directions, no orthogonalization)

Planes are spanned by the two normalized SAE decoder directions d1, d2 as supplied, at their
true mutual angle phi. Directions are NOT orthogonalized (departure from the exploratory
holonomy-probe instrument, which applied Gram-Schmidt; chosen to preserve the native learned
geometry, consistent with the perturbation-response assay's stated philosophy).

The loop is traced as:

    gamma(t) = c + rho * ( cos(2 pi t) * d1 + sin(2 pi t) * d2 ),   t in [0,1]

where c is the loop center (Section 7.3, magnitude-matching) and rho is the radius.

Radius convention: rho = radius_relative * ||base activation||, with radius_relative = 6.0e-3,
inherited from the validated Iteration 7/8 instrument for continuity.

### 7.2 Response variable: holonomy

H = theta / A_enclosed

- theta: rotation angle extracted from the antisymmetric part of the loop transport operator
  (same extraction as the frozen instrument).
- A_enclosed = rho^2 * sqrt(det M), the pullback-metric (G-)area of the loop, where
  M = (JD)^T (JD) is the 2x2 plane Gram matrix under the pullback metric (Section 7.3).
  This supersedes the earlier Euclidean expression rho^2 * sin(phi). All geometric quantities
  (magnitude, area, transport) are now computed under the single pullback metric G = J^T J,
  consistent with the one-metric principle (Curved Inference 2025).
- At the orthogonal Euclidean limit (J = I, phi = 90 deg) this reduces to the prior expression,
  so continuity with the validated instrument holds at that limit.

### 7.2.1 Degeneracy floor (pre-registered, blind, applied at SELECTION)

- Floor: det(M) > tau_detM for every plane, all three arms, where M is the pullback-metric plane
  Gram (Section 7.3). Rationale: A_enclosed = rho^2 sqrt(det M) -> 0 as det M -> 0 (the Jacobian
  maps the two directions toward collinear), making H ill-defined. det M is the correct
  degeneracy quantity under the pullback metric; the earlier Euclidean phi floor (pi/8) is
  retired as primary and may be kept only as a cheap pre-filter before the JVP step.
- tau_detM: TODO value (set from the throwaway bench distribution of det M on synthetic points,
  NOT from the experiment sample).
- Application point, all-arms uniformity, and exclusion-rate reporting: unchanged from prior
  7.2.1 (enforced at pair selection, before any holonomy observed, reported per arm).

### 7.3 In-plane magnitude and magnitude-matching (pullback metric)

All geometric quantities use the model's pullback metric G = J^T J at the base point, where J is
the local Jacobian already computed for loop transport (cf. pullback-metric interpretability
literature: FishBack 2026; Curved Inference 2025). G is never materialized; only J applied to
d1, d2, h is needed.

Let D = [d1 d2] (raw normalized decoder directions, NOT orthogonalized).

- Compute JD = [J d1, J d2] and Jh (three Jacobian-vector products).
- Plane metric Gram: M = (JD)^T (JD)   (2x2).
- G-projection coefficients of h onto the plane: a = M^{-1} (JD)^T (Jh).
- IN-PLANE MAGNITUDE (the matched scalar): mag(h) = sqrt(a^T M a) = G-norm of h's G-orthogonal
  projection onto the plane. This is the scalar Factor B percentiles are cut on AND the scalar
  the center-placement resets.

Center-placement for magnitude level m:

- In-plane vector v = D a; out-of-plane part = h - v (G-orthogonal to the plane by construction).
- c = (h - v) + m * (v / mag(h)).
  Preserves out-of-plane context exactly, preserves in-plane direction, sets in-plane G-magnitude
  to m.
- Degeneracy floor: if mag(h) < eps_mag (TODO value), set the in-plane offset direction to the
  G-normalized d1 instead of v/mag(h). Record when this fallback fires.

### 7.x Semantic contrast: covariate-adjusted by default (pre-registered)

The load-bearing semantic contrast is real-feature vs shuffled-feature (NOT real vs random).
It is estimated by covariate adjustment in all cases, with no balance gate:

- PRIMARY estimate: real-vs-shuffled holonomy from a regression including manifold distance and phi
  as covariate terms. This adjusted estimate is the pre-registered semantic effect.
- SECONDARY (transparency): report the UNADJUSTED real-vs-shuffled contrast alongside, so the
  reader sees the effect of adjustment.
- Balance diagnostics (Section 4.3) are reported to characterize imbalance and confirm common
  support. If common support fails (positivity < 90%), the adjusted estimate is flagged as
  partially extrapolated.
- NULL branch (retained): if the semantic effect is present unadjusted but VANISHES after
  adjustment, report this as the finding — the apparent semantic effect is attributable to the
  covariates (manifold distance and/or phi), not to meaning. Pre-registered, publishable null.

The random arm is the noise floor and the lower anchor of the random < shuffled < real gradient
(H-grad) only; the semantic claim is never drawn from a real-vs-random comparison.

## 8. Reproducibility

- Data release plan: TODO
- Code release plan: TODO
- Seed handling: TODO. Seeds fixed and recorded; bitwise reproducibility guaranteed within a single compute backend only.
- Hardware: Apple Silicon / MPS, the sole backend for this study. All results are produced on MPS; no CUDA alternative is part of this pre-registration. The reproducibility claim is bitwise-within-MPS. See ENVIRONMENT.md.
