# Pre-Registration — Gemma Holonomy DOE
Status: FROZEN 2026-06-14 (prior content commit: 61a0581fdb51be46e411b03705fdee24a60bf2d2)

## 1. Purpose and scope

This study tests whether measured holonomy in the Gemma residual stream is explained by activation magnitude alone, by semantic structure associated with Gemma Scope SAE feature directions, or by a graded combination of both. The model is Gemma 2 2B with Gemma Scope SAEs, measured in the residual stream at layer 12. This is a confirmatory study, not exploratory analysis; hypotheses, thresholds, factors, exclusions, and stopping rules are to be frozen before data collection.

## 2. Design

Factorial design, shared blocked base points.

- Factor A — Plane type (3 levels): random; magnitude-matched shuffled-feature; magnitude-matched real-feature.
- Magnitude levels (Factor B) are the 25th (low) and 75th (high) percentiles of pullback-metric
  in-plane magnitude mag(h) (Section 7.3), cut NOT over the real-feature arm but over the
  COMMON-SUPPORT BAND: the magnitude range where all three plane arms have overlapping mass. Rationale:
  the throwaway mag(h) bench showed the arms are widely separated (real p50 ~65, shuffled ~59,
  random ~13); cutting levels on the real arm alone would force random planes 4-5x beyond their
  native magnitude, making the "matched" comparison an extrapolation rather than a match. Matching
  is only meaningful where the distributions overlap.
- Construction of the band (pre-registered): after plane selection and magnitude evaluation on the
  stage-1 sample, compute mag(h) for all three arms; define the common-support band as the overlap
  of the three arms' [p5, p95] ranges; cut m_low, m_high at the 25th/75th percentiles of the POOLED
  magnitudes restricted to that band. The two absolute m values are applied identically to all
  three arms. Record the band bounds and the resulting m values.
- Band-collapse contingency (pre-registered, decided blind): the three-arm common-support band may
  be narrow or empty, because the arms are magnitude-separated (bench: real p50 ~65, random p50
  ~13). Define the band as the overlap of the three arms' [p5, p95] ranges, with REQUIRED MINIMUM
  WIDTH >= 0.5 * (pooled IQR of mag(h)). Two cases:
  1. Band valid (width >= threshold): proceed as specified — cut m_low, m_high at the 25th/75th
     percentiles of pooled magnitudes within the three-arm band; all three hypotheses evaluated at
     matched magnitude as planned.
  2. Band collapsed (empty, or width < threshold): the three plane types do not share a common
     magnitude regime, which is itself a reported finding. Fall back by hypothesis:
     - H-sem (real vs shuffled): use the TWO-ARM common-support band of real and shuffled only
       (the bench shows these overlap: real p50 ~65, shuffled p50 ~59, shuffled tail reaching ~10).
       Cut m_low, m_high at the 25th/75th percentiles of pooled real+shuffled magnitudes within the
       two-arm band. H-sem — the powered, load-bearing claim — proceeds normally.
       - Two-arm terminal case (pre-data clarification, added post-freeze but blind to all
         holonomy): if the two-arm real+shuffled band ALSO fails the minimum width
         (>= 0.5 * pooled IQR of real+shuffled mag(h)), then real and shuffled features do not
         share a common magnitude regime either. H-sem is then reported as UNDEFINED AT MATCHED
         MAGNITUDE — the same honest status H-grad takes under three-arm collapse — and no matched
         semantic verdict is produced. This outcome is itself a reportable finding. The unmatched
         real-vs-shuffled contrast is reported descriptively only, explicitly flagged as
         magnitude-confounded and NOT a corroboration of H-sem.
     - H-grad (random < shuffled < real): reported as UNDEFINED AT MATCHED MAGNITUDE, because no
       magnitude is shared by all three arms. The natural (unmatched) gradient is reported
       descriptively only, explicitly flagged as confounded by magnitude and NOT a corroboration
       of H-grad.
     - H-mag: unaffected; evaluated across the magnitude levels actually used.
  Which case obtains is determined by the blind magnitude evaluation (pipeline step 3, before any
  holonomy is observed) and recorded.
- Sequencing constraint: because mag(h) depends on each plane's Jacobian, m levels cannot be fixed
  until planes are selected and JVPs evaluated. Pipeline order: (1) select planes under the
  det M degeneracy floor, (2) evaluate mag(h) for all arms, (3) compute the common-support band and
  fix m_low, m_high, (4) only then run all three arms at those m. This ordering is pre-registered.
- Response variable: holonomy (loop transport rotation); operational definition in Section 7.
- Blocking unit: base point. Each of 96 fresh base points is evaluated across all 3 plane types × 2 magnitude levels.
- Full design: 96 base points × 3 plane types × 2 magnitude levels = 576 plane evaluations.

## 3. Hypotheses (frozen predictions)

All three are evaluated against a single materiality threshold (Section 6). Each hypothesis has a
three-way verdict: CORROBORATED, FALSIFIED, or INCONCLUSIVE, decided by where the pre-registered
confidence interval sits relative to the materiality threshold.

### H-mag — magnitude main effect

Prediction: holonomy increases with in-plane pullback-metric magnitude (Section 7.3); the high
magnitude level shows materially greater holonomy than the low level, pooled across plane types.

- CORROBORATED: estimated magnitude main effect exceeds materiality AND the CI lower bound lies
  above the materiality threshold.
- FALSIFIED: CI upper bound lies below the materiality threshold (magnitude does not move holonomy
  materially).
- INCONCLUSIVE: CI spans the materiality threshold.

### H-sem — semantic main effect (the powered effect)

Prediction: after covariate adjustment for manifold distance and phi (Section 7.4), real-feature
planes show materially greater holonomy than magnitude-matched shuffled-feature planes.

- CORROBORATED: adjusted real-vs-shuffled contrast exceeds materiality AND CI lower bound above
  the materiality threshold.
- FALSIFIED: CI upper bound below the materiality threshold.
- NULL-ATTRIBUTED (special case of falsification): an unadjusted effect present but vanishing under
  covariate adjustment is reported as attributable to covariates, not meaning (Section 7.4 null
  branch).
- INCONCLUSIVE: CI spans materiality. (Powered to make this outcome unlikely for the material
  effect size; see Section 5.)

### H-grad — monotone gradient

Prediction: holonomy rises monotonically across plane types, random < shuffled < real.

- CORROBORATED: both ordered steps (shuffled - random; real - shuffled) are positive and the
  overall trend across the three levels is material.
- FALSIFIED: the ordering is violated, or the overall trend is immaterial.
- INCONCLUSIVE: ordering holds in point estimates but the trend CI spans materiality.

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

## 6. Materiality threshold and stopping rules (pre-registered)

### 6.1 Materiality threshold

A holonomy effect is MATERIAL if it reaches log(1.25) = 0.2231 on the log scale (a 25%
multiplicative change). This single threshold applies to all three hypotheses (Section 3) and is
the same threshold the Section 5 power calculation was built on.

### 6.2 Decision rule (per hypothesis)

Using the pre-registered confidence interval for each effect:

- CORROBORATED: CI lower bound > materiality threshold.
- FALSIFIED: CI upper bound < materiality threshold.
- INCONCLUSIVE: CI spans the materiality threshold.

(H-sem additionally carries the NULL-ATTRIBUTED branch of Section 7.4.)

### 6.3 Fixed-N, analyze-once

The design is fixed-N. All 96 stage-1 base points are run to completion and analyzed ONCE.
No peeking, no result-dependent early stopping, no result-dependent extension of stage 1.
N was fixed in advance (Section 5); the null verdict is interpretable as material absence
BECAUSE the design was powered at 0.90 for the material effect.

### 6.4 Pre-specified two-stage rule (the ONLY permitted extension)

Decided in advance, before any data:

- TRIGGER: stage-1 H-sem verdict is INCONCLUSIVE (CI spans materiality). No other outcome, for
  any hypothesis, triggers a second stage.
- STAGE 2: run a second batch of N2 = 96 fresh base points, drawn by the identical Section 9
  procedure from the pre-recorded stage-2 reserve (seed-ordered, fixed before stage 1 was seen).
- ANALYSIS: report the stage-1 result STANDALONE regardless. Then report the pooled
  (stage 1 + stage 2, N = 192) result alongside, with a stage indicator term so stage-1/stage-2
  consistency is checkable. The pooled result supplements, never replaces, the standalone stage-1
  result.
- N2 and the trigger are fixed here; nothing about stage 2 is sized or decided from stage-1 values.

## 7. Analysis plan

### Primary contrast

The pre-registered primary semantic estimate is the covariate-adjusted real-feature vs
shuffled-feature holonomy difference, computed as a PAIRED within-base-point contrast (real minus
shuffled at each base point), adjusted for the per-base-point differences in manifold distance and
phi. H-sem's official verdict rests on this paired test. It is the test the Section 5 power
calculation was built on (tau = 0.649 is the across-base-point SD of this per-base contrast), so
the primary test and the power basis are the same object.

### Model / regression form

- PRIMARY (H-sem): paired within-base-point contrast. Per base point,
  d_b = log H_real,b - log H_shuffled,b (the contrast is taken on LOG holonomy, so a difference of
  log(1.25) is a 25% multiplicative change in H and matches the log-scale tau = 0.649 the power
  calc is built on); regress d_b on the per-base covariate differences (manifold distance, phi);
  test whether the adjusted mean of d_b exceeds the materiality threshold. One-sample structure on
  96 paired differences.
- SECONDARY (full factorial, reported alongside): mixed-effects model
  H ~ plane_type * magnitude + manifold_distance + phi + (1 | base_point)
  providing the magnitude main effect (H-mag), the plane-type gradient (H-grad), the
  plane_type x magnitude interaction, and covariate-adjusted effects in one frame. Reported as
  corroboration; agreement across primary and secondary is reported as robustness, disagreement is
  reported and discussed.
- H-mag and H-grad take their verdicts from the secondary model (they are not within-base paired
  contrasts), against the same materiality threshold.

### Decided before data vs reported as-is

- DECIDED IN ADVANCE (frozen here): the three hypotheses and their verdict rules; materiality
  threshold log(1.25); the primary paired test and its covariate adjustment; the secondary model
  form; all degeneracy floors (tau_detM, eps_mag); corpus seed (42) and selection rule; the
  two-stage trigger and N2.
- REPORTED AS-IS (no decision threshold attached): all effect-size point estimates and CIs;
  balance diagnostics (SMD, overlap); per-arm floor exclusion rates; eps_mag fallback counts;
  stage-1/stage-2 consistency.

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
- All contrasts and the materiality threshold are defined on log H. Holonomy H is strictly positive
  by construction (theta and A_enclosed are positive), so log H is well-defined.
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
- tau_detM = 0.413. This is the lowest probed candidate from the throwaway det M bench; it excludes
  0% of real-feature, 3.1% of shuffled-feature, and 0% of random planes in that throwaway probe,
  catching degeneracy without asymmetrically thinning any arm. Calibrated blind on throwaway
  synthetic points (bench/, gitignored), never on the experiment sample.
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
- Degeneracy floor: if mag(h) < eps_mag = 2.66, set the in-plane offset direction to the
  G-normalized d1 instead of v/mag(h). Record when this fallback fires. This value gives a
  real-feature fallback rate of 0% in the throwaway bench; fallback is driven only by the floor/random
  arm. Calibrated blind on throwaway synthetic points (bench/, gitignored), never on the experiment
  sample.

### 7.4 Semantic contrast: covariate-adjustment principle (pre-registered)

The load-bearing semantic contrast is real-feature vs shuffled-feature (NOT real vs random).
It is estimated by covariate adjustment in all cases, with no balance gate:

- Covariate adjustment: the real-vs-shuffled contrast is adjusted for manifold distance and phi.
  The PRIMARY (paired within-base-point) and SECONDARY (mixed-effects) tests that carry the
  verdicts are defined in Section 7 "Model / regression form"; this subsection states only the
  adjustment principle and the null branch, not a separate primary test.
- Always report the unadjusted contrast alongside the adjusted one.
- Balance diagnostics (Section 4.3) are reported to characterize imbalance and confirm common
  support. If common support fails (positivity < 90%), the adjusted estimate is flagged as
  partially extrapolated.
- NULL branch (retained): if the semantic effect is present unadjusted but VANISHES after
  adjustment, report this as the finding — the apparent semantic effect is attributable to the
  covariates (manifold distance and/or phi), not to meaning. Pre-registered, publishable null.

The random arm is the noise floor and the lower anchor of the random < shuffled < real gradient
(H-grad) only; the semantic claim is never drawn from a real-vs-random comparison.

## 8. Reproducibility

- Data release: the 96 (and any stage-2) base-point activations, plane definitions, per-plane
  holonomy, det M, mag(h), and covariate values released as artifacts; corpus referenced by
  WikiText-103 version + seed 42 + recorded indices (passages not redistributed, regenerable from
  the rule).
- Code release: full run pipeline, analysis scripts, and bench/probe scripts released; repo public
  on freeze of results.
- Seed handling: SEED_CORPUS = 42 for passage draw; all stochastic steps (plane sampling, any RNG
  in control construction) seeded and recorded. Bitwise reproducibility guaranteed within the MPS
  backend only (see ENVIRONMENT.md).
- Hardware: Apple Silicon / MPS, the sole backend for this study. All results are produced on MPS; no CUDA alternative is part of this pre-registration. The reproducibility claim is bitwise-within-MPS. See ENVIRONMENT.md.

## 9. Base-point corpus and sampling (pre-registered)

- Corpus: WikiText-103 (declarative English prose; license CC BY-SA). Chosen over instruction-style
  data (e.g. Alpaca) so base-point activations match the declarative prose on which the degeneracy
  floors and the exploratory variance (tau) were calibrated. Scope of the resulting claim is
  therefore English natural prose; cross-lingual holonomy is named as future work (limitation).
- Selection rule (fully reproducible):
  1. Fix random seed SEED_CORPUS = 42 (set before any run; record value).
  2. Draw candidate passages from WikiText-103 (record exact dataset version/revision).
  3. Truncate each passage to the first 64 tokens (Gemma tokenizer). Passages shorter than 64
     tokens are DROPPED (recorded), never padded.
  4. Base point = the layer-12 residual-stream activation at the final (64th) token position.
  5. Oversample candidates (target ~240) so that >= 192 base points survive the degeneracy floors
     (Section 7.2.1) and short-passage drops. Record the actual survivor count.
  6. Record: dataset version, seed, drawn indices, dropped indices with reason, and the final
     retained indices. The first 96 survivors in seed order are the stage-1 experiment sample;
     the next 96 survivors are the stage-2 reserve; any remainder is held unused, in recorded order.
- Language: English only (pre-registered scope boundary).

## Addenda (post-freeze, blind to holonomy)

- 2026-06-14: Added the two-arm terminal-collapse case for H-sem (Section 2 band-collapse
  contingency). Justification: the frozen text specified the three-arm collapse fallback but was
  silent on whether the two-arm fallback could itself fail the width threshold. This addendum
  defines that bottom case. It is outcome-independent: the band determination occurs at pipeline
  step 3, before any holonomy is computed, so this clarification cannot have been influenced by
  results. The pairing rule (real = jointly active features at the base point; shuffled = one
  active feature paired with one inactive real dictionary feature; random = two random unit
  directions) is also recorded here as the operational definition used by planes/, fixed before
  any holonomy computation.
