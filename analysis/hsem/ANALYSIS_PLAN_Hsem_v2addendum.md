H-sem Analysis Plan — Addendum to PREREGISTRATION_v2

Study: Gemma Holonomy DOE
Governs: Frozen PREREGISTRATION_v2 (this note adds no new hypotheses; it pins implementation choices the prereg left discretionary)
Stage A band decision: TWO_ARM_FALLBACK
Results artifact under analysis: run/holonomy_results_n390.parquet (companion: .json, .checkpoint.jsonl)
Frozen results commit: 51931280063d30c8cef2bf05d75a3e44cf996b8c — the anchor for the analyzed bytes. The repo HEAD has since advanced (draft-addendum commits); the results commit is an ancestor of current HEAD, and HEAD == origin/main. The integrity check (§4) confirms the results file is byte-identical to its blob at the results commit, which is the invariant that matters.
Status of this note: FREEZE-READY pending one scope decision (§5). §1 reconciled verbatim against v2; R6 resolved from the v2 text.


0. Status of each hypothesis under the actual Stage A branch

HypothesisStatus this runVerdict sourceVerdict eligible?H-semLive: real-feature vs shuffled-feature at matched magnitudePrimary paired test (this note)Yes — primaryH-magEvaluable across the magnitude levels actually usedSecondary mixed-effects model (v2 Section, see §6)Yes, from the secondary modelH-gradUNDEFINED_AT_MATCHED_MAGNITUDE—No — receives no corroboration/falsification verdict

This note's primary subject is H-sem. Because v2 makes the secondary mixed-effects model the official verdict source for H-mag, and H-mag remains evaluable this run, §6 pins the fallback-case implementation of that secondary model. H-grad is excluded from any verdict; its term in the secondary model is fit-but-suppressed (§6).


1. Frozen H-sem test (restated from PREREGISTRATION_v2, reconciled verbatim)

The following are carried verbatim from v2 and are not discretionary. Each was confirmed against the v2 text in reports/prereg_reconciliation.md (status: matches).


Primary estimate: paired within-base-point contrast, real minus shuffled.
Scale: d_b = log H_real,b − log H_shuffled,b.
Adjustment: regress d_b on per-base covariate differences — manifold distance and phi.
Materiality threshold: δ = log(1.25) = 0.2231 (v2: single threshold, all three hypotheses, same threshold the power calc was built on).
Verdict rule (read off the adjusted contrast):

Corroborated if CI lower bound > 0.2231
Falsified if CI upper bound < 0.2231
Inconclusive if CI spans 0.2231



Unadjusted contrast reported alongside adjusted.
Balance diagnostics reported, not used as gates.
No parity/violation filter. The only floor is the pre-measurement Stage A selection rule det(M) > τ_detM, τ_detM = 0.413, enforced at pair selection before any holonomy was observed. No post-hoc exclusions are introduced.



2. Resolved discretionary points (new decisions recorded here)

R1 — Two magnitude levels collapse to one d_b

Decision: Within each arm and base point, average the two log-H values across magnitude level, then form one paired contrast:

d_b = mean_m(log H_real,b,m) − mean_m(log H_shuffled,b,m)

equivalently the log geometric-mean ratio of real to shuffled at base b.

Rationale. Averaging the logs within arm before differencing is algebraically identical to running the low and high paired contrasts separately and combining them with equal weight. The only alternative that would differ is precision-/inverse-variance-weighting the two magnitude contrasts — which is data-dependent and therefore a forking path we explicitly decline. Equal weight is the pre-registerable choice. H-sem is entitled to average magnitude out because magnitude-dependence is owned by H-mag, a separately evaluated hypothesis.

Completeness. The run is 2340/2340 with 0 non-finite and 0 nonpositive H, so all four cells (real/shuffled × low/high) exist for every base point; the average is defined everywhere and no missing-cell rule is required.

Reported alongside (descriptive, not a gate): the two within-magnitude contrasts (low-only, high-only), so any arm×magnitude interaction the averaging would mask is visible to a reviewer.

R2 — CI construction and covariate centering

Decision:


Estimator: OLS of d_b on the (centered) covariate differences; intercept = adjusted mean effect.
Center the covariate differences at their sample means so the intercept is the standard ANCOVA adjusted mean (effect at the sample covariate centroid), not an extrapolation to covariate_diff = 0.
Primary interval: two-sided 95% OLS t-interval on the intercept.
Pre-committed robustness: HC3 robust SE on the same fit, reported alongside. If the classical and HC3 intervals disagree on the verdict relative to 0.2231, report both and label the verdict sensitive to SE specification.


Rationale. With raw (uncentered) covariate diffs, the intercept estimates the effect at covariate parity (diff = 0), a different estimand that diverges from the adjusted mean exactly under systematic real-vs-shuffled covariate imbalance (by slope × mean covariate diff), with a different — generally wider — SE that can flip the verdict against the threshold. Centering makes the intercept unambiguously the adjusted mean reviewers expect. HC3 nests the homoskedastic case and covers realistic heteroskedasticity across base points; it is a check, not a replacement. Bootstrap is declined as primary to avoid the percentile-vs-BCa knob.

Reported alongside (secondary descriptive): the uncentered "effect at covariate parity" intercept, since it directly answers the imbalance critique.

R3 — Covariate difference definition

Decision: Per base point, each covariate difference is real_covariate − shuffled_covariate, for manifold distance and phi, taken from the plane-level manifest covariates (already aggregated over the low/high centers).

Sign convention: covariate diffs are real-minus-shuffled, matching the sign of d_b (also real-minus-shuffled), so the adjustment is coherent.

Aggregation consistency: the outcome is averaged over low/high (R1) and the covariates are aggregated over low/high in the manifest — both outcome and covariates are collapsed across magnitude the same way; no level-of-aggregation mismatch.

Standardization: optional and cosmetic only (rescales slope interpretability; leaves intercept, its SE, and the verdict CI unchanged). Default: do not standardize.

R4 — NULL-ATTRIBUTED operationalization

Decision. Define using only the already-frozen materiality + CI machinery; introduce no new number.


NULL-ATTRIBUTED fires iff the unadjusted contrast meets the Corroborated rule (unadjusted CI LB > 0.2231) AND the adjusted contrast meets the Falsified rule (adjusted CI UB < 0.2231). I.e. the raw effect is materially present and the adjusted effect is materially gone.
Structural constraint: NULL-ATTRIBUTED is not a fourth primary verdict. The primary verdict is always read off the adjusted contrast (§1). NULL-ATTRIBUTED is an explanatory annotation that can only co-occur with a primary verdict of Falsified; it records why the primary landed there.
In-between case (unadjusted Corroborated, adjusted Inconclusive): this is real attenuation but not a clean attribution. Label it separately as attenuated/inconclusive; do not fold it into NULL-ATTRIBUTED, so the clean NULL-ATTRIBUTED claim stays clean.



3. Additional locks (recorded for completeness)

R5 — No new exclusions; influence reported not gated

The only floor is the pre-measurement det(M) > 0.413 from Stage A. No post-hoc outlier or influence filter is added. Consistent with the frozen "diagnostics reported, not gated" stance, report leverage and Cook's distance descriptively so any high-leverage base points that move the intercept are visible — but take no action on them.

R6 — Model form and multiplicity — RESOLVED


Model form: linear in the covariate differences — the faithful reading of "regress d_b on per-base covariate differences." Pre-committed as linear. Report a residual-vs-covariate diagnostic; do not adapt the functional form to it.
Internal multiplicity: H-sem has a single adjusted contrast — no within-hypothesis correction needed.
Cross-hypothesis multiplicity — RESOLVED FROM v2: no correction. v2 gives each hypothesis its own three-way verdict against a single shared materiality threshold, names H-sem's paired test as the official H-sem verdict, and routes H-mag/H-grad to the secondary model. v2 specifies no familywise error budget and no multiplicity correction. We honor the frozen design: no correction is applied. This is principled as well as procedural — v2's design is an effect-size gate (does a CI clear a 25% multiplicative change), and the three hypotheses answer distinct scientific questions rather than testing one composite null, so a classical familywise correction is not indicated. Adding one now would be a deviation from the frozen design requiring its own recorded justification; we decline it. (Source: reports/prereg_reconciliation.md §1b.)


R7 — Positivity / common-support flag

v2 specifies a named diagnostic flag: if positivity (common support) < 90%, the adjusted estimate is flagged as partially extrapolated. This is consistent with the reported-not-gated stance — it annotates the estimate, it does not switch the analysis path or gate the verdict. The harness computes the common-support / positivity fraction and emits the partially_extrapolated flag when it falls below 90%. The verdict is still read off the adjusted contrast regardless; the flag travels with it.


4. Execution order (once this note is frozen)


Integrity (re)check. Load run/holonomy_results_n390.parquet; assert 2340 rows, 390 base points, all four real/shuffled × low/high cells per base, 0 non-finite, 0 nonpositive H. Additionally hash the results-file blob and confirm it is byte-identical to its version at the frozen results commit 5193128 (not just "manifest unchanged" — prove the analyzed bytes are the committed bytes).
Form per-base d_b per R1; form covariate diffs per R3.
H-sem primary: fit centered-covariate OLS (R2); read adjusted-mean intercept + 95% t CI; compute HC3 CI.
Compute unadjusted contrast (mean and CI of d_b) for the NULL-ATTRIBUTED rule and required side-by-side reporting.
Apply the frozen verdict rule (§1) to the adjusted contrast; annotate NULL-ATTRIBUTED / attenuated-inconclusive per R4.
Secondary model (if in scope — see §5/§6): fit the v2 mixed-effects model in its TWO_ARM_FALLBACK form per §6; read H-mag's verdict from it; report H-sem agreement across primary/secondary as robustness.
Emit reported-not-gated diagnostics: within-magnitude contrasts (R1), uncentered parity intercept (R2), balance diagnostics (SMD, overlap), positivity / partially_extrapolated flag (R7), leverage/Cook's distance (R5), residual-vs-covariate plot (R6).
No other analysis is run against this artifact. Any further question is a new, separately-recorded plan.



5. The one open item before freeze

Secondary-model scope decision (Larry). v2 freezes the secondary mixed-effects model and makes it the official verdict source for H-mag, which is evaluable this run. Two defensible options:


(A) Single pass: include the secondary model in this analysis pass under the §6 fallback specification. H-sem and H-mag verdicts emerge together; H-sem agreement across primary/secondary is reported as robustness.
(B) Split pass: freeze and run H-sem primary now; treat the secondary model (H-mag verdict + H-sem robustness corroboration) as a separate, separately-frozen pass.


§6 below specifies the fallback implementation so that (A) is freeze-ready if you choose it. If you choose (B), §6 still stands; it just freezes with the second pass instead.

(Everything else is closed: §1 reconciled verbatim; R6 resolved to no-correction from the v2 text.)


6. Secondary mixed-effects model — TWO_ARM_FALLBACK implementation note

Frozen v2 model form (not modified):

H ~ plane_type * magnitude + manifold_distance + phi + (1 | base_point)

reported alongside the primary; it provides the magnitude main effect (H-mag), the plane-type gradient (H-grad), the plane_type × magnitude interaction, and covariate-adjusted effects in one frame. v2: H-mag and H-grad take their verdicts from this secondary model (they are not within-base paired contrasts), against the same materiality threshold; agreement across primary and secondary is reported as robustness, disagreement is reported and discussed.

The following pin the points the fallback branch leaves discretionary. None changes the frozen model form.

S1 — plane_type levels. The data contain three arms (real, shuffled, random). Keep all three levels in plane_type; the random arm is informative for the gradient and for common-support context and v2 does not drop it. The reference level is set to shuffled so the plane_type contrasts read as "relative to the magnitude-matched shuffled control," matching the primary's framing. (Reference-level choice is presentational; it leaves the fitted model and all verdicts invariant.)

S2 — H-grad term under UNDEFINED_AT_MATCHED_MAGNITUDE. H-grad is excluded from any verdict this run. Fit the full model as frozen (do not drop terms), but suppress H-grad's verdict in the output — report its coefficient as a descriptive estimate explicitly labeled UNDEFINED_AT_MATCHED_MAGNITUDE / no verdict, and do not apply the corroborated/falsified/inconclusive rule to it. Rationale: dropping the term would change the fit and contaminate H-mag and the interaction; suppressing only the verdict honors the band decision without perturbing the other estimates.

S3 — H-mag verdict extraction. H-mag's verdict is the magnitude main-effect coefficient (on the log-H scale) read against the same materiality threshold 0.2231 under the same three-way rule (§1). Report it with its pre-registered CI.

S4 — CI construction for the mixed model. Distinct from the OLS-t pinned for H-sem. Use the model's profile / Wald CI as produced by the fitting library's standard method, two-sided 95%, applied to the magnitude main effect for H-mag. Pre-commit the method now (Wald on REML fit) and record the library/version in the run log; do not switch methods after seeing the estimate. If the random-effect variance is estimated at the boundary (≈ 0), report it and fall back to the OLS-equivalent fixed-effects CI, flagged.

S5 — Estimand scale consistency. The mixed model is fit on log H (same scale as the primary d_b), so the magnitude main effect and the materiality threshold are on the same log scale. No back-transformation before the verdict.

S6 — Reported, not gated. All secondary-model effect sizes and CIs are reported as-is. The plane_type × magnitude interaction is reported descriptively (it also bears on R1's averaging assumption — a large interaction is the thing the within-magnitude contrasts would surface). Agreement/disagreement between the primary H-sem paired verdict and the secondary model's covariate-adjusted plane-type effect is reported as robustness per v2.