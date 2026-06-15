# Pre-Analysis Reconciliation Against PREREGISTRATION_v2

Prepared before running any H-sem estimate. This report reconciles
`analysis/hsem/HSEM_ANALYSIS_PLAN_ADDENDUM_DRAFT.md` against the frozen
`PREREGISTRATION_v2.md` text.

## 1a. Verbatim Section 1 Check

### Materiality Value

Addendum item:

> **Materiality threshold:** `delta = log(1.25) = 0.2231`.

Frozen v2 text:

> A holonomy effect is MATERIAL if it reaches log(1.25) = 0.2231 on the log scale (a 25%
> multiplicative change). This single threshold applies to all three hypotheses (Section 3) and is
> the same threshold the Section 5 power calculation was built on.

Status: matches.

### Covariate List

Addendum item:

> **Adjustment:** regress `d_b` on per-base covariate differences — manifold distance and phi.

Frozen v2 text:

> Prediction: after covariate adjustment for manifold distance and phi (Section 7.4), real-feature
> planes show materially greater holonomy than magnitude-matched shuffled-feature planes.

Frozen v2 text:

> The pre-registered primary semantic estimate is the covariate-adjusted real-feature vs
> shuffled-feature holonomy difference, computed as a PAIRED within-base-point contrast (real minus
> shuffled at each base point), adjusted for the per-base-point differences in manifold distance and
> phi.

Status: matches.

### Verdict Inequalities

Addendum item:

> **Corroborated** if CI lower bound > `0.2231`
>
> **Falsified** if CI upper bound < `0.2231`
>
> **Inconclusive** if CI spans `0.2231`

Frozen v2 text:

> Using the pre-registered confidence interval for each effect:
>
> - CORROBORATED: CI lower bound > materiality threshold.
> - FALSIFIED: CI upper bound < materiality threshold.
> - INCONCLUSIVE: CI spans the materiality threshold.

Status: matches.

### Balance Diagnostics Reported, Not Gated

Addendum item:

> **Balance diagnostics reported, not used as gates.**

Frozen v2 text:

> Balance is reported as a diagnostic, NOT used as a pass/fail gate. The semantic contrast is
> covariate-adjusted by default (see Section 7), so these diagnostics characterize imbalance and
> verify common support; they do not switch the analysis path.

Frozen v2 text:

> Balance diagnostics (Section 4.3) are reported to characterize imbalance and confirm common
> support. If common support fails (positivity < 90%), the adjusted estimate is flagged as
> partially extrapolated.

Status: matches.

### Stage A det(M) Floor as the Only Filter

Addendum item:

> **No parity/violation filter.** The only floor is the pre-measurement Stage A selection rule
> `det(M) > 0.413`, already applied. No post-hoc exclusions are introduced.

Frozen v2 text:

> Floor: det(M) > tau_detM for every plane, all three arms, where M is the pullback-metric plane
> Gram (Section 7.3).

Frozen v2 text:

> tau_detM = 0.413.

Frozen v2 text:

> Application point, all-arms uniformity, and exclusion-rate reporting: unchanged from prior
> 7.2.1 (enforced at pair selection, before any holonomy observed, reported per arm).

Frozen v2 text:

> REPORTED AS-IS (no decision threshold attached): all effect-size point estimates and CIs;
> balance diagnostics (SMD, overlap); per-arm floor exclusion rates; eps_mag fallback counts; reserve/dropout handling.

Status: matches. v2 names no parity or violation filter for this DOE analysis; the addendum's
"no post-hoc exclusions" statement is consistent with v2's selection-stage det(M) floor and
reported-not-gated diagnostics.

## 1b. R6 Cross-Hypothesis Multiplicity

Question: does v2 designate H-sem and H-mag as co-primary sharing an error budget?

Frozen v2 text:

> All three are evaluated against a single materiality threshold (Section 6). Each hypothesis has a
> three-way verdict: CORROBORATED, FALSIFIED, or INCONCLUSIVE, decided by where the pre-registered
> confidence interval sits relative to the materiality threshold.

Frozen v2 text:

> H-sem's official verdict rests on this paired test.

Frozen v2 text:

> SECONDARY (full factorial, reported alongside): mixed-effects model
> H ~ plane_type * magnitude + manifold_distance + phi + (1 | base_point)
> providing the magnitude main effect (H-mag), the plane-type gradient (H-grad), the
> plane_type x magnitude interaction, and covariate-adjusted effects in one frame. Reported as
> corroboration; agreement across primary and secondary is reported as robustness, disagreement is
> reported and discussed.

Frozen v2 text:

> H-mag and H-grad take their verdicts from the secondary model (they are not within-base paired
> contrasts), against the same materiality threshold.

Frozen v2 text:

> DECIDED IN ADVANCE (frozen here): the three hypotheses and their verdict rules; materiality
> threshold log(1.25); the primary paired test and its covariate adjustment; the secondary model
> form; all degeneracy floors (tau_detM, eps_mag); corpus seed (42) and selection rule; the
> single-stage fixed-N rule.

Finding: v2 does not state that H-sem and H-mag are co-primary hypotheses sharing a familywise
error budget. It gives each hypothesis its own verdict rule against the shared materiality
threshold, identifies H-sem's paired test as the official H-sem verdict test, and assigns H-mag and
H-grad to the secondary model.

Consequence: v2 names no alpha-sharing correction and no multiplicity correction to apply here.
If Larry elects to add such a correction, that would be a new pre-analysis decision to record
before running estimates; it is not specified by the frozen v2 text.

