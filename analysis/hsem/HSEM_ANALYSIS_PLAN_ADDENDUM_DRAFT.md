# H-sem Analysis Plan — Addendum to PREREGISTRATION_v2

**Study:** Gemma Holonomy DOE
**Governs:** Frozen PREREGISTRATION_v2 (this note adds no new hypotheses; it pins implementation choices the prereg left discretionary)
**Stage A band decision:** `TWO_ARM_FALLBACK`
**Results artifact under analysis:** `run/holonomy_results_n390.parquet` (companion: `.json`, `.checkpoint.jsonl`)
**Frozen at commit:** `51931280063d30c8cef2bf05d75a3e44cf996b8c` (local `HEAD` == `origin/main`)
**Status of this note:** DRAFT — to be checked against v2, then frozen *before* any H value is read.

---

## 0. Status of each hypothesis under the actual Stage A branch

| Hypothesis | Status this run | Verdict eligible? |
|---|---|---|
| **H-sem** | Live: real-feature vs shuffled-feature at matched magnitude | **Yes** — primary |
| **H-grad** | `UNDEFINED_AT_MATCHED_MAGNITUDE` | **No** — receives no corroboration/falsification verdict |
| **H-mag** | Evaluable across the magnitude levels actually used | Yes, per its own frozen rule (out of scope for this note) |

This note resolves **H-sem only**. H-grad is explicitly excluded from any verdict. H-mag is evaluated under its own frozen rule and is not modified here.

---

## 1. Frozen H-sem test (restated from PREREGISTRATION_v2, not modified)

The following are carried verbatim from v2 and are **not** discretionary:

- **Primary estimate:** paired within-base-point contrast, real minus shuffled.
- **Scale:** `d_b = log H_real,b − log H_shuffled,b`.
- **Adjustment:** regress `d_b` on per-base covariate differences — manifold distance and phi.
- **Materiality threshold:** `δ = log(1.25) = 0.2231`.
- **Verdict rule (read off the *adjusted* contrast):**
  - **Corroborated** if CI lower bound > `0.2231`
  - **Falsified** if CI upper bound < `0.2231`
  - **Inconclusive** if CI spans `0.2231`
- **Unadjusted contrast reported alongside adjusted.**
- **Balance diagnostics reported, not used as gates.**
- **No parity/violation filter.** The only floor is the pre-measurement Stage A selection rule `det(M) > 0.413`, already applied. No post-hoc exclusions are introduced.

---

## 2. Resolved discretionary points (new decisions recorded here)

### R1 — Two magnitude levels collapse to one `d_b`

**Decision:** Within each arm and base point, average the two log-H values across magnitude level, *then* form one paired contrast:

```
d_b = mean_m(log H_real,b,m) − mean_m(log H_shuffled,b,m)
```

equivalently the log geometric-mean ratio of real to shuffled at base `b`.

**Rationale.** Averaging the logs within arm before differencing is algebraically identical to running the low and high paired contrasts separately and combining them with **equal weight**. The only alternative that would differ is precision-/inverse-variance-weighting the two magnitude contrasts — which is data-dependent and therefore a forking path we explicitly decline. Equal weight is the pre-registerable choice. H-sem is entitled to average magnitude out because magnitude-dependence is owned by H-mag, a separately evaluated hypothesis.

**Completeness.** The run is 2340/2340 with 0 non-finite and 0 nonpositive H, so all four cells (real/shuffled × low/high) exist for every base point; the average is defined everywhere and no missing-cell rule is required.

**Reported alongside (descriptive, not a gate):** the two within-magnitude contrasts (low-only, high-only), so any arm×magnitude interaction the averaging would mask is visible to a reviewer.

### R2 — CI construction and covariate centering

**Decision:**
- Estimator: **OLS** of `d_b` on the (centered) covariate differences; intercept = adjusted mean effect.
- **Center the covariate differences at their sample means** so the intercept is the standard ANCOVA adjusted mean (effect at the sample covariate centroid), not an extrapolation to `covariate_diff = 0`.
- Primary interval: two-sided **95% OLS t-interval** on the intercept.
- Pre-committed robustness: **HC3** robust SE on the same fit, reported alongside. If the classical and HC3 intervals disagree on the verdict relative to `0.2231`, report both and label the verdict **sensitive to SE specification**.

**Rationale.** With raw (uncentered) covariate diffs, the intercept estimates the effect at covariate parity (`diff = 0`), a different estimand that diverges from the adjusted mean exactly under systematic real-vs-shuffled covariate imbalance (by slope × mean covariate diff), with a different — generally wider — SE that can flip the verdict against the threshold. Centering makes the intercept unambiguously the adjusted mean reviewers expect. HC3 nests the homoskedastic case and covers realistic heteroskedasticity across base points; it is a check, not a replacement. Bootstrap is declined as primary to avoid the percentile-vs-BCa knob.

**Reported alongside (secondary descriptive):** the uncentered "effect at covariate parity" intercept, since it directly answers the imbalance critique.

### R3 — Covariate difference definition

**Decision:** Per base point, each covariate difference is `real_covariate − shuffled_covariate`, for manifold distance and phi, taken from the plane-level manifest covariates (already aggregated over the low/high centers).

**Sign convention:** covariate diffs are real-minus-shuffled, matching the sign of `d_b` (also real-minus-shuffled), so the adjustment is coherent.

**Aggregation consistency:** the outcome is averaged over low/high (R1) and the covariates are aggregated over low/high in the manifest — both outcome and covariates are collapsed across magnitude the same way; no level-of-aggregation mismatch.

**Standardization:** optional and cosmetic only (rescales slope interpretability; leaves intercept, its SE, and the verdict CI unchanged). Default: do not standardize.

### R4 — NULL-ATTRIBUTED operationalization

**Decision.** Define using only the already-frozen materiality + CI machinery; introduce no new number.

- **NULL-ATTRIBUTED** fires iff the **unadjusted** contrast meets the *Corroborated* rule (unadjusted CI LB > `0.2231`) **AND** the **adjusted** contrast meets the *Falsified* rule (adjusted CI UB < `0.2231`). I.e. the raw effect is materially present and the adjusted effect is materially gone.
- **Structural constraint:** NULL-ATTRIBUTED is **not** a fourth primary verdict. The primary verdict is always read off the **adjusted** contrast (R-section 1). NULL-ATTRIBUTED is an explanatory annotation that can only co-occur with a primary verdict of **Falsified**; it records *why* the primary landed there.
- **In-between case** (unadjusted Corroborated, adjusted **Inconclusive**): this is real attenuation but not a clean attribution. Label it separately as **attenuated/inconclusive**; do **not** fold it into NULL-ATTRIBUTED, so the clean NULL-ATTRIBUTED claim stays clean.

---

## 3. Additional locks (not in the original four, recorded for completeness)

### R5 — No new exclusions; influence reported not gated

The only floor is the pre-measurement `det(M) > 0.413` from Stage A. **No** post-hoc outlier or influence filter is added. Consistent with the frozen "diagnostics reported, not gated" stance, report leverage and Cook's distance descriptively so any high-leverage base points that move the intercept are visible — but take no action on them.

### R6 — Model form and multiplicity

- **Model form:** linear in the covariate differences — the faithful reading of "regress `d_b` on per-base covariate differences." Pre-committed as linear. Report a residual-vs-covariate diagnostic; do not adapt the functional form to it.
- **Internal multiplicity:** H-sem has a single adjusted contrast — no within-hypothesis correction needed.
- **Cross-hypothesis multiplicity — OPEN, must be confirmed against v2:** if v2 designates H-sem and H-mag as co-primary sharing an error budget, the correction must be fixed *now*, before reading values. If v2 treats them as independently reported, no correction. **This is the one item that cannot be resolved from this note alone — it requires the v2 text.**

---

## 4. Execution order (once this note is frozen)

1. Load `run/holonomy_results_n390.parquet`; assert 2340 rows, 390 base points, 0 non-finite, 0 nonpositive H (re-confirm the committed completion checks).
2. Form per-base `d_b` per R1; form covariate diffs per R3.
3. Fit centered-covariate OLS (R2); read adjusted-mean intercept + 95% t CI; compute HC3 CI.
4. Compute unadjusted contrast (mean and CI of `d_b`) for the NULL-ATTRIBUTED rule and required side-by-side reporting.
5. Apply the frozen verdict rule (§1) to the **adjusted** contrast; annotate NULL-ATTRIBUTED / attenuated-inconclusive per R4.
6. Emit reported-not-gated diagnostics: within-magnitude contrasts (R1), uncentered parity intercept (R2), balance diagnostics, leverage/Cook's distance (R5), residual-vs-covariate plot (R6).
7. No other analysis is run against this artifact. Any further question is a new, separately-recorded plan.

---

## 5. Open items requiring the v2 text before freeze

1. **R6 cross-hypothesis multiplicity** — confirm whether H-sem and H-mag are co-primary sharing an error budget.
2. Confirm the verbatim §1 restatements match v2 exactly (materiality value, covariate list, verdict inequalities, "reported not gated" language).

Resolve these two, then freeze.