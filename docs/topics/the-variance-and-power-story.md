# The Variance and Power Story

<span style="background-color: #7c4dff; color: white; padding: 0.2em 0.6em; border-radius: 0.25em; font-size: 0.85em; font-weight: 600;">TOPIC</span>

## Terms covered

- [planning tau](../glossary/planning-tau.md)
- [97% deviation metric](../glossary/97-deviation-metric.md)
- [holonomy-magnitude response](../glossary/holonomy-magnitude-response.md)
- [fixed-N design](../glossary/fixed-n-design.md)
- [materiality threshold](../glossary/materiality-threshold.md)
- [resid_post](../glossary/resid-post.md)
- [burned pilot points](../glossary/burned-pilot-points.md)

## Narrative

The v1 pre-registration was frozen with τ = 0.649 and N = 96, derived from pilot data at resid_pre. Then the hook-site error was discovered: the Gemma Scope SAE is trained on [resid_post](../glossary/resid-post.md), not resid_pre. The [97% deviation metric](../glossary/97-deviation-metric.md) from FishBack shows these sites are geometrically different — the pullback metric deviates over 97% from Euclidean.

Correcting the extraction site required re-piloting the contrast variance. The [burned pilot points](../glossary/burned-pilot-points.md) (draw orders 96-111) yielded τ_pilot = 1.11 with 95% CI [0.82, 1.72]. The [planning tau](../glossary/planning-tau.md) of 1.30 is a conservative choice near the 72% upper bound, and N was re-derived to 390 for 0.90 [power](../glossary/power.md) at the [materiality threshold](../glossary/materiality-threshold.md) δ = 0.223.

The [fixed-N design](../glossary/fixed-n-design.md) commits to running all 390 points before analysis, with no peeking or early stopping. v1 included a [two-stage rule (retired)](../glossary/two-stage-rule-retired.md) that would trigger a second batch if H-sem was inconclusive at N = 96 — but with the corrected variance and N = 390, this rescue mechanism became unnecessary and was removed in v2. The governing design is single-stage: all 390 base points run to completion, then analyzed once.

---

[← Back to Topics](index.md) · [← Back to Glossary](../glossary/index.md)
