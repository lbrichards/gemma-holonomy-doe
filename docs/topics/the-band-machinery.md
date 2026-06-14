# The Band Machinery

<span style="background-color: #7c4dff; color: white; padding: 0.2em 0.6em; border-radius: 0.25em; font-size: 0.85em; font-weight: 600;">TOPIC</span>

## Terms covered

- [common-support band](../glossary/common-support-band.md)
- [three-arm band](../glossary/three-arm-band.md)
- [two-arm band](../glossary/two-arm-band.md)
- [band-collapse contingency](../glossary/band-collapse-contingency.md)
- [two-arm terminal collapse](../glossary/two-arm-terminal-collapse.md)
- [minimum band width threshold](../glossary/minimum-band-width-threshold.md)
- [pooled IQR](../glossary/pooled-iqr.md)
- [magnitude level](../glossary/magnitude-level.md)
- [undefined at matched magnitude](../glossary/undefined-at-matched-magnitude.md)
- [BandDecision](../glossary/banddecision.md)

## Narrative

The three [plane arm](../glossary/plane-arm.md)s have wildly different native magnitudes: real-feature planes cluster around p50 ~65, shuffled-feature around ~59, and random planes down at ~13. Comparing holonomy across arms without accounting for this would confound semantic structure with magnitude. The [common-support band](../glossary/common-support-band.md) solves this by finding the magnitude range where all three distributions overlap.

The [three-arm band](../glossary/three-arm-band.md) is computed as the overlap of each arm's [p5, p95] [in-plane magnitude (pullback)](../glossary/in-plane-magnitude-pullback.md) range. For this band to be valid, its width must exceed the [minimum band width threshold](../glossary/minimum-band-width-threshold.md) — defined as 0.5 × [pooled IQR](../glossary/pooled-iqr.md) of the concatenated magnitudes. If it does, [magnitude level](../glossary/magnitude-level.md)s (m_low at p25, m_high at p75) are cut from the pooled distribution within the band and applied identically to all arms.

When the [three-arm band](../glossary/three-arm-band.md) is too narrow or empty, the [band-collapse contingency](../glossary/band-collapse-contingency.md) fires. [H-grad](../glossary/h-grad.md) is marked [undefined at matched magnitude](../glossary/undefined-at-matched-magnitude.md) because no magnitude is shared by all three arms. [H-sem](../glossary/h-sem.md) falls back to the [two-arm band](../glossary/two-arm-band.md) (real + shuffled only), which the bench showed overlap comfortably. If even this narrower band fails the width threshold, [two-arm terminal collapse](../glossary/two-arm-terminal-collapse.md) fires and H-sem also becomes undefined. The full audit evidence — band bounds, widths, thresholds, and per-hypothesis status — is captured in the [BandDecision](../glossary/banddecision.md) record.

---

[← Back to Topics](index.md) · [← Back to Glossary](../glossary/index.md)
