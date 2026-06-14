# The Three Plane Arms

<span style="background-color: #7c4dff; color: white; padding: 0.2em 0.6em; border-radius: 0.25em; font-size: 0.85em; font-weight: 600;">TOPIC</span>

## Terms covered

- [plane arm](../glossary/plane-arm.md)
- [real-feature plane](../glossary/real-feature-plane.md)
- [shuffled-feature plane](../glossary/shuffled-feature-plane.md)
- [random plane](../glossary/random-plane.md)
- [pairing rule](../glossary/pairing-rule.md)
- [active feature](../glossary/active-feature.md)
- [inactive feature](../glossary/inactive-feature.md)
- [magnitude-matched shuffled-feature plane](../glossary/magnitude-matched-shuffled-feature-plane.md)
- [Factor A (plane type)](../glossary/factor-a-plane-type.md)

## Narrative

[Factor A (plane type)](../glossary/factor-a-plane-type.md) is the semantic dimension of the factorial design, with three levels forming a gradient from noise floor to intact semantic structure. The [pairing rule](../glossary/pairing-rule.md) specifies exactly how each [plane arm](../glossary/plane-arm.md) is constructed before any holonomy is computed.

The [random plane](../glossary/random-plane.md) uses two random unit directions with no relationship to the base point — pure geometric noise providing the lower anchor. The [shuffled-feature plane](../glossary/shuffled-feature-plane.md) pairs one [active feature](../glossary/active-feature.md) (code > 0 at the base point) with one [inactive feature](../glossary/inactive-feature.md) (code ≤ 0), breaking joint activation structure while preserving real dictionary directions. The [real-feature plane](../glossary/real-feature-plane.md) uses two jointly active SAE features, capturing intact semantic geometry.

This ordering — random < shuffled < real — forms the semantic gradient tested by [H-grad](../glossary/h-grad.md). The key control is the [magnitude-matched shuffled-feature plane](../glossary/magnitude-matched-shuffled-feature-plane.md): by matching shuffled planes to real planes in [in-plane magnitude (pullback)](../glossary/in-plane-magnitude-pullback.md), the [H-sem](../glossary/h-sem.md) contrast isolates semantic coherence from the confounding effect of activation magnitude. If shuffled planes had systematically different magnitudes, any holonomy difference might reflect that rather than meaning.

---

[← Back to Topics](index.md) · [← Back to Glossary](../glossary/index.md)
