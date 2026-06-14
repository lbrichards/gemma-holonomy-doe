# The Blind Apparatus

<span style="background-color: #7c4dff; color: white; padding: 0.2em 0.6em; border-radius: 0.25em; font-size: 0.85em; font-weight: 600;">TOPIC</span>

## Terms covered

- [frozen (pre-registration status)](../glossary/frozen-pre-registration-status.md)
- [blind (to holonomy)](../glossary/blind-to-holonomy.md)
- [blind manifest boundary](../glossary/blind-manifest-boundary.md)
- [run manifest](../glossary/run-manifest.md)
- [seed-42 determinism](../glossary/seed-42-determinism.md)
- [burned pilot points](../glossary/burned-pilot-points.md)
- [pre-data clarification](../glossary/pre-data-clarification.md)
- [blind handoff](../glossary/blind-handoff.md)
- [Stage A](../glossary/stage-a.md)
- [Stage B](../glossary/stage-b.md)

## Narrative

The integrity of the study rests on decisions being [blind (to holonomy)](../glossary/blind-to-holonomy.md) — made without access to any response values. The pre-registration is [frozen (pre-registration status)](../glossary/frozen-pre-registration-status.md), meaning only [pre-data clarification](../glossary/pre-data-clarification.md)s (timestamped addenda provably recorded before any θ is computed) may refine it.

The pipeline enforces blindness structurally. [Stage A](../glossary/stage-a.md) extracts activations, selects planes, computes the band, places centers, and measures covariates — producing the [run manifest](../glossary/run-manifest.md), a JSON artifact that physically cannot carry holonomy fields (the schema rejects them). [Stage B](../glossary/stage-b.md) reads this manifest and computes holonomy, writing results to a separate artifact. This [blind handoff](../glossary/blind-handoff.md) makes result-dependent design modification impossible.

[seed-42 determinism](../glossary/seed-42-determinism.md) ensures the [corpus draw](../glossary/corpus-draw.md) is bitwise reproducible: any researcher with the seed can regenerate the exact same passages and draw order. The [blind manifest boundary](../glossary/blind-manifest-boundary.md) partitions the pool into experiment sample and reserve by draw order, fixed before any holonomy. The [burned pilot points](../glossary/burned-pilot-points.md) (draw orders 96-111) were used in the resid_post variance re-pilot and are permanently excluded, preserving blindness of the experiment sample.

---

[← Back to Topics](index.md) · [← Back to Glossary](../glossary/index.md)
