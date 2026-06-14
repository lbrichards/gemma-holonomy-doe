# The Corpus Pipeline

<span style="background-color: #7c4dff; color: white; padding: 0.2em 0.6em; border-radius: 0.25em; font-size: 0.85em; font-weight: 600;">TOPIC</span>

## Terms covered

- [base point](../glossary/base-point.md)
- [corpus draw](../glossary/corpus-draw.md)
- [article reconstruction](../glossary/article-reconstruction.md)
- [draw order](../glossary/draw-order.md)
- [survivor](../glossary/survivor.md)
- [drop](../glossary/drop.md)
- [experiment sample](../glossary/experiment-sample.md)
- [reserve pool](../glossary/reserve-pool.md)
- [WikiText-103](../glossary/wikitext-103.md)
- [resid_post](../glossary/resid-post.md)
- [forward hook](../glossary/forward-hook.md)
- [survivor-targeted draw](../glossary/survivor-targeted-draw.md)

## Narrative

Each [base point](../glossary/base-point.md) is a layer-12 [resid_post](../glossary/resid-post.md) activation vector extracted from a [WikiText-103](../glossary/wikitext-103.md) passage. The [corpus draw](../glossary/corpus-draw.md) procedure is fully deterministic: [article reconstruction](../glossary/article-reconstruction.md) joins raw WikiText rows between top-level headers into coherent articles, a seed-42 shuffle assigns [draw order](../glossary/draw-order.md), and the [survivor-targeted draw](../glossary/survivor-targeted-draw.md) continues until ≥700 candidates pass the 64-token filter (sufficient for the 390-point [experiment sample](../glossary/experiment-sample.md) plus [reserve pool](../glossary/reserve-pool.md)).

Not every article yields a usable base point. Articles too short after tokenization become [drop](../glossary/drop.md)s with logged reasons. Points whose planes fail the [det M degeneracy floor](../glossary/det-m-degeneracy-floor.md) are also dropped. A [survivor](../glossary/survivor.md) is a point that clears all filters and enters the [experiment sample](../glossary/experiment-sample.md) (first 390 by draw order) or [reserve pool](../glossary/reserve-pool.md).

Extraction uses a [forward hook](../glossary/forward-hook.md) registered on `model.model.layers[12]` to capture the residual-stream output matching the Gemma Scope SAE's training site. The v1→v2 correction (resid_pre to [resid_post](../glossary/resid-post.md)) was critical: the SAE encodes resid_post, so base points must come from the same site for feature activations to be meaningful.

---

[← Back to Topics](index.md) · [← Back to Glossary](../glossary/index.md)
