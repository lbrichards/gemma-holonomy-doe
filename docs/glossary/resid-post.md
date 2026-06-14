# resid_post

<span style="background-color: #4051b5; color: white; padding: 0.2em 0.6em; border-radius: 0.25em; font-size: 0.85em; font-weight: 600;">COINED</span>

The residual-stream activation at the output of a transformer block (post-layernorm, pre-next-block), the extraction site for this study.

## Why this term exists

The Gemma Scope SAE is trained on layer-12 resid_post, so base points must be extracted at the same site for SAE encoding to be meaningful. v1's resid_pre error motivated the v2 correction.

## Source

extraction/extract.py

## Depends on

- [residual stream](residual-stream.md)

## Appears in topics

- [The Corpus Pipeline](../topics/the-corpus-pipeline.md)
- [The Variance and Power Story](../topics/the-variance-and-power-story.md)

---

[← Back to Glossary](index.md)
