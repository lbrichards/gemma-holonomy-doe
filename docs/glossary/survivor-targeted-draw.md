# survivor-targeted draw

<span style="background-color: #4051b5; color: white; padding: 0.2em 0.6em; border-radius: 0.25em; font-size: 0.85em; font-weight: 600;">COINED</span>

A draw procedure that continues drawing candidates until a target number (240) pass the 64-token filter, rather than drawing a fixed number of candidates.

## Why this term exists

Targeting survivors ensures the pool contains exactly the needed number after filtering, avoiding both over-draw waste and under-draw sample size shortfalls.

## Source

corpus/draw.py

## Depends on

- [corpus draw](corpus-draw.md)
- [seed-42 determinism](seed-42-determinism.md)

## Appears in topics

- [The Corpus Pipeline](../topics/the-corpus-pipeline.md)

---

[← Back to Glossary](index.md)
