# corpus draw

<span style="background-color: #4051b5; color: white; padding: 0.2em 0.6em; border-radius: 0.25em; font-size: 0.85em; font-weight: 600;">COINED</span>

The deterministic procedure for selecting WikiText passages: article reconstruction, seed-42 shuffle, raw word filtering, token truncation to 64 Gemma tokens.

## Why this term exists

Fully reproducible passage selection from WikiText-103 ensures any researcher can regenerate the exact same base-point inputs given the seed, revision, and filtering rules.

## Source

corpus/draw.py

## Depends on

- [WikiText-103](wikitext-103.md)

## Appears in topics

- [The Corpus Pipeline](../topics/the-corpus-pipeline.md)

---

[← Back to Glossary](index.md)
