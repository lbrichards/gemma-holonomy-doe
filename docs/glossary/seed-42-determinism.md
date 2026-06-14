# seed-42 determinism

<span style="background-color: #4051b5; color: white; padding: 0.2em 0.6em; border-radius: 0.25em; font-size: 0.85em; font-weight: 600;">COINED</span>

The property that the corpus draw with SEED_CORPUS = 42 produces bitwise-identical article indices, token sequences, and draw order across independent runs.

## Why this term exists

Full seed-42 reproducibility means any researcher can regenerate the exact same candidate pool and partition, making the experiment verifiable without access to original artifacts.

## Source

corpus/draw.py

## Depends on

- [random seed](random-seed.md)
- [bitwise reproducibility](bitwise-reproducibility.md)
- [corpus draw](corpus-draw.md)

---

[← Back to Glossary](index.md)
