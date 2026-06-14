# article reconstruction

<span style="background-color: #4051b5; color: white; padding: 0.2em 0.6em; border-radius: 0.25em; font-size: 0.85em; font-weight: 600;">COINED</span>

The process of joining WikiText-103 raw rows between consecutive top-level headers into coherent articles, prior to passage sampling.

## Why this term exists

Raw WikiText rows are line-level fragments; reconstruction yields contiguous prose passages suitable for 64-token truncation, avoiding the length/format bias of raw-row sampling.

## Source

corpus/draw.py

## Depends on

- [WikiText-103](wikitext-103.md)

---

[← Back to Glossary](index.md)
