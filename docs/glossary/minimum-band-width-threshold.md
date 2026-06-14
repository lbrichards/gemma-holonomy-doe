# minimum band width threshold

<span style="background-color: #4051b5; color: white; padding: 0.2em 0.6em; border-radius: 0.25em; font-size: 0.85em; font-weight: 600;">COINED</span>

The requirement that a common-support band have width >= 0.5 × pooled IQR to be considered valid for matched comparisons.

$$
\text{width} \geq 0.5 \times \text{IQR}_{\text{pooled}}
$$

## Why this term exists

A narrow overlap that is technically non-empty but very thin would force magnitude levels into an artificially compressed range, making the matched comparison meaningless.

## Source

PREREGISTRATION_v2.md Section 2

## Depends on

- [pooled IQR](pooled-iqr.md)
- [common-support band](common-support-band.md)

## Appears in topics

- [The Band Machinery](../topics/the-band-machinery.md)

---

[← Back to Glossary](index.md)
