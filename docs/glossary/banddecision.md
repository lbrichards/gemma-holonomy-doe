# BandDecision

<span style="background-color: #4051b5; color: white; padding: 0.2em 0.6em; border-radius: 0.25em; font-size: 0.85em; font-weight: 600;">COINED</span>

The structured record returned by bands/ containing band bounds, threshold, status (THREE_ARM_VALID/TWO_ARM_FALLBACK/TWO_ARM_TERMINAL), m_low, m_high, and per-hypothesis matched flags.

## Why this term exists

The BandDecision schema captures the full evidence for the band decision — not just which branch fired but the three-arm and two-arm widths that forced it, enabling audit verification.

## Source

bands/support.py

## Depends on

- [common-support band](common-support-band.md)
- [three-arm band](three-arm-band.md)
- [two-arm band](two-arm-band.md)

---

[← Back to Glossary](index.md)
