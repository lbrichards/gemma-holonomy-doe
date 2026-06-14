# eps_mag fallback

<span style="background-color: #4051b5; color: white; padding: 0.2em 0.6em; border-radius: 0.25em; font-size: 0.85em; font-weight: 600;">COINED</span>

When in-plane magnitude is below eps_mag = 2.66, the offset direction switches to G-normalized d1 instead of the projection direction.

## Why this term exists

If the base activation has negligible in-plane component, the projection direction is numerically unstable. The fallback uses a deterministic alternative to avoid division-by-near-zero.

## Source

PREREGISTRATION_v2.md Section 7.3

## Depends on

- [in-plane magnitude (pullback)](in-plane-magnitude-pullback.md)
- [center placement](center-placement.md)

---

[← Back to Glossary](index.md)
