# loop sweep

<span style="background-color: #4051b5; color: white; padding: 0.2em 0.6em; border-radius: 0.25em; font-size: 0.85em; font-weight: 600;">COINED</span>

The n_steps + 1 points traced along the γ(t) = c + ρ(cos(2πt) d₁ + sin(2πt) d₂) circle for transport and covariate measurement.

## Why this term exists

The loop discretization must be fine enough for accurate transport accumulation (n_steps = 200) but coarse covariate sampling (32 points) suffices for mean manifold distance.

## Source

transport/loop.py

## Depends on

- [loop center](loop-center.md)
- [loop radius (ρ)](loop-radius.md)

---

[← Back to Glossary](index.md)
