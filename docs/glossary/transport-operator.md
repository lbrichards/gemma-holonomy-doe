# transport operator

<span style="background-color: #4051b5; color: white; padding: 0.2em 0.6em; border-radius: 0.25em; font-size: 0.85em; font-weight: 600;">COINED</span>

The 2×2 matrix H accumulating the effect of parallel-transporting a probe frame around the loop via iterated restricted Jacobian steps.

$$
H = \prod_{i=1}^{n} J_{i+1}^+ J_i
$$

## Why this term exists

H encodes how much a tangent frame rotates after traversing the closed loop. The deviation from identity measures local curvature; the holonomy is the rotation angle per enclosed area.

## Source

transport/loop.py

## Depends on

- [restricted Jacobian](restricted-jacobian.md)
- [parallel transport](parallel-transport.md)

---

[← Back to Glossary](index.md)
