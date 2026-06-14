# The Pullback-Metric Geometry

<span style="background-color: #7c4dff; color: white; padding: 0.2em 0.6em; border-radius: 0.25em; font-size: 0.85em; font-weight: 600;">TOPIC</span>

## Terms covered

- [pullback metric](../glossary/pullback-metric.md)
- [Gram matrix](../glossary/gram-matrix.md)
- [Jacobian-vector product (JVP)](../glossary/jacobian-vector-product-jvp.md)
- [in-plane magnitude (pullback)](../glossary/in-plane-magnitude-pullback.md)
- [enclosed area (wedge)](../glossary/enclosed-area-wedge.md)
- [one-metric principle](../glossary/one-metric-principle.md)
- [restricted Jacobian](../glossary/restricted-jacobian.md)
- [G-projection coefficients](../glossary/g-projection-coefficients.md)
- [det M degeneracy floor](../glossary/det-m-degeneracy-floor.md)
- [97% deviation metric](../glossary/97-deviation-metric.md)

## Narrative

The model's learned geometry differs drastically from Euclidean space — the [97% deviation metric](../glossary/97-deviation-metric.md) from FishBack quantifies this as over 97% deviation in relative spectral norm. This motivates the [one-metric principle](../glossary/one-metric-principle.md): all geometric quantities must be computed under a single consistent metric, the [pullback metric](../glossary/pullback-metric.md) G = JᵀJ induced by the [readout map](../glossary/readout-map.md)'s Jacobian.

Computing the full Jacobian is prohibitive, so we use [Jacobian-vector product (JVP)](../glossary/jacobian-vector-product-jvp.md)s to build the [restricted Jacobian](../glossary/restricted-jacobian.md) — the 2-column matrix [J d₁, J d₂] capturing how the readout map acts on the plane directions. From this we derive the [Gram matrix](../glossary/gram-matrix.md) M = DᵀGD, whose entries encode inner products under the pullback metric.

The [G-projection coefficients](../glossary/g-projection-coefficients.md) a = M⁻¹(JD)ᵀ(Jh) express the base activation's G-orthogonal projection onto the plane, yielding [in-plane magnitude (pullback)](../glossary/in-plane-magnitude-pullback.md) mag(h) = √(aᵀMa). The [enclosed area (wedge)](../glossary/enclosed-area-wedge.md) A = ρ²√(det M) normalizes the rotation angle to produce [holonomy](../glossary/holonomy.md). When det(M) approaches zero the plane directions become G-collinear; the [det M degeneracy floor](../glossary/det-m-degeneracy-floor.md) (τ = 0.413) excludes such degenerate planes before any holonomy is measured.

---

[← Back to Topics](index.md) · [← Back to Glossary](../glossary/index.md)
