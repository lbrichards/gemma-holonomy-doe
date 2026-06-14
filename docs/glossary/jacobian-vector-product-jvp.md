# Jacobian-vector product (JVP)

<span style="background-color: #448aff; color: white; padding: 0.2em 0.6em; border-radius: 0.25em; font-size: 0.85em; font-weight: 600;">INHERITED</span>

The product Jv of the Jacobian matrix J with a vector v, computed efficiently via forward-mode autodiff; enables computing the restricted Jacobian and pullback metric without materializing the full Jacobian.

$$
Jv = \frac{\partial f}{\partial x} v
$$

## Source

Automatic differentiation (forward mode); JAX/PyTorch documentation

## Appears in topics

- [The Pullback-Metric Geometry](../topics/the-pullback-metric-geometry.md)

---

[← Back to Glossary](index.md)
