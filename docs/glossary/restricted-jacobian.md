# restricted Jacobian

<span style="background-color: #4051b5; color: white; padding: 0.2em 0.6em; border-radius: 0.25em; font-size: 0.85em; font-weight: 600;">COINED</span>

The 2-column matrix [J d1, J d2] giving the Jacobian of the readout map restricted to the plane directions, computed via JVP.

## Why this term exists

Full Jacobian materialization is prohibitive for high-dimensional activations. The restricted Jacobian captures the action of J on the plane subspace, sufficient for pullback metric and transport.

## Source

transport/loop.py

## Depends on

- [Jacobian-vector product (JVP)](jacobian-vector-product-jvp.md)

---

[← Back to Glossary](index.md)
