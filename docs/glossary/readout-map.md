# readout map

<span style="background-color: #4051b5; color: white; padding: 0.2em 0.6em; border-radius: 0.25em; font-size: 0.85em; font-weight: 600;">COINED</span>

The function F: resid_post_12 → resid_post_13 computed by patching layer-12 output and capturing layer-13 output, whose Jacobian drives transport.

## Why this term exists

The JVP of the readout map gives the local linear approximation of how layer-13 responds to perturbations at layer-12, defining the pullback metric and connection for transport.

## Source

stage_a/run.py

## Depends on

- [forward hook](forward-hook.md)
- [Jacobian-vector product (JVP)](jacobian-vector-product-jvp.md)

---

[← Back to Glossary](index.md)
