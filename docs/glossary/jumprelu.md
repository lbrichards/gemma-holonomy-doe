# JumpReLU

<span style="background-color: #448aff; color: white; padding: 0.2em 0.6em; border-radius: 0.25em; font-size: 0.85em; font-weight: 600;">INHERITED</span>

An activation function that outputs zero below a learned threshold and the pre-activation above it, used in Gemma Scope SAEs.

$$
\text{JumpReLU}(x) = x \cdot \mathbf{1}[x > \tau]
$$

## Source

Gemma Scope (Google DeepMind)

## Depends on

- [sparse autoencoder (SAE)](sparse-autoencoder-sae.md)

---

[← Back to Glossary](index.md)
