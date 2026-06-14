# forward hook

<span style="background-color: #4051b5; color: white; padding: 0.2em 0.6em; border-radius: 0.25em; font-size: 0.85em; font-weight: 600;">COINED</span>

A PyTorch hook registered on a module to capture intermediate activations during the forward pass, used for extraction and JVP computation.

## Why this term exists

Forward hooks provide read access to internal layer outputs without modifying the model. They enable extracting resid_post and computing the readout-map JVP for transport.

## Source

extraction/extract.py

## Depends on

- [forward pass](forward-pass.md)

---

[← Back to Glossary](index.md)
