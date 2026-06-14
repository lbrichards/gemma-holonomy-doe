# run manifest

<span style="background-color: #4051b5; color: white; padding: 0.2em 0.6em; border-radius: 0.25em; font-size: 0.85em; font-weight: 600;">COINED</span>

The JSON artifact serializing all Stage A decisions (planes, centers, covariates, band) but structurally forbidding response fields.

## Why this term exists

The manifest is the blind handoff between stages. Schema validation rejects any field whose name contains "holonomy", "theta", or "transport", enforcing the blind boundary at the type level.

## Source

manifest/schema.py

## Depends on

- [Stage A](stage-a.md)

---

[← Back to Glossary](index.md)
