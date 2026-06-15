# Claude Code Project Context

## Glossary System

This project has a cross-linked MkDocs glossary for project terminology. The glossary is **additive documentation tooling** — do not modify experiment code when working on it.

### Key Files

| File | Purpose |
|------|---------|
| `glossary/glossary.yaml` | Term definitions (136 terms) |
| `glossary/topics.yaml` | Topic bundles linking related terms |
| `glossary/build_glossary.py` | Generates Markdown into `docs/glossary/` and `docs/topics/` |
| `tests/test_glossary.py` | Validation tests (13 tests) |
| `mkdocs.yml` | MkDocs config with Material theme + MathJax |
| `docs/` | Generated site content |

### Term Schema

```yaml
- name: term name
  kind: coined | inherited
  status: current | superseded | retired
  one_line: Single sentence definition
  formula: LaTeX (optional, no surrounding $)
  why: 1-3 sentence justification (required for coined; null for inherited)
  source: repo location (coined) or external citation (inherited)
  depends_on:
    - other term name
```

### Commands

```bash
# Validate
uv run pytest tests/test_glossary.py

# Build glossary + site
uv run python -m glossary.build_glossary && uv run mkdocs build

# Preview
uv run mkdocs serve  # http://127.0.0.1:8000/
```

### Constraints (enforced by tests)

- **Anti-leak**: Current terms cannot depend on retired/superseded terms
- **No dangling edges**: Every `depends_on` target must exist
- **No cycles**: Dependency graph must be acyclic
- **Coined terms**: Must have non-empty `why`
- **Inherited terms**: Must have `source` and `why: null`
- **Status required**: Every term needs `status: current|superseded|retired`
- **Formula if present**: Must be non-empty string

### Status Badges

- `current` — Unobtrusive, part of live v2 design
- `superseded` — Orange warning badge, replaced by newer value
- `retired` — Red warning badge, removed from design with no replacement

### Protected Directories

When working on glossary, do NOT touch: `run/`, `geometry/`, `bands/`, `corpus/`, `extraction/`, `planes/`, `covariates/`, `transport/`, `manifest/`, `stage_a/`, `stage_b/`, `pilot/`, `pyproject.toml`, `PREREGISTRATION_v2.md`
