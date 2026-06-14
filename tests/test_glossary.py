"""
Validation tests for the glossary graph.

Run: uv run pytest tests/test_glossary.py
"""

from pathlib import Path
import yaml


GLOSSARY_YAML = Path(__file__).parent.parent / "glossary" / "glossary.yaml"


def load_terms() -> list[dict]:
    """Load glossary terms from YAML."""
    with open(GLOSSARY_YAML) as f:
        data = yaml.safe_load(f)
    return data["terms"]


def test_no_dangling_edges():
    """Every depends_on target must exist as a term name."""
    terms = load_terms()
    all_names = {t["name"] for t in terms}

    dangling = []
    for term in terms:
        for dep in term.get("depends_on") or []:
            if dep not in all_names:
                dangling.append((term["name"], dep))

    assert not dangling, f"Dangling dependency edges: {dangling}"


def test_no_cycles():
    """The dependency graph must be acyclic (DAG)."""
    terms = load_terms()
    name_to_deps = {t["name"]: set(t.get("depends_on") or []) for t in terms}

    # Kahn's algorithm for cycle detection
    in_degree = {name: 0 for name in name_to_deps}
    for deps in name_to_deps.values():
        for dep in deps:
            if dep in in_degree:
                in_degree[dep] += 1

    queue = [name for name, deg in in_degree.items() if deg == 0]
    visited = 0

    while queue:
        node = queue.pop()
        visited += 1
        for dep in name_to_deps.get(node, []):
            if dep in in_degree:
                in_degree[dep] -= 1
                if in_degree[dep] == 0:
                    queue.append(dep)

    has_cycle = visited != len(name_to_deps)
    assert not has_cycle, "Dependency graph contains a cycle"


def test_coined_terms_have_why():
    """Every coined term must have a non-empty 'why' field."""
    terms = load_terms()
    coined = [t for t in terms if t["kind"] == "coined"]

    missing_why = [t["name"] for t in coined if not t.get("why")]
    assert not missing_why, f"Coined terms missing 'why': {missing_why}"


def test_inherited_terms_have_source_and_null_why():
    """Every inherited term must have a non-empty 'source' and 'why' must be null."""
    terms = load_terms()
    inherited = [t for t in terms if t["kind"] == "inherited"]

    missing_source = [t["name"] for t in inherited if not t.get("source")]
    non_null_why = [t["name"] for t in inherited if t.get("why") is not None]

    assert not missing_source, f"Inherited terms missing 'source': {missing_source}"
    assert not non_null_why, f"Inherited terms with non-null 'why': {non_null_why}"


def test_names_are_unique():
    """All term names must be unique."""
    terms = load_terms()
    names = [t["name"] for t in terms]
    duplicates = [name for name in names if names.count(name) > 1]

    assert not duplicates, f"Duplicate term names: {set(duplicates)}"


def test_term_count():
    """Verify we have a reasonable number of terms with both kinds."""
    terms = load_terms()
    coined = [t for t in terms if t["kind"] == "coined"]
    inherited = [t for t in terms if t["kind"] == "inherited"]

    # Minimum threshold to ensure glossary has meaningful content
    assert len(terms) >= 20, f"Expected at least 20 terms, got {len(terms)}"
    assert len(coined) >= 10, f"Expected at least 10 coined terms, got {len(coined)}"
    assert len(inherited) >= 10, f"Expected at least 10 inherited terms, got {len(inherited)}"


def test_formula_is_optional_but_nonempty_if_present():
    """Formula field is optional; if present, must be a non-empty string."""
    terms = load_terms()

    invalid_formulas = []
    for term in terms:
        if "formula" in term:
            formula = term["formula"]
            if not isinstance(formula, str) or not formula.strip():
                invalid_formulas.append(term["name"])

    assert not invalid_formulas, f"Terms with invalid formula (must be non-empty string): {invalid_formulas}"
