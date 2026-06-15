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


# =============================================================================
# Status field validation tests
# =============================================================================

VALID_STATUSES = {"current", "superseded", "retired"}


def test_every_term_has_valid_status():
    """Every term must have a status field with a valid value."""
    terms = load_terms()

    missing_status = []
    invalid_status = []

    for term in terms:
        if "status" not in term:
            missing_status.append(term["name"])
        elif term["status"] not in VALID_STATUSES:
            invalid_status.append((term["name"], term["status"]))

    assert not missing_status, f"Terms missing 'status' field: {missing_status}"
    assert not invalid_status, f"Terms with invalid status (must be current/superseded/retired): {invalid_status}"


def test_no_current_term_depends_on_dead_term():
    """Anti-leak: no current term may depend on a retired or superseded term.

    A live concept defined in terms of a dead one is a structural error.
    This prevents obsolete concepts from seeding further evolution.
    """
    terms = load_terms()
    name_to_status = {t["name"]: t.get("status", "current") for t in terms}

    violations = []
    for term in terms:
        if term.get("status") == "current":
            for dep in term.get("depends_on") or []:
                dep_status = name_to_status.get(dep, "current")
                if dep_status in ("retired", "superseded"):
                    violations.append((term["name"], dep, dep_status))

    assert not violations, (
        f"Anti-leak violation: current terms depending on retired/superseded terms: "
        f"{[(v[0], v[1], v[2]) for v in violations]}"
    )


# =============================================================================
# Topic validation tests
# =============================================================================

TOPICS_YAML = Path(__file__).parent.parent / "glossary" / "topics.yaml"


def load_topics() -> list[dict]:
    """Load topics from YAML."""
    if not TOPICS_YAML.exists():
        return []
    with open(TOPICS_YAML) as f:
        data = yaml.safe_load(f)
    return data.get("topics", [])


def test_topic_bundles_reference_existing_terms():
    """Every term name in a topic's bundles must exist in glossary.yaml."""
    terms = load_terms()
    topics = load_topics()
    all_term_names = {t["name"] for t in terms}

    dangling = []
    for topic in topics:
        for term_name in topic.get("bundles", []):
            if term_name not in all_term_names:
                dangling.append((topic["title"], term_name))

    assert not dangling, f"Topic bundles reference non-existent terms: {dangling}"


def test_topic_narrative_refs_resolve():
    """Every [[term name]] in a topic narrative must resolve to an existing term."""
    import re

    terms = load_terms()
    topics = load_topics()
    all_term_names = {t["name"] for t in terms}

    unresolved = []
    for topic in topics:
        narrative = topic.get("narrative", "")
        refs = re.findall(r"\[\[([^\]]+)\]\]", narrative)
        for ref in refs:
            if ref not in all_term_names:
                unresolved.append((topic["title"], ref))

    assert not unresolved, f"Topic narratives reference non-existent terms: {unresolved}"


def test_topic_titles_unique():
    """All topic titles must be unique."""
    topics = load_topics()
    titles = [t["title"] for t in topics]
    duplicates = [title for title in titles if titles.count(title) > 1]

    assert not duplicates, f"Duplicate topic titles: {set(duplicates)}"


def test_topics_have_required_fields():
    """Every topic must have title, bundles, and narrative."""
    topics = load_topics()

    missing_fields = []
    for topic in topics:
        title = topic.get("title", "<untitled>")
        if not topic.get("title"):
            missing_fields.append((title, "title"))
        if not topic.get("bundles"):
            missing_fields.append((title, "bundles"))
        if not topic.get("narrative"):
            missing_fields.append((title, "narrative"))

    assert not missing_fields, f"Topics missing required fields: {missing_fields}"
