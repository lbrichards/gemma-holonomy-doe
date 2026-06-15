#!/usr/bin/env python3
"""
Build cross-linked glossary Markdown from glossary.yaml and topics.yaml.

Emits:
  - docs/glossary/index.md: grouped listing of all terms
  - docs/glossary/<slug>.md: one page per term with cross-links
  - docs/topics/<slug>.md: one page per topic with term links
  - docs/topics/index.md: listing of all topics

Run: uv run python -m glossary.build_glossary
"""

from pathlib import Path
import re
import yaml


GLOSSARY_YAML = Path(__file__).parent / "glossary.yaml"
TOPICS_YAML = Path(__file__).parent / "topics.yaml"
GLOSSARY_OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "glossary"
TOPICS_OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "topics"

# Badge HTML for coined vs inherited (visible styling)
BADGE_COINED = '<span style="background-color: #4051b5; color: white; padding: 0.2em 0.6em; border-radius: 0.25em; font-size: 0.85em; font-weight: 600;">COINED</span>'
BADGE_INHERITED = '<span style="background-color: #448aff; color: white; padding: 0.2em 0.6em; border-radius: 0.25em; font-size: 0.85em; font-weight: 600;">INHERITED</span>'
BADGE_TOPIC = '<span style="background-color: #7c4dff; color: white; padding: 0.2em 0.6em; border-radius: 0.25em; font-size: 0.85em; font-weight: 600;">TOPIC</span>'

# Status badges - superseded/retired are prominent warnings
BADGE_SUPERSEDED = '<span style="background-color: #ff9800; color: black; padding: 0.2em 0.6em; border-radius: 0.25em; font-size: 0.85em; font-weight: 600;">⚠ SUPERSEDED</span>'
BADGE_RETIRED = '<span style="background-color: #f44336; color: white; padding: 0.2em 0.6em; border-radius: 0.25em; font-size: 0.85em; font-weight: 600;">⛔ RETIRED</span>'


def slugify(name: str) -> str:
    """Convert term/topic name to a URL-safe slug."""
    slug = name.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    return slug


def load_glossary() -> list[dict]:
    """Load and return the list of term records."""
    with open(GLOSSARY_YAML) as f:
        data = yaml.safe_load(f)
    return data["terms"]


def load_topics() -> list[dict]:
    """Load and return the list of topic records."""
    if not TOPICS_YAML.exists():
        return []
    with open(TOPICS_YAML) as f:
        data = yaml.safe_load(f)
    return data.get("topics", [])


def build_term_to_topics_map(topics: list[dict]) -> dict[str, list[str]]:
    """Build a mapping from term name to list of topic titles that bundle it."""
    term_to_topics: dict[str, list[str]] = {}
    for topic in topics:
        for term_name in topic.get("bundles", []):
            if term_name not in term_to_topics:
                term_to_topics[term_name] = []
            term_to_topics[term_name].append(topic["title"])
    return term_to_topics


def convert_term_refs_to_links(text: str, all_names: set[str]) -> str:
    """Convert [[term name]] markup to Markdown links."""
    def replace_ref(match):
        term_name = match.group(1)
        if term_name in all_names:
            slug = slugify(term_name)
            return f"[{term_name}](../glossary/{slug}.md)"
        else:
            # Term not found - leave as plain text with warning marker
            return f"{term_name} *(missing)*"

    return re.sub(r"\[\[([^\]]+)\]\]", replace_ref, text)


def build_glossary_index(terms: list[dict], topics: list[dict]) -> str:
    """Build the index.md content grouped by kind."""
    coined = [t for t in terms if t["kind"] == "coined"]
    inherited = [t for t in terms if t["kind"] == "inherited"]

    lines = [
        "# Glossary",
        "",
        "Cross-linked terminology for the Gemma Holonomy DOE project.",
        "",
    ]

    if topics:
        lines.extend([
            f"**[→ View Topics](../topics/index.md)** — {len(topics)} connective articles bundling related terms.",
            "",
        ])

    lines.extend([
        "---",
        "",
        f"## {BADGE_COINED} Coined Terms",
        "",
        "*Terms introduced by this project.*",
        "",
    ])

    for t in sorted(coined, key=lambda x: x["name"].lower()):
        slug = slugify(t["name"])
        lines.append(f"- [{t['name']}]({slug}.md) — {t['one_line']}")

    lines.extend([
        "",
        "---",
        "",
        f"## {BADGE_INHERITED} Inherited Terms",
        "",
        "*Standard terms from prior literature.*",
        "",
    ])

    for t in sorted(inherited, key=lambda x: x["name"].lower()):
        slug = slugify(t["name"])
        lines.append(f"- [{t['name']}]({slug}.md) — {t['one_line']}")

    lines.append("")
    return "\n".join(lines)


def build_term_page(term: dict, all_names: set[str], term_to_topics: dict[str, list[str]], all_terms: list[dict]) -> str:
    """Build a single term's page with cross-links, badge, formula, and topic backlinks."""
    kind_badge = BADGE_COINED if term["kind"] == "coined" else BADGE_INHERITED
    status = term.get("status", "current")

    lines = [
        f"# {term['name']}",
        "",
    ]

    # Show status badge prominently for non-current terms
    if status == "superseded":
        lines.extend([
            BADGE_SUPERSEDED,
            "",
            "> **This term is superseded.** It was valid under v1 but has been replaced. See the replacement below.",
            "",
            kind_badge,
            "",
        ])
    elif status == "retired":
        lines.extend([
            BADGE_RETIRED,
            "",
            "> **This term is retired.** It is not part of the current v2 design and has no replacement.",
            "",
            kind_badge,
            "",
        ])
    else:
        lines.extend([
            kind_badge,
            "",
        ])

    lines.extend([
        f"{term['one_line']}",
        "",
    ])

    # Add formula as block math if present
    if term.get("formula"):
        lines.extend([
            "$$",
            term["formula"],
            "$$",
            "",
        ])

    if term["kind"] == "coined" and term.get("why"):
        lines.extend([
            "## Why this term exists",
            "",
            term["why"].strip(),
            "",
        ])

    lines.extend([
        "## Source",
        "",
        f"{term['source']}",
        "",
    ])

    depends = term.get("depends_on") or []
    if depends:
        lines.extend([
            "## Depends on",
            "",
        ])
        for dep in depends:
            if dep in all_names:
                dep_slug = slugify(dep)
                lines.append(f"- [{dep}]({dep_slug}.md)")
            else:
                # Should not happen if validation passes
                lines.append(f"- {dep} *(missing)*")
        lines.append("")

    # Add topic backlinks
    topic_titles = term_to_topics.get(term["name"], [])
    if topic_titles:
        lines.extend([
            "## Appears in topics",
            "",
        ])
        for title in sorted(topic_titles):
            topic_slug = slugify(title)
            lines.append(f"- [{title}](../topics/{topic_slug}.md)")
        lines.append("")

    lines.extend([
        "---",
        "",
        "[← Back to Glossary](index.md)",
        "",
    ])

    return "\n".join(lines)


def build_topics_index(topics: list[dict]) -> str:
    """Build the topics index page."""
    lines = [
        "# Topics",
        "",
        "Connective articles bundling related glossary terms into explanatory narratives.",
        "",
        "---",
        "",
    ]

    for topic in topics:
        slug = slugify(topic["title"])
        bundle_count = len(topic.get("bundles", []))
        lines.append(f"- **[{topic['title']}]({slug}.md)** — bundles {bundle_count} terms")

    lines.extend([
        "",
        "---",
        "",
        "[← Back to Glossary](../glossary/index.md)",
        "",
    ])

    return "\n".join(lines)


def build_topic_page(topic: dict, all_names: set[str]) -> str:
    """Build a single topic's page with linked narrative."""
    lines = [
        f"# {topic['title']}",
        "",
        BADGE_TOPIC,
        "",
    ]

    # Bundled terms
    lines.extend([
        "## Terms covered",
        "",
    ])
    for term_name in topic.get("bundles", []):
        if term_name in all_names:
            slug = slugify(term_name)
            lines.append(f"- [{term_name}](../glossary/{slug}.md)")
        else:
            lines.append(f"- {term_name} *(missing)*")
    lines.append("")

    # Narrative with term links
    lines.extend([
        "## Narrative",
        "",
    ])
    narrative = topic.get("narrative", "")
    narrative_with_links = convert_term_refs_to_links(narrative, all_names)
    lines.append(narrative_with_links)
    lines.append("")

    # Optional run outcome
    if topic.get("run_outcome"):
        lines.extend([
            "---",
            "",
            f"*Run outcome: {topic['run_outcome']}*",
            "",
        ])

    lines.extend([
        "---",
        "",
        "[← Back to Topics](index.md) · [← Back to Glossary](../glossary/index.md)",
        "",
    ])

    return "\n".join(lines)


def main():
    """Build all glossary and topic Markdown files."""
    terms = load_glossary()
    topics = load_topics()
    all_names = {t["name"] for t in terms}
    term_to_topics = build_term_to_topics_map(topics)

    # Ensure output directories exist
    GLOSSARY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TOPICS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Write glossary index
    index_content = build_glossary_index(terms, topics)
    (GLOSSARY_OUTPUT_DIR / "index.md").write_text(index_content)
    print(f"Wrote {GLOSSARY_OUTPUT_DIR / 'index.md'}")

    # Write individual term pages
    for term in terms:
        slug = slugify(term["name"])
        page_content = build_term_page(term, all_names, term_to_topics, terms)
        page_path = GLOSSARY_OUTPUT_DIR / f"{slug}.md"
        page_path.write_text(page_content)
        print(f"Wrote {page_path}")

    print(f"\nGenerated {len(terms)} term pages + index in {GLOSSARY_OUTPUT_DIR}")

    # Write topics if any exist
    if topics:
        # Write topics index
        topics_index = build_topics_index(topics)
        (TOPICS_OUTPUT_DIR / "index.md").write_text(topics_index)
        print(f"\nWrote {TOPICS_OUTPUT_DIR / 'index.md'}")

        # Write individual topic pages
        for topic in topics:
            slug = slugify(topic["title"])
            page_content = build_topic_page(topic, all_names)
            page_path = TOPICS_OUTPUT_DIR / f"{slug}.md"
            page_path.write_text(page_content)
            print(f"Wrote {page_path}")

        print(f"\nGenerated {len(topics)} topic pages + index in {TOPICS_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
