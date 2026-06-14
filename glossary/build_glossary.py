#!/usr/bin/env python3
"""
Build cross-linked glossary Markdown from glossary.yaml.

Emits:
  - docs/glossary/index.md: grouped listing of all terms
  - docs/glossary/<slug>.md: one page per term with cross-links

Run: uv run python -m glossary.build_glossary
"""

from pathlib import Path
import re
import yaml


GLOSSARY_YAML = Path(__file__).parent / "glossary.yaml"
OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "glossary"


def slugify(name: str) -> str:
    """Convert term name to a URL-safe slug."""
    slug = name.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    return slug


def load_glossary() -> list[dict]:
    """Load and return the list of term records."""
    with open(GLOSSARY_YAML) as f:
        data = yaml.safe_load(f)
    return data["terms"]


def build_index(terms: list[dict]) -> str:
    """Build the index.md content grouped by kind."""
    coined = [t for t in terms if t["kind"] == "coined"]
    inherited = [t for t in terms if t["kind"] == "inherited"]

    lines = [
        "# Glossary",
        "",
        "Cross-linked terminology for the Gemma Holonomy DOE project.",
        "",
        "---",
        "",
        "## Coined Terms",
        "",
        "*Terms introduced by this project.*",
        "",
    ]

    for t in sorted(coined, key=lambda x: x["name"].lower()):
        slug = slugify(t["name"])
        lines.append(f"- [{t['name']}]({slug}.md) — {t['one_line']}")

    lines.extend([
        "",
        "---",
        "",
        "## Inherited Terms",
        "",
        "*Standard terms from prior literature.*",
        "",
    ])

    for t in sorted(inherited, key=lambda x: x["name"].lower()):
        slug = slugify(t["name"])
        lines.append(f"- [{t['name']}]({slug}.md) — {t['one_line']}")

    lines.append("")
    return "\n".join(lines)


def build_term_page(term: dict, all_names: set[str]) -> str:
    """Build a single term's page with cross-links."""
    kind_badge = "**`coined`**" if term["kind"] == "coined" else "**`inherited`**"

    lines = [
        f"# {term['name']}",
        "",
        f"{kind_badge}",
        "",
        f"{term['one_line']}",
        "",
    ]

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

    lines.extend([
        "---",
        "",
        "[← Back to Glossary](index.md)",
        "",
    ])

    return "\n".join(lines)


def main():
    """Build all glossary Markdown files."""
    terms = load_glossary()
    all_names = {t["name"] for t in terms}

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Write index
    index_content = build_index(terms)
    (OUTPUT_DIR / "index.md").write_text(index_content)
    print(f"Wrote {OUTPUT_DIR / 'index.md'}")

    # Write individual term pages
    for term in terms:
        slug = slugify(term["name"])
        page_content = build_term_page(term, all_names)
        page_path = OUTPUT_DIR / f"{slug}.md"
        page_path.write_text(page_content)
        print(f"Wrote {page_path}")

    print(f"\nGenerated {len(terms)} term pages + index in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
