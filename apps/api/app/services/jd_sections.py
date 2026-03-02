"""JD-aware section detection. Canonical section map + alias mapping."""

import re

# Canonical JD section keys (order hints for extraction)
CANONICAL_SECTIONS = [
    "about",
    "position_summary",
    "responsibilities",
    "tools_technologies",
    "qualifications",
    "preferred_qualifications",
    "compensation",
    "location",
    "company_info",
]

# Alias mapping: normalized heading text -> canonical section
# Lowercase, stripped; we match first word or common phrases
JD_SECTION_ALIASES: dict[str, str] = {
    "about": "about",
    "about the role": "about",
    "about the position": "about",
    "about us": "company_info",
    "company": "company_info",
    "company info": "company_info",
    "company information": "company_info",
    "who we are": "company_info",
    "compensation": "compensation",
    "salary": "compensation",
    "salary range": "compensation",
    "pay": "compensation",
    "benefits": "compensation",
    "total rewards": "compensation",
    "location": "location",
    "work location": "location",
    "remote": "location",
    "full-time": "location",
    "part-time": "location",
    "hybrid": "location",
    "position_summary": "position_summary",
    "position summary": "position_summary",
    "job summary": "position_summary",
    "role summary": "position_summary",
    "summary": "position_summary",
    "overview": "position_summary",
    "responsibilities": "responsibilities",
    "key responsibilities": "responsibilities",
    "job responsibilities": "responsibilities",
    "duties": "responsibilities",
    "what you'll do": "responsibilities",
    "tools": "tools_technologies",
    "technologies": "tools_technologies",
    "tools & technologies": "tools_technologies",
    "tools and technologies": "tools_technologies",
    "tech stack": "tools_technologies",
    "required skills": "qualifications",
    "qualifications": "qualifications",
    "requirements": "qualifications",
    "must have": "qualifications",
    "minimum qualifications": "qualifications",
    "basic qualifications": "qualifications",
    "preferred": "preferred_qualifications",
    "preferred qualifications": "preferred_qualifications",
    "nice to have": "preferred_qualifications",
    "pluses": "preferred_qualifications",
}


def _normalize_heading(text: str) -> str:
    """Normalize heading for alias lookup."""
    t = text.strip().lower()
    t = re.sub(r"[^\w\s&]", "", t)  # remove punctuation except &
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _match_section_heading(line: str) -> str | None:
    """Return canonical section if line matches a JD heading, else None."""
    norm = _normalize_heading(line)
    if not norm or len(norm) > 60:
        return None
    # Exact match
    if norm in JD_SECTION_ALIASES:
        return JD_SECTION_ALIASES[norm]
    # First word / prefix match: require short line to avoid treating
    # content (e.g. "Remote - US. Some travel required.", "Hybrid - San Francisco, CA") as heading
    stripped_len = len(line.strip())
    # Allow longer lines for "Salary Range:" / "Salary:" style headings
    if norm.startswith("salary") and stripped_len < 60:
        return "compensation"
    if stripped_len > 35:
        return None
    # "remote"/"hybrid" as first word often appear in location content, not as headings
    words = norm.split()
    first = words[0] if words else ""
    if first in ("remote", "hybrid") and stripped_len > 15:
        return None
    for alias, canonical in JD_SECTION_ALIASES.items():
        if alias == first or alias.startswith(first + " "):
            return canonical
        if len(alias) <= len(norm) and norm.startswith(alias):
            return canonical
    return None


def sectionize_jd_text(text: str) -> list[tuple[str, str]]:
    """
    Split JD text into (section_type, content) tuples.
    Uses heading detection + alias mapping; falls back to keyword grouping.
    Returns list of (canonical_section, section_content).
    """
    lines = text.split("\n")
    sections: list[tuple[str, str]] = []
    current_section = "about"  # default for content before first heading
    current_content: list[str] = []

    for ln in lines:
        stripped = ln.strip()
        if not stripped:
            if current_content:
                content = "\n".join(current_content).strip()
                if content:
                    sections.append((current_section, content))
                current_content = []
            continue

        matched = _match_section_heading(stripped)
        if matched:
            if current_content:
                content = "\n".join(current_content).strip()
                if content:
                    sections.append((current_section, content))
                current_content = []
            current_section = matched
            # Heading line: if short, treat as header only; if long, might be header + first line
            if len(stripped) < 50:
                current_content = []  # header only
            else:
                current_content = [stripped]
        else:
            current_content.append(stripped)

    if current_content:
        content = "\n".join(current_content).strip()
        if content:
            sections.append((current_section, content))

    return sections


def normalize_jd_text(text: str) -> str:
    """Normalize JD text: artifacts, bullets, repeated headers/footers."""
    if not text:
        return ""
    t = text.replace("\u00a0", " ")
    t = t.replace("Â", "")
    # Fix UTF-8 bullet mojibake (• decoded wrong): â¢, â€¢, ΓÇó -> •
    t = t.replace("\u00e2\u20ac\u00a2", "\u2022")  # â€¢
    t = t.replace("\u00e2\u00a2", "\u2022")  # â¢
    t = t.replace("\u0393\u00c7\u00f3", "\u2022")  # ΓÇó (U+0393 U+00C7 U+00F3)
    t = re.sub(r"â¢|â€¢|ΓÇó", "\u2022", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    # Normalize bullets
    t = re.sub(r"^\s*[•\-*]\s+", "• ", t, flags=re.MULTILINE)
    t = re.sub(r"^\s*[\u2022\u2023]\s+", "• ", t, flags=re.MULTILINE)
    # Collapse repeated "Page X of Y" / "Confidential" footer lines
    t = re.sub(r"(?m)^\s*Page\s+\d+\s+of\s+\d+\s*$", "", t)
    t = re.sub(r"(?m)^\s*\d+\s*$", "", t)  # lone page numbers
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()
