"""Rule-based structured extraction from JD text. No LLM."""

import re
from dataclasses import asdict, dataclass, field

from app.services.jd_sections import sectionize_jd_text
from app.services.jd_sections import normalize_jd_text


@dataclass
class JDExtraction:
    company: str = ""
    role_title: str = ""
    location: str = ""
    salary_range: str | None = None
    required_skills: list[str] = field(default_factory=list)
    preferred_skills: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    cloud_platforms: list[str] = field(default_factory=list)
    experience_years_required: str | None = None
    education_requirements: str | None = None
    raw_sections: dict = field(default_factory=lambda: {
        "responsibilities": [],
        "qualifications": [],
        "tools_technologies": [],
    })

    def to_json(self) -> dict:
        return asdict(self)


# Regex patterns for extraction
SALARY_RE = re.compile(
    r"\$[\d,]+(?:\.\d{2})?\s*[-–—to]+\s*\$[\d,]+(?:\.\d{2})?|"
    r"\$[\d,]+(?:\.\d{2})?\s*(?:per\s+year|/yr|annually)|"
    r"[\d,]+\s*[-–—to]+\s*[\d,]+\s*(?:k|K|USD)",
    re.IGNORECASE,
)
YEAR_RE = re.compile(
    r"(?:[\d]+)\s*\+?\s*years?\s*(?:of\s+)?(?:experience|exp)",
    re.IGNORECASE,
)
EDUCATION_RE = re.compile(
    r"(?:bachelor|b\.?s\.?|master|m\.?s\.?|phd|mba|degree)\s*(?:in\s+)?[\w\s,]+",
    re.IGNORECASE,
)
# Common skills / tools keywords
SKILL_KEYWORDS = frozenset(
    "python java javascript typescript sql spark aws azure gcp "
    "machine learning ml ai llm tensorflow pytorch sklearn "
    "react node api rest api agile scrum".split()
)
TOOL_KEYWORDS = frozenset(
    "jupyter pandas numpy docker kubernetes git jenkins "
    "postgres mysql mongodb redis s3 lambda".split()
)
CLOUD_KEYWORDS = frozenset("aws azure gcp google cloud".split())


def _extract_bullets(text: str) -> list[str]:
    """Extract bullet points from text."""
    bullets = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Remove leading bullet char
        m = re.match(r"^[•\-*]\s*(.+)", line)
        if m:
            bullets.append(m.group(1).strip())
        elif re.match(r"^\d+[.)]\s+", line):
            bullets.append(re.sub(r"^\d+[.)]\s+", "", line).strip())
    return [b for b in bullets if len(b) > 10]


def _extract_skills_from_text(text: str) -> list[str]:
    """Extract skill-like tokens from text."""
    words = re.findall(r"\b[A-Za-z][a-zA-Z0-9.#+]+\b", text.lower())
    seen: set[str] = set()
    result = []
    for w in words:
        if w in SKILL_KEYWORDS or w in TOOL_KEYWORDS:
            if w not in seen:
                seen.add(w)
                result.append(w)
    return result


def _extract_cloud(text: str) -> list[str]:
    """Extract cloud platform mentions."""
    t = text.lower()
    result = []
    for c in CLOUD_KEYWORDS:
        if c in t:
            result.append(c)
    if "gcp" in t or "google cloud" in t:
        result.append("gcp")
    return list(dict.fromkeys(result))  # preserve order, no dupes


def extract_jd_struct(full_text: str) -> dict:
    """
    Extract structured JSON from JD text. Rule-based, no LLM.
    Returns dict matching JDExtraction schema.
    """
    norm_text = normalize_jd_text(full_text)
    sections = sectionize_jd_text(norm_text)
    section_map = {k: v for k, v in sections}

    extraction = JDExtraction()

    # Role title: often first line or in position_summary / about
    lines = norm_text.split("\n")
    for i, ln in enumerate(lines[:15]):
        ln = ln.strip()
        if len(ln) > 10 and len(ln) < 120:
            if not re.match(r"^[•\-*]\s", ln) and "job" in norm_text[:500].lower():
                # Heuristic: first substantial line might be title
                if "engineer" in ln.lower() or "analyst" in ln.lower() or "manager" in ln.lower():
                    extraction.role_title = ln
                    break

    # Company: often at top; look for known patterns
    for ln in lines[:20]:
        ln = ln.strip()
        if 5 < len(ln) < 80 and extraction.company == "":
            if any(x in ln.lower() for x in ["inc", "llc", "corp", "ltd", "scientific"]):
                extraction.company = ln
                break

    # Location
    if "location" in section_map:
        extraction.location = section_map["location"][:200].replace("\n", " ")
    # Fallback: search for city/state patterns
    if not extraction.location:
        loc_m = re.search(r"([A-Za-z\s]+,\s*[A-Za-z]{2})|(remote|hybrid|onsite)", norm_text, re.IGNORECASE)
        if loc_m:
            extraction.location = loc_m.group(0)[:100]

    # Salary
    salary_m = SALARY_RE.search(norm_text)
    if salary_m:
        extraction.salary_range = salary_m.group(0).strip()

    # Experience
    exp_m = YEAR_RE.search(norm_text)
    if exp_m:
        extraction.experience_years_required = exp_m.group(0).strip()

    # Education
    edu_m = EDUCATION_RE.search(norm_text)
    if edu_m:
        extraction.education_requirements = edu_m.group(0).strip()

    # Raw sections + skills from them
    if "responsibilities" in section_map:
        extraction.raw_sections["responsibilities"] = _extract_bullets(section_map["responsibilities"])
    if "qualifications" in section_map:
        extraction.raw_sections["qualifications"] = _extract_bullets(section_map["qualifications"])
        extraction.required_skills = _extract_skills_from_text(section_map["qualifications"])
    if "tools_technologies" in section_map or "qualifications" in section_map:
        combined = section_map.get("tools_technologies", "") + " " + section_map.get("qualifications", "")
        extraction.raw_sections["tools_technologies"] = _extract_bullets(combined)
        extraction.tools = _extract_skills_from_text(combined)
    if "preferred_qualifications" in section_map:
        extraction.preferred_skills = _extract_skills_from_text(section_map["preferred_qualifications"])

    extraction.cloud_platforms = _extract_cloud(norm_text)

    return extraction.to_json()
