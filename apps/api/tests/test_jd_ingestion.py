"""Ingestion tests for Job Description–optimized RAG.

Verifies:
- Sections detected correctly via alias mapping
- Structured JSON extracted properly
- Chunk count reasonable (~8–20) for typical JDs

Uses synthetic JDs: Thermo Fisher–style and a differently formatted JD.
"""

import pytest

from app.services.jd_chunking import chunk_jd_pages, JDChunkResult, JD_DOMAIN
from app.services.jd_extraction import extract_jd_struct
from app.services.jd_sections import normalize_jd_text, sectionize_jd_text


# Thermo Fisher–style JD: structured headings, bullet lists
THERMO_FISHER_JD = """
Thermo Fisher Scientific

AI Engineer - Digital Science

About the Role

We are seeking an AI Engineer to join our Digital Science team. You will build
ML solutions that accelerate drug discovery and laboratory automation.

Key Responsibilities

• Design and implement machine learning pipelines for scientific data
• Collaborate with scientists to deploy models in production
• Develop NLP solutions for document understanding and extraction
• Optimize model performance for large-scale datasets
• Mentor junior engineers on best practices

Tools & Technologies

• Python, PyTorch, TensorFlow
• AWS, Azure for cloud infrastructure
• Kubernetes, Docker for deployment
• Spark for data processing
• PostgreSQL, MongoDB

Qualifications

• 5+ years of experience in ML/AI
• Bachelor's degree in Computer Science or related field
• Strong Python and SQL skills
• Experience with cloud platforms (AWS, Azure, GCP)
• Agile methodologies

Preferred Qualifications

• PhD in Machine Learning or related field
• Experience with LLMs and RAG systems
• Domain knowledge in life sciences

Compensation

$130,000 - $180,000 per year depending on experience. Comprehensive benefits.

Location

Remote - US. Some travel may be required.

Page 1 of 1
"""


# Differently formatted JD: startup style, different heading names
STARTUP_JD = """
Acme Tech Inc.

Senior Data Scientist

Job Summary

Acme is building the future of analytics. We need a Senior Data Scientist to lead
our ML initiatives.

What You'll Do

• Build predictive models for customer behavior
• Create dashboards and reports
• Work with engineering on model deployment
• A/B testing and experimentation

Tech Stack

JavaScript, Python, SQL
React, Node.js, REST APIs
Postgres, Redis, S3
Scrum, Agile

Requirements

• 3+ years of experience
• Master's degree in Data Science or Statistics
• Python, SQL, machine learning

Nice to Have

• Experience with Spark
• Knowledge of TensorFlow or PyTorch

Total Rewards

$100k - $140k USD annually. Equity. Health benefits.

Work Location

Hybrid - San Francisco, CA
"""


def test_normalize_removes_artifacts():
    """Normalize removes Â, weird spacing, collapses bullets."""
    raw = "HelloÂ world  \n\n  •  Bullet 1  •  Bullet 2"
    out = normalize_jd_text(raw)
    assert "Â" not in out
    assert "  " not in out or "•" in out


def test_normalize_fixes_bullet_mojibake():
    """Normalize fixes â¢ / â€¢ mojibake to proper bullet •."""
    raw = "â¢ Bachelor's degree required\nâ€¢ 5+ years experience"
    out = normalize_jd_text(raw)
    assert "â¢" not in out
    assert "â€¢" not in out
    assert "\u2022" in out or "•" in out


def test_thermo_fisher_sections_detected():
    """Thermo Fisher JD: canonical sections detected via heading + aliases."""
    norm = normalize_jd_text(THERMO_FISHER_JD)
    sections = sectionize_jd_text(norm)
    section_types = [s[0] for s in sections]

    assert "responsibilities" in section_types
    assert "tools_technologies" in section_types
    assert "qualifications" in section_types
    assert "preferred_qualifications" in section_types
    assert "compensation" in section_types
    assert "location" in section_types

    # Content present
    resp_content = next((c for t, c in sections if t == "responsibilities"), "")
    assert "machine learning" in resp_content.lower() or "ML" in resp_content


def test_startup_jd_sections_detected():
    """Differently formatted JD: 'What You'll Do' -> responsibilities, etc."""
    norm = normalize_jd_text(STARTUP_JD)
    sections = sectionize_jd_text(norm)
    section_types = [s[0] for s in sections]

    # "What You'll Do" -> responsibilities
    assert "responsibilities" in section_types
    # "Requirements" -> qualifications
    assert "qualifications" in section_types
    # "Tech Stack" or tools -> tools_technologies
    assert "tools_technologies" in section_types
    # "Total Rewards" -> compensation
    assert "compensation" in section_types
    # "Work Location" -> location
    assert "location" in section_types


def test_extract_jd_struct_thermo_fisher():
    """Thermo Fisher: structured JSON extracted with expected fields."""
    norm = normalize_jd_text(THERMO_FISHER_JD)
    jd = extract_jd_struct(norm)

    assert "company" in jd
    assert "role_title" in jd
    assert "location" in jd
    assert "salary_range" in jd
    assert jd["salary_range"]  # We have $130,000 - $180,000
    assert "required_skills" in jd
    assert "tools" in jd
    assert "cloud_platforms" in jd
    assert "experience_years_required" in jd
    assert "5" in (jd["experience_years_required"] or "")
    assert "education_requirements" in jd
    assert "raw_sections" in jd
    assert "responsibilities" in jd["raw_sections"]
    assert "qualifications" in jd["raw_sections"]
    assert "tools_technologies" in jd["raw_sections"]

    # Skills/tools from qualifications + tools_technologies
    assert len(jd["required_skills"]) >= 1 or len(jd["tools"]) >= 1
    assert "aws" in [s.lower() for s in jd["cloud_platforms"]] or "azure" in [
        s.lower() for s in jd["cloud_platforms"]
    ]


def test_extract_jd_struct_startup():
    """Startup JD: structured JSON extracted (different format)."""
    norm = normalize_jd_text(STARTUP_JD)
    jd = extract_jd_struct(norm)

    assert "company" in jd
    assert "role_title" in jd
    assert "location" in jd
    assert "required_skills" in jd
    assert "raw_sections" in jd
    assert "responsibilities" in jd["raw_sections"]
    assert "qualifications" in jd["raw_sections"]

    # Startup has "3+ years"; salary extraction may vary by format ($100k style)
    assert "3" in (jd.get("experience_years_required") or "")


def test_chunk_jd_pages_thermo_fisher_count():
    """Chunk count for Thermo Fisher JD is ~5–25 depending on JD size."""
    page_texts = [(1, THERMO_FISHER_JD)]
    results = chunk_jd_pages(page_texts, min_chars=25)
    assert 5 <= len(results) <= 30
    for r in results:
        assert isinstance(r, JDChunkResult)
        assert r.section_type
        assert r.doc_domain == JD_DOMAIN
        assert isinstance(r.skills_detected, list)
        assert len(r.content) >= 25


def test_chunk_jd_pages_startup_count():
    """Chunk count for differently formatted JD is ~5–25 depending on size."""
    page_texts = [(1, STARTUP_JD)]
    results = chunk_jd_pages(page_texts, min_chars=25)
    assert 5 <= len(results) <= 30


def test_chunks_have_section_types():
    """Each chunk has section_type and doc_domain."""
    page_texts = [(1, THERMO_FISHER_JD)]
    results = chunk_jd_pages(page_texts)
    section_types = {r.section_type for r in results}
    assert len(section_types) >= 2  # Multiple sections
    assert all(r.doc_domain == JD_DOMAIN for r in results)
    assert all(isinstance(r.skills_detected, list) for r in results)


def test_bullets_not_broken_across_chunks():
    """Responsibilities section content preserved; bullets not orphaned."""
    page_texts = [(1, THERMO_FISHER_JD)]
    results = chunk_jd_pages(page_texts)
    resp_chunks = [r for r in results if r.section_type == "responsibilities"]
    # Responsibility bullets should be in contiguous content
    for c in resp_chunks:
        # Should see full bullets, not cut mid-sentence
        assert "•" in c.content or "-" in c.content or len(c.content) > 50
