"""Tests for line-based paragraph fallback (single-block PDFs)."""

import pytest

from app.services.chunking import chunk_pages, _paragraphs


def test_line_fallback_produces_more_paragraphs():
    """Single-block text (no blank lines) triggers line-based fallback and produces multiple paragraphs."""
    # Dense text: no \n\n, single newlines only; >100 chars to trigger fallback
    text = """Contact
203 Morell Dr.
8649784905 (Home)
trey@example.com
Education
Clemson University
BS Computer Science, 2024
Experience
WebstaurantStore
Software Engineer Intern
May 2024 - August 2024
TD SYNNEX
Capstone Program AI Engineer
August 2025 - December 2025"""
    paras = _paragraphs(text)
    assert len(paras) >= 4, f"Expected >=4 paragraphs for single-block text, got {len(paras)}"


def test_chunking_produces_many_chunks_for_single_block():
    """A 2-page PDF with single-block extraction should produce many chunks."""
    # Long enough to trigger line fallback (>100 chars per page)
    page1 = """Contact
203 Morell Dr.
trey@example.com
www.linkedin.com/in/trey
Summary
Hi, I'm Trey. Recent grad from Clemson. Software engineer with experience in full stack and AI."""
    page2 = """Experience
WebstaurantStore
Software Engineer Intern
May 2024 - Aug 2024
TD SYNNEX
Capstone AI Engineer
Aug 2025 - Dec 2025
Education
Clemson University
BS Computer Science"""
    results = chunk_pages(
        [(1, page1), (2, page2)],
        chunk_size=120,  # Small enough to split into multiple chunks
        overlap_paragraphs=1,
        min_chars=25,
    )
    assert len(results) >= 4, f"Expected >=4 chunks for 2-page doc, got {len(results)}"
    pages = {r.page_number for r in results}
    assert 1 in pages and 2 in pages
