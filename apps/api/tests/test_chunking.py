"""Unit tests for paragraph-based chunking and quality scoring."""

import pytest

from app.services.chunking import (
    chunk_pages,
    normalize_text,
)


def test_normalize_text_removes_artifacts():
    """Normalize removes Â and non-breaking spaces."""
    assert "hello world" in normalize_text("hello\u00a0world")
    assert "Aug 2024" in normalize_text("Aug 2024Â -Â August 2024")
    assert "Â" not in normalize_text("textÂwithÂartifacts")


def test_paragraph_based_chunking():
    """Chunks are built by accumulating paragraphs, not raw char windows."""
    page_texts = [
        (1, "First paragraph with some content.\n\nSecond paragraph here.\n\nThird paragraph."),
    ]
    results = chunk_pages(page_texts, chunk_size=200, overlap_paragraphs=1, min_chars=10)
    assert len(results) >= 1
    for r in results:
        assert "\n\n" in r.content or len(r.content) < 200


def test_quality_score_and_low_signal():
    """Low-signal chunks (contact info, very short) are marked."""
    page_texts = [(1, "me@example.com 555-123-4567 www.site.com")]
    results = chunk_pages(page_texts, chunk_size=500, min_chars=5)
    assert len(results) >= 1
    assert results[0].is_low_signal is True
    assert 0 <= results[0].quality_score <= 1


def test_high_signal_chunk():
    """Substantive content is not marked low-signal."""
    page_texts = [
        (1, "This paragraph discusses machine learning and Python in depth with many unique words."),
    ]
    results = chunk_pages(page_texts, chunk_size=500, min_chars=10)
    assert len(results) >= 1
    assert results[0].is_low_signal is False


def test_chunk_pages_returns_chunk_results():
    """chunk_pages returns ChunkResult with quality_score, is_low_signal, content_hash."""
    page_texts = [(1, "Experienced engineer with machine learning skills.\n\nWorked at Acme.")]
    results = chunk_pages(page_texts, chunk_size=500, min_chars=10)
    assert len(results) >= 1
    r = results[0]
    assert r.page_number == 1
    assert "engineer" in r.content
    assert 0 <= r.quality_score <= 1
    assert isinstance(r.is_low_signal, bool)
    assert r.chunk_index >= 0
    assert len(r.content_hash) == 40  # sha1 hex


def test_repeated_content_marked_low_signal():
    """Short chunks (<=200) repeated 3+ times (headers/footers) are marked low-signal."""
    identical = "Page footer."  # short, <=200
    page_texts = [
        (1, f"Intro text here.\n\n{identical}"),
        (2, f"Middle section.\n\n{identical}"),
        (3, f"End section.\n\n{identical}"),
    ]
    # chunk_size=20 keeps each paragraph intact and separate (avoids oversized split)
    results = chunk_pages(page_texts, chunk_size=20, overlap_paragraphs=0, min_chars=5)
    dupes = [r for r in results if r.content == identical]
    assert len(dupes) >= 3
    assert all(r.is_low_signal for r in dupes)


def test_long_repeated_content_not_marked_low_signal():
    """Long repeated chunks (>200 chars) are NOT marked low-signal due to repetition alone."""
    # Varied words to avoid unique_word_ratio < 0.25; >200 chars
    long_text = (
        "Machine learning and Python development experience at Acme Corp. "
        "Built scalable systems. Led team of five engineers. Shipped products. "
    ) * 2
    page_texts = [(1, long_text), (2, long_text), (3, long_text)]
    results = chunk_pages(page_texts, chunk_size=600, min_chars=10)
    # Chunking strips content; match by normalized text
    normalized = long_text.strip()
    dupes = [r for r in results if len(r.content) > 200 and r.content == normalized]
    assert len(dupes) >= 3
    # length>200 so repetition rule does NOT apply
    assert not any(r.is_low_signal for r in dupes)
