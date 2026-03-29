"""Unit tests for analyze-fit cache fingerprints."""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.fit_analysis_cache import (
    DEFAULT_ANALYZE_FIT_QUESTION,
    analyze_fit_query_fingerprint,
    document_chunk_fingerprints,
    normalize_analyze_fit_question,
)


def test_normalize_analyze_fit_question_default_when_empty():
    assert normalize_analyze_fit_question(None) == DEFAULT_ANALYZE_FIT_QUESTION
    assert normalize_analyze_fit_question("") == DEFAULT_ANALYZE_FIT_QUESTION
    assert normalize_analyze_fit_question("   ") == DEFAULT_ANALYZE_FIT_QUESTION


def test_normalize_analyze_fit_question_strips():
    assert normalize_analyze_fit_question("  leadership  ") == "leadership"


def test_analyze_fit_query_fingerprint_stable():
    q = "Focus on Python"
    a = analyze_fit_query_fingerprint(q)
    b = analyze_fit_query_fingerprint(q)
    assert len(a) == 64
    assert a == b
    assert analyze_fit_query_fingerprint("other") != a


@pytest.mark.asyncio
async def test_document_chunk_fingerprints_partitions_by_document():
    jd_id = uuid.uuid4()
    rs_id = uuid.uuid4()
    db = MagicMock()
    mock_result = MagicMock()
    mock_result.all.return_value = [
        (jd_id, 0, "aaa", "jd text"),
        (jd_id, 1, None, "more jd"),
        (rs_id, 0, None, "resume line"),
    ]
    db.execute = AsyncMock(return_value=mock_result)

    jd_fp, rs_fp = await document_chunk_fingerprints(db, jd_id, rs_id)

    assert len(jd_fp) == 64
    assert len(rs_fp) == 64
    assert jd_fp != rs_fp

    db.execute.assert_awaited_once()
