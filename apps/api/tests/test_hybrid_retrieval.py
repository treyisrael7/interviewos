"""Tests for shared hybrid retrieval behavior."""

import uuid

import pytest

from app.core.config import settings


@pytest.mark.asyncio
async def test_retrieve_chunks_hybrid_marks_both_sources(demo_key_off, monkeypatch, seed_document_bundle):
    """Hybrid retrieval merges semantic and keyword hits for the same chunk."""
    from app.services.retrieval import retrieve_chunks

    dim = 1536
    mock_vec = [0.1] * dim

    monkeypatch.setattr(settings, "hybrid_retrieval_enabled", True)

    seeded = await seed_document_bundle(
        user_email="hybrid-both@t.local",
        filename="jd.pdf",
        doc_domain="job_description",
        sources=[{
            "key": "jd",
            "source_type": "jd",
            "title": "Job Description",
            "original_file_name": "jd.pdf",
        }],
        chunks=[{
            "source_key": "jd",
            "content": "Python and AWS are required for this backend role.",
            "page_number": 1,
            "section_type": "qualifications",
            "doc_domain": "job_description",
            "content_hash": "same-content",
            "embedding": mock_vec,
        }],
    )
    doc_id = seeded["document_id"]
    chunk_id = seeded["chunk_ids"][0]

    async def _mock_keyword(**kwargs):
        return [{
            "chunk_id": str(chunk_id),
            "chunkId": str(chunk_id),
            "page_number": 1,
            "page": 1,
            "snippet": "Python and AWS are required for this backend role.",
            "text": "Python and AWS are required for this backend role.",
            "score": 0.4,
            "is_low_signal": False,
            "section_type": "qualifications",
            "sourceType": "jd",
            "sourceTitle": "Job Description",
            "content_hash": "same-content",
            "embedding": mock_vec,
        }]

    monkeypatch.setattr("app.services.retrieval.retrieve_chunks_keyword", _mock_keyword)

    from app.db.base import async_session_maker
    async with async_session_maker() as db:
        chunks = await retrieve_chunks(
            db=db,
            document_id=doc_id,
            query_embedding=mock_vec,
            query_text="python aws backend",
            top_k=3,
            include_low_signal=False,
            doc_domain="job_description",
        )

    assert len(chunks) == 1
    assert chunks[0]["chunkId"] == str(chunk_id)
    assert chunks[0]["retrieval_source"] == "both"
    assert chunks[0]["retrievalSource"] == "both"
    assert chunks[0]["semantic_score"] is not None
    assert chunks[0]["keyword_score"] is not None
    assert chunks[0]["final_score"] == chunks[0]["score"]


@pytest.mark.asyncio
async def test_retrieve_chunks_hybrid_falls_back_to_semantic_on_keyword_error(demo_key_off, monkeypatch, seed_document_bundle):
    """Hybrid retrieval preserves current behavior if keyword retrieval is unavailable."""
    from app.services.retrieval import retrieve_chunks

    dim = 1536
    mock_vec = [0.1] * dim

    monkeypatch.setattr(settings, "hybrid_retrieval_enabled", True)

    seeded = await seed_document_bundle(
        user_email="hybrid-fallback@t.local",
        filename="jd.pdf",
        doc_domain="job_description",
        sources=[{
            "key": "jd",
            "source_type": "jd",
            "title": "Job Description",
            "original_file_name": "jd.pdf",
        }],
        chunks=[{
            "source_key": "jd",
            "content": "Python experience is required.",
            "page_number": 1,
            "section_type": "qualifications",
            "doc_domain": "job_description",
            "embedding": mock_vec,
        }],
    )
    doc_id = seeded["document_id"]

    async def _failing_keyword(**kwargs):
        raise RuntimeError("fts unavailable")

    monkeypatch.setattr("app.services.retrieval.retrieve_chunks_keyword", _failing_keyword)

    from app.db.base import async_session_maker
    async with async_session_maker() as db:
        chunks = await retrieve_chunks(
            db=db,
            document_id=doc_id,
            query_embedding=mock_vec,
            query_text="python",
            top_k=3,
            include_low_signal=False,
            doc_domain="job_description",
        )

    assert len(chunks) == 1
    assert chunks[0]["retrieval_source"] == "semantic"
    assert chunks[0]["semantic_score"] == chunks[0]["score"]
    assert chunks[0]["keyword_score"] is None
    assert chunks[0]["final_score"] == chunks[0]["score"]


@pytest.mark.asyncio
async def test_retrieve_chunks_semantic_only_when_hybrid_disabled(demo_key_off, monkeypatch, seed_document_bundle):
    """Disabling hybrid retrieval preserves the original semantic-only path."""
    from app.services.retrieval import retrieve_chunks

    dim = 1536
    mock_vec = [0.1] * dim

    monkeypatch.setattr(settings, "hybrid_retrieval_enabled", False)

    async def _unexpected_keyword(**kwargs):
        raise AssertionError("keyword retrieval should not run when hybrid is disabled")

    monkeypatch.setattr("app.services.retrieval.retrieve_chunks_keyword", _unexpected_keyword)

    seeded = await seed_document_bundle(
        user_email="semantic-only@t.local",
        filename="jd.pdf",
        doc_domain="job_description",
        sources=[{
            "key": "jd",
            "source_type": "jd",
            "title": "Job Description",
            "original_file_name": "jd.pdf",
        }],
        chunks=[{
            "source_key": "jd",
            "content": "Python and AWS are required for this backend role.",
            "page_number": 1,
            "section_type": "qualifications",
            "doc_domain": "job_description",
            "embedding": mock_vec,
        }],
    )
    doc_id = seeded["document_id"]

    from app.db.base import async_session_maker
    async with async_session_maker() as db:
        chunks = await retrieve_chunks(
            db=db,
            document_id=doc_id,
            query_embedding=mock_vec,
            query_text="python aws backend",
            top_k=3,
            include_low_signal=False,
            doc_domain="job_description",
        )

    assert len(chunks) == 1
    assert chunks[0]["retrieval_source"] == "semantic"
    assert chunks[0]["semantic_score"] == chunks[0]["score"]
    assert chunks[0]["keyword_score"] is None


@pytest.mark.asyncio
async def test_retrieve_chunks_hybrid_can_surface_keyword_only_hits(demo_key_off, monkeypatch):
    """Hybrid retrieval can return exact keyword matches even when semantic candidates are empty."""
    from app.services.retrieval import retrieve_chunks

    dim = 1536
    mock_vec = [0.1] * dim
    doc_id = uuid.uuid4()
    keyword_chunk_id = uuid.uuid4()

    monkeypatch.setattr(settings, "hybrid_retrieval_enabled", True)

    async def _mock_semantic(**kwargs):
        return []

    async def _mock_keyword(**kwargs):
        return [{
            "chunk_id": str(keyword_chunk_id),
            "chunkId": str(keyword_chunk_id),
            "page_number": 2,
            "page": 2,
            "snippet": "Experience with pgvector in PostgreSQL and Node.js services is required.",
            "text": "Experience with pgvector in PostgreSQL and Node.js services is required.",
            "score": 0.8,
            "is_low_signal": False,
            "section_type": "qualifications",
            "sourceType": "jd",
            "sourceTitle": "Platform Engineer JD",
        }]

    monkeypatch.setattr("app.services.retrieval._retrieve_semantic_candidates", _mock_semantic)
    monkeypatch.setattr("app.services.retrieval.retrieve_chunks_keyword", _mock_keyword)

    chunks = await retrieve_chunks(
        db=None,
        document_id=doc_id,
        query_embedding=mock_vec,
        query_text="pgvector node.js postgres exact match",
        top_k=3,
        include_low_signal=False,
        doc_domain="job_description",
    )

    assert len(chunks) == 1
    assert chunks[0]["chunkId"] == str(keyword_chunk_id)
    assert chunks[0]["retrieval_source"] == "keyword"
    assert chunks[0]["semantic_score"] is None
    assert chunks[0]["keyword_score"] == 0.8
    assert chunks[0]["score"] == 0.9
    assert chunks[0]["final_score"] == chunks[0]["score"]
    assert "pgvector" in chunks[0]["text"].lower()
