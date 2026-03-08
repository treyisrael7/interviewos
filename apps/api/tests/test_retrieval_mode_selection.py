"""Unit tests for explicit retrieval mode selection."""

import uuid

import pytest

from app.services.retrieval import retrieve_chunks_for_mode


def _candidate(*, chunk_id: str, score: float, retrieval_source: str = "semantic") -> dict:
    return {
        "chunk_id": chunk_id,
        "chunkId": chunk_id,
        "page_number": 1,
        "page": 1,
        "snippet": "Python and AWS are required for this backend platform role.",
        "text": "Python and AWS are required for this backend platform role.",
        "score": score,
        "is_low_signal": False,
        "section_type": "qualifications",
        "sourceType": "jd",
        "sourceTitle": "Job Description",
        "content_hash": f"hash-{chunk_id}",
        "embedding": [0.1] * 4,
        "retrieval_source": retrieval_source,
        "retrievalSource": retrieval_source,
    }


@pytest.mark.asyncio
async def test_retrieve_chunks_for_mode_semantic_returns_semantic_results(monkeypatch):
    """Semantic mode should use the semantic candidate path only."""
    async def _mock_semantic(**kwargs):
        return [_candidate(chunk_id="semantic-1", score=0.95)]

    monkeypatch.setattr("app.services.retrieval._retrieve_semantic_candidates", _mock_semantic)

    chunks = await retrieve_chunks_for_mode(
        db=None,
        document_id=uuid.uuid4(),
        query_embedding=[0.1] * 4,
        query_text="python aws backend",
        top_k=3,
        mode="semantic",
    )

    assert len(chunks) == 1
    assert chunks[0]["chunkId"] == "semantic-1"
    assert chunks[0]["retrieval_source"] == "semantic"
    assert chunks[0]["semantic_score"] == chunks[0]["score"]


@pytest.mark.asyncio
async def test_retrieve_chunks_for_mode_keyword_returns_keyword_results(monkeypatch):
    """Keyword mode should use the keyword retrieval path directly."""
    async def _mock_keyword(**kwargs):
        return [_candidate(chunk_id="keyword-1", score=0.75, retrieval_source="keyword")]

    monkeypatch.setattr("app.services.retrieval.retrieve_chunks_keyword", _mock_keyword)

    chunks = await retrieve_chunks_for_mode(
        db=None,
        document_id=uuid.uuid4(),
        query_embedding=None,
        query_text="python aws backend",
        top_k=3,
        mode="keyword",
    )

    assert len(chunks) == 1
    assert chunks[0]["chunkId"] == "keyword-1"
    assert chunks[0]["retrieval_source"] == "keyword"
    assert chunks[0]["keyword_score"] == chunks[0]["score"]


@pytest.mark.asyncio
async def test_retrieve_chunks_for_mode_hybrid_merges_semantic_and_keyword(monkeypatch):
    """Hybrid mode should merge semantic and keyword hits through the shared merge path."""
    async def _mock_semantic(**kwargs):
        return [_candidate(chunk_id="shared-1", score=0.95, retrieval_source="semantic")]

    async def _mock_keyword(**kwargs):
        candidate = _candidate(chunk_id="shared-1", score=0.80, retrieval_source="keyword")
        candidate["content_hash"] = "same-content"
        return [candidate]

    monkeypatch.setattr("app.services.retrieval._retrieve_semantic_candidates", _mock_semantic)
    monkeypatch.setattr("app.services.retrieval.retrieve_chunks_keyword", _mock_keyword)

    chunks = await retrieve_chunks_for_mode(
        db=None,
        document_id=uuid.uuid4(),
        query_embedding=[0.1] * 4,
        query_text="python aws backend",
        top_k=3,
        mode="hybrid",
    )

    assert len(chunks) == 1
    assert chunks[0]["chunkId"] == "shared-1"
    assert chunks[0]["retrieval_source"] == "both"
    assert chunks[0]["semantic_score"] is not None
    assert chunks[0]["keyword_score"] is not None
