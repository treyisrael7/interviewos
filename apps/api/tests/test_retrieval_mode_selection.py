"""Unit tests for explicit retrieval mode selection."""

import uuid

import pytest

from app.services.retrieval import (
    MAX_RETRIEVAL_CHUNKS,
    _allocate_jd_resume_slots,
    _split_slot_targets,
    retrieve_chunks_for_mode,
)


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


def test_split_slot_targets_eight_total_is_four_four() -> None:
    assert _split_slot_targets(8, has_additional=True) == (4, 4)


def test_allocate_jd_resume_slots_reallocates_when_one_side_is_short() -> None:
    """Sparse resume pool: spare slots fill from remaining JD chunks by score."""
    primary = [{"chunk_id": f"p{i}", "score": 1.0 - i * 0.01} for i in range(7)]
    additional = [{"chunk_id": "a0", "score": 0.5}]
    slot_p, slot_a = _split_slot_targets(8, has_additional=True)
    out = _allocate_jd_resume_slots(
        primary,
        additional,
        max_total=MAX_RETRIEVAL_CHUNKS,
        slot_primary=slot_p,
        slot_additional=slot_a,
    )
    assert len(out) == MAX_RETRIEVAL_CHUNKS
    # 7 JD + 1 resume (only one resume exists; three JD slots were reallocated).
    assert sum(1 for c in out if c["chunk_id"].startswith("p")) == 7
    assert sum(1 for c in out if c["chunk_id"] == "a0") == 1


@pytest.mark.asyncio
async def test_production_budget_invokes_scoped_keyword_for_resume(monkeypatch) -> None:
    """With enforce_production_chunk_budget + additional docs, keyword runs per scope."""
    calls: list[dict] = []

    async def _mock_keyword(**kwargs):
        calls.append({"_scope": kwargs.get("_scope"), "_sql_limit_override": kwargs.get("_sql_limit_override")})
        st = kwargs.get("_scope", "union")
        if st == "primary":
            return [_candidate(chunk_id="jd-kw", score=0.9, retrieval_source="keyword")]
        if st == "additional":
            return [_candidate(chunk_id="cv-kw", score=0.85, retrieval_source="keyword")]
        return []

    async def _mock_semantic(**kwargs):
        st = kwargs.get("_scope", "union")
        if st == "primary":
            return [_candidate(chunk_id="jd-sem", score=0.95)]
        if st == "additional":
            return [_candidate(chunk_id="cv-sem", score=0.9)]
        return []

    monkeypatch.setattr("app.services.retrieval.retrieve_chunks_keyword", _mock_keyword)
    monkeypatch.setattr("app.services.retrieval._retrieve_semantic_candidates", _mock_semantic)

    primary_id = uuid.uuid4()
    resume_id = uuid.uuid4()

    chunks = await retrieve_chunks_for_mode(
        db=None,
        document_id=primary_id,
        query_embedding=[0.1] * 4,
        query_text="python",
        top_k=8,
        mode="hybrid",
        additional_document_ids=[resume_id],
        enforce_production_chunk_budget=True,
    )

    scopes = {c["_scope"] for c in calls}
    assert scopes == {"primary", "additional"}
    assert len(chunks) <= MAX_RETRIEVAL_CHUNKS
    assert {c["chunkId"] for c in chunks} <= {"jd-sem", "jd-kw", "cv-sem", "cv-kw"}
