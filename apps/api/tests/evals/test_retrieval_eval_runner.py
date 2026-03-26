"""Tests for the internal retrieval eval runner."""

import uuid

import pytest

from evals.retrieval.runner import run_eval_case, run_eval_dataset_for_modes
from evals.retrieval.schema import RetrievalEvalCase, RetrievalEvalDataset


def _mock_returned_chunk(*, retrieval_source: str, section_type: str = "qualifications") -> dict:
    return {
        "chunk_id": "chunk-1",
        "chunkId": "chunk-1",
        "page_number": 1,
        "page": 1,
        "snippet": "Python and AWS are required for this backend platform role.",
        "text": "Python and AWS are required for this backend platform role.",
        "score": 0.95,
        "sourceType": "jd",
        "sourceTitle": "Job Description",
        "section_type": section_type,
        "retrieval_source": retrieval_source,
        "retrievalSource": retrieval_source,
    }


@pytest.mark.asyncio
async def test_run_eval_dataset_for_modes_supports_semantic_keyword_and_hybrid(monkeypatch):
    """The eval runner should execute the same case across all supported retrieval modes."""
    document_id = uuid.uuid4()
    dim = 1536
    mock_vec = [0.1] * dim

    monkeypatch.setattr("evals.retrieval.runner.embed_query", lambda query: mock_vec)

    async def _mock_retrieve_chunks_for_mode(**kwargs):
        mode = kwargs["mode"]
        retrieval_source = "both" if mode == "hybrid" else mode
        return [_mock_returned_chunk(retrieval_source=retrieval_source)]

    monkeypatch.setattr(
        "evals.retrieval.runner.retrieve_chunks_for_mode",
        _mock_retrieve_chunks_for_mode,
    )

    dataset = RetrievalEvalDataset.model_validate(
        {
            "version": 1,
            "dataset": "runner_smoke",
            "cases": [
                {
                    "id": "skills-case",
                    "document_id": str(document_id),
                    "query": "python aws backend",
                    "expected_content_substrings": ["Python and AWS", "backend platform role"],
                    "expected_section_types": ["qualifications"],
                    "expected_source_types": ["jd"],
                    "top_k": 3,
                }
            ],
        }
    )

    runs = await run_eval_dataset_for_modes(
        db=None,
        dataset=dataset,
        modes=["semantic", "keyword", "hybrid"],
    )

    assert [run.mode for run in runs] == ["semantic", "keyword", "hybrid"]
    assert all(run.total_cases == 1 for run in runs)
    assert all(run.passed_cases == 1 for run in runs)
    assert all(run.failed_cases == 0 for run in runs)
    assert all(run.summary_metrics.hit_at_1 == 1.0 for run in runs)
    assert all(run.summary_metrics.mrr == 1.0 for run in runs)
    assert all(run.results[0].passed for run in runs)
    assert all(run.results[0].returned_chunks for run in runs)
    assert all(run.results[0].metrics.hit_at_1 == 1.0 for run in runs)


@pytest.mark.asyncio
async def test_run_eval_case_resolves_fixture_reference(monkeypatch):
    """Fixture-backed eval cases should resolve document ids through the supplied resolver."""
    document_id = uuid.uuid4()
    dim = 1536
    mock_vec = [0.1] * dim

    monkeypatch.setattr("evals.retrieval.runner.embed_query", lambda query: mock_vec)

    async def _mock_retrieve_chunks_for_mode(**kwargs):
        return [_mock_returned_chunk(retrieval_source="semantic", section_type="compensation")]

    monkeypatch.setattr(
        "evals.retrieval.runner.retrieve_chunks_for_mode",
        _mock_retrieve_chunks_for_mode,
    )

    case = RetrievalEvalCase.model_validate(
        {
            "id": "salary-case",
            "fixture_ref": "platform_engineer_jd",
            "query": "What is the salary range?",
            "expected_content_substrings": ["Python and AWS", "backend platform role"],
            "expected_section_types": ["compensation"],
            "expected_source_types": ["jd"],
            "top_k": 3,
        }
    )

    async def _resolve_fixture(name: str) -> uuid.UUID:
        assert name == "platform_engineer_jd"
        return document_id

    result = await run_eval_case(
        db=None,
        case=case,
        mode="semantic",
        fixture_resolver=_resolve_fixture,
    )

    assert result.passed is True
    assert result.document_id == document_id
    assert result.expectations.missing_content_substrings == []
    assert result.expectations.missing_section_types == []
    assert result.expectations.missing_source_types == []
    assert result.metrics.hit_at_1 == 1.0
    assert result.metrics.section_type_match_rate == 1.0
    assert result.metrics.source_type_match_rate == 1.0


@pytest.mark.asyncio
async def test_run_eval_case_includes_failure_reasons_and_expected_evidence(monkeypatch):
    """Failed cases should carry structured debugging context."""
    document_id = uuid.uuid4()
    dim = 1536
    mock_vec = [0.1] * dim

    monkeypatch.setattr("evals.retrieval.runner.embed_query", lambda query: mock_vec)

    async def _mock_retrieve_chunks_for_mode(**kwargs):
        return [_mock_returned_chunk(retrieval_source="semantic", section_type="about")]

    monkeypatch.setattr(
        "evals.retrieval.runner.retrieve_chunks_for_mode",
        _mock_retrieve_chunks_for_mode,
    )

    case = RetrievalEvalCase.model_validate(
        {
            "id": "failed-case",
            "document_id": str(document_id),
            "query": "What is the salary range?",
            "expected_chunk_ids": ["missing-chunk"],
            "expected_content_substrings": ["$160,000"],
            "expected_section_types": ["compensation"],
            "expected_source_types": ["jd"],
            "top_k": 3,
        }
    )

    result = await run_eval_case(
        db=None,
        case=case,
        mode="semantic",
    )

    assert result.passed is False
    assert result.expected_evidence.chunk_ids == ["missing-chunk"]
    assert result.expected_evidence.content_substrings == ["$160,000"]
    assert any("missing expected chunk ids" in reason for reason in result.failure_reasons)
    assert any("missing expected text" in reason for reason in result.failure_reasons)
    assert any("missing expected section types" in reason for reason in result.failure_reasons)


@pytest.mark.asyncio
async def test_run_eval_case_surfaces_retrieval_errors_as_failed_results(monkeypatch):
    """Runner should convert retrieval exceptions into debuggable failed case results."""
    document_id = uuid.uuid4()
    dim = 1536
    mock_vec = [0.1] * dim

    monkeypatch.setattr("evals.retrieval.runner.embed_query", lambda query: mock_vec)

    async def _failing_retrieve_chunks_for_mode(**kwargs):
        raise RuntimeError("fts unavailable")

    monkeypatch.setattr(
        "evals.retrieval.runner.retrieve_chunks_for_mode",
        _failing_retrieve_chunks_for_mode,
    )

    case = RetrievalEvalCase.model_validate(
        {
            "id": "error-case",
            "document_id": str(document_id),
            "query": "What is the salary range?",
            "expected_content_substrings": ["salary range"],
            "expected_section_types": ["compensation"],
            "top_k": 3,
        }
    )

    result = await run_eval_case(
        db=None,
        case=case,
        mode="semantic",
    )

    assert result.passed is False
    assert result.error == "fts unavailable"
    assert result.returned_chunks == []
    assert any("retrieval error: fts unavailable" in reason for reason in result.failure_reasons)
