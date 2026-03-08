"""Tests for retrieval eval metric computation."""

from evals.retrieval.metrics import aggregate_case_metrics, compute_case_metrics, evaluate_case_expectations
from evals.retrieval.schema import RetrievalEvalCase


def _chunk(
    chunk_id: str,
    text: str,
    *,
    section_type: str | None = None,
    source_type: str = "jd",
) -> dict:
    return {
        "chunk_id": chunk_id,
        "chunkId": chunk_id,
        "text": text,
        "snippet": text,
        "section_type": section_type,
        "sourceType": source_type,
    }


def test_compute_case_metrics_supports_mixed_relevance_definitions():
    """Metrics should combine exact, substring, and metadata-based relevance."""
    case = RetrievalEvalCase.model_validate(
        {
            "id": "mixed-relevance",
            "fixture_ref": "platform_engineer_jd",
            "query": "python aws salary",
            "expected_chunk_ids": ["chunk-3"],
            "expected_content_substrings": ["salary range"],
            "expected_section_types": ["compensation"],
            "expected_source_types": ["jd"],
            "top_k": 5,
        }
    )
    returned = [
        _chunk("chunk-1", "This job description is for a platform engineer.", section_type="about"),
        _chunk("chunk-2", "The salary range is $160,000 to $190,000.", section_type="compensation"),
        _chunk("chunk-3", "Python and AWS are required.", section_type="qualifications"),
    ]

    expectations = evaluate_case_expectations(case, returned)
    metrics = compute_case_metrics(case, returned)

    assert expectations.matched_chunk_ids == ["chunk-3"]
    assert expectations.matched_content_substrings == ["salary range"]
    assert expectations.matched_section_types == ["compensation"]
    assert expectations.matched_source_types == ["jd"]
    assert metrics.hit_at_1 == 1.0
    assert metrics.hit_at_3 == 1.0
    assert metrics.hit_at_5 == 1.0
    assert metrics.recall_at_k == 1.0
    assert metrics.mrr == 1.0
    assert metrics.first_relevant_rank == 1
    assert metrics.section_type_match_rate == 1.0
    assert metrics.source_type_match_rate == 1.0


def test_compute_case_metrics_reports_partial_recall_and_rank():
    """Recall and MRR should reflect partial success and the first relevant rank."""
    case = RetrievalEvalCase.model_validate(
        {
            "id": "partial-relevance",
            "fixture_ref": "platform_engineer_jd",
            "query": "what tools are used",
            "expected_content_substrings": ["Kubernetes", "Terraform"],
            "expected_section_types": ["tools"],
            "top_k": 5,
        }
    )
    returned = [
        _chunk("chunk-1", "Responsibilities include collaborating with platform teams.", section_type="responsibilities"),
        _chunk("chunk-2", "We use Kubernetes for orchestration.", section_type="tools"),
    ]

    metrics = compute_case_metrics(case, returned)

    assert metrics.hit_at_1 == 0.0
    assert metrics.hit_at_3 == 1.0
    assert metrics.hit_at_5 == 1.0
    assert metrics.first_relevant_rank == 2
    assert metrics.recall_at_k == 2 / 3
    assert metrics.mrr == 0.5
    assert metrics.section_type_match_rate == 1.0


def test_aggregate_case_metrics_returns_macro_summary():
    """Aggregate metrics should average per-case results and skip absent metadata rates."""
    case_a = RetrievalEvalCase.model_validate(
        {
            "id": "case-a",
            "fixture_ref": "platform_engineer_jd",
            "query": "salary",
            "expected_content_substrings": ["salary range"],
            "expected_section_types": ["compensation"],
            "top_k": 5,
        }
    )
    case_b = RetrievalEvalCase.model_validate(
        {
            "id": "case-b",
            "fixture_ref": "platform_engineer_jd",
            "query": "responsibilities",
            "expected_content_substrings": ["collaborate"],
            "top_k": 5,
        }
    )

    metrics_a = compute_case_metrics(
        case_a,
        [_chunk("chunk-1", "The salary range is competitive.", section_type="compensation")],
    )
    metrics_b = compute_case_metrics(
        case_b,
        [_chunk("chunk-2", "General company information.", section_type="about")],
    )

    aggregate = aggregate_case_metrics([metrics_a, metrics_b])

    assert aggregate.total_cases == 2
    assert aggregate.hit_at_1 == 0.5
    assert aggregate.hit_at_3 == 0.5
    assert aggregate.hit_at_5 == 0.5
    assert aggregate.recall_at_k == 0.5
    assert aggregate.mrr == 0.5
    assert aggregate.section_type_match_rate == 1.0
    assert aggregate.cases_with_section_type_expectations == 1
