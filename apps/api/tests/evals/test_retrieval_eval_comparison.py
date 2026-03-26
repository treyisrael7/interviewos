"""Tests for retrieval eval comparison helpers."""

import uuid
from pathlib import Path

from evals.retrieval.comparison import (
    build_comparison_result,
    format_comparison_summary,
    write_comparison_result_json,
)
from evals.retrieval.report import format_failed_case_details, format_single_run_summary
from evals.retrieval.metrics import RetrievalEvalAggregateMetrics, RetrievalEvalCaseMetrics, RetrievalEvalExpectationResult
from evals.retrieval.runner import (
    RetrievalEvalCaseResult,
    RetrievalEvalExpectedEvidence,
    RetrievalEvalReturnedChunk,
    RetrievalEvalRunResult,
)


def _case_result(
    *,
    case_id: str,
    mode: str,
    passed: bool,
    score: float,
    hit_at_1: float,
    mrr: float,
) -> RetrievalEvalCaseResult:
    return RetrievalEvalCaseResult(
        case_id=case_id,
        mode=mode,
        document_id=uuid.uuid4(),
        query=f"query for {case_id}",
        top_k=5,
        passed=passed,
        score=score,
        expected_evidence=RetrievalEvalExpectedEvidence(),
        expectations=RetrievalEvalExpectationResult(total_expectations=1, matched_expectations=int(passed)),
        metrics=RetrievalEvalCaseMetrics(
            hit_at_1=hit_at_1,
            hit_at_3=hit_at_1,
            hit_at_5=hit_at_1,
            recall_k=5,
            recall_at_k=score,
            mrr=mrr,
            first_relevant_rank=1 if passed else None,
            relevant_ranks=[1] if passed else [],
            total_relevance_targets=1,
            matched_relevance_targets=int(passed),
        ),
        returned_chunks=[
            RetrievalEvalReturnedChunk(
                chunk_id="chunk-1",
                score=0.9,
                text="Python and AWS are required.",
                section_type="qualifications",
                source_type="jd",
                source_title="Job Description",
                retrieval_source=mode,
            )
        ],
        failure_reasons=[] if passed else ["missing expected evidence"],
    )


def _run_result(mode: str, case_results: list[RetrievalEvalCaseResult]) -> RetrievalEvalRunResult:
    return RetrievalEvalRunResult(
        dataset="job_description_starter",
        mode=mode,
        total_cases=len(case_results),
        passed_cases=sum(1 for result in case_results if result.passed),
        failed_cases=sum(1 for result in case_results if not result.passed),
        score=(sum(result.score for result in case_results) / len(case_results)) if case_results else 0.0,
        summary_metrics=RetrievalEvalAggregateMetrics(
            hit_at_1=sum(result.metrics.hit_at_1 for result in case_results) / len(case_results),
            hit_at_3=sum(result.metrics.hit_at_3 for result in case_results) / len(case_results),
            hit_at_5=sum(result.metrics.hit_at_5 for result in case_results) / len(case_results),
            recall_at_k=sum(result.metrics.recall_at_k for result in case_results) / len(case_results),
            mrr=sum(result.metrics.mrr for result in case_results) / len(case_results),
            total_cases=len(case_results),
        ),
        results=case_results,
    )


def test_build_comparison_result_highlights_hybrid_improvements():
    """Comparison output should flag cases where hybrid beats semantic."""
    semantic_run = _run_result(
        "semantic",
        [
            _case_result(case_id="case-a", mode="semantic", passed=False, score=0.0, hit_at_1=0.0, mrr=0.0),
            _case_result(case_id="case-b", mode="semantic", passed=True, score=1.0, hit_at_1=1.0, mrr=1.0),
        ],
    )
    hybrid_run = _run_result(
        "hybrid",
        [
            _case_result(case_id="case-a", mode="hybrid", passed=True, score=1.0, hit_at_1=1.0, mrr=1.0),
            _case_result(case_id="case-b", mode="hybrid", passed=True, score=1.0, hit_at_1=1.0, mrr=1.0),
        ],
    )
    keyword_run = _run_result(
        "keyword",
        [
            _case_result(case_id="case-a", mode="keyword", passed=True, score=1.0, hit_at_1=1.0, mrr=1.0),
            _case_result(case_id="case-b", mode="keyword", passed=False, score=0.0, hit_at_1=0.0, mrr=0.0),
        ],
    )

    comparison = build_comparison_result([semantic_run, hybrid_run, keyword_run])

    assert comparison.summary.hybrid_improved_case_ids == ["case-a"]
    assert comparison.summary.hybrid_regressed_case_ids == []
    assert comparison.summary.hybrid_only_success_case_ids == ["case-a"]
    case_a = next(case for case in comparison.cases if case.case_id == "case-a")
    assert case_a.hybrid_improved_over_semantic is True
    assert case_a.hybrid_vs_semantic == "improved"


def test_format_comparison_summary_is_concise_and_side_by_side():
    """Formatted summary should include per-mode metrics and per-case statuses."""
    semantic_run = _run_result(
        "semantic",
        [_case_result(case_id="case-a", mode="semantic", passed=False, score=0.0, hit_at_1=0.0, mrr=0.0)],
    )
    hybrid_run = _run_result(
        "hybrid",
        [_case_result(case_id="case-a", mode="hybrid", passed=True, score=1.0, hit_at_1=1.0, mrr=1.0)],
    )

    comparison = build_comparison_result([semantic_run, hybrid_run])
    summary_text = format_comparison_summary(comparison)

    assert "Dataset: job_description_starter" in summary_text
    assert "- semantic:" in summary_text
    assert "- hybrid:" in summary_text
    assert "improved cases: case-a" in summary_text
    assert "case-a: semantic=fail hybrid=pass" in summary_text


def test_write_comparison_result_json_creates_report_file(tmp_path: Path):
    """Comparison results should be writable as JSON for later inspection."""
    semantic_run = _run_result(
        "semantic",
        [_case_result(case_id="case-a", mode="semantic", passed=True, score=1.0, hit_at_1=1.0, mrr=1.0)],
    )
    comparison = build_comparison_result([semantic_run])

    output_path = write_comparison_result_json(comparison, tmp_path / "comparison.json")

    assert output_path.exists()
    contents = output_path.read_text(encoding="utf-8")
    assert '"dataset": "job_description_starter"' in contents
    assert '"case_id": "case-a"' in contents


def test_format_failed_case_details_shows_query_expectations_and_chunks():
    """Failed case details should include the main debugging context."""
    failed = _case_result(
        case_id="case-a",
        mode="semantic",
        passed=False,
        score=0.0,
        hit_at_1=0.0,
        mrr=0.0,
    )
    failed.failure_reasons = ["missing expected text: salary range"]
    failed.expected_evidence.chunk_ids = ["chunk-99"]
    failed.expected_evidence.content_substrings = ["salary range"]
    failed.expected_evidence.section_types = ["compensation"]
    failed.expected_evidence.source_types = ["jd"]

    detail = format_failed_case_details(failed)

    assert "query: query for case-a" in detail
    assert "expected:" in detail
    assert "why_failed: missing expected text: salary range" in detail
    assert "chunk_id=chunk-1" in detail
    assert "score=0.900000" in detail
    assert "source=jd" in detail


def test_format_single_run_summary_includes_failed_case_details():
    """Single-run text output should expand failed cases for debugging."""
    failed = _case_result(
        case_id="case-a",
        mode="semantic",
        passed=False,
        score=0.0,
        hit_at_1=0.0,
        mrr=0.0,
    )
    failed.failure_reasons = ["missing expected section types: compensation"]
    failed.expected_evidence.section_types = ["compensation"]

    summary = format_single_run_summary(_run_result("semantic", [failed]))

    assert "Failed case details:" in summary
    assert "why_failed: missing expected section types: compensation" in summary
    assert "returned_top_k:" in summary
