"""Retrieval evaluation runner built on top of the shared retrieval service."""

from __future__ import annotations

import inspect
import uuid
from collections.abc import Awaitable, Callable, Sequence

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.retrieval import RetrievalMode, embed_query, retrieve_chunks_for_mode

from .metrics import (
    RetrievalEvalAggregateMetrics,
    RetrievalEvalCaseMetrics,
    RetrievalEvalExpectationResult,
    aggregate_case_metrics,
    compute_case_metrics,
    evaluate_case_expectations,
)
from .schema import RetrievalEvalCase, RetrievalEvalDataset

FixtureResolver = Callable[[str], uuid.UUID | Awaitable[uuid.UUID]]


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


class RetrievalEvalRequest(BaseModel):
    """Generic retrieval request used by the eval runner."""

    document_id: uuid.UUID
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=6, ge=1, le=50)
    include_low_signal: bool = False
    section_types: list[str] | None = None
    doc_domain: str | None = None
    source_types: list[str] | None = None
    additional_document_ids: list[uuid.UUID] | None = None


CaseRequestBuilder = Callable[
    [RetrievalEvalCase, uuid.UUID],
    RetrievalEvalRequest | Awaitable[RetrievalEvalRequest],
]


class RetrievalEvalReturnedChunk(BaseModel):
    """Compact chunk summary included in eval results."""

    chunk_id: str
    score: float
    text: str
    section_type: str | None = None
    source_type: str = "jd"
    source_title: str = ""
    retrieval_source: str = "semantic"


class RetrievalEvalExpectedEvidence(BaseModel):
    """Expected evidence definition echoed into run results for debugging."""

    chunk_ids: list[str] = Field(default_factory=list)
    content_substrings: list[str] = Field(default_factory=list)
    section_types: list[str] = Field(default_factory=list)
    source_types: list[str] = Field(default_factory=list)


class RetrievalEvalCaseResult(BaseModel):
    """Outcome for one case under one retrieval mode."""

    case_id: str
    mode: RetrievalMode
    document_id: uuid.UUID
    query: str
    top_k: int
    passed: bool
    score: float
    notes: str | None = None
    expected_evidence: RetrievalEvalExpectedEvidence
    expectations: RetrievalEvalExpectationResult
    metrics: RetrievalEvalCaseMetrics
    returned_chunks: list[RetrievalEvalReturnedChunk] = Field(default_factory=list)
    failure_reasons: list[str] = Field(default_factory=list)
    error: str | None = None


class RetrievalEvalRunResult(BaseModel):
    """Dataset-level summary for one retrieval mode."""

    dataset: str
    mode: RetrievalMode
    total_cases: int
    passed_cases: int
    failed_cases: int
    score: float
    summary_metrics: RetrievalEvalAggregateMetrics
    results: list[RetrievalEvalCaseResult] = Field(default_factory=list)


def build_default_request(case: RetrievalEvalCase, document_id: uuid.UUID) -> RetrievalEvalRequest:
    """Map a stored eval case to the generic retrieval request shape."""
    return RetrievalEvalRequest(
        document_id=document_id,
        query=case.query,
        top_k=case.top_k,
    )


def build_expected_evidence(case: RetrievalEvalCase) -> RetrievalEvalExpectedEvidence:
    """Mirror the case relevance definition into the result payload."""
    return RetrievalEvalExpectedEvidence(
        chunk_ids=list(case.expected_chunk_ids),
        content_substrings=list(case.expected_content_substrings),
        section_types=list(case.expected_section_types),
        source_types=list(case.expected_source_types),
    )


def build_failure_reasons(
    expectations: RetrievalEvalExpectationResult,
    *,
    error: str | None = None,
) -> list[str]:
    """Build concise, actionable reasons for a failed eval case."""
    reasons: list[str] = []
    if error:
        reasons.append(f"retrieval error: {error}")
    if expectations.missing_chunk_ids:
        reasons.append(
            "missing expected chunk ids: " + ", ".join(expectations.missing_chunk_ids)
        )
    if expectations.missing_content_substrings:
        reasons.append(
            "missing expected text: " + ", ".join(expectations.missing_content_substrings)
        )
    if expectations.missing_section_types:
        reasons.append(
            "missing expected section types: " + ", ".join(expectations.missing_section_types)
        )
    if expectations.missing_source_types:
        reasons.append(
            "missing expected source types: " + ", ".join(expectations.missing_source_types)
        )
    return reasons


async def resolve_case_document_id(
    case: RetrievalEvalCase,
    fixture_resolver: FixtureResolver | None = None,
) -> uuid.UUID:
    """Resolve a case target from either a direct document id or a fixture reference."""
    if case.document_id is not None:
        return case.document_id
    if not case.fixture_ref:
        raise ValueError(f"Eval case '{case.id}' is missing document_id and fixture_ref")
    if fixture_resolver is None:
        raise ValueError(
            f"Eval case '{case.id}' uses fixture_ref='{case.fixture_ref}' but no fixture_resolver was provided"
        )
    resolved = await _maybe_await(fixture_resolver(case.fixture_ref))
    if not isinstance(resolved, uuid.UUID):
        raise ValueError(
            f"Fixture resolver returned an invalid document id for '{case.fixture_ref}': {resolved!r}"
        )
    return resolved

async def run_eval_case(
    db: AsyncSession,
    case: RetrievalEvalCase,
    *,
    mode: RetrievalMode,
    fixture_resolver: FixtureResolver | None = None,
    request_builder: CaseRequestBuilder | None = None,
) -> RetrievalEvalCaseResult:
    """Run one retrieval eval case against the shared retrieval pipeline."""
    document_id = await resolve_case_document_id(case, fixture_resolver=fixture_resolver)
    builder = request_builder or build_default_request
    request = await _maybe_await(builder(case, document_id))
    expected_evidence = build_expected_evidence(case)

    query_embedding = None
    if mode in ("semantic", "hybrid"):
        query_embedding = embed_query(request.query)

    try:
        returned_chunks = await retrieve_chunks_for_mode(
            db=db,
            document_id=request.document_id,
            query_embedding=query_embedding,
            top_k=request.top_k,
            mode=mode,
            include_low_signal=request.include_low_signal,
            section_types=request.section_types,
            doc_domain=request.doc_domain,
            source_types=request.source_types,
            additional_document_ids=request.additional_document_ids,
            query_text=request.query,
        )
    except Exception as exc:
        expectations = evaluate_case_expectations(case, [])
        metrics = compute_case_metrics(case, [])
        failure_reasons = build_failure_reasons(expectations, error=str(exc))
        return RetrievalEvalCaseResult(
            case_id=case.id,
            mode=mode,
            document_id=document_id,
            query=request.query,
            top_k=request.top_k,
            passed=False,
            score=0.0,
            notes=case.notes,
            expected_evidence=expected_evidence,
            expectations=expectations,
            metrics=metrics,
            returned_chunks=[],
            failure_reasons=failure_reasons,
            error=str(exc),
        )

    expectations = evaluate_case_expectations(case, returned_chunks)
    metrics = compute_case_metrics(case, returned_chunks)
    passed = expectations.matched_expectations == expectations.total_expectations
    failure_reasons = [] if passed else build_failure_reasons(expectations)
    score = (
        expectations.matched_expectations / expectations.total_expectations
        if expectations.total_expectations > 0 else 0.0
    )

    returned = [
        RetrievalEvalReturnedChunk(
            chunk_id=str(chunk.get("chunkId") or chunk.get("chunk_id") or ""),
            score=float(chunk.get("score") or 0.0),
            text=str(chunk.get("text") or chunk.get("snippet") or ""),
            section_type=chunk.get("section_type"),
            source_type=str(chunk.get("sourceType") or chunk.get("source_type") or "jd"),
            source_title=str(chunk.get("sourceTitle") or chunk.get("source_title") or ""),
            retrieval_source=str(
                chunk.get("retrieval_source") or chunk.get("retrievalSource") or mode
            ),
        )
        for chunk in returned_chunks
    ]

    return RetrievalEvalCaseResult(
        case_id=case.id,
        mode=mode,
        document_id=document_id,
        query=request.query,
        top_k=request.top_k,
        passed=passed,
        score=score,
        notes=case.notes,
        expected_evidence=expected_evidence,
        expectations=expectations,
        metrics=metrics,
        returned_chunks=returned,
        failure_reasons=failure_reasons,
    )


async def run_eval_dataset(
    db: AsyncSession,
    dataset: RetrievalEvalDataset,
    *,
    mode: RetrievalMode,
    fixture_resolver: FixtureResolver | None = None,
    request_builder: CaseRequestBuilder | None = None,
) -> RetrievalEvalRunResult:
    """Run all cases in a dataset for one retrieval mode."""
    results: list[RetrievalEvalCaseResult] = []
    for case in dataset.cases:
        results.append(
            await run_eval_case(
                db=db,
                case=case,
                mode=mode,
                fixture_resolver=fixture_resolver,
                request_builder=request_builder,
            )
        )

    passed_cases = sum(1 for result in results if result.passed)
    total_cases = len(results)
    failed_cases = total_cases - passed_cases
    score = (
        sum(result.score for result in results) / total_cases
        if total_cases > 0 else 0.0
    )
    summary_metrics = aggregate_case_metrics([result.metrics for result in results])

    return RetrievalEvalRunResult(
        dataset=dataset.dataset,
        mode=mode,
        total_cases=total_cases,
        passed_cases=passed_cases,
        failed_cases=failed_cases,
        score=score,
        summary_metrics=summary_metrics,
        results=results,
    )


async def run_eval_dataset_for_modes(
    db: AsyncSession,
    dataset: RetrievalEvalDataset,
    *,
    modes: Sequence[RetrievalMode],
    fixture_resolver: FixtureResolver | None = None,
    request_builder: CaseRequestBuilder | None = None,
) -> list[RetrievalEvalRunResult]:
    """Run the same dataset across multiple retrieval modes."""
    return [
        await run_eval_dataset(
            db=db,
            dataset=dataset,
            mode=mode,
            fixture_resolver=fixture_resolver,
            request_builder=request_builder,
        )
        for mode in modes
    ]
