"""Metric computation for retrieval eval cases and run summaries."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from .schema import RetrievalEvalCase


class RetrievalEvalExpectationResult(BaseModel):
    """Per-case expectation matching summary."""

    matched_chunk_ids: list[str] = Field(default_factory=list)
    missing_chunk_ids: list[str] = Field(default_factory=list)
    matched_content_substrings: list[str] = Field(default_factory=list)
    missing_content_substrings: list[str] = Field(default_factory=list)
    matched_section_types: list[str] = Field(default_factory=list)
    missing_section_types: list[str] = Field(default_factory=list)
    matched_source_types: list[str] = Field(default_factory=list)
    missing_source_types: list[str] = Field(default_factory=list)
    total_expectations: int = 0
    matched_expectations: int = 0


class RetrievalEvalCaseMetrics(BaseModel):
    """Rank-based and metadata-based metrics for one eval case."""

    hit_at_1: float = 0.0
    hit_at_3: float = 0.0
    hit_at_5: float = 0.0
    recall_k: int = 0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    section_type_match_rate: float | None = None
    source_type_match_rate: float | None = None
    first_relevant_rank: int | None = None
    relevant_ranks: list[int] = Field(default_factory=list)
    total_relevance_targets: int = 0
    matched_relevance_targets: int = 0


class RetrievalEvalAggregateMetrics(BaseModel):
    """Macro summary metrics for one dataset/mode run."""

    hit_at_1: float = 0.0
    hit_at_3: float = 0.0
    hit_at_5: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    section_type_match_rate: float | None = None
    source_type_match_rate: float | None = None
    total_cases: int = 0
    cases_with_section_type_expectations: int = 0
    cases_with_source_type_expectations: int = 0


def _first_match_ranks(case: RetrievalEvalCase, returned_chunks: list[dict]) -> dict[str, dict[str, int]]:
    """Return first-match ranks for each expectation item in the case."""
    chunk_id_ranks: dict[str, int] = {}
    substring_ranks: dict[str, int] = {}
    section_type_ranks: dict[str, int] = {}
    source_type_ranks: dict[str, int] = {}

    expected_chunk_ids = set(case.expected_chunk_ids)
    expected_substrings = list(case.expected_content_substrings)
    expected_section_types = set(case.expected_section_types)
    expected_source_types = set(case.expected_source_types)

    for rank, chunk in enumerate(returned_chunks, start=1):
        chunk_id = str(chunk.get("chunkId") or chunk.get("chunk_id") or "")
        text = str(chunk.get("text") or chunk.get("snippet") or "")
        section_type = chunk.get("section_type")
        source_type = chunk.get("sourceType") or chunk.get("source_type")

        if chunk_id in expected_chunk_ids and chunk_id not in chunk_id_ranks:
            chunk_id_ranks[chunk_id] = rank

        lowered_text = text.lower()
        for substring in expected_substrings:
            if substring not in substring_ranks and substring.lower() in lowered_text:
                substring_ranks[substring] = rank

        if section_type in expected_section_types and section_type not in section_type_ranks:
            section_type_ranks[str(section_type)] = rank

        if source_type in expected_source_types and source_type not in source_type_ranks:
            source_type_ranks[str(source_type)] = rank

    return {
        "chunk_ids": chunk_id_ranks,
        "content_substrings": substring_ranks,
        "section_types": section_type_ranks,
        "source_types": source_type_ranks,
    }


def evaluate_case_expectations(
    case: RetrievalEvalCase,
    returned_chunks: list[dict],
) -> RetrievalEvalExpectationResult:
    """Compare expected evidence against returned retrieval results."""
    ranks = _first_match_ranks(case, returned_chunks)

    matched_chunk_ids = [item for item in case.expected_chunk_ids if item in ranks["chunk_ids"]]
    missing_chunk_ids = [item for item in case.expected_chunk_ids if item not in ranks["chunk_ids"]]

    matched_content_substrings = [
        item for item in case.expected_content_substrings if item in ranks["content_substrings"]
    ]
    missing_content_substrings = [
        item for item in case.expected_content_substrings if item not in ranks["content_substrings"]
    ]

    matched_section_types = [
        item for item in case.expected_section_types if item in ranks["section_types"]
    ]
    missing_section_types = [
        item for item in case.expected_section_types if item not in ranks["section_types"]
    ]

    matched_source_types = [
        item for item in case.expected_source_types if item in ranks["source_types"]
    ]
    missing_source_types = [
        item for item in case.expected_source_types if item not in ranks["source_types"]
    ]

    total_expectations = (
        len(case.expected_chunk_ids)
        + len(case.expected_content_substrings)
        + len(case.expected_section_types)
        + len(case.expected_source_types)
    )
    matched_expectations = (
        len(matched_chunk_ids)
        + len(matched_content_substrings)
        + len(matched_section_types)
        + len(matched_source_types)
    )

    return RetrievalEvalExpectationResult(
        matched_chunk_ids=matched_chunk_ids,
        missing_chunk_ids=missing_chunk_ids,
        matched_content_substrings=matched_content_substrings,
        missing_content_substrings=missing_content_substrings,
        matched_section_types=matched_section_types,
        missing_section_types=missing_section_types,
        matched_source_types=matched_source_types,
        missing_source_types=missing_source_types,
        total_expectations=total_expectations,
        matched_expectations=matched_expectations,
    )


def compute_case_metrics(
    case: RetrievalEvalCase,
    returned_chunks: list[dict],
) -> RetrievalEvalCaseMetrics:
    """Compute first-pass retrieval metrics for one eval case."""
    ranks = _first_match_ranks(case, returned_chunks)

    all_ranks = (
        list(ranks["chunk_ids"].values())
        + list(ranks["content_substrings"].values())
        + list(ranks["section_types"].values())
        + list(ranks["source_types"].values())
    )
    all_ranks.sort()

    total_relevance_targets = (
        len(case.expected_chunk_ids)
        + len(case.expected_content_substrings)
        + len(case.expected_section_types)
        + len(case.expected_source_types)
    )
    matched_relevance_targets = len(all_ranks)
    first_relevant_rank = all_ranks[0] if all_ranks else None

    recall_k = case.top_k
    recall_at_k = (
        matched_relevance_targets / total_relevance_targets
        if total_relevance_targets > 0 else 0.0
    )
    mrr = 1.0 / first_relevant_rank if first_relevant_rank else 0.0

    section_type_match_rate = None
    if case.expected_section_types:
        section_type_match_rate = (
            len(ranks["section_types"]) / len(case.expected_section_types)
        )

    source_type_match_rate = None
    if case.expected_source_types:
        source_type_match_rate = (
            len(ranks["source_types"]) / len(case.expected_source_types)
        )

    return RetrievalEvalCaseMetrics(
        hit_at_1=1.0 if first_relevant_rank is not None and first_relevant_rank <= 1 else 0.0,
        hit_at_3=1.0 if first_relevant_rank is not None and first_relevant_rank <= 3 else 0.0,
        hit_at_5=1.0 if first_relevant_rank is not None and first_relevant_rank <= 5 else 0.0,
        recall_k=recall_k,
        recall_at_k=recall_at_k,
        mrr=mrr,
        section_type_match_rate=section_type_match_rate,
        source_type_match_rate=source_type_match_rate,
        first_relevant_rank=first_relevant_rank,
        relevant_ranks=all_ranks,
        total_relevance_targets=total_relevance_targets,
        matched_relevance_targets=matched_relevance_targets,
    )


def aggregate_case_metrics(
    case_metrics: Sequence[RetrievalEvalCaseMetrics],
) -> RetrievalEvalAggregateMetrics:
    """Aggregate per-case metrics into a dataset-level summary."""
    total_cases = len(case_metrics)
    if total_cases == 0:
        return RetrievalEvalAggregateMetrics(total_cases=0)

    section_rates = [
        metric.section_type_match_rate
        for metric in case_metrics
        if metric.section_type_match_rate is not None
    ]
    source_rates = [
        metric.source_type_match_rate
        for metric in case_metrics
        if metric.source_type_match_rate is not None
    ]

    return RetrievalEvalAggregateMetrics(
        hit_at_1=sum(metric.hit_at_1 for metric in case_metrics) / total_cases,
        hit_at_3=sum(metric.hit_at_3 for metric in case_metrics) / total_cases,
        hit_at_5=sum(metric.hit_at_5 for metric in case_metrics) / total_cases,
        recall_at_k=sum(metric.recall_at_k for metric in case_metrics) / total_cases,
        mrr=sum(metric.mrr for metric in case_metrics) / total_cases,
        section_type_match_rate=(
            sum(section_rates) / len(section_rates) if section_rates else None
        ),
        source_type_match_rate=(
            sum(source_rates) / len(source_rates) if source_rates else None
        ),
        total_cases=total_cases,
        cases_with_section_type_expectations=len(section_rates),
        cases_with_source_type_expectations=len(source_rates),
    )
