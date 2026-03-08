"""Comparison helpers for multi-strategy retrieval eval runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.retrieval import RetrievalMode

from .runner import (
    CaseRequestBuilder,
    FixtureResolver,
    RetrievalEvalCaseResult,
    RetrievalEvalRunResult,
    run_eval_dataset_for_modes,
)
from .schema import RetrievalEvalDataset

HybridComparisonStatus = Literal["improved", "regressed", "same", "hybrid_only", "semantic_only"]


class RetrievalModeCaseSnapshot(BaseModel):
    """Compact per-mode view of a single eval case."""

    passed: bool
    score: float
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    recall_at_k: float
    mrr: float
    first_relevant_rank: int | None = None
    error: str | None = None


class RetrievalEvalCaseComparison(BaseModel):
    """Per-case comparison across retrieval strategies."""

    case_id: str
    query: str
    notes: str | None = None
    modes: dict[str, RetrievalModeCaseSnapshot] = Field(default_factory=dict)
    succeeded_modes: list[str] = Field(default_factory=list)
    failed_modes: list[str] = Field(default_factory=list)
    best_modes: list[str] = Field(default_factory=list)
    hybrid_vs_semantic: HybridComparisonStatus | None = None
    hybrid_improved_over_semantic: bool = False


class RetrievalEvalComparisonSummary(BaseModel):
    """Top-level comparison summary across retrieval modes."""

    dataset: str
    compared_modes: list[RetrievalMode]
    summaries_by_mode: dict[str, dict] = Field(default_factory=dict)
    total_cases: int = 0
    hybrid_improved_case_ids: list[str] = Field(default_factory=list)
    hybrid_regressed_case_ids: list[str] = Field(default_factory=list)
    semantic_only_success_case_ids: list[str] = Field(default_factory=list)
    hybrid_only_success_case_ids: list[str] = Field(default_factory=list)


class RetrievalEvalComparisonResult(BaseModel):
    """Developer-friendly comparison result for multiple strategy runs."""

    dataset: str
    compared_modes: list[RetrievalMode]
    summary: RetrievalEvalComparisonSummary
    runs: list[RetrievalEvalRunResult]
    cases: list[RetrievalEvalCaseComparison] = Field(default_factory=list)


def _snapshot_case_result(result: RetrievalEvalCaseResult) -> RetrievalModeCaseSnapshot:
    return RetrievalModeCaseSnapshot(
        passed=result.passed,
        score=result.score,
        hit_at_1=result.metrics.hit_at_1,
        hit_at_3=result.metrics.hit_at_3,
        hit_at_5=result.metrics.hit_at_5,
        recall_at_k=result.metrics.recall_at_k,
        mrr=result.metrics.mrr,
        first_relevant_rank=result.metrics.first_relevant_rank,
        error=result.error,
    )


def _compare_hybrid_vs_semantic(
    *,
    semantic: RetrievalEvalCaseResult | None,
    hybrid: RetrievalEvalCaseResult | None,
) -> tuple[HybridComparisonStatus | None, bool]:
    if semantic is None and hybrid is not None:
        return "hybrid_only", False
    if hybrid is None and semantic is not None:
        return "semantic_only", False
    if semantic is None or hybrid is None:
        return None, False
    if hybrid.passed and not semantic.passed:
        return "improved", True
    if semantic.passed and not hybrid.passed:
        return "regressed", False
    if hybrid.score > semantic.score:
        return "improved", True
    if hybrid.score < semantic.score:
        return "regressed", False
    if hybrid.passed and semantic.passed:
        return "same", False
    return "same", False


def build_comparison_result(runs: list[RetrievalEvalRunResult]) -> RetrievalEvalComparisonResult:
    """Convert per-mode run results into a side-by-side comparison view."""
    if not runs:
        raise ValueError("At least one run result is required to build a comparison")

    dataset = runs[0].dataset
    compared_modes = [run.mode for run in runs]

    all_case_ids: list[str] = []
    seen: set[str] = set()
    for run in runs:
        for result in run.results:
            if result.case_id not in seen:
                seen.add(result.case_id)
                all_case_ids.append(result.case_id)

    cases: list[RetrievalEvalCaseComparison] = []
    hybrid_improved_case_ids: list[str] = []
    hybrid_regressed_case_ids: list[str] = []
    semantic_only_success_case_ids: list[str] = []
    hybrid_only_success_case_ids: list[str] = []

    for case_id in all_case_ids:
        result_by_mode: dict[str, RetrievalEvalCaseResult] = {}
        for run in runs:
            match = next((result for result in run.results if result.case_id == case_id), None)
            if match is not None:
                result_by_mode[run.mode] = match

        modes = {
            mode: _snapshot_case_result(result)
            for mode, result in result_by_mode.items()
        }
        succeeded_modes = [
            mode for mode, result in result_by_mode.items()
            if result.passed
        ]
        failed_modes = [
            mode for mode, result in result_by_mode.items()
            if not result.passed
        ]

        best_score = max((result.score for result in result_by_mode.values()), default=0.0)
        best_modes = sorted(
            [
                mode for mode, result in result_by_mode.items()
                if result.score == best_score
            ]
        )

        semantic_result = result_by_mode.get("semantic")
        hybrid_result = result_by_mode.get("hybrid")
        hybrid_vs_semantic, hybrid_improved = _compare_hybrid_vs_semantic(
            semantic=semantic_result,
            hybrid=hybrid_result,
        )

        if hybrid_vs_semantic == "improved":
            hybrid_improved_case_ids.append(case_id)
        elif hybrid_vs_semantic == "regressed":
            hybrid_regressed_case_ids.append(case_id)

        if semantic_result and semantic_result.passed and hybrid_result and not hybrid_result.passed:
            semantic_only_success_case_ids.append(case_id)
        if hybrid_result and hybrid_result.passed and semantic_result and not semantic_result.passed:
            hybrid_only_success_case_ids.append(case_id)

        exemplar = next(iter(result_by_mode.values()))
        cases.append(
            RetrievalEvalCaseComparison(
                case_id=case_id,
                query=exemplar.query,
                notes=exemplar.notes,
                modes=modes,
                succeeded_modes=sorted(succeeded_modes),
                failed_modes=sorted(failed_modes),
                best_modes=best_modes,
                hybrid_vs_semantic=hybrid_vs_semantic,
                hybrid_improved_over_semantic=hybrid_improved,
            )
        )

    summaries_by_mode = {
        run.mode: {
            "passed_cases": run.passed_cases,
            "failed_cases": run.failed_cases,
            "score": run.score,
            "summary_metrics": run.summary_metrics.model_dump(),
        }
        for run in runs
    }

    summary = RetrievalEvalComparisonSummary(
        dataset=dataset,
        compared_modes=compared_modes,
        summaries_by_mode=summaries_by_mode,
        total_cases=len(all_case_ids),
        hybrid_improved_case_ids=hybrid_improved_case_ids,
        hybrid_regressed_case_ids=hybrid_regressed_case_ids,
        semantic_only_success_case_ids=semantic_only_success_case_ids,
        hybrid_only_success_case_ids=hybrid_only_success_case_ids,
    )

    return RetrievalEvalComparisonResult(
        dataset=dataset,
        compared_modes=compared_modes,
        summary=summary,
        runs=runs,
        cases=cases,
    )


async def compare_eval_dataset(
    db: AsyncSession,
    dataset: RetrievalEvalDataset,
    *,
    modes: tuple[RetrievalMode, ...] = ("semantic", "hybrid", "keyword"),
    fixture_resolver: FixtureResolver | None = None,
    request_builder: CaseRequestBuilder | None = None,
) -> RetrievalEvalComparisonResult:
    """Run a dataset across multiple retrieval modes and compare the results."""
    runs = await run_eval_dataset_for_modes(
        db=db,
        dataset=dataset,
        modes=modes,
        fixture_resolver=fixture_resolver,
        request_builder=request_builder,
    )
    return build_comparison_result(runs)


def format_comparison_summary(result: RetrievalEvalComparisonResult) -> str:
    """Backward-compatible wrapper around the shared comparison formatter."""
    from .report import format_comparison_summary as _format_comparison_summary

    return _format_comparison_summary(result)


def write_comparison_result_json(
    result: RetrievalEvalComparisonResult,
    output_path: str | Path,
) -> Path:
    """Write a comparison result to a JSON file for later inspection."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    return path
