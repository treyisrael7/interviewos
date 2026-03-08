"""Readable text reporting for retrieval eval runs and comparisons."""

from __future__ import annotations

from .comparison import RetrievalEvalComparisonResult
from .runner import RetrievalEvalCaseResult, RetrievalEvalRunResult


def _truncate(text: str, limit: int = 120) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def format_expected_evidence(case_result: RetrievalEvalCaseResult) -> str:
    """Render expected evidence in one short line."""
    parts: list[str] = []
    if case_result.expected_evidence.chunk_ids:
        parts.append("chunk_ids=" + ", ".join(case_result.expected_evidence.chunk_ids))
    if case_result.expected_evidence.content_substrings:
        parts.append(
            "text="
            + ", ".join(f'"{item}"' for item in case_result.expected_evidence.content_substrings)
        )
    if case_result.expected_evidence.section_types:
        parts.append("sections=" + ", ".join(case_result.expected_evidence.section_types))
    if case_result.expected_evidence.source_types:
        parts.append("sources=" + ", ".join(case_result.expected_evidence.source_types))
    return "; ".join(parts) if parts else "(none)"


def format_returned_chunks(case_result: RetrievalEvalCaseResult, limit: int | None = None) -> list[str]:
    """Render returned top-k chunks as concise bullet lines."""
    chunks = case_result.returned_chunks[:limit] if limit is not None else case_result.returned_chunks
    if not chunks:
        return ["(no chunks returned)"]

    lines = []
    for idx, chunk in enumerate(chunks, start=1):
        meta = (
            f"chunk_id={chunk.chunk_id} "
            f"score={chunk.score:.6f} "
            f"section={chunk.section_type or '-'} "
            f"source={chunk.source_type} "
            f"retrieval_source={chunk.retrieval_source}"
        )
        lines.append(f"{idx}. {meta} | {_truncate(chunk.text)}")
    return lines


def format_failed_case_details(case_result: RetrievalEvalCaseResult) -> str:
    """Render a concise but debuggable failure block for one case result."""
    lines = [
        f"- {case_result.case_id} ({case_result.mode})",
        f"  query: {case_result.query}",
        f"  expected: {format_expected_evidence(case_result)}",
        f"  why_failed: {'; '.join(case_result.failure_reasons) or 'case did not meet all expectations'}",
        "  returned_top_k:",
    ]
    for line in format_returned_chunks(case_result, limit=case_result.top_k):
        lines.append(f"    {line}")
    return "\n".join(lines)


def format_single_run_summary(run_result: RetrievalEvalRunResult) -> str:
    """Render a single-mode run with compact failure details."""
    metrics = run_result.summary_metrics
    lines = [
        f"Dataset: {run_result.dataset}",
        f"Mode: {run_result.mode}",
        f"Cases: {run_result.passed_cases}/{run_result.total_cases} passed",
        (
            f"Summary: score={run_result.score:.3f} "
            f"hit@1={metrics.hit_at_1:.3f} "
            f"hit@3={metrics.hit_at_3:.3f} "
            f"hit@5={metrics.hit_at_5:.3f} "
            f"recall@k={metrics.recall_at_k:.3f} "
            f"mrr={metrics.mrr:.3f}"
        ),
        "",
        "Per-case outcomes:",
    ]
    for result in run_result.results:
        status = "pass" if result.passed else "fail"
        lines.append(
            (
                f"- {result.case_id}: {status} "
                f"score={result.score:.3f} "
                f"hit@1={result.metrics.hit_at_1:.1f} "
                f"recall@k={result.metrics.recall_at_k:.3f} "
                f"mrr={result.metrics.mrr:.3f}"
            )
        )

    failed_results = [result for result in run_result.results if not result.passed]
    if failed_results:
        lines.extend(["", "Failed case details:"])
        for result in failed_results:
            lines.append(format_failed_case_details(result))

    return "\n".join(lines)


def format_comparison_summary(result: RetrievalEvalComparisonResult) -> str:
    """Render comparison output with concise failed-case debugging details."""
    lines = [
        f"Dataset: {result.dataset}",
        f"Modes: {', '.join(result.compared_modes)}",
        "",
        "Summary:",
    ]
    for mode in result.compared_modes:
        summary = result.summary.summaries_by_mode[mode]
        metrics = summary["summary_metrics"]
        lines.append(
            (
                f"- {mode}: passed={summary['passed_cases']}/{result.summary.total_cases} "
                f"score={summary['score']:.3f} "
                f"hit@1={metrics['hit_at_1']:.3f} "
                f"hit@3={metrics['hit_at_3']:.3f} "
                f"recall@k={metrics['recall_at_k']:.3f} "
                f"mrr={metrics['mrr']:.3f}"
            )
        )

    lines.extend(
        [
            "",
            "Hybrid vs semantic:",
            f"- improved cases: {', '.join(result.summary.hybrid_improved_case_ids) or '(none)'}",
            f"- regressed cases: {', '.join(result.summary.hybrid_regressed_case_ids) or '(none)'}",
            "",
            "Per-case outcomes:",
        ]
    )
    for case in result.cases:
        statuses = " ".join(
            f"{mode}={'pass' if snapshot.passed else 'fail'}"
            for mode, snapshot in case.modes.items()
        )
        hybrid_note = (
            f" hybrid_vs_semantic={case.hybrid_vs_semantic}"
            if case.hybrid_vs_semantic else ""
        )
        lines.append(f"- {case.case_id}: {statuses}{hybrid_note}")

    failed_case_results = [
        case_result
        for run in result.runs
        for case_result in run.results
        if not case_result.passed
    ]
    if failed_case_results:
        lines.extend(["", "Failed case details:"])
        for case_result in failed_case_results:
            lines.append(format_failed_case_details(case_result))

    return "\n".join(lines)
