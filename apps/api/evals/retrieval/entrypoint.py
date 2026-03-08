"""Developer-friendly entry point for retrieval eval runs and comparisons."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path

from app.db.base import async_session_maker

from .comparison import (
    compare_eval_dataset,
    write_comparison_result_json,
)
from .loader import load_builtin_dataset, load_eval_dataset
from .report import format_comparison_summary, format_single_run_summary
from .runner import run_eval_dataset
from .schema import RetrievalEvalDataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PrepPilot retrieval evals in single-mode or comparison mode."
    )
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument(
        "--dataset",
        help="Built-in dataset name, e.g. job_description_starter",
    )
    dataset_group.add_argument(
        "--dataset-path",
        help="Path to a retrieval eval dataset JSON file",
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--mode",
        choices=("semantic", "hybrid", "keyword"),
        help="Run a single retrieval mode",
    )
    mode_group.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple retrieval modes side by side",
    )

    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("semantic", "hybrid", "keyword"),
        default=["semantic", "hybrid", "keyword"],
        help="Modes to compare when --compare is used",
    )
    parser.add_argument(
        "--fixture-map",
        help="Optional JSON file mapping fixture_ref -> document UUID",
    )
    parser.add_argument(
        "--document-id",
        help="Only run cases targeting this concrete document UUID",
    )
    parser.add_argument(
        "--fixture-ref",
        help="Only run cases using this fixture_ref",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Only run the specified case id(s). Repeat the flag to select multiple cases.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to save the JSON result",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Skip printing the console summary",
    )
    return parser.parse_args()


def _load_dataset(args: argparse.Namespace) -> RetrievalEvalDataset:
    return (
        load_builtin_dataset(args.dataset)
        if args.dataset
        else load_eval_dataset(args.dataset_path)
    )


def _load_fixture_map(path: str | None) -> dict[str, uuid.UUID]:
    if not path:
        return {}
    file_path = Path(path)
    if not file_path.exists():
        raise ValueError(f"Fixture map file not found: {file_path}")

    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Fixture map is not valid JSON: {file_path}") from exc

    if not isinstance(raw, dict):
        raise ValueError("Fixture map must be a JSON object of {fixture_ref: document_uuid}")

    try:
        return {str(key): uuid.UUID(str(value)) for key, value in raw.items()}
    except ValueError as exc:
        raise ValueError(
            "Fixture map values must be valid document UUID strings"
        ) from exc


def _fixture_resolver_from_map(mapping: dict[str, uuid.UUID]):
    async def _resolve(name: str) -> uuid.UUID:
        try:
            return mapping[name]
        except KeyError as exc:
            available = ", ".join(sorted(mapping))
            raise ValueError(
                f"Missing fixture_ref mapping for '{name}'. Available mappings: {available or '(none)'}"
            ) from exc

    return _resolve


def _filter_dataset(dataset: RetrievalEvalDataset, args: argparse.Namespace) -> RetrievalEvalDataset:
    document_id = uuid.UUID(args.document_id) if args.document_id else None
    case_ids = set(args.case_id or [])

    filtered_cases = []
    for case in dataset.cases:
        if document_id and case.document_id != document_id:
            continue
        if args.fixture_ref and case.fixture_ref != args.fixture_ref:
            continue
        if case_ids and case.id not in case_ids:
            continue
        filtered_cases.append(case)

    if not filtered_cases:
        filters = []
        if document_id:
            filters.append(f"document_id={document_id}")
        if args.fixture_ref:
            filters.append(f"fixture_ref={args.fixture_ref}")
        if case_ids:
            filters.append(f"case_ids={', '.join(sorted(case_ids))}")
        filter_text = " and ".join(filters) if filters else "the requested filters"
        raise ValueError(f"No eval cases matched {filter_text}")

    return RetrievalEvalDataset(
        version=dataset.version,
        dataset=dataset.dataset,
        description=dataset.description,
        cases=filtered_cases,
    )


def _validate_fixture_requirements(
    dataset: RetrievalEvalDataset,
    fixture_map: dict[str, uuid.UUID],
) -> None:
    needed_fixture_refs = sorted({case.fixture_ref for case in dataset.cases if case.fixture_ref})
    if not needed_fixture_refs:
        return
    if not fixture_map:
        refs = ", ".join(needed_fixture_refs)
        raise ValueError(
            "This dataset uses fixture_ref values but no --fixture-map was provided. "
            f"Required fixture refs: {refs}"
        )

    missing = [ref for ref in needed_fixture_refs if ref not in fixture_map]
    if missing:
        raise ValueError(
            "Fixture map is missing required fixture refs: "
            + ", ".join(missing)
        )

async def _main() -> int:
    args = _parse_args()
    dataset = _filter_dataset(_load_dataset(args), args)
    fixture_map = _load_fixture_map(args.fixture_map)
    _validate_fixture_requirements(dataset, fixture_map)
    fixture_resolver = _fixture_resolver_from_map(fixture_map) if fixture_map else None

    async with async_session_maker() as db:
        if args.compare:
            result = await compare_eval_dataset(
                db=db,
                dataset=dataset,
                modes=tuple(args.modes),
                fixture_resolver=fixture_resolver,
            )
            if not args.quiet:
                print(format_comparison_summary(result))
            if args.output_json:
                output_path = write_comparison_result_json(result, args.output_json)
                if not args.quiet:
                    print(f"\nWrote JSON report to {output_path}")
            return 0

        mode = args.mode or "hybrid"
        result = await run_eval_dataset(
            db=db,
            dataset=dataset,
            mode=mode,
            fixture_resolver=fixture_resolver,
        )
        if not args.quiet:
            print(format_single_run_summary(result))
        if args.output_json:
            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(result.model_dump(mode="json"), indent=2),
                encoding="utf-8",
            )
            if not args.quiet:
                print(f"\nWrote JSON report to {output_path}")
        return 0


def main() -> int:
    try:
        return asyncio.run(_main())
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
