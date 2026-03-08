"""Helpers for loading internal retrieval eval datasets."""

from __future__ import annotations

import json
from pathlib import Path

from .schema import RetrievalEvalDataset

CASES_DIR = Path(__file__).resolve().parent / "cases"

BUILTIN_DATASETS: dict[str, str] = {
    "job_description_starter": "job_description_starter.json",
}


def get_builtin_dataset_path(name: str) -> Path:
    """Resolve a built-in dataset name to its JSON file path."""
    try:
        filename = BUILTIN_DATASETS[name]
    except KeyError as exc:
        available = ", ".join(sorted(BUILTIN_DATASETS))
        raise ValueError(f"Unknown builtin dataset '{name}'. Available: {available}") from exc
    return CASES_DIR / filename


def load_eval_dataset(path: str | Path) -> RetrievalEvalDataset:
    """Load and validate a retrieval eval dataset from disk."""
    dataset_path = Path(path)
    with dataset_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return RetrievalEvalDataset.model_validate(raw)


def load_builtin_dataset(name: str) -> RetrievalEvalDataset:
    """Load one of the versioned datasets stored under evals/retrieval/cases."""
    return load_eval_dataset(get_builtin_dataset_path(name))
