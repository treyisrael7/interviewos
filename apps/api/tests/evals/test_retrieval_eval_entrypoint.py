"""Tests for the retrieval eval CLI entry point helpers."""

import argparse
import json
import uuid

import pytest

from evals.retrieval.entrypoint import (
    _filter_dataset,
    _load_fixture_map,
    _validate_fixture_requirements,
)
from evals.retrieval.schema import RetrievalEvalDataset


def _dataset() -> RetrievalEvalDataset:
    return RetrievalEvalDataset.model_validate(
        {
            "version": 1,
            "dataset": "entrypoint-test",
            "cases": [
                {
                    "id": "fixture-case",
                    "fixture_ref": "platform_engineer_jd",
                    "query": "salary",
                    "expected_content_substrings": ["salary range"],
                },
                {
                    "id": "doc-case",
                    "document_id": "11111111-1111-1111-1111-111111111111",
                    "query": "skills",
                    "expected_content_substrings": ["Python"],
                },
            ],
        }
    )


def test_filter_dataset_can_select_by_fixture_ref():
    """CLI filters should narrow the dataset to matching cases."""
    args = argparse.Namespace(
        document_id=None,
        fixture_ref="platform_engineer_jd",
        case_id=[],
    )

    filtered = _filter_dataset(_dataset(), args)

    assert len(filtered.cases) == 1
    assert filtered.cases[0].id == "fixture-case"


def test_filter_dataset_fails_clearly_when_no_cases_match():
    """CLI should explain when the requested filters exclude all cases."""
    args = argparse.Namespace(
        document_id=None,
        fixture_ref="missing-fixture",
        case_id=[],
    )

    with pytest.raises(ValueError, match="No eval cases matched"):
        _filter_dataset(_dataset(), args)


def test_validate_fixture_requirements_lists_missing_refs():
    """Missing fixture mappings should raise an actionable error."""
    with pytest.raises(ValueError, match="platform_engineer_jd"):
        _validate_fixture_requirements(_dataset(), {})


def test_load_fixture_map_validates_json_and_uuid(tmp_path):
    """Fixture map loader should validate both file shape and UUID values."""
    path = tmp_path / "fixture-map.json"
    path.write_text(
        json.dumps({"platform_engineer_jd": str(uuid.uuid4())}),
        encoding="utf-8",
    )

    mapping = _load_fixture_map(str(path))

    assert "platform_engineer_jd" in mapping
    assert isinstance(mapping["platform_engineer_jd"], uuid.UUID)
