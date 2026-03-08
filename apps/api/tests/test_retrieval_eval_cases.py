"""Tests for internal retrieval eval dataset schema and starter cases."""

from pathlib import Path

import pytest

from evals.retrieval import get_builtin_dataset_path, load_eval_dataset
from evals.retrieval.schema import RetrievalEvalDataset


def test_builtin_job_description_dataset_loads():
    """Starter dataset should remain valid and loadable."""
    dataset = load_eval_dataset(get_builtin_dataset_path("job_description_starter"))

    assert dataset.dataset == "job_description_starter"
    assert dataset.version == 1
    assert len(dataset.cases) >= 4
    assert all(case.fixture_ref == "platform_engineer_jd" for case in dataset.cases)


def test_retrieval_eval_case_requires_document_or_fixture():
    """Each eval case must target either a real document or a fixture."""
    with pytest.raises(ValueError, match="document_id or fixture_ref"):
        RetrievalEvalDataset.model_validate(
            {
                "version": 1,
                "dataset": "invalid",
                "cases": [
                    {
                        "id": "missing-target",
                        "query": "What skills are required?",
                        "expected_content_substrings": ["Python"],
                    }
                ],
            }
        )


def test_retrieval_eval_case_requires_expectation():
    """Cases without expected_* fields are not measurable."""
    with pytest.raises(ValueError, match="at least one expected_"):
        RetrievalEvalDataset.model_validate(
            {
                "version": 1,
                "dataset": "invalid",
                "cases": [
                    {
                        "id": "missing-expectation",
                        "fixture_ref": "platform_engineer_jd",
                        "query": "What skills are required?",
                    }
                ],
            }
        )


def test_builtin_dataset_path_points_to_json_file():
    """Builtin dataset mapping should resolve to a checked-in JSON asset."""
    path = get_builtin_dataset_path("job_description_starter")

    assert isinstance(path, Path)
    assert path.name == "job_description_starter.json"
    assert path.exists()


def test_retrieval_eval_dataset_rejects_duplicate_case_ids():
    """Datasets should fail validation when case ids are duplicated."""
    with pytest.raises(ValueError, match="Duplicate eval case ids"):
        RetrievalEvalDataset.model_validate(
            {
                "version": 1,
                "dataset": "invalid",
                "cases": [
                    {
                        "id": "duplicate-id",
                        "fixture_ref": "platform_engineer_jd",
                        "query": "What skills are required?",
                        "expected_content_substrings": ["Python"],
                    },
                    {
                        "id": "duplicate-id",
                        "fixture_ref": "platform_engineer_jd",
                        "query": "What is the salary range?",
                        "expected_content_substrings": ["salary range"],
                    },
                ],
            }
        )
