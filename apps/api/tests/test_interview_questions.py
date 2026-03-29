"""Tests for interview_questions.generate_interview_questions."""

import json
from unittest.mock import MagicMock, patch

import pytest

from app.services.interview_questions import generate_interview_questions


def test_generate_interview_questions_empty_inputs_no_llm(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    called = []

    def _boom(**kwargs):
        called.append(1)
        raise AssertionError("OpenAI should not run")

    with patch("app.services.interview_questions.OpenAI") as m:
        m.return_value.chat.completions.create.side_effect = _boom
        out = generate_interview_questions([], [], [])
    assert not called
    assert out == {"questions": []}


def test_generate_interview_questions_parses_structured_response(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "openai_api_key", "sk-test")

    payload = {
        "questions": [
            {"type": "behavioral", "question": "Describe a time you aligned stakeholders on a delayed release."},
            {"type": "technical", "question": "How would you debug elevated p95 latency on a Python API?"},
            {"type": "gap", "question": "The role expects AWS; what cloud or infra work have you owned, and how would you close gaps on AWS?"},
        ]
    }
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content=json.dumps(payload)))]

    with patch("app.services.interview_questions.OpenAI") as m:
        m.return_value.chat.completions.create.return_value = mock_resp
        out = generate_interview_questions(
            [{"snippet": "We use AWS Lambda and Python."}],
            [{"text": "Backend engineer, Docker, on-prem deployments."}],
            [{"requirement": "AWS experience", "reason": "Not on resume", "importance": "high"}],
            max_questions=10,
        )

    assert len(out["questions"]) == 3
    types = {q["type"] for q in out["questions"]}
    assert types == {"behavioral", "technical", "gap"}
    assert "AWS" in out["questions"][2]["question"]


def test_requires_api_key(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "openai_api_key", None)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        generate_interview_questions([{"snippet": "x"}], [], [])
