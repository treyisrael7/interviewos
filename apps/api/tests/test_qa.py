"""Unit tests for generate_grounded_answer (structured JSON, temperature)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from app.services.qa import QA_TEMPERATURE, generate_grounded_answer


def test_generate_grounded_answer_empty_chunks():
    answer, cites = generate_grounded_answer("q?", [])
    assert "don't have enough information" in answer.lower()
    assert cites == []


def test_generate_grounded_answer_requires_api_key(monkeypatch):
    import app.services.qa as qa_mod

    monkeypatch.setattr(qa_mod.settings, "openai_api_key", None)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        generate_grounded_answer("q", [{"chunk_id": "1", "page_number": 1, "snippet": "x"}])


def test_generate_grounded_answer_json_and_temperature(monkeypatch):
    import app.services.qa as qa_mod

    monkeypatch.setattr(qa_mod.settings, "openai_api_key", "sk-test")
    monkeypatch.setattr(qa_mod.settings, "model_fast", "gpt-4o-mini")
    monkeypatch.setattr(qa_mod.settings, "max_completion_tokens", 500)

    payload = {
        "key_job_requirements": ["Python"],
        "matches": [
            {
                "requirement": "Python",
                "candidate_experience": "5y Python",
                "alignment_notes": "Strong match per resume excerpt.",
            }
        ],
        "gaps": [],
        "fit_score": 85,
        "reasoning": "Resume supports Python requirement shown in JD excerpts.",
    }

    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content=json.dumps(payload)))]

    captured: dict = {}

    def _capture_create(**kwargs):
        captured.update(kwargs)
        return mock_resp

    chunks = [
        {"chunk_id": "a", "page_number": 1, "snippet": "Need Python.", "source_type": "JD"},
        {"chunk_id": "b", "page_number": 2, "snippet": "Python 5 years.", "source_type": "RESUME"},
    ]

    with patch("app.services.qa.OpenAI") as m_cli:
        m_cli.return_value.chat.completions.create.side_effect = _capture_create
        answer, cites = generate_grounded_answer("How is my fit?", chunks)

    assert 0 <= QA_TEMPERATURE <= 0.3
    assert captured.get("temperature") == QA_TEMPERATURE
    assert captured.get("response_format") == {"type": "json_object"}
    sys_msg = captured["messages"][0]["content"]
    assert "AI recruiter" in sys_msg
    assert "Job description excerpts" in captured["messages"][1]["content"]
    assert "Candidate resume excerpts" in captured["messages"][1]["content"]

    out = json.loads(answer)
    assert out["fit_score"] == 85
    assert len(out["matches"]) == 1
    assert len(cites) == 2
