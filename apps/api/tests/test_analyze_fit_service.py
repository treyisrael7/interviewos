"""Tests for analyze_fit_service (splitting, validation, empty path, scoring)."""

import json
import uuid
from unittest.mock import MagicMock, patch

import pytest

from app.services import analyze_fit_service as afs
from app.services.analyze_fit_service import (
    AnalyzeFitLLMResult,
    AnalyzeFitResult,
    FitGap,
    FitRecommendation,
    _align_recommendations_to_gaps,
    analyze_fit,
    compress_chunks,
    compute_fit_score,
    split_chunks_for_fit,
)


def test_compress_chunks_keeps_skills_and_drops_fluff():
    chunk = {
        "source_type": "JD",
        "snippet": (
            "We are an equal opportunity employer committed to diversity. "
            "You must have 5+ years of Python and Kubernetes experience in production. "
            "Nice to have AWS certification. Please apply online today."
        ),
    }
    out = compress_chunks([chunk], max_combined_estimated_tokens=8000)
    assert len(out) == 1
    text = (out[0].get("snippet") or "").lower()
    assert "python" in text or "kubernetes" in text
    assert len(out[0]["snippet"]) < len(chunk["snippet"])


def test_compress_chunks_respects_combined_token_budget():
    long_body = (
        "Must have strong experience with distributed systems and Python. "
        "Responsible for building APIs and mentoring engineers. " * 15
    )
    chunks = [
        {"source_type": "JD", "chunk_id": "j1", "snippet": long_body},
        {"source_type": "JD", "chunk_id": "j2", "snippet": long_body},
        {"source_type": "RESUME", "chunk_id": "r1", "snippet": long_body},
        {"source_type": "RESUME", "chunk_id": "r2", "snippet": long_body},
    ]
    out = compress_chunks(chunks, max_combined_estimated_tokens=450)
    jd, rs = split_chunks_for_fit(out)
    jt = afs._format_side_for_prompt("JOB_EXCERPTS (requirements / role)", jd)
    rt = afs._format_side_for_prompt("RESUME_EXCERPTS (candidate)", rs)
    from app.services.token_budget import estimate_tokens

    combined = estimate_tokens(jt) + estimate_tokens(rt)
    excerpt_budget = max(400, 450 - afs._ANALYZE_FIT_USER_FRAME_TOKEN_EST)
    assert combined <= excerpt_budget + 8  # slack for char-based token estimate


def test_split_chunks_by_source_type():
    jd = {"chunk_id": "a", "source_type": "JD", "text": "Must know Python."}
    rs = {"chunk_id": "b", "source_type": "RESUME", "text": "Python 5 years."}
    other = {"chunk_id": "c", "source_type": "OTHER", "text": "Extra."}
    none_st = {"chunk_id": "d", "text": "No source_type"}

    jd_chunks, resume_chunks = split_chunks_for_fit([jd, rs, other, none_st])
    assert len(jd_chunks) == 3
    assert len(resume_chunks) == 1
    assert resume_chunks[0]["chunk_id"] == "b"


def test_compute_fit_score_weighted_and_gap_penalty():
    matches = [
        {"confidence": 0.9, "importance": "medium"},
    ]
    gaps = [
        {"importance": "medium"},
    ]
    out = compute_fit_score(matches, gaps)
    # total_w = 1.5 + 1.5 = 3; raw = 0.9*1.5/3*100 = 45; penalty = 2.5 -> 42
    assert out["matched_count"] == 1
    assert out["gap_count"] == 1
    assert out["total_requirements"] == 2
    assert out["fit_score"] == 42
    assert out["gap_penalty"] == 2.5


def test_compute_fit_score_critical_gap_extra_penalty():
    matches = [{"confidence": 1.0, "importance": "high"}]
    gaps = [{"importance": "high"}]
    out = compute_fit_score(matches, gaps)
    # total_w = 2.5+2.5=5; coverage = 2.5/5*100=50; penalty = 2.5+6=8.5 -> 41
    assert out["gap_penalty"] == 8.5
    # 50% weighted coverage minus 8.5 penalty → 41.5 rounds to 42
    assert out["fit_score"] == 42


def test_analyze_fit_no_excerpt_text_returns_without_llm(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    called = []

    def _no_create(*args, **kwargs):
        called.append(kwargs)
        raise AssertionError("OpenAI should not be called when excerpts are empty")

    with patch("app.services.analyze_fit_service.OpenAI") as m_client:
        m_client.return_value.chat.completions.create.side_effect = _no_create
        out = analyze_fit(
            query="fit?",
            retrieved_chunks=[{"source_type": "JD", "snippet": "   "}],
            user_id=uuid.uuid4(),
        )

    assert not called
    assert out["fit_score"] == 0
    assert out["fit_score_hint"] == 0
    assert out["matches"] == []
    assert out["gaps"] == []
    assert out["matched_count"] == 0
    assert out["total_requirements"] == 0
    assert out["gap_count"] == 0
    assert out["recommendations"] == []
    assert "no job description or resume text" in out["summary"].lower()


def test_analyze_fit_structured_response_parsed(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "openai_api_key", "sk-test")

    payload = {
        "matches": [
            {
                "requirement": "Python",
                "resume_evidence": "5 years Python",
                "confidence": 0.9,
                "importance": "medium",
            }
        ],
        "gaps": [{"requirement": "Kubernetes", "reason": "Not mentioned.", "importance": "medium"}],
        "fit_score_hint": 72,
        "summary": "Strong Python match; gap on Kubernetes.",
        "recommendations": [
            {
                "gap": "Kubernetes",
                "suggestion": "Add the exact term Kubernetes; reframe your Docker work as container orchestration exposure and study one cluster setup you can describe.",
                "example_resume_line": "Deployed services with Docker; building hands-on experience with Kubernetes (minikube) for orchestration.",
            }
        ],
    }

    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content=json.dumps(payload)))]

    with patch("app.services.analyze_fit_service.OpenAI") as m_client:
        m_client.return_value.chat.completions.create.return_value = mock_resp
        out = analyze_fit(
            query="How do I fit?",
            retrieved_chunks=[
                {"source_type": "JD", "snippet": "Need Python and Kubernetes."},
                {"source_type": "RESUME", "snippet": "Python developer 5 years."},
            ],
            user_id=uuid.uuid4(),
        )

    m_client.return_value.chat.completions.create.assert_called_once()
    assert out["fit_score_hint"] == 72
    assert out["fit_score"] == 42
    assert len(out["matches"]) == 1
    assert out["matches"][0]["confidence"] == 0.9
    assert len(out["gaps"]) == 1
    assert out["matched_count"] == 1
    assert out["total_requirements"] == 2
    assert out["gap_count"] == 1
    assert len(out["recommendations"]) == 1
    assert out["recommendations"][0]["gap"] == "Kubernetes"
    assert "Kubernetes" in out["recommendations"][0]["example_resume_line"]


def test_align_recommendations_to_gaps_uses_requirement_text():
    gaps = [
        FitGap(requirement="Know Rust", reason="x", importance="high"),
        FitGap(requirement="AWS Lambda", reason="y", importance="medium"),
    ]
    recs = [
        FitRecommendation(gap="wrong label", suggestion="s1", example_resume_line="e1"),
        FitRecommendation(gap="x", suggestion="s2", example_resume_line="e2"),
    ]
    aligned = _align_recommendations_to_gaps(gaps, recs)
    assert aligned[0].gap == "Know Rust"
    assert aligned[1].gap == "AWS Lambda"
    assert aligned[0].suggestion == "s1"


def test_analyze_fit_rejects_extra_fields_from_model():
    raw = {
        "matches": [],
        "gaps": [],
        "fit_score_hint": 50,
        "summary": "x",
        "recommendations": [],
        "hallucinated": "no",
    }
    with pytest.raises(Exception):
        AnalyzeFitLLMResult.model_validate(raw)


def test_analyze_fit_api_failure_returns_once_no_second_call(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "openai_api_key", "sk-test")

    with patch("app.services.analyze_fit_service.OpenAI") as m_client:
        m_client.return_value.chat.completions.create.side_effect = RuntimeError("API down")
        out = analyze_fit(
            query="fit?",
            retrieved_chunks=[
                {"source_type": "JD", "snippet": "Need Python."},
                {"source_type": "RESUME", "snippet": "Rust expert."},
            ],
            user_id=uuid.uuid4(),
        )

    m_client.return_value.chat.completions.create.assert_called_once()
    assert out["matches"] == []
    assert out["gaps"] == []
    assert "model request failed" in out["summary"].lower()


def test_analyze_fit_requires_openai_key(monkeypatch):
    import app.services.analyze_fit_service as afs_mod

    monkeypatch.setattr(afs_mod.settings, "openai_api_key", None)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        analyze_fit(
            "q",
            [{"source_type": "JD", "snippet": "text"}],
            uuid.uuid4(),
        )


def test_final_analyze_fit_result_shape():
    r = AnalyzeFitResult(
        matches=[],
        gaps=[],
        summary="s",
        fit_score=10,
        fit_score_hint=50,
        matched_count=0,
        total_requirements=0,
        gap_count=0,
        gap_penalty=0.0,
        coverage_raw=0.0,
        recommendations=[],
    )
    d = r.model_dump()
    assert set(d.keys()) >= {
        "matches",
        "gaps",
        "summary",
        "fit_score",
        "fit_score_hint",
        "matched_count",
        "total_requirements",
        "gap_count",
        "gap_penalty",
        "coverage_raw",
        "recommendations",
    }
