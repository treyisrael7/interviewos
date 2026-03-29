"""Token budgeting helpers (grounded QA path)."""

from app.core.config import settings
from app.services.qa import _SYSTEM_PROMPT, _build_user_content_for_budget, _split_jd_resume_chunks
from app.services.token_budget import (
    TOKEN_BUDGET_SAFETY_SLACK,
    budget_grounded_qa_prompt,
    clamp_completion_to_budget,
    estimate_tokens,
    retrieval_score,
)


def test_estimate_tokens_empty_and_rounding():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2


def test_retrieval_score_prefers_final_score():
    assert retrieval_score({"final_score": 0.9, "score": 0.1}) == 0.9
    assert retrieval_score({"score": 0.5}) == 0.5
    assert retrieval_score({}) == 0.0


def test_clamp_completion_respects_budget():
    cap = clamp_completion_to_budget(
        estimated_input_tokens=3500,
        requested_completion_tokens=1200,
        total_budget=4000,
        slack=TOKEN_BUDGET_SAFETY_SLACK,
    )
    assert cap <= 4000 - 3500 - TOKEN_BUDGET_SAFETY_SLACK


def test_budget_grounded_qa_drops_lower_scoring_chunks_first(monkeypatch):
    monkeypatch.setattr(settings, "max_llm_budget_tokens", 900)
    q = "Fit?"
    chunks = [
        {"chunk_id": "j_hi", "page_number": 1, "snippet": "Alpha " * 80, "source_type": "JD", "score": 0.99},
        {"chunk_id": "j_lo", "page_number": 2, "snippet": "Beta " * 80, "source_type": "JD", "score": 0.1},
        {"chunk_id": "r_hi", "page_number": 3, "snippet": "Gamma " * 80, "source_type": "RESUME", "score": 0.98},
        {"chunk_id": "r_lo", "page_number": 4, "snippet": "Delta " * 80, "source_type": "RESUME", "score": 0.05},
    ]
    used, user_content, eff = budget_grounded_qa_prompt(
        question=q,
        chunks=chunks,
        system_prompt=_SYSTEM_PROMPT,
        split_chunks=_split_jd_resume_chunks,
        build_user_content=lambda jd_c, rs_c: _build_user_content_for_budget(jd_c, rs_c, q),
        requested_completion_tokens=1200,
        total_budget=settings.max_llm_budget_tokens,
    )
    ids = {c["chunk_id"] for c in used}
    assert ids == {"j_hi", "r_hi"}
    assert eff >= 1
    inp = estimate_tokens(_SYSTEM_PROMPT) + estimate_tokens(user_content)
    assert inp + eff + TOKEN_BUDGET_SAFETY_SLACK <= settings.max_llm_budget_tokens + 2
