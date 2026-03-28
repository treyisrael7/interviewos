"""Service entry points used by interview HTTP handlers.

Tests monkeypatch ``app.routers.interview.runtime.generate_questions``; the
``generate`` route calls it via ``interview_runtime.generate_questions`` so patches apply.
"""

from app.services.interview import evaluate_answer_with_retrieval, generate_questions

__all__ = ["evaluate_answer_with_retrieval", "generate_questions"]
