"""Service entry points used by interview HTTP handlers.

Tests monkeypatch attributes on this module (e.g. ``generate_questions``).
"""

from app.services.interview import evaluate_answer_with_retrieval, generate_questions

__all__ = ["evaluate_answer_with_retrieval", "generate_questions"]
