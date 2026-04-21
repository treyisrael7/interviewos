"""Unit tests for study plan generation helpers."""

from types import SimpleNamespace

from app.services.study_plan import generate_role_study_plan


def _doc(**kwargs):
    base = {
        "filename": "jd.pdf",
        "doc_domain": "job_description",
        "role_profile": {"roleTitleGuess": "Backend Engineer", "focusAreas": ["API design"]},
        "jd_extraction_json": {"required_skills": ["python", "sql"], "tools": ["docker"]},
        "competencies": [{"id": "system-design", "label": "System Design"}],
        "rubric_json": [{"name": "Communication", "description": "Clear communication"}],
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_generate_role_study_plan_fallback_without_openai(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "openai_api_key", None)
    plan = generate_role_study_plan(document=_doc(), days=7, focus="system design")
    assert plan["duration_days"] == 7
    assert len(plan["daily_plan"]) == 7
    assert plan["role_title"] == "Backend Engineer"
    assert any("system design" in " ".join(day["topics"]).lower() for day in plan["daily_plan"])


def test_generate_role_study_plan_has_daily_drills_and_mock_targets(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "openai_api_key", None)
    plan = generate_role_study_plan(document=_doc(), days=10)
    assert all(day["drills"] for day in plan["daily_plan"])
    assert all(day["mock_target"] for day in plan["daily_plan"])

