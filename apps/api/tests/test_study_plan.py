"""Tests for JD role-specific study plan generation."""

import uuid

import pytest

from app.core.rate_limit import RATE_LIMITS, _path_to_route


def test_rate_limit_maps_study_plan_path():
    assert _path_to_route("/documents/11111111-1111-1111-1111-111111111111/study-plan") == "study-plan"
    assert "study-plan" in RATE_LIMITS


@pytest.mark.asyncio
async def test_study_plan_requires_job_description_document(client, demo_key_off, force_auth):
    """Study plan returns 400 for non-JD documents."""
    from app.db.base import async_session_maker
    from app.models import Document, User

    user_id = uuid.uuid4()
    async with async_session_maker() as db:
        db.add(User(id=user_id, email="study-general@t.local"))
        await db.commit()
    await force_auth(user_id=user_id, email="study-general@t.local")

    async with async_session_maker() as db:
        doc = Document(
            user_id=user_id,
            filename="notes.pdf",
            s3_key="x",
            status="ready",
            doc_domain="general",
        )
        db.add(doc)
        await db.flush()
        doc_id = doc.id
        await db.commit()

    resp = await client.post(f"/documents/{doc_id}/study-plan", json={"days": 10})
    assert resp.status_code == 400
    assert "job description" in str(resp.json().get("detail", "")).lower()


@pytest.mark.asyncio
async def test_study_plan_returns_fallback_shape_when_openai_missing(
    client,
    demo_key_off,
    force_auth,
    monkeypatch,
):
    """Study plan still returns valid output using deterministic fallback."""
    from app.core.config import settings
    from app.db.base import async_session_maker
    from app.models import Document, User

    monkeypatch.setattr(settings, "openai_api_key", None)

    user_id = uuid.uuid4()
    async with async_session_maker() as db:
        db.add(User(id=user_id, email="study-fallback@t.local"))
        await db.commit()
    await force_auth(user_id=user_id, email="study-fallback@t.local")

    async with async_session_maker() as db:
        doc = Document(
            user_id=user_id,
            filename="jd.pdf",
            s3_key="x",
            status="ready",
            doc_domain="job_description",
            jd_extraction_json={
                "required_skills": ["python", "sql"],
                "tools": ["docker"],
            },
            role_profile={"roleTitleGuess": "Backend Engineer", "focusAreas": ["API design"]},
            competencies=[{"id": "system-design", "label": "System Design"}],
        )
        db.add(doc)
        await db.flush()
        doc_id = doc.id
        await db.commit()

    resp = await client.post(
        f"/documents/{doc_id}/study-plan",
        json={"days": 7, "focus": "system design"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["duration_days"] == 7
    assert len(data["daily_plan"]) == 7
    assert all(day["day"] == idx + 1 for idx, day in enumerate(data["daily_plan"]))
    assert any("system design" in ", ".join(day["topics"]).lower() for day in data["daily_plan"])


@pytest.mark.asyncio
async def test_study_plan_rejects_invalid_days(client, demo_key_off, force_auth):
    """Study plan input validates days between 7 and 14."""
    from app.db.base import async_session_maker
    from app.models import Document, User

    user_id = uuid.uuid4()
    async with async_session_maker() as db:
        db.add(User(id=user_id, email="study-days@t.local"))
        await db.commit()
    await force_auth(user_id=user_id, email="study-days@t.local")

    async with async_session_maker() as db:
        doc = Document(
            user_id=user_id,
            filename="jd.pdf",
            s3_key="x",
            status="ready",
            doc_domain="job_description",
        )
        db.add(doc)
        await db.flush()
        doc_id = doc.id
        await db.commit()

    resp = await client.post(f"/documents/{doc_id}/study-plan", json={"days": 5})
    assert resp.status_code == 422

