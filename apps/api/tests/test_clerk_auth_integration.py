"""Integration tests for Clerk auth: 401 when no auth, Bearer flow."""

import uuid

import pytest

from app.core.config import settings
from app.main import app
from app.core.auth import get_user_id_from_bearer


@pytest.fixture
def clerk_jwks_off(monkeypatch):
    monkeypatch.setattr(settings, "clerk_jwks_url", None)


@pytest.mark.asyncio
async def test_ask_returns_401_when_no_user_id_and_no_bearer(client, demo_key_off, clerk_jwks_off):
    """POST /ask returns 401 when neither user_id in body nor Bearer token provided."""
    resp = await client.post(
        "/ask",
        json={
            "document_id": "11111111-1111-1111-1111-111111111111",
            "question": "What is the salary?",
        },
    )
    assert resp.status_code == 401
    assert "Authentication" in resp.text or "401" in str(resp.status_code)


@pytest.mark.asyncio
async def test_ask_succeeds_with_user_id_in_body(client, demo_key_off, clerk_jwks_off, monkeypatch):
    """POST /ask accepts user_id in body (demo mode) when no Bearer."""
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    resp = await client.post(
        "/ask",
        json={
            "user_id": "11111111-1111-1111-1111-111111111111",
            "document_id": "11111111-1111-1111-1111-111111111111",
            "question": "What is the salary?",
        },
    )
    # 404 = doc not found (we passed auth)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_interview_sessions_returns_401_without_user_id(client, demo_key_off, clerk_jwks_off):
    """GET /interview/sessions returns 401 when user_id not in query and no Bearer."""
    resp = await client.get("/interview/sessions")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_interview_sessions_succeeds_with_user_id_query(client, demo_key_off, clerk_jwks_off):
    """GET /interview/sessions accepts user_id in query (demo mode)."""
    from app.db.base import async_session_maker
    from app.models import Document, InterviewQuestion, InterviewSession, User

    user_id = uuid.uuid4()
    async with async_session_maker() as db:
        user = User(id=user_id, email="sessions-auth@t.local")
        db.add(user)
        await db.commit()
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
        session = InterviewSession(
            user_id=user_id,
            document_id=doc.id,
            mode="technical",
            difficulty="junior",
        )
        db.add(session)
        await db.flush()
        q = InterviewQuestion(
            session_id=session.id,
            type="technical",
            question="Q?",
            rubric_json={"bullets": [], "evidence": [], "key_topics": []},
        )
        db.add(q)
        await db.commit()

    resp = await client.get(f"/interview/sessions?user_id={user_id}")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_bearer_token_via_dependency_override(client, demo_key_off, clerk_jwks_off):
    """When Bearer present, dependency override can inject user_id (simulates valid Clerk)."""
    from fastapi import Request
    from httpx import ASGITransport, AsyncClient

    user_id = uuid.uuid4()

    async def override_bearer(request: Request):
        if request.headers.get("authorization", "").startswith("Bearer "):
            return user_id
        return None

    app.dependency_overrides[get_user_id_from_bearer] = override_bearer

    from app.db.base import async_session_maker
    from app.models import Document, User

    async with async_session_maker() as db:
        user = User(id=user_id, email="bearer-test@t.local")
        db.add(user)
        await db.commit()
    doc_id = None
    async with async_session_maker() as db:
        doc = Document(
            user_id=user_id,
            filename="x.pdf",
            s3_key="x",
            status="ready",
        )
        db.add(doc)
        await db.flush()
        doc_id = doc.id
        await db.commit()

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as ac:
            # Without user_id in query, Bearer override provides it
            resp = await ac.get(
                f"/documents/{doc_id}",
                headers={"Authorization": "Bearer fake-but-overridden"},
            )
            assert resp.status_code == 200
    finally:
        app.dependency_overrides.pop(get_user_id_from_bearer, None)
