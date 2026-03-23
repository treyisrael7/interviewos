"""Integration tests for Clerk auth: 401 when no auth, Bearer flow."""

import uuid

import pytest

from app.core.config import settings
from app.main import app
from app.core.auth import get_current_user


@pytest.fixture
def clerk_jwks_off(monkeypatch):
    monkeypatch.setattr(settings, "clerk_jwks_url", None)


@pytest.mark.asyncio
async def test_ask_returns_401_when_no_user_id_and_no_bearer(client, demo_key_off, clerk_jwks_off):
    """POST /ask returns 401 when no Bearer token is provided."""
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
    """POST /ask ignores client-supplied user_id and still requires auth."""
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    resp = await client.post(
        "/ask",
        json={
            "user_id": "11111111-1111-1111-1111-111111111111",
            "document_id": "11111111-1111-1111-1111-111111111111",
            "question": "What is the salary?",
        },
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_interview_sessions_returns_401_without_user_id(client, demo_key_off, clerk_jwks_off):
    """GET /interview/sessions returns 401 when no Bearer token is provided."""
    resp = await client.get("/interview/sessions")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_interview_sessions_succeeds_with_authenticated_user(
    client, demo_key_off, clerk_jwks_off, force_auth
):
    """GET /interview/sessions uses the authenticated user instead of a query param."""
    from app.db.base import async_session_maker
    from app.models import Document, InterviewQuestion, InterviewSession, User

    user_id = uuid.uuid4()
    async with async_session_maker() as db:
        user = User(id=user_id, email="sessions-auth@t.local")
        db.add(user)
        await db.commit()
    await force_auth(user_id=user_id, email="sessions-auth@t.local")
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

    resp = await client.get("/interview/sessions")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_bearer_token_via_dependency_override(client, demo_key_off, clerk_jwks_off):
    """A dependency override can inject the authenticated user (simulates valid Clerk)."""
    from fastapi import Request
    from httpx import ASGITransport, AsyncClient
    from app.models import User

    user_id = uuid.uuid4()

    async def override_bearer(request: Request):
        if request.headers.get("authorization", "").startswith("Bearer "):
            return User(id=user_id, email="bearer-test@t.local")
        raise AssertionError("Authorization header missing in override")

    app.dependency_overrides[get_current_user] = override_bearer

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
        app.dependency_overrides.pop(get_current_user, None)
