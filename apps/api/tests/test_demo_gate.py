"""Tests for demo gate middleware."""

from unittest.mock import AsyncMock, patch

import pytest

from app.core.config import settings


@pytest.mark.asyncio
async def test_health_public_without_key(client):
    """Health is public and does not require x-demo-key."""
    resp = await client.get("/health")
    assert resp.status_code in (200, 503)


@pytest.mark.asyncio
async def test_protected_route_without_demo_key(client, monkeypatch, force_auth):
    """When demo mode is off, protected routes work without x-demo-key."""
    monkeypatch.setattr(settings, "demo_mode_enabled", False)
    monkeypatch.setattr(settings, "demo_key", None)
    await force_auth()
    resp = await client.post(
        "/ask",
        json={
            "document_id": "11111111-1111-1111-1111-111111111111",
            "question": "test",
        },
    )
    # 404 = doc not found; anything but 401 means we passed the gate
    assert resp.status_code != 401


@pytest.mark.asyncio
async def test_protected_route_with_demo_key(client, monkeypatch, force_auth):
    """When demo mode is on and DEMO_KEY is set, valid key allows access."""
    monkeypatch.setattr(settings, "demo_mode_enabled", True)
    monkeypatch.setattr(settings, "demo_key", "test-secret")
    await force_auth()
    resp = await client.post(
        "/ask",
        json={
            "document_id": "11111111-1111-1111-1111-111111111111",
            "question": "test",
        },
        headers={"x-demo-key": "test-secret"},
    )
    assert resp.status_code != 401


@pytest.mark.asyncio
async def test_protected_route_rejects_missing_key(client, monkeypatch):
    """When demo mode is on, missing x-demo-key returns 401."""
    monkeypatch.setattr(settings, "demo_mode_enabled", True)
    monkeypatch.setattr(settings, "demo_key", "test-secret")
    resp = await client.post("/ask", json={})
    assert resp.status_code == 401
    assert "x-demo-key" in resp.text.lower() or "invalid" in resp.text.lower()


@pytest.mark.asyncio
async def test_protected_route_rejects_wrong_key(client, monkeypatch):
    """When demo mode is on, wrong x-demo-key returns 401."""
    monkeypatch.setattr(settings, "demo_mode_enabled", True)
    monkeypatch.setattr(settings, "demo_key", "test-secret")
    resp = await client.post("/ask", json={}, headers={"x-demo-key": "wrong"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_public_paths_when_demo_key_set(client, monkeypatch):
    """When demo mode is on, public paths still work without header."""
    monkeypatch.setattr(settings, "demo_mode_enabled", True)
    monkeypatch.setattr(settings, "demo_key", "test-secret")
    for path in ["/", "/health", "/openapi.json"]:
        resp = await client.get(path)
        assert resp.status_code in (200, 503), f"{path} should be public"


@pytest.mark.asyncio
async def test_demo_key_ignored_when_demo_mode_off(client, monkeypatch):
    """DEMO_KEY in env must not enable demo auth when DEMO_MODE_ENABLED is false."""
    monkeypatch.setattr(settings, "demo_mode_enabled", False)
    monkeypatch.setattr(settings, "demo_key", "should-not-matter")
    monkeypatch.setattr(settings, "clerk_jwks_url", None)
    resp = await client.post(
        "/ask",
        json={"document_id": "11111111-1111-1111-1111-111111111111", "question": "x"},
        headers={"x-demo-key": "should-not-matter"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_bearer_plus_demo_key_returns_400(client, monkeypatch):
    """Sending both Bearer and x-demo-key is rejected (middleware)."""
    monkeypatch.setattr(settings, "demo_mode_enabled", True)
    monkeypatch.setattr(settings, "demo_key", "test-secret")
    resp = await client.post(
        "/ask",
        json={"document_id": "11111111-1111-1111-1111-111111111111", "question": "x"},
        headers={
            "Authorization": "Bearer any-token",
            "x-demo-key": "test-secret",
        },
    )
    assert resp.status_code == 400
    assert "Bearer" in resp.text or "demo" in resp.text.lower()


@pytest.mark.asyncio
async def test_demo_key_authenticates_without_clerk(client, monkeypatch):
    """Valid x-demo-key resolves sandbox user and reaches the handler (no force_auth)."""
    import uuid

    from app.db.base import async_session_maker
    from app.models import Document, User

    demo_uid = uuid.uuid4()
    monkeypatch.setattr(settings, "demo_user_id", demo_uid)
    monkeypatch.setattr(settings, "demo_mode_enabled", True)
    monkeypatch.setattr(settings, "demo_key", "demo-secret")
    monkeypatch.setattr(settings, "clerk_jwks_url", None)
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    doc_id = uuid.uuid4()
    async with async_session_maker() as db:
        db.add(User(id=demo_uid, email="demo@sandbox.local"))
        await db.commit()
    async with async_session_maker() as db:
        db.add(
            Document(
                id=doc_id,
                user_id=demo_uid,
                filename="jd.pdf",
                s3_key="x",
                status="ready",
                doc_domain="job_description",
            )
        )
        await db.commit()

    with patch("app.routers.ask.retrieve_chunks", new_callable=AsyncMock, return_value=[]):
        with patch("app.routers.ask.embed_query", return_value=[0.1] * 1536):
            resp = await client.post(
                "/ask",
                json={"document_id": str(doc_id), "question": "What is the role?"},
                headers={"x-demo-key": "demo-secret"},
            )
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
