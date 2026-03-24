"""Tests for basic route responses."""

import pytest

from app.core.config import settings


@pytest.mark.asyncio
async def test_root(client):
    """Root returns API message."""
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "message" in resp.json()


@pytest.mark.asyncio
async def test_ask_requires_body(client, monkeypatch, force_auth):
    """Ask returns 422 for missing body (validation error)."""
    monkeypatch.setattr(settings, "demo_mode_enabled", False)
    monkeypatch.setattr(settings, "demo_key", None)  # ensure no gate
    await force_auth()
    resp = await client.post("/ask", json={})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_ingest_requires_document_and_user(client, monkeypatch, force_auth):
    """Ingest returns 401 when no auth, 404 for unknown document."""
    monkeypatch.setattr(settings, "demo_mode_enabled", False)
    monkeypatch.setattr(settings, "demo_key", None)
    monkeypatch.setattr(settings, "clerk_jwks_url", None)  # No Bearer
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")  # Avoid 503
    # 401: no Bearer token
    resp = await client.post(
        "/documents/11111111-1111-1111-1111-111111111111/ingest",
        json={},
    )
    assert resp.status_code == 401
    # 404: document not found
    await force_auth()
    resp = await client.post(
        "/documents/11111111-1111-1111-1111-111111111111/ingest",
        json={},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_presign_with_invalid_body_returns_422(client, monkeypatch, force_auth):
    """Presign returns 422 for invalid/missing body (validation error)."""
    monkeypatch.setattr(settings, "demo_mode_enabled", False)
    monkeypatch.setattr(settings, "demo_key", None)
    await force_auth()
    resp = await client.post("/documents/presign", json={})
    assert resp.status_code == 422
