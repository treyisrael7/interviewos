"""Tests for demo gate middleware."""

import pytest

from app.core.config import settings


@pytest.mark.asyncio
async def test_health_public_without_key(client):
    """Health is public and does not require x-demo-key."""
    resp = await client.get("/health")
    assert resp.status_code in (200, 503)


@pytest.mark.asyncio
async def test_protected_route_without_demo_key(client, monkeypatch, force_auth):
    """When DEMO_KEY is not set, protected routes work without header."""
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
    """When DEMO_KEY is set, valid key allows access."""
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
    """When DEMO_KEY is set, missing key returns 401."""
    monkeypatch.setattr(settings, "demo_key", "test-secret")
    resp = await client.post("/ask", json={})
    assert resp.status_code == 401
    assert "x-demo-key" in resp.text.lower() or "invalid" in resp.text.lower()


@pytest.mark.asyncio
async def test_protected_route_rejects_wrong_key(client, monkeypatch):
    """When DEMO_KEY is set, wrong key returns 401."""
    monkeypatch.setattr(settings, "demo_key", "test-secret")
    resp = await client.post("/ask", json={}, headers={"x-demo-key": "wrong"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_public_paths_when_demo_key_set(client, monkeypatch):
    """When DEMO_KEY is set, public paths still work without header."""
    monkeypatch.setattr(settings, "demo_key", "test-secret")
    for path in ["/", "/health", "/openapi.json"]:
        resp = await client.get(path)
        assert resp.status_code in (200, 503), f"{path} should be public"
