"""Unit tests for Clerk auth module."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.auth import (
    get_current_user,
    get_or_create_user_by_clerk_id,
    verify_clerk_token,
)
from app.core.config import settings
from app.db.base import async_session_maker
from app.models import User


# --- verify_clerk_token ---


def test_verify_clerk_token_returns_none_when_no_jwks_url(monkeypatch):
    """verify_clerk_token returns None when clerk_jwks_url is not configured."""
    monkeypatch.setattr(settings, "clerk_jwks_url", None)
    assert verify_clerk_token("any-token") is None


def test_verify_clerk_token_returns_none_for_invalid_token(monkeypatch):
    """verify_clerk_token returns None for malformed or invalid JWT."""
    monkeypatch.setattr(settings, "clerk_jwks_url", "https://test.clerk.accounts.dev/.well-known/jwks.json")

    with patch("app.core.auth._get_jwks_client") as mock_get_jwks:
        mock_jwks = MagicMock()
        mock_get_jwks.return_value = mock_jwks
        mock_jwks.get_signing_key_from_jwt.side_effect = Exception("Invalid token")

        assert verify_clerk_token("invalid-jwt") is None


def test_verify_clerk_token_returns_sub_for_valid_token(monkeypatch):
    """verify_clerk_token returns 'sub' from JWT payload when valid."""
    monkeypatch.setattr(settings, "clerk_jwks_url", "https://test.clerk.accounts.dev/.well-known/jwks.json")

    with patch("app.core.auth._get_jwks_client") as mock_get_jwks:
        mock_key = MagicMock()
        mock_jwks = MagicMock()
        mock_jwks.get_signing_key_from_jwt.return_value = mock_key
        mock_get_jwks.return_value = mock_jwks

        with patch("app.core.auth.jwt.decode", return_value={"sub": "user_2abc123"}):
            assert verify_clerk_token("valid.jwt.here") == "user_2abc123"


# --- get_or_create_user_by_clerk_id ---


@pytest.mark.asyncio
async def test_get_or_create_user_returns_existing(monkeypatch):
    """get_or_create_user_by_clerk_id returns existing user when clerk_id matches."""
    user_id = uuid.uuid4()
    clerk_id = "user_existing123"
    async with async_session_maker() as db:
        user = User(id=user_id, clerk_id=clerk_id, email="existing@test.local")
        db.add(user)
        await db.commit()

    async with async_session_maker() as db2:
        result = await get_or_create_user_by_clerk_id(db2, clerk_id)
        assert result is not None
        assert result.id == user_id
        assert result.clerk_id == clerk_id


@pytest.mark.asyncio
async def test_get_or_create_user_creates_new():
    """get_or_create_user_by_clerk_id creates user when none exists."""
    clerk_id = f"user_new_{uuid.uuid4().hex[:8]}"
    async with async_session_maker() as db:
        result = await get_or_create_user_by_clerk_id(db, clerk_id)
        assert result is not None
        assert result.clerk_id == clerk_id
        assert result.email == f"{clerk_id}@clerk.user"
        await db.commit()


# --- get_current_user ---


@pytest.mark.asyncio
async def test_get_current_user_rejects_invalid_token(monkeypatch):
    """get_current_user returns 401 when verify_clerk_token fails."""
    monkeypatch.setattr(settings, "clerk_jwks_url", "https://test.clerk.accounts.dev/.well-known/jwks.json")

    with patch("app.core.auth.verify_clerk_token", return_value=None):
        from fastapi import FastAPI, Depends

        app = FastAPI()

        @app.get("/test")
        async def route(user=Depends(get_current_user)):
            return {"user_id": str(user.id)}

        from httpx import ASGITransport, AsyncClient
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.get("/test", headers={"Authorization": "Bearer invalid-token"})
            assert resp.status_code == 401


@pytest.mark.asyncio
async def test_get_current_user_requires_bearer_token():
    """get_current_user returns 401 when the Authorization header is missing."""
    from fastapi import FastAPI, Depends
    from httpx import ASGITransport, AsyncClient

    app = FastAPI()

    @app.get("/test")
    async def route(user=Depends(get_current_user)):
        return {"user_id": str(user.id)}

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        resp = await client.get("/test")
        assert resp.status_code == 401
