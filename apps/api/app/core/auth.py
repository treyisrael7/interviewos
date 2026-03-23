"""Clerk JWT verification and authenticated user helpers."""

import logging
from typing import Any
import uuid
from urllib.parse import urlparse

import jwt
from fastapi import Depends, HTTPException, Request
from jwt import PyJWKClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.session import get_db
from app.models import User

logger = logging.getLogger(__name__)

_jwks_client: PyJWKClient | None = None


def _issuer_from_jwks_url(url: str) -> str:
    """Derive issuer from JWKS URL: https://x.clerk.accounts.dev/.well-known/jwks.json -> https://x.clerk.accounts.dev"""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def _get_jwks_client() -> PyJWKClient:
    global _jwks_client
    if _jwks_client is None:
        url = settings.clerk_jwks_url
        if not url:
            raise ValueError("CLERK_JWKS_URL is not configured")
        _jwks_client = PyJWKClient(url, cache_jwk_set=True)
    return _jwks_client


def verify_clerk_token(token: str) -> str | None:
    """
    Verify Clerk JWT and return the subject (Clerk user ID).
    Returns None on invalid/expired token.
    """
    if not settings.clerk_jwks_url:
        return None
    try:
        jwks = _get_jwks_client()
        signing_key = jwks.get_signing_key_from_jwt(token)
    except Exception as e:
        logger.warning("Clerk token signing key lookup failed: %s", e)
        return None
    issuer = settings.clerk_issuer or _issuer_from_jwks_url(settings.clerk_jwks_url)
    # Try with issuer first; fallback without issuer for some Clerk setups
    for verify_iss in (True, False):
        try:
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                issuer=issuer if verify_iss else None,
                leeway=120,  # Allow 2min clock skew (iat/nbf/exp) — Docker on Windows often lags
                options={
                    "verify_exp": True,
                    "verify_aud": False,
                    "verify_iss": verify_iss,
                },
            )
            return payload.get("sub")
        except jwt.InvalidIssuerError as e:
            if verify_iss:
                logger.info("Clerk issuer mismatch, retrying without issuer check: %s", e)
                continue
            raise
        except Exception as e:
            logger.warning("Clerk token verification failed: %s", e)
            return None
    return None


async def get_or_create_user_by_clerk_id(db: AsyncSession, clerk_id: str) -> User:
    """Get an existing user by Clerk ID, or create one."""
    result = await db.execute(select(User).where(User.clerk_id == clerk_id))
    user = result.scalar_one_or_none()
    if user:
        return user
    user = User(id=uuid.uuid4(), clerk_id=clerk_id, email=f"{clerk_id}@clerk.user")
    db.add(user)
    await db.flush()
    return user


async def get_current_user(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Validate the Clerk Bearer token, resolve the database user, and create it if needed.
    """
    auth_header = request.headers.get("authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Authentication required")
    scheme, _, token = auth_header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(status_code=401, detail="Authentication required")
    if not settings.clerk_jwks_url:
        raise HTTPException(
            status_code=401,
            detail="Clerk is not configured. Add CLERK_JWKS_URL to your API environment.",
        )
    clerk_id = verify_clerk_token(token.strip())
    if not clerk_id:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired session. Please sign in again.",
        )
    user = await get_or_create_user_by_clerk_id(db, clerk_id)
    await db.commit()
    await db.refresh(user)
    return user


def assert_resource_ownership(resource: Any, current_user: User) -> None:
    """Raise when a resource is missing or belongs to another user."""
    if resource is None:
        raise HTTPException(status_code=404, detail="Resource not found")

    owner_id = getattr(resource, "user_id", None)
    if owner_id is None:
        raise ValueError("Resource does not expose user_id for ownership checks")

    if owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Forbidden")
