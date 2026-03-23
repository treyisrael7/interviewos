"""Demo gate and rate limit middleware."""

import json

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings
from app.core.rate_limit import RATE_LIMITS, _path_to_route, check_rate_limit

PUBLIC_PATHS = {"/", "/health", "/openapi.json", "/docs", "/redoc"}


def _path_matches_route(path: str) -> str | None:
    """Return route key if path is rate-limited."""
    return _path_to_route(path)


class DemoGateMiddleware(BaseHTTPMiddleware):
    """Optional demo gate; never provides user identity."""

    async def dispatch(self, request: Request, call_next) -> Response:
        if not settings.demo_key:
            return await call_next(request)

        # Allow OPTIONS (CORS preflight) - browser doesn't send custom headers on preflight
        if request.method == "OPTIONS":
            return await call_next(request)

        path = (request.url.path or "/").rstrip("/") or "/"
        if path in PUBLIC_PATHS:
            return await call_next(request)

        # When Clerk is configured, allow Bearer token (routers will validate JWT)
        if settings.clerk_jwks_url and request.headers.get("authorization", "").lower().startswith("bearer "):
            return await call_next(request)

        key = request.headers.get("x-demo-key")
        if key != settings.demo_key:
            return Response(
                content='{"detail":"Missing or invalid x-demo-key header"}',
                status_code=401,
                media_type="application/json",
            )
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """In-memory rate limiting keyed by IP only."""

    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path
        route = _path_matches_route(path)
        if not route:
            return await call_next(request)

        ip = request.client.host if request.client else "0.0.0.0"
        limit, window_seconds = RATE_LIMITS[route]
        window_name = "hour" if window_seconds == 3600 else "day"

        allowed, retry_after = check_rate_limit(ip, path, None)

        if not allowed:
            body = json.dumps({
                "detail": "Rate limit exceeded",
                "retry_after_seconds": retry_after,
                "limit": limit,
                "window": window_name,
            })
            return Response(
                content=body,
                status_code=429,
                media_type="application/json",
                headers={"Retry-After": str(retry_after)},
            )
        return await call_next(request)
