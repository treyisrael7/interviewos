import logging

from fastapi import FastAPI

# Ensure ingestion/chunking logs appear in Docker console (propagate=False avoids duplicates)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(levelname)s:     %(name)s - %(message)s"))
_app_log = logging.getLogger("app.services")
_app_log.setLevel(logging.INFO)
_app_log.addHandler(_handler)
_app_log.propagate = False
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from starlette.responses import JSONResponse

from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.middleware import DemoGateMiddleware, RateLimitMiddleware
from app.routers import analyze_fit, ask, documents, fit_history, interview, retrieve, user_resume

app = FastAPI(title="InterviewOS API", version="0.1.0")

# CORS: allow_credentials=True cannot be used with allow_origins=["*"] — browser blocks it.
_DEFAULT_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
]
_EXTRA = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
_CORS_ORIGINS = list(dict.fromkeys(_DEFAULT_ORIGINS + _EXTRA))
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(DemoGateMiddleware)


async def _check_db() -> tuple[bool, str | None]:
    """Returns (ok, error_msg). Uses fresh engine to avoid event loop issues in tests."""
    engine = create_async_engine(settings.database_url, echo=False)
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True, None
    except Exception as e:
        return False, str(e)
    finally:
        await engine.dispose()


@app.get("/health", tags=["health"])
async def health():
    ok, err = await _check_db()
    clerk_configured = bool(settings.clerk_jwks_url)
    if ok:
        return {"status": "ok", "database": "connected", "clerk_configured": clerk_configured}
    return JSONResponse(
        status_code=503,
        content={"status": "error", "database": "disconnected", "detail": err or "unknown", "clerk_configured": clerk_configured},
    )


@app.get("/", tags=["meta"])
async def root():
    return {"message": "InterviewOS API"}


app.include_router(analyze_fit.router)
app.include_router(fit_history.router)
app.include_router(ask.router)
app.include_router(documents.router)
app.include_router(interview.router)
app.include_router(retrieve.router)
app.include_router(user_resume.router)
