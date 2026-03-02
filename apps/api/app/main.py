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

from app.core.config import settings
from app.core.middleware import DemoGateMiddleware, RateLimitMiddleware
from app.routers import ask, documents, retrieve

app = FastAPI(title="RAG Assistant API", version="0.1.0")

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


@app.get("/health")
async def health():
    ok, err = await _check_db()
    if ok:
        return {"status": "ok", "database": "connected"}
    return JSONResponse(
        status_code=503,
        content={"status": "error", "database": "disconnected", "detail": err or "unknown"},
    )


@app.get("/")
async def root():
    return {"message": "RAG Assistant API"}


app.include_router(ask.router)
app.include_router(documents.router)
app.include_router(retrieve.router)
