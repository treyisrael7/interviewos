import uuid
from pathlib import Path
from typing import Self

from dotenv import load_dotenv
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Fixed sandbox identity when demo mode is enabled (isolated from real Clerk users).
_DEFAULT_DEMO_USER_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")

# Project root (apps/api/app/core/config.py -> 5 levels up: core->app->api->apps->root)
_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
# API dir (apps/api)
_API_ROOT = _ROOT / "apps" / "api"

# Explicitly load .env files into os.environ (ensures CLERK_JWKS_URL etc. are available)
for env_path in (_ROOT / ".env", _API_ROOT / ".env"):
    if env_path.exists():
        load_dotenv(env_path, override=False)  # earlier files take precedence


class Settings(BaseSettings):
    # Load .env from project root and apps/api (api dir takes precedence for overrides)
    model_config = SettingsConfigDict(
        env_file=(_ROOT / ".env", _API_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/interview_os"
    database_url_sync: str = "postgresql://postgres:postgres@localhost:5432/interview_os"

    # Demo sandbox (OFF in production). When False, x-demo-key and demo user identity are disabled entirely.
    demo_mode_enabled: bool = False  # DEMO_MODE_ENABLED
    demo_key: str | None = None  # DEMO_KEY; required when demo_mode_enabled (validated below)
    demo_user_id: uuid.UUID = _DEFAULT_DEMO_USER_ID  # DEMO_USER_ID — DB user for sandbox data only

    # Clerk JWT (Bearer). Real users always authenticate here; demo never overrides a valid Bearer path.
    clerk_jwks_url: str | None = None  # CLERK_JWKS_URL e.g. https://<xxx>.clerk.accounts.dev/.well-known/jwks.json
    clerk_issuer: str | None = None  # CLERK_ISSUER override; if unset, derived from JWKS URL

    # CORS origins (comma-separated; when empty, uses default localhost list)
    cors_origins: str = ""  # CORS_ORIGINS e.g. http://localhost:3000,http://192.168.1.5:3000

    # AWS S3 (production)
    aws_region: str = "us-east-1"
    s3_bucket: str | None = None
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None

    # Hard limits (config via env)
    max_pdf_mb: int = 10  # MAX_PDF_MB
    max_pdf_pages: int = 20  # MAX_PDF_PAGES
    max_chunks_per_doc: int = 300  # MAX_CHUNKS_PER_DOC
    top_k_max: int = 8  # TOP_K_MAX
    max_completion_tokens: int = 500  # MAX_COMPLETION_TOKENS

    # Chunking (job description uses jd_chunking; these retained for potential generic docs)
    chunk_size: int = 512  # CHUNK_SIZE (legacy)
    min_chunk_chars: int = 25  # MIN_CHUNK_CHARS
    top_n_candidates: int = 50  # Fetch N by pgvector similarity before MMR
    mmr_lambda: float = 0.7  # MMR: lambda*sim(q,d) - (1-lambda)*max_sim(d,selected)
    hybrid_retrieval_enabled: bool = True  # HYBRID_RETRIEVAL_ENABLED; safe because retrieval falls back to semantic-only on FTS errors

    # OpenAI embeddings
    openai_api_key: str | None = None  # OPENAI_API_KEY
    openai_embedding_model: str = "text-embedding-3-small"  # OPENAI_EMBEDDING_MODEL
    openai_embedding_dim: int = 1536  # OPENAI_EMBEDDING_DIM (must match DB vector column)

    # OpenAI chat (Q&A)
    openai_chat_model: str = "gpt-4o-mini"  # OPENAI_CHAT_MODEL

    @property
    def demo_auth_active(self) -> bool:
        """True when demo header auth is allowed (demo mode on and DEMO_KEY set)."""
        return bool(self.demo_mode_enabled and self.demo_key)

    @model_validator(mode="after")
    def _validate_demo_config(self) -> Self:
        if self.demo_mode_enabled and not (self.demo_key and str(self.demo_key).strip()):
            raise ValueError("DEMO_MODE_ENABLED requires a non-empty DEMO_KEY")
        return self


settings = Settings()
