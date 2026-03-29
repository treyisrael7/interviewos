import uuid
from pathlib import Path
from typing import Self

from dotenv import load_dotenv
from pydantic import AliasChoices, Field, model_validator
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
        populate_by_name=True,
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
    # Estimated-token cap for one LLM call (system + user/context + completion reserve).
    # Env aliases: MAX_LLM_BUDGET_TOKENS, MAX_TOKENS.
    max_llm_budget_tokens: int = Field(
        default=4000,
        validation_alias=AliasChoices("max_llm_budget_tokens", "MAX_LLM_BUDGET_TOKENS", "MAX_TOKENS"),
    )

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

    # Chat models: fast default for most requests; high tier for optional complex paths.
    model_fast: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices("model_fast", "MODEL_FAST", "OPENAI_CHAT_MODEL"),
    )
    model_high_quality: str = Field(
        default="gpt-4o",
        validation_alias=AliasChoices(
            "model_high_quality",
            "MODEL_HIGH_QUALITY",
            "OPENAI_CHAT_MODEL_EVAL_HIGH",
        ),
    )

    # Final answer evaluation: use ``model_high_quality`` only when enabled
    use_high_quality_eval: bool = False  # USE_HIGH_QUALITY_EVAL

    # Structured fit analysis (/analyze-fit): use ``model_high_quality`` for complex reasoning
    use_high_quality_fit_analysis: bool = False  # USE_HIGH_QUALITY_FIT_ANALYSIS

    # LLM cache (Redis optional; falls back to in-process memory)
    redis_url: str | None = None  # REDIS_URL e.g. redis://localhost:6379/0
    cache_ttl_retrieval_seconds: int = 86400  # CACHE_TTL_RETRIEVAL_SECONDS (0 = disable retrieval cache)
    cache_ttl_evaluation_seconds: int = 3600  # CACHE_TTL_EVALUATION_SECONDS (0 = disable eval cache)

    # Interview answer evaluation quotas (per calendar month, UTC). Plan is stored on users.plan.
    plan_limit_free: int = 30  # PLAN_LIMIT_FREE
    plan_limit_pro: int = 500  # PLAN_LIMIT_PRO
    plan_limit_enterprise: int = 100_000  # PLAN_LIMIT_ENTERPRISE
    # Demo sandbox user (x-demo-key): separate cap so local testing does not hit free tier.
    demo_monthly_evaluation_limit: int = 5_000  # DEMO_MONTHLY_EVALUATION_LIMIT

    @property
    def demo_auth_active(self) -> bool:
        """True when demo header auth is allowed (demo mode on and DEMO_KEY set)."""
        return bool(self.demo_mode_enabled and self.demo_key)

    @property
    def openai_eval_chat_model(self) -> str:
        """Chat model for final interview evaluation (see USE_HIGH_QUALITY_EVAL)."""
        if self.use_high_quality_eval:
            return self.model_high_quality
        return self.model_fast

    def chat_model_fit_analysis(self) -> str:
        """Standard fit analysis uses ``model_fast``; enable ``use_high_quality_fit_analysis`` for complex reasoning."""
        if self.use_high_quality_fit_analysis:
            return self.model_high_quality
        return self.model_fast

    @model_validator(mode="after")
    def _validate_demo_config(self) -> Self:
        if self.demo_mode_enabled and not (self.demo_key and str(self.demo_key).strip()):
            raise ValueError("DEMO_MODE_ENABLED requires a non-empty DEMO_KEY")
        return self


settings = Settings()
