from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/rag_assistant"
    database_url_sync: str = "postgresql://postgres:postgres@localhost:5432/rag_assistant"

    # Demo gate
    demo_key: str | None = None  # DEMO_KEY env; if set, require x-demo-key header on non-public routes

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

    # Chunking (JD uses jd_chunking; these retained for potential generic docs)
    chunk_size: int = 512  # CHUNK_SIZE (legacy)
    min_chunk_chars: int = 25  # MIN_CHUNK_CHARS
    top_n_candidates: int = 50  # Fetch N by pgvector similarity before MMR
    mmr_lambda: float = 0.7  # MMR: lambda*sim(q,d) - (1-lambda)*max_sim(d,selected)

    # OpenAI embeddings
    openai_api_key: str | None = None  # OPENAI_API_KEY
    openai_embedding_model: str = "text-embedding-3-small"  # OPENAI_EMBEDDING_MODEL
    openai_embedding_dim: int = 1536  # OPENAI_EMBEDDING_DIM (must match DB vector column)


settings = Settings()
