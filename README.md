# RAG Assistant – Job Description Q&A

LLM-powered RAG system optimized for **Job Description (JD) PDFs**. Upload JDs, extract structured data, and query with grounded, cited answers.

## Tech stack

- **Frontend:** Next.js, TypeScript
- **API:** FastAPI, Python, async
- **DB:** PostgreSQL, pgvector for embeddings
- **Embeddings:** OpenAI text-embedding-3-small
- **Storage:** AWS S3 (production) or local `./uploads`

## Quick start

```bash
make up
```

Then:

- **Web:** http://localhost:3000
- **API:** http://localhost:8000
- **pgAdmin:** http://localhost:5050

## Migrations

```bash
make db-migrate
```

Or locally: `cd apps/api && alembic upgrade head`

## JD features

- **Section detection:** Responsibilities, qualifications, compensation, location, tools, etc.
- **Structured extraction:** Company, role, salary range, required skills, experience (rule-based)
- **Section-aware chunking:** Keeps bullets intact, tags chunks with `section_type`
- **Smart retrieval:** Filters by section for queries like "What is the salary?" or "What skills are required?"

## Testing with JD PDFs

1. Edit `scripts/test-upload.ps1` – set `$pdfPath` to your JD PDF
2. Run:
   ```powershell
   .\scripts\test-upload.ps1
   ```
3. After ingestion completes, run:
   ```powershell
   .\scripts\test-retrieve.ps1
   ```

Or use `scripts/reingest.ps1` to re-process, `scripts/chunk-stats.ps1` for ingestion stats.

## Demo gate & API access

When `DEMO_KEY` is set, non-public routes require an `x-demo-key` header.

```bash
# Public
curl http://localhost:8000/health

# Protected
curl -X POST http://localhost:8000/retrieve \
  -H "x-demo-key: your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"user_id":"uuid","document_id":"uuid","query":"What are the qualifications?"}'
```

## Document upload flow

1. **POST /documents/presign** – Get presigned PUT URL
2. **PUT** to `upload_url` – Upload the PDF
3. **POST /documents/confirm** – Mark uploaded
4. **POST /documents/{id}/ingest** – Extract, chunk, embed (async)

## Rate limits

| Route              | Limit    |
|--------------------|----------|
| POST /ask          | 10/hour  |
| POST /documents/ingest | 3/day  |
| POST /documents/presign | 10/day |
| POST /documents/confirm | 20/day |

## Tests

```bash
make test-docker   # in Docker
make test         # locally: cd apps/api && pytest -v
```
