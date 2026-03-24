# InterviewOS

Upload a job description PDF, add your resume if you want, and get a RAG-powered interview prep tool that actually uses the JD. Ask questions, practice answers, and get feedback that’s grounded in the real requirements—not generic advice.

Built with Next.js, FastAPI, PostgreSQL + pgvector, and OpenAI. Local dev uses a file-based upload folder; production can use S3.

## Getting started

Copy `.env.example` to `.env`, then add your `OPENAI_API_KEY` (you’ll need it for ingestion and Q&A). The web app signs in with **Clerk** — create an app at [clerk.com](https://clerk.com), then add `CLERK_JWKS_URL` and `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` to `.env`. The API can also run a separate **demo sandbox** (`DEMO_MODE_ENABLED` + `DEMO_KEY`) for tools like `curl`; the Next.js client always uses Clerk Bearer tokens only.

Then:

```bash
make up
```

Web runs at http://localhost:3000, API at http://localhost:8000. If you’re using pgAdmin, it’s on port 5050.

Run migrations with `make db-migrate` (or `cd apps/api && alembic upgrade head` if you’re running things locally).

## What it does

**Job descriptions** – Parses PDFs, detects sections (responsibilities, qualifications, comp, etc.), chunks them sensibly, and stores embeddings so you can query things like “What’s the salary?” or “What skills do they want?”

**Interview prep** – Upload one resume on the dashboard and it’s reused for every JD. The system pulls competencies from the job description, generates behavioral and role-specific questions with rubrics, and evaluates your answers using both the JD and your resume. You can also add company notes or URLs per document.

**Ask** – RAG Q&A over your documents with citations.

## Testing with real PDFs

There are PowerShell scripts in `scripts/` for upload, retrieval, reingest, chunk stats, and Q&A. Edit `scripts/test-upload.ps1` to point `$pdfPath` at a JD, run it, then use `scripts/test-retrieve.ps1` after ingestion finishes. `scripts/test-ask.ps1` runs the full Q&A flow with citations.

## API access

`/health` is public. Everything else needs auth: either a Clerk Bearer token, or (in demo mode) an `x-demo-key` header. Example:

```bash
curl -X POST http://localhost:8000/ask \
  -H "x-demo-key: your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"user_id":"uuid","document_id":"uuid","question":"What is the salary range?"}'
```

Document flow: `POST /documents/presign` → PUT to the URL → `POST /documents/confirm` → `POST /documents/{id}/ingest`.

## Rate limits

| Route | Limit |
|-------|-------|
| POST /ask | 10/hour |
| POST /documents/ingest | 3/day |
| POST /documents/presign | 10/day |
| POST /documents/confirm | 20/day |

## Tests

Backend: `cd apps/api && pytest -v` (Postgres + migrations required). Or `make test` / `make test-docker`.

Frontend: `cd apps/web && npm run test` (or `npm run test:web` from the root).
