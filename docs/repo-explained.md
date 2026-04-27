# InterviewOS Repository Guide

This document explains the `rag-assistant` repository as it exists today. The product is branded as **InterviewOS**: a focused interview-practice app that ingests a job description and resume, builds a retrieval index, generates role-aware interview questions, and evaluates practice answers with cited evidence.

## 1. Product at a Glance

InterviewOS centers on a job description PDF and an account-level resume. A signed-in user uploads both, the API extracts and chunks their text, stores embeddings in PostgreSQL with pgvector, and then uses those chunks for the core interview-practice loop.

Core scope:

- **Interview practice**: Generate role-aware questions from the JD, collect answers, and evaluate them against rubrics and retrieved JD/resume evidence.
- **Grounded prep support**: Use cited Q&A over the JD or resume to check details while preparing.

Secondary or deferred surfaces already present in the repo:

- **Resume fit analysis**: Compare a resume against a JD and produce matches, gaps, score, and recommendations.
- **Study plans and gap analysis**: Generate preparation guidance from retrieved document evidence.
- **Analytics**: Track interview scores, competency trends, and recent sessions.
- **Extra kit sources**: Add company or notes sources alongside the JD.

The stack is a monorepo with a Next.js web app, FastAPI backend, PostgreSQL + pgvector database, Alembic migrations, OpenAI embeddings/chat calls, and Docker-based local orchestration.

## 2. Top-Level Layout

- `README.md`: short product and setup overview.
- `package.json`: root npm workspace definition for `apps/*` and `packages/*`.
- `docker-compose.yml`: local services for Postgres, API, web, and pgAdmin.
- `Makefile`: local commands for Docker, migrations, tests, and retrieval evals.
- `.env.example`: root environment template used by Docker and the API/web apps.
- `.github/workflows/ci.yml`: backend and frontend CI jobs.
- `apps/api`: FastAPI backend, database models, services, routers, tests, migrations, and retrieval eval tooling.
- `apps/web`: Next.js frontend, React components, hooks, API client, tests, and styling.
- `packages/shared`: small TypeScript shared package; currently only exposes a minimal shared type.
- `scripts`: PowerShell/Python helpers for upload, retrieval, ask, reingest, and chunk stats.
- `docs`: repository notes and deployment-related snippets such as S3 CORS config.

## 3. Runtime Architecture

The app has four main runtime components:

- **Web app** in `apps/web`: Next.js 15 and React 19. It handles Clerk auth, page routing, document upload UI, interview UI, dashboards, and API calls.
- **API app** in `apps/api`: FastAPI service exposing the core document, resume, retrieval, and interview endpoints, plus secondary fit analysis and analytics endpoints.
- **Database**: PostgreSQL 16 with pgvector. It stores users, documents, chunks, interview sessions/questions/answers, fit analysis cache rows, and retrieval feedback.
- **Object storage**: S3 in production when configured; local filesystem storage for development when S3 env vars are absent.

OpenAI is used for:

- Text embeddings during ingestion and query retrieval.
- Grounded Q&A responses.
- Secondary fit analysis.
- Interview question generation.
- Interview answer evaluation.
- Rubric extraction, with secondary study-plan and fit-analysis paths.

## 4. Local Development

The intended local path is:

1. Copy `.env.example` to `.env`.
2. Set at least `OPENAI_API_KEY`.
3. Configure Clerk with `CLERK_JWKS_URL` for the API and `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` for the web app.
4. Run `make up`.
5. Run `make db-migrate`.

Important ports:

- Web: `http://localhost:3000` in local Next dev; Docker maps web as `3001:3000`.
- API: `http://localhost:8000`.
- Postgres: `localhost:5432`.
- pgAdmin: `http://localhost:5050`.

Useful Make targets:

- `make up`: build and start all Docker services detached.
- `make up-minimal`: start Postgres and API only.
- `make up-verbose`: build with visible progress and run foreground.
- `make db-migrate`: run Alembic migrations in the API container.
- `make test`: ensure Postgres is up, migrate, then run backend pytest locally.
- `make test-docker`: build API image and run backend tests in Docker.
- `make retrieval-eval`: run offline retrieval evals.
- `make retrieval-eval-compare`: compare retrieval modes.

## 5. Configuration

Backend settings live in `apps/api/app/core/config.py`. The API loads environment from the repo root `.env` and `apps/api/.env`.

Key config areas:

- **Database**: `DATABASE_URL`, `DATABASE_URL_SYNC`.
- **Auth**: `CLERK_JWKS_URL`, optional `CLERK_ISSUER`, `DEMO_MODE_ENABLED`, `DEMO_KEY`, optional `DEMO_USER_ID`.
- **Storage**: `AWS_REGION`, `S3_BUCKET`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`.
- **PDF limits**: `MAX_PDF_MB`, `MAX_PDF_PAGES`, `MAX_RESUME_PDF_PAGES`.
- **Chunk/retrieval limits**: `MAX_CHUNKS_PER_DOC`, `TOP_K_MAX`, `TOP_N_CANDIDATES`, `MMR_LAMBDA`, `HYBRID_RETRIEVAL_ENABLED`.
- **OpenAI**: `OPENAI_API_KEY`, `OPENAI_EMBEDDING_MODEL`, `OPENAI_EMBEDDING_DIM`, `MODEL_FAST`, `MODEL_HIGH_QUALITY`.
- **LLM budget/caching**: `MAX_LLM_BUDGET_TOKENS`, `REDIS_URL`, `CACHE_TTL_RETRIEVAL_SECONDS`, `CACHE_TTL_EVALUATION_SECONDS`.
- **Plan quotas**: `PLAN_LIMIT_FREE`, `PLAN_LIMIT_PRO`, `PLAN_LIMIT_ENTERPRISE`, `DEMO_MONTHLY_EVALUATION_LIMIT`.
- **Web**: `NEXT_PUBLIC_API_BASE_URL`, `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`, optional `BASIC_AUTH_USER`, `BASIC_AUTH_PASSWORD`.

## 6. Backend Application Shell

The FastAPI entrypoint is `apps/api/app/main.py`.

It creates the app, wires middleware, exposes `/health`, and includes these routers:

- `analyze_fit`
- `fit_history`
- `ask`
- `documents`
- `interview`
- `retrieve`
- `user_resume`

Middleware order matters:

- `CORSMiddleware` allows localhost web origins plus `CORS_ORIGINS`.
- `RateLimitMiddleware` applies in-memory rate limits to expensive routes.
- `DemoGateMiddleware` enforces `x-demo-key` when demo mode is enabled, while still allowing Clerk Bearer auth.

The `/health` endpoint checks DB connectivity with `SELECT 1` and reports whether Clerk is configured.

## 7. Auth and Ownership

Authentication is implemented in `apps/api/app/core/auth.py`.

The API supports two auth modes:

- **Clerk Bearer token**: The frontend obtains a Clerk JWT and sends `Authorization: Bearer <token>`. The API verifies it through Clerk JWKS and creates or loads a `users` row keyed by Clerk user id.
- **Demo mode**: If `DEMO_MODE_ENABLED=true` and `DEMO_KEY` is set, tools such as curl can send `x-demo-key`. This maps to a fixed sandbox user id.

The API rejects requests that mix Bearer auth and `x-demo-key`.

Authorization is mostly ownership-based. `assert_resource_ownership` checks that a loaded resource has `user_id == current_user.id`, returning `403` when a user tries to access another user's resource.

On the frontend, `apps/web/src/lib/api.ts` intentionally does not send client-supplied user ids for identity. It gets auth headers from `apps/web/src/lib/auth.ts`, which is populated by `ClerkAuthProvider`.

## 8. Database Model Overview

The SQLAlchemy models are under `apps/api/app/models`.

Important tables:

- `users`: Clerk id, email, plan, monthly evaluation usage.
- `documents`: Uploaded PDFs or account resume records. Stores status, storage key, page count, domain, extracted JD structure, role profile, competencies, and rubric JSON.
- `interview_sources`: Sources attached to a document: JD, resume, company, or notes.
- `document_chunks`: Chunk text and metadata, embedding vector, generated full-text search vector, section type, quality flags, and source linkage.
- `interview_sessions`: A generated practice session for one user and one JD, including role profile and secondary adaptive-performance metadata.
- `interview_questions`: Generated questions and rubric JSON.
- `interview_answers`: User answers, scores, structured feedback, and evaluation snapshots.
- `interview_retrieval_feedback`: User feedback that retrieved evidence missed the mark.
- `fit_analyses`: Persisted/cached fit-analysis results keyed by user, JD, resume, query fingerprint, and chunk fingerprints.

`document_chunks` is the core RAG table. It has a pgvector HNSW cosine index on `embedding` and a GIN index on the generated `search_vector`, enabling hybrid semantic plus keyword retrieval.

Alembic migrations live in `apps/api/alembic/versions`.

## 9. Storage Model

Storage is abstracted in `apps/api/app/services/storage.py`.

- If S3 bucket and credentials are present, `S3Storage` generates S3 presigned PUT URLs and downloads/deletes S3 objects.
- Otherwise, `LocalStorage` stores files under a local `uploads` directory. In this mode, presigned upload URLs point back to the API's local upload endpoint.

Standard document upload flow:

1. `POST /documents/presign`
2. Client `PUT`s the PDF to returned URL.
3. `POST /documents/confirm`
4. `POST /documents/{document_id}/ingest`

Account resume upload has its own flow under `/user/resume/*` and uses a fixed key: `users/{user_id}/resume.pdf`.

## 10. Document Ingestion Pipeline

The main ingestion function is `run_ingestion` in `apps/api/app/services/ingestion.py`.

For a JD or general PDF:

1. Load the `Document` row.
2. Download PDF bytes from storage.
3. Count pages and enforce `MAX_PDF_PAGES`.
4. Extract per-page text with PyMuPDF.
5. Normalize text and detect document domain.
6. If it is a job description, extract JD structure and use section-aware JD chunking.
7. Otherwise, use generic page chunking.
8. Infer a role profile from the full text.
9. Create OpenAI embeddings for every chunk.
10. Delete old chunks for reingestion and insert new `document_chunks` rows.
11. Create or reuse the JD `interview_sources` row.
12. Mark the document `ready` or `failed`.
13. For job descriptions, run additional extraction such as competencies and rubric metadata.

The document status progression is generally:

`pending` -> `uploaded` -> `processing` -> `ready` or `failed`.

Account resume ingestion is in `apps/api/app/services/source_ingestion.py`. It validates that the PDF looks like a resume, caps resume pages more strictly, chunks and embeds it, stores a `resume` source, extracts resume profile metadata, and deletes the uploaded object after ingestion.

## 11. Retrieval Architecture

Retrieval code lives in `apps/api/app/services/retrieval`.

The public production entry is `retrieve_chunks`. The lower-level mode-aware function is `retrieve_chunks_for_mode` in `orchestration.py`.

Supported retrieval modes:

- **Semantic**: Uses pgvector cosine similarity, then MMR for diversity.
- **Keyword**: Uses PostgreSQL full-text search over generated `search_vector`, optionally followed by MMR when an embedding is available.
- **Hybrid**: Runs semantic and keyword candidates, merges/dedupes them, then applies MMR. If keyword retrieval fails, it falls back to semantic-only.

Important retrieval behaviors:

- Production LLM paths enforce a maximum returned chunk budget.
- When an additional document is included, such as a resume alongside a JD, retrieval splits slots between the primary and additional documents and reallocates if one side lacks evidence.
- `suggest_section_filters` maps queries to JD section types such as compensation or qualifications. `/ask` retries without the filter if the filtered pass returns no chunks.
- Chunk payloads preserve source metadata so downstream prompts can separate JD and resume evidence.

Key files:

- `orchestration.py`: mode selection, hybrid flow, production chunk budget.
- `semantic.py`: vector retrieval SQL.
- `keyword_db.py`: full-text retrieval SQL.
- `merge_mmr.py`: score normalization, candidate merge, slot allocation, MMR.
- `payloads.py`: final chunk dictionaries for citations and prompts.
- `embeddings.py`: query embedding creation.
- `keyword_query.py`: section filter and keyword normalization helpers.

## 12. Q&A Flow

The Q&A endpoint is `POST /ask` in `apps/api/app/routers/ask.py`.

Flow:

1. Validate authenticated user and document ownership.
2. Require the document to be `ready`.
3. Validate any `additional_document_ids`.
4. Embed the user's question.
5. Retrieve chunks with a default top-k of 6, optionally searching additional documents such as a resume.
6. For job descriptions, infer section filters from the question and retry without filters on no-hit.
7. Call `generate_grounded_answer` in `apps/api/app/services/qa.py`.
8. Return an `answer` string and citations.

The Q&A prompt is strict. It asks the model to act like a hiring analyst, compare JD excerpts against resume excerpts, avoid loose analogies, and return JSON when possible.

The frontend parses structured answer JSON with `parseAskStructuredAnswer` in `apps/web/src/lib/api.ts` and displays it through `AskAnswerDisplay`.

## 13. Raw Retrieval Endpoint

`POST /retrieve` exposes the retrieval layer without LLM summarization.

It returns chunks with:

- text
- score
- source type
- source title
- page
- chunk id
- low-signal flag
- section type

This is useful for debugging retrieval quality, inspecting section filters, and testing citation behavior.

## 14. Secondary Fit Analysis

Fit analysis is a secondary prep surface implemented by `POST /analyze-fit` and `GET /analyze-fit/latest` in `apps/api/app/routers/analyze_fit.py`.

The user supplies a JD document id and a resume document id. The API:

1. Validates ownership and readiness for both documents.
2. Computes a query fingerprint from the optional question.
3. Computes chunk fingerprints for the JD and resume.
4. Checks `fit_analyses` for a cached result with matching fingerprints.
5. If no cache hit exists, embeds the normalized fit question.
6. Retrieves JD plus resume chunks.
7. Augments resume education evidence when needed.
8. Calls `analyze_fit` in `apps/api/app/services/analyze_fit_service.py`.
9. Persists and returns matches, gaps, fit score, summary, and recommendations.

The cache invalidates naturally when the JD or resume is reingested because chunk fingerprints change.

The frontend exposes this through `Analyze Fit` on `apps/web/src/app/documents/[id]/page.tsx` when the document is a ready JD and the user has an account resume.

## 15. Interview Practice

The interview subsystem is under `apps/api/app/routers/interview` and `apps/api/app/services/interview`.

Registered routes include:

- `POST /interview/generate`
- `POST /interview/evaluate`
- `GET /interview/sessions`
- `GET /interview/sessions/{session_id}`
- `GET /interview/questions/{question_id}`
- `GET /interview/analytics/overview`
- `GET /interview/{session_id}/analytics`
- `POST /interview/retrieval-feedback`

Question generation flow:

1. Require a ready `job_description` document.
2. Validate optional setup overrides such as domain, seniority, and question mix preset.
3. Create an `interview_sessions` row.
4. Generate questions using document role profile, competencies, and retrieved evidence.
5. Store `interview_questions` with rubric JSON, focus areas, competency ids, and evidence chunk ids.

Answer evaluation flow:

1. Load the question and session.
2. Enforce ownership.
3. Consume monthly evaluation quota for the user's plan.
4. Retrieve evaluation evidence from the JD and optional account resume.
5. Call the evaluation service.
6. Normalize rubric scores and compute score breakdown.
7. Store an `interview_answers` row.
8. Recompute session `performance_profile` for secondary adaptive hints.

Frontend routes:

- `apps/web/src/app/interview/setup/[id]/page.tsx`: setup screen for a ready JD.
- `apps/web/src/app/interview/session/[sessionId]/page.tsx`: live session page.
- `apps/web/src/components/interview/InterviewSessionView.tsx`: main interview interaction UI.
- `apps/web/src/components/interview/EvaluationDrawer.tsx`: answer feedback display.
- `apps/web/src/components/interview/ReferenceDrawer.tsx`: evidence/citation display.

## 16. Secondary Prep Surfaces

The documents router is large because it owns several prep surfaces around a JD. These exist in the implementation, but they are not the lean product's primary promise.

Besides CRUD and ingestion, `apps/api/app/routers/documents.py` includes:

- Source listing.
- Adding pasted text sources.
- Presigning and ingesting per-document resume sources.
- Adding company information from URL.
- Chunk stats.
- Gap analysis.
- Study plan generation.

The source types are represented by `interview_sources`:

- `jd`
- `resume`
- `company`
- `notes`

These sources produce chunks tied to the same document. They can enrich retrieval, but the core scope should continue to work with just the JD plus the user's account resume.

## 17. Account Resume

The account-level resume routes are in `apps/api/app/routers/user_resume.py`.

Endpoints:

- `GET /user/resume`
- `POST /user/resume/presign`
- `POST /user/resume`
- `DELETE /user/resume`
- `POST /user/resume/ask`

There is one account resume per user. It is stored as a special `documents` row with `doc_domain = "user_resume"` and is hidden from normal document listings.

The resume is reused across job descriptions for interview evidence and secondary fit analysis. The dashboard's `AccountResumeSection` manages this flow in the frontend.

## 18. Frontend Architecture

The web app is in `apps/web`.

Framework and libraries:

- Next.js 15 App Router.
- React 19.
- Clerk for auth.
- TanStack React Query for server state.
- Tailwind CSS for styling.
- Framer Motion for motion.
- Recharts for analytics.

Important app routes:

- `src/app/page.tsx`: landing page; signed-in users redirect to `/dashboard`.
- `src/app/layout.tsx`: global providers and chrome.
- `src/app/sign-in/[[...sign-in]]/page.tsx`: Clerk sign-in.
- `src/app/sign-up/[[...sign-up]]/page.tsx`: Clerk sign-up.
- `src/app/dashboard/page.tsx`: main dashboard with JD uploads and account resume.
- `src/app/dashboard/analytics/page.tsx`: secondary interview analytics dashboard.
- `src/app/documents/[id]/page.tsx`: document detail page with Interview plus supporting Ask, Analyze Fit, and Study Plan tabs.
- `src/app/interview/setup/[id]/page.tsx`: session setup.
- `src/app/interview/session/[sessionId]/page.tsx`: interview session.
- `src/app/resume/coach/page.tsx`: account resume coaching.

Important support files:

- `src/lib/api.ts`: typed API client and response parsers.
- `src/lib/auth.ts`: token provider consumed by the API client.
- `src/middleware.ts`: Clerk-protected route handling, with Basic Auth fallback if Clerk publishable key is absent.
- `src/providers/QueryProvider.tsx`: React Query client config.
- `src/components/ClerkAuthProvider.tsx`: wires Clerk token retrieval into the API client.
- `src/contexts/LibraryContext.tsx`: document library modal state.
- `src/components/LibraryModal.tsx`: quick document switcher.
- `src/components/AppChrome.tsx` and `src/components/AppHeader.tsx`: shared app shell.

## 19. Frontend Data Fetching Pattern

The frontend generally follows this pattern:

1. `src/lib/api.ts` defines the fetch function and TypeScript types.
2. A hook in `src/hooks` wraps it with React Query.
3. A page or component consumes the hook and renders loading/error/success states.
4. Mutations invalidate query keys from `src/lib/query-keys.ts`.

Representative hooks:

- `use-documents`
- `use-user-resume`
- `use-ask-question`
- `use-analyze-fit`
- `use-study-plan`
- `use-interview-setup`
- `use-interview-session`
- `use-interview-evaluate`
- `use-interview-analytics`

## 20. Rate Limits and Quotas

Route rate limiting is in `apps/api/app/core/rate_limit.py`.

Current route limits include:

- `/ask`: 10 per hour.
- `/user/resume/ask`: same bucket as ask.
- `/analyze-fit`: 10 per hour.
- `/documents/{id}/study-plan`: 10 per hour.
- `/fit-history` and `/analyze-fit/latest`: 120 per hour.
- `/retrieve`: 60 per hour.
- `/documents/{id}/ingest`: 3 per day.
- `/documents/presign`: 10 per day.
- `/documents/confirm`: 20 per day.

The rate limiter is in-memory and keyed by IP in middleware. It is suitable for single-instance development but would need Redis or another shared store for horizontally scaled production.

Interview answer evaluation also has monthly plan quotas managed by `apps/api/app/services/evaluation_usage.py`, using fields on the `users` table.

## 21. Caching and Token Budgeting

`apps/api/app/core/llm_cache.py` provides LLM-adjacent caches:

- Retrieval chunks keyed by document and question hash.
- Session JD pool keyed by interview session.
- Evaluation payloads keyed by document, question, answer, rubric fingerprint, and evaluation mode.

If `REDIS_URL` is configured, it uses Redis. Otherwise it falls back to in-memory TTL dictionaries.

`apps/api/app/services/token_budget.py` uses character-based token estimates to keep Q&A prompts under `MAX_LLM_BUDGET_TOKENS`. It trims chunks and clamps completion tokens before calling OpenAI.

Fit analysis has its own compression and budget logic in `analyze_fit_service.py`.

## 22. Tests and CI

Backend tests are under `apps/api/tests`.

They cover:

- Health and dependencies.
- Auth and demo mode.
- Rate limits.
- Chunking and document domain behavior.
- Retrieval modes and hybrid retrieval.
- Q&A.
- Fit analysis and fit cache.
- Gap analysis.
- Interview question/evaluation/scoring paths.
- Token budget behavior.
- Evaluation usage and model routing.
- Offline retrieval eval cases.

Backend test command:

```bash
cd apps/api && pytest -v
```

Frontend tests are colocated under `apps/web/src` and run with Vitest.

Frontend test command:

```bash
cd apps/web && npm run test
```

CI runs:

- Backend on Python 3.11 with a pgvector Postgres service, Alembic migrations, then pytest.
- Frontend on Node 22 with `npm ci`, web tests, then Next build.

## 23. Offline Retrieval Evals

Offline eval tooling lives in `apps/api/evals/retrieval`.

It compares semantic, keyword, and hybrid retrieval modes against fixture cases. The production API does not import this package directly; it is invoked through:

```bash
make retrieval-eval
make retrieval-eval-compare
```

This is the right place to validate changes to retrieval ranking, section filtering, query expansion, and hybrid merge behavior.

## 24. End-to-End User Flow

A typical happy path:

1. User signs in through Clerk.
2. User uploads an account resume on the dashboard.
3. User uploads a JD PDF.
4. Web calls presign, uploads the PDF, confirms it, then starts ingestion.
5. API extracts text, chunks it, embeds chunks, infers role profile, stores competencies, and marks the document ready.
6. User opens the JD detail page.
7. User starts interview setup from the ready JD.
8. Interview generation creates a session and questions grounded in JD evidence.
9. User answers questions in the session UI.
10. API retrieves JD/resume evidence, evaluates answers, and stores feedback.

Supporting flows such as cited Q&A, fit analysis, study plans, and analytics are available in the repo, but should not be treated as the main product path.

## 25. Where to Start When Changing the Repo

For backend app wiring:

- Start with `apps/api/app/main.py`.
- Then inspect the router under `apps/api/app/routers`.
- Follow the call into `apps/api/app/services`.
- Check models in `apps/api/app/models` and migrations in `apps/api/alembic/versions`.

For retrieval changes:

- Start with `apps/api/app/services/retrieval/orchestration.py`.
- Review `semantic.py`, `keyword_db.py`, `merge_mmr.py`, and `payloads.py`.
- Run focused tests plus retrieval evals.

For ingestion changes:

- Start with `apps/api/app/services/ingestion.py` for JD/general docs.
- Use `apps/api/app/services/source_ingestion.py` for resume/company/notes sources.
- Check chunking helpers and tests before changing thresholds.

For interview changes:

- Start with `apps/api/app/routers/interview/generate_evaluate.py`.
- Then inspect `apps/api/app/services/interview/questions.py`, `runtime.py`, `evaluation.py`, `evidence.py`, and `interview_scoring.py`.
- Verify UI expectations in `apps/web/src/components/interview`.

For frontend product changes:

- Start with the route file under `apps/web/src/app`.
- Use `apps/web/src/lib/api.ts` to find the API function.
- Use `apps/web/src/hooks` to understand query/mutation behavior.
- Follow into shared components under `apps/web/src/components`.

## 26. Notable Engineering Notes

- The repo has strong separation between HTTP routers and service code, but some routers, especially `documents.py`, are intentionally large because they own several secondary prep workflows.
- Retrieval is the most important shared backend subsystem. Many product features depend on chunk payload shape and source metadata.
- The app has both per-JD supplemental sources and an account-level resume model. For the lean product path, the account resume under `/user/resume` is the important one.
- The frontend assumes API identity comes from Clerk tokens. Avoid reintroducing `user_id` fields as trusted client input.
- Local storage is a development fallback. S3 CORS and presigned PUT behavior matter for production.
- Rate limiting and cache fallback are process-local when Redis is absent, so multi-instance deployments need shared infrastructure for consistency.
- pgvector embedding dimensions must match the configured OpenAI embedding dimension and database vector column.

