"""Stable IDs for CI-backed retrieval eval fixtures.

The UUID must match evals/retrieval/cases/ci_fixture_map.json so local CLI runs can use:

  python -m evals.retrieval.entrypoint --dataset job_description_starter \\
    --compare --fixture-map evals/retrieval/cases/ci_fixture_map.json

after seeding the document (pytest seeds it automatically; for manual runs use the API or SQL).
"""

from __future__ import annotations

import uuid

PLATFORM_ENGINEER_JD_DOCUMENT_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
PLATFORM_ENGINEER_JD_USER_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")
PLATFORM_ENGINEER_JD_FIXTURE_REF = "platform_engineer_jd"

# Shared query/chunk embedding in CI tests: avoids OpenAI while keeping cosine scores stable.
CI_SHARED_EMBEDDING_VALUE = 0.02
