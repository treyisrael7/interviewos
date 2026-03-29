"""Add cache fingerprint columns and index for fit_analyses."""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20250328140000"
down_revision: Union[str, None] = "20250328130000"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "fit_analyses",
        sa.Column("query_fingerprint", sa.String(length=64), server_default="", nullable=False),
    )
    op.add_column(
        "fit_analyses",
        sa.Column("jd_chunk_fingerprint", sa.String(length=64), server_default="", nullable=False),
    )
    op.add_column(
        "fit_analyses",
        sa.Column("resume_chunk_fingerprint", sa.String(length=64), server_default="", nullable=False),
    )
    op.create_index(
        "ix_fit_analyses_cache_lookup",
        "fit_analyses",
        [
            "user_id",
            "job_description_id",
            "resume_id",
            "query_fingerprint",
            "jd_chunk_fingerprint",
            "resume_chunk_fingerprint",
        ],
    )
    op.alter_column("fit_analyses", "query_fingerprint", server_default=None)
    op.alter_column("fit_analyses", "jd_chunk_fingerprint", server_default=None)
    op.alter_column("fit_analyses", "resume_chunk_fingerprint", server_default=None)


def downgrade() -> None:
    op.drop_index("ix_fit_analyses_cache_lookup", table_name="fit_analyses")
    op.drop_column("fit_analyses", "resume_chunk_fingerprint")
    op.drop_column("fit_analyses", "jd_chunk_fingerprint")
    op.drop_column("fit_analyses", "query_fingerprint")
