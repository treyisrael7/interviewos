"""Add fit_analyses table for persisted analyze-fit results."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "20250328130000"
down_revision: Union[str, None] = "20250326140000"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "fit_analyses",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("job_description_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("resume_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("fit_score", sa.Integer(), nullable=False),
        sa.Column("matches_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("gaps_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("recommendations_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("summary", sa.Text(), server_default="", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["job_description_id"], ["documents.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["resume_id"], ["documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_fit_analyses_user_created", "fit_analyses", ["user_id", "created_at"])
    op.create_index(
        "ix_fit_analyses_user_jd_resume",
        "fit_analyses",
        ["user_id", "job_description_id", "resume_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_fit_analyses_user_jd_resume", table_name="fit_analyses")
    op.drop_index("ix_fit_analyses_user_created", table_name="fit_analyses")
    op.drop_table("fit_analyses")
