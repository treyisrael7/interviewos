"""Typed schema for internal retrieval evaluation datasets."""

from __future__ import annotations

import uuid

from pydantic import BaseModel, Field, field_validator, model_validator


class RetrievalEvalCase(BaseModel):
    """One retrieval expectation for a document fixture or concrete document."""

    id: str = Field(..., min_length=1)
    document_id: uuid.UUID | None = None
    fixture_ref: str | None = None
    query: str = Field(..., min_length=1)
    expected_chunk_ids: list[str] = Field(default_factory=list)
    expected_content_substrings: list[str] = Field(default_factory=list)
    expected_section_types: list[str] = Field(default_factory=list)
    expected_source_types: list[str] = Field(default_factory=list)
    top_k: int = Field(default=6, ge=1, le=50)
    notes: str | None = None

    @field_validator(
        "id",
        "fixture_ref",
        "query",
        "notes",
        mode="before",
    )
    @classmethod
    def _strip_optional_strings(cls, value: object) -> object:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @field_validator(
        "expected_chunk_ids",
        "expected_content_substrings",
        "expected_section_types",
        "expected_source_types",
        mode="before",
    )
    @classmethod
    def _normalize_string_lists(cls, value: object) -> object:
        if value is None:
            return []
        if not isinstance(value, list):
            raise TypeError("Expected a list of strings")

        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, str):
                raise TypeError("Expected a list of strings")
            stripped = item.strip()
            if stripped and stripped not in seen:
                seen.add(stripped)
                normalized.append(stripped)
        return normalized

    @model_validator(mode="after")
    def _validate_case_shape(self) -> "RetrievalEvalCase":
        if not self.document_id and not self.fixture_ref:
            raise ValueError("Each eval case must set either document_id or fixture_ref")

        has_expectation = any(
            [
                self.expected_chunk_ids,
                self.expected_content_substrings,
                self.expected_section_types,
                self.expected_source_types,
            ]
        )
        if not has_expectation:
            raise ValueError(
                "Each eval case must include at least one expected_* field so the case is measurable"
            )
        return self


class RetrievalEvalDataset(BaseModel):
    """A named collection of retrieval eval cases loaded from JSON."""

    version: int = Field(default=1, ge=1)
    dataset: str = Field(..., min_length=1)
    description: str | None = None
    cases: list[RetrievalEvalCase] = Field(default_factory=list)

    @field_validator("dataset", "description", mode="before")
    @classmethod
    def _strip_dataset_strings(cls, value: object) -> object:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @model_validator(mode="after")
    def _validate_ids_unique(self) -> "RetrievalEvalDataset":
        ids = [case.id for case in self.cases]
        duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
        if duplicates:
            raise ValueError(f"Duplicate eval case ids are not allowed: {', '.join(duplicates)}")
        return self
