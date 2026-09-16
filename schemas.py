"""Pydantic schemas for structured model output (Phase 2, Phase 3, integration)."""

from pydantic import BaseModel, Field


class StructuredAnswer(BaseModel):
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class QualityScore(BaseModel):
    score: int = Field(ge=1, le=5)
    justification: str


class RAGAnswer(BaseModel):
    answer: str
    citations: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


class RewrittenQuery(BaseModel):
    query: str
