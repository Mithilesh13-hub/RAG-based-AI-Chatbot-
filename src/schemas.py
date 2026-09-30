"""Pydantic request/response models for the API."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    query: str = Field(..., description="The user's question.", max_length=2000)

    @field_validator("query")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be empty.")
        return value


class RetrievedChunk(BaseModel):
    content: str
    page: Optional[int] = None
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    source: Optional[str] = None
    chunk_id: Optional[str] = None


class Citation(BaseModel):
    source: str
    page: Optional[int] = None


class ChatResponse(BaseModel):
    final_answer: str
    retrieved_context: List[RetrievedChunk]
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    citations: List[Citation]
