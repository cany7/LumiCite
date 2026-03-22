from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.core.schemas import RAGAnswer, SearchResult


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    num_papers: int
    num_chunks: int
    embedding_model: str
    retrieval_modes_available: list[str] = Field(default_factory=list)


class PaperSummary(BaseModel):
    doc_id: str
    num_chunks: int


class PapersResponse(BaseModel):
    papers: list[PaperSummary] = Field(default_factory=list)
    total: int


class SearchRequest(BaseModel):
    question: str
    top_k: int = 10
    retrieval_mode: Literal["dense", "sparse", "hybrid"] | None = None
    rerank: bool = False


class SearchResponse(BaseModel):
    results: list[SearchResult] = Field(default_factory=list)
    retrieval_latency_ms: float
    retrieval_mode: str
    total_results: int


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    retrieval_mode: Literal["dense", "sparse", "hybrid"] | None = None
    rerank: bool = True
    llm_backend: Literal["api", "ollama"] | None = None
    llm_model: str | None = None


class ErrorDetail(BaseModel):
    code: str
    message: str
    detail: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


QueryResponse = RAGAnswer
