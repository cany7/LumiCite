from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from src.core.schemas import RAGAnswer


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    num_papers: int
    num_chunks: int
    embedding_model: str
    retrieval_modes_available: list[str] = Field(default_factory=list)


class PaperSummary(BaseModel):
    id: str
    title: str
    year: int
    num_chunks: int


class PapersResponse(BaseModel):
    papers: list[PaperSummary] = Field(default_factory=list)
    total: int


class SearchRequest(BaseModel):
    question: str
    top_k: int = 10
    retrieval_mode: Literal["dense", "sparse", "hybrid"] | None = None
    rerank: bool = False


class SearchResult(BaseModel):
    rank: int
    ref_id: str
    score: float
    text: str
    page: int | None = None
    source_file: str
    headings: list[str] = Field(default_factory=list)


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
    llm_backend: Literal["gemini", "ollama"] | None = None


class IngestRequest(BaseModel):
    source: Literal["metadata_csv", "local_dir", "url_list"] = "metadata_csv"
    path: str = ""

    @model_validator(mode="after")
    def validate_path_requirements(self) -> "IngestRequest":
        if self.source == "url_list" and not self.path.strip():
            raise ValueError("path is required when source=url_list")
        return self


class IngestAcceptedResponse(BaseModel):
    status: str
    message: str
    task_id: str


class ErrorDetail(BaseModel):
    code: str
    message: str
    detail: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


QueryResponse = RAGAnswer
