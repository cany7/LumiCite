from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 64

    chunk_size: int = 512
    chunk_overlap: int = 50

    retrieval_top_k: int = 5
    distance_threshold: float = 1.2
    retrieval_mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    llm_backend: Literal["gemini", "ollama"] = "gemini"
    gemini_model: str = "gemini-2.5-flash"
    gcp_project: str = ""
    gcp_location: str = "us-central1"
    gcp_credentials_path: str = ""
    ollama_base_url: str = "http://localhost:11434"

    ingest_workers: int = 4
    vector_db: str = "faiss"

    log_format: Literal["json", "console"] = "console"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
