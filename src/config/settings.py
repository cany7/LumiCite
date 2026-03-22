from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


def normalize_reasoning_effort(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    return normalized or None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 64

    chunk_size: int = 1000
    chunk_overlap: int = 100

    download_timeout_seconds: int = 60
    request_retry_attempts: int = 3
    request_retry_delay_seconds: float = 3.0

    retrieval_top_k: int = 10
    distance_threshold: float = 1.2
    retrieval_mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    llm_backend: Literal["api", "ollama"] = "api"
    api_base_url: str = ""
    api_key: str = ""
    api_model: str = "qwen/qwen3.5-27b"
    api_timeout_seconds: int = 120
    query_explanation_reasoning_effort: str = "none"
    visual_api_base_url: str = ""
    visual_api_key: str = ""
    visual_api_model: str = "qwen/qwen2.5-vl-7b-instruct"
    visual_api_timeout_seconds: int = 120
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:4b"
    ollama_startup_timeout_seconds: int = 180
    ollama_poll_interval_seconds: float = 2.0
    mineru_command: str = "mineru"
    mineru_model_source: str = "huggingface"
    mineru_timeout_seconds: int = 1800

    vector_db: str = "faiss"

    log_format: Literal["json", "console"] = "console"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
