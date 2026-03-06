from __future__ import annotations

import hashlib
import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from src.core.logging import get_logger
from src.core.paths import find_project_root

logger = get_logger(__name__)

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class BM25Index:
    def __init__(self, index_path: Path | None = None, chunks_path: Path | None = None) -> None:
        root = find_project_root()
        self.index_path = index_path or (root / "data" / "bm25_index.pkl")
        self.chunks_path = chunks_path or (root / "data" / "JSON" / "chunks.jsonl")
        self.metadata_path = root / "data" / "bm25_index.meta.json"
        self.bm25: BM25Okapi | None = None
        self.records: list[dict[str, Any]] = []
        self.metadata: dict[str, Any] = {}

    def is_loaded(self) -> bool:
        return self.bm25 is not None and bool(self.records)

    def load(self) -> bool:
        if not self.index_path.exists():
            return False
        if not self.metadata_path.exists():
            logger.warning("bm25_metadata_missing", path=str(self.metadata_path))
            return False

        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if self.chunks_path.exists():
            current_hash = _file_sha256(self.chunks_path)
            if metadata.get("chunks_hash") != current_hash:
                logger.warning(
                    "bm25_chunks_changed",
                    stored_hash=metadata.get("chunks_hash", ""),
                    current_hash=current_hash,
                )
                return False

        with self.index_path.open("rb") as handle:
            payload = pickle.load(handle)
        self.bm25 = payload["bm25"]
        self.records = payload["records"]
        self.metadata = metadata
        logger.info("bm25_index_loaded", path=str(self.index_path), rows=len(self.records))
        return True

    def build(self) -> None:
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {self.chunks_path}")

        corpus: list[list[str]] = []
        records: list[dict[str, Any]] = []
        with self.chunks_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                text = str(payload.get("text", "")).strip()
                if not text:
                    continue
                tokens = tokenize(text)
                if not tokens:
                    continue
                corpus.append(tokens)
                records.append(payload)

        if not corpus:
            raise ValueError("No chunk text found to build BM25 index")

        self.bm25 = BM25Okapi(corpus)
        self.records = records
        self.metadata = {
            "chunks_hash": _file_sha256(self.chunks_path),
            "rows": len(self.records),
        }
        self.save()
        logger.info("bm25_index_built", path=str(self.index_path), rows=len(self.records))

    def save(self) -> None:
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built")

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with self.index_path.open("wb") as handle:
            pickle.dump({"bm25": self.bm25, "records": self.records}, handle)
        self.metadata_path.write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        logger.info("bm25_index_saved", path=str(self.index_path), rows=len(self.records))

    def ensure_loaded(self) -> None:
        if self.is_loaded():
            return
        if self.load():
            return
        self.build()

    def search(self, query: str, top_k: int) -> list[tuple[float, dict[str, Any]]]:
        if top_k <= 0:
            return []

        self.ensure_loaded()
        if self.bm25 is None:
            raise RuntimeError("BM25 index not loaded")

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = np.asarray(self.bm25.get_scores(tokens), dtype=float)
        if scores.size == 0:
            return []

        top_indices = np.argsort(scores)[::-1][:top_k]
        results: list[tuple[float, dict[str, Any]]] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0.0:
                continue
            results.append((score, self.records[int(idx)]))
        return results
