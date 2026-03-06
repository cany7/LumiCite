from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.config.settings import get_settings
from src.core.logging import get_logger
from src.core.paths import find_project_root

logger = get_logger(__name__)


def locate_embeddings_file(root_dir: Path) -> Path:
    json_dir = root_dir / "data" / "JSON"
    json_dir.mkdir(parents=True, exist_ok=True)
    return json_dir / "embeddings.jsonl"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FaissStore:
    def __init__(
        self,
        index_path: Path | None = None,
        text_data_path: Path | None = None,
        embeddings_path: Path | None = None,
        embedding_model: str | None = None,
    ) -> None:
        root = find_project_root()
        data_dir = root / "data"
        settings = get_settings()
        self.index_path = index_path or data_dir / "my_faiss.index"
        self.text_data_path = text_data_path or data_dir / "text_data.pkl"
        self.embeddings_path = embeddings_path or locate_embeddings_file(root)
        self.metadata_path = data_dir / "my_faiss.meta.json"
        self.embedding_model = embedding_model or settings.embedding_model
        self.index: faiss.Index | None = None
        self.text_data: list[dict[str, Any]] = []
        self.metadata: dict[str, Any] = {}

    def is_loaded(self) -> bool:
        return self.index is not None and bool(self.text_data)

    def load(self) -> bool:
        if not self.index_path.exists() or not self.text_data_path.exists():
            return False
        if not self.metadata_path.exists():
            logger.warning("faiss_store_metadata_missing", path=str(self.metadata_path))
            return False

        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        expected_model = str(metadata.get("embedding_model", ""))
        if expected_model != self.embedding_model:
            raise RuntimeError(
                f"FAISS index was built with embedding model '{expected_model}', "
                f"but current settings require '{self.embedding_model}'. Re-ingest/rebuild is required."
            )
        if self.embeddings_path.exists():
            current_hash = _file_sha256(self.embeddings_path)
            if metadata.get("embeddings_hash") != current_hash:
                logger.warning(
                    "faiss_store_embeddings_changed",
                    stored_hash=metadata.get("embeddings_hash", ""),
                    current_hash=current_hash,
                )
                return False

        self.index = faiss.read_index(str(self.index_path))
        expected_dim = int(metadata.get("embedding_dim", -1))
        if expected_dim > 0 and self.index.d != expected_dim:
            raise RuntimeError(
                f"FAISS index dimension {self.index.d} does not match stored metadata dimension {expected_dim}."
            )
        with self.text_data_path.open("rb") as handle:
            self.text_data = pickle.load(handle)
        self.metadata = metadata
        logger.info(
            "faiss_store_loaded",
            index=str(self.index_path),
            text_data=str(self.text_data_path),
            rows=len(self.text_data),
        )
        return True

    def build(self, embeddings_path: Path | None = None) -> None:
        source = embeddings_path or self.embeddings_path
        if not source.exists():
            raise FileNotFoundError(f"Embeddings file not found: {source}")

        embeddings: list[list[float]] = []
        text_data: list[dict[str, Any]] = []
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                embeddings.append(record["embedding"])
                text_data.append(
                    {
                        "id": record["id"],
                        "text": record["text"],
                        "metadata": record.get("metadata", {}),
                    }
                )

        if not embeddings:
            raise ValueError("No embeddings found to build FAISS index")

        vectors = np.array(embeddings, dtype="float32")
        self.index = faiss.IndexFlatL2(vectors.shape[1])
        self.index.add(vectors)
        self.text_data = text_data
        self.metadata = {
            "embedding_model": self.embedding_model,
            "embedding_dim": vectors.shape[1],
            "embeddings_hash": _file_sha256(source),
            "vectors": int(self.index.ntotal),
        }
        self.save()
        logger.info(
            "faiss_store_built",
            embeddings_path=str(source),
            dim=vectors.shape[1],
            vectors=int(self.index.ntotal),
        )

    def save(self) -> None:
        if self.index is None:
            raise RuntimeError("FAISS index not built")

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with self.text_data_path.open("wb") as handle:
            pickle.dump(self.text_data, handle)
        self.metadata_path.write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        logger.info(
            "faiss_store_saved",
            index=str(self.index_path),
            text_data=str(self.text_data_path),
            metadata=str(self.metadata_path),
        )

    def ensure_loaded(self) -> None:
        if self.is_loaded():
            return
        if self.load():
            return
        self.build()

    def search(self, query_vec: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            raise RuntimeError("FAISS index not loaded")
        return self.index.search(query_vec, top_k)
