from __future__ import annotations

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Return the repository root by searching upwards for src/ and data/."""
    path = (start or Path(__file__)).resolve()
    if path.is_file():
        path = path.parent

    while True:
        if (path / "src").exists() and (path / "data").exists():
            return path
        if path.parent == path:
            raise FileNotFoundError("Could not find project root containing src/ and data/.")
        path = path.parent


def data_dir(root: Path | None = None) -> Path:
    project_root = root or find_project_root()
    path = project_root / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def metadata_dir(root: Path | None = None) -> Path:
    path = data_dir(root) / "metadata"
    path.mkdir(parents=True, exist_ok=True)
    return path


def chunks_dir(root: Path | None = None) -> Path:
    path = metadata_dir(root) / "chunks"
    path.mkdir(parents=True, exist_ok=True)
    return path


def embeddings_dir(root: Path | None = None) -> Path:
    path = metadata_dir(root) / "embeddings"
    path.mkdir(parents=True, exist_ok=True)
    return path


def faiss_dir(root: Path | None = None) -> Path:
    path = metadata_dir(root) / "faiss"
    path.mkdir(parents=True, exist_ok=True)
    return path


def bm25_dir(root: Path | None = None) -> Path:
    path = metadata_dir(root) / "bm25"
    path.mkdir(parents=True, exist_ok=True)
    return path


def chunks_jsonl_path(root: Path | None = None) -> Path:
    return chunks_dir(root) / "chunks.jsonl"


def embeddings_jsonl_path(root: Path | None = None) -> Path:
    return embeddings_dir(root) / "embeddings.jsonl"


def faiss_index_path(root: Path | None = None) -> Path:
    return faiss_dir(root) / "my_faiss.index"


def faiss_metadata_path(root: Path | None = None) -> Path:
    return faiss_dir(root) / "my_faiss.meta.json"


def faiss_text_data_path(root: Path | None = None) -> Path:
    return faiss_dir(root) / "text_data.pkl"


def bm25_index_path(root: Path | None = None) -> Path:
    return bm25_dir(root) / "bm25_index.pkl"


def bm25_metadata_path(root: Path | None = None) -> Path:
    return bm25_dir(root) / "bm25_index.meta.json"


def manifest_path(root: Path | None = None) -> Path:
    return data_dir(root) / "manifest.json"


def rag_log_path(root: Path | None = None) -> Path:
    return (root or find_project_root()) / "rag.log"


def model_cache_dir(root: Path | None = None) -> Path:
    path = data_dir(root) / "model_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def huggingface_cache_dir(root: Path | None = None) -> Path:
    path = model_cache_dir(root) / "huggingface"
    path.mkdir(parents=True, exist_ok=True)
    return path


def huggingface_hub_cache_dir(root: Path | None = None) -> Path:
    path = huggingface_cache_dir(root) / "hub"
    path.mkdir(parents=True, exist_ok=True)
    return path


def transformers_cache_dir(root: Path | None = None) -> Path:
    path = huggingface_cache_dir(root) / "transformers"
    path.mkdir(parents=True, exist_ok=True)
    return path


def sentence_transformers_cache_dir(root: Path | None = None) -> Path:
    path = model_cache_dir(root) / "sentence_transformers"
    path.mkdir(parents=True, exist_ok=True)
    return path


def torch_cache_dir(root: Path | None = None) -> Path:
    path = model_cache_dir(root) / "torch"
    path.mkdir(parents=True, exist_ok=True)
    return path


def mineru_cache_dir(root: Path | None = None) -> Path:
    path = model_cache_dir(root) / "mineru"
    path.mkdir(parents=True, exist_ok=True)
    return path


def mineru_config_path(root: Path | None = None) -> Path:
    return mineru_cache_dir(root) / "mineru.json"


def mineru_ready_marker_path(root: Path | None = None) -> Path:
    return mineru_cache_dir(root) / ".models_ready"


def intermediate_root(root: Path | None = None) -> Path:
    path = data_dir(root) / "intermediate" / "mineru"
    path.mkdir(parents=True, exist_ok=True)
    return path


def mineru_output_dir(doc_id: str, root: Path | None = None) -> Path:
    return intermediate_root(root) / doc_id


def assets_root(root: Path | None = None) -> Path:
    path = data_dir(root) / "assets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def doc_assets_dir(doc_id: str, root: Path | None = None) -> Path:
    path = assets_root(root) / doc_id
    path.mkdir(parents=True, exist_ok=True)
    return path
