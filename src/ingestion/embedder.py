"""
embedder.py

Reads chunks.json (docling output) and writes embeddings.jsonl using
sentence-transformers/all-MiniLM-L6-v2 (local embeddings).

Output format: JSONL where each line is:
{
  "id": "<chunk_id>",
  "text": "<chunk text>",
  "metadata": { ... },
  "embedding": [ ... ]   # list of floats
}

Usage:
    pip install sentence-transformers tqdm
    python embedder.py
"""
from __future__ import annotations
import argparse
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List
from tqdm import tqdm


# ------------------------------
# 1. Locate project root
# ------------------------------
def find_project_root(project_name: str | None = None) -> Path:

    path = Path(__file__).resolve()
    while True:
        if project_name:
            if path.name.lower() == project_name.lower():
                return path
        elif (path / "src").exists() and (path / "data").exists():
            return path
        if path.parent == path:
            label = project_name or "project root"
            print(f"Warning: {label} not found. Using parent directory.")
            return Path(__file__).resolve().parent
        path = path.parent


# ------------------------------
# 2. Load chunks.json properly
# ------------------------------
def load_chunks(path: Path) -> List[Dict[str, Any]]:
    """
    Loads chunks from chunks.json.
    Handles two main structures:
    1. A single list of chunks: [ {...}, {...} ]
    2. A dictionary of lists: { "file1": [ {...} ], "file2": [ {...} ] }
    """
    raw = json.loads(path.read_text(encoding="utf-8"))


    if isinstance(raw, list):
        print("Detected chunk format: List")
        return raw


    if isinstance(raw, dict):
        print("Detected chunk format: Dictionary of lists")
        all_chunks = []
        for v in raw.values():
            if isinstance(v, list):
                all_chunks.extend(v)

        if all_chunks:
            return all_chunks

    raise ValueError(f"Could not find a valid list of chunks in {path}")


# ------------------------------
# 3. Normalize each chunk
# ------------------------------
def deterministic_id(prefix: str, text: str) -> str:
    h = hashlib.md5((text or "").encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{h}"


def normalize_chunk(raw: Dict[str, Any]) -> Dict[str, Any]:
    cid = raw.get("chunk_id") or raw.get("id") or raw.get("chunkId")
    text = raw.get("text") or raw.get("page_content") or ""

    metadata = {}

    # merge known metadata fields
    for k in ("headings", "page_number", "page", "source_file", "source", "type"):
        if k in raw:
            metadata[k] = raw[k]

    # merge nested metadata
    if "metadata" in raw and isinstance(raw["metadata"], dict):
        metadata.update(raw["metadata"])

    if not cid:
        cid = deterministic_id("chunk", text[:200])

    return {
        "id": cid,
        "text": (text or "").strip(),
        "metadata": metadata,
    }


# ------------------------------
# 4. Load JSONL safely
# ------------------------------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Skipping malformed JSONL line: {line[:80]}...")
    return data


# ------------------------------
# 5. Write embeddings.jsonl
# ------------------------------
def write_jsonl(records: List[Dict[str, Any]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ------------------------------
# 6. Compute embeddings
# ------------------------------
def embed_local(records: List[Dict[str, Any]], model_name: str, batch_size: int) -> List[List[float]]:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    texts = [r["text"] for r in records]

    vectors = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch_texts = texts[i: i + batch_size]
        embs = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
        vectors.extend(embs.tolist())
    return vectors


# ------------------------------
# 7. Main execution
# ------------------------------
def main():
    try:
        root = find_project_root()
        print(f"Project root found at: {root}")
    except RuntimeError as e:
        print(f"Error: {e}. Defaulting to current script's parent directory.")
        root = Path(__file__).resolve().parent

    json_dir = root / "data" / "JSON"
    json_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = json_dir / "chunks.json"
    alt_text_path = json_dir / "alt_text.json"

    all_raw_chunks: List[Dict[str, Any]] = []


    try:
        print("Loading chunks from:", chunks_path)
        chunks_data = load_chunks(chunks_path)
        all_raw_chunks.extend(chunks_data)
    except FileNotFoundError as e:
        print(f"Warning: {e}")


    try:
        print("Loading alt_text from:", alt_text_path)
        alt_text_data = load_chunks(alt_text_path)
        all_raw_chunks.extend(alt_text_data)
    except FileNotFoundError as e:
        print(f"Warning: {e}")

    if not all_raw_chunks:
        raise FileNotFoundError("No chunks were loaded from alt_text.json or chunks.json.")

    print(f"Loaded a total of {len(all_raw_chunks)} raw chunks from all sources.")

    records = [normalize_chunk(c) for c in all_raw_chunks]
    records = [r for r in records if r["text"]]
    print(f"{len(records)} chunks after normalizing and dropping empty text")

    print("Computing embeddings...")
    vectors = embed_local(records, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=64)

    if len(vectors) != len(records):
        raise RuntimeError("Embedding count mismatch")

    for rec, vec in zip(records, vectors):
        rec["embedding"] = vec


    out_path = json_dir / "embeddings.jsonl"
    print("Writing combined_embeddings.jsonl...")
    write_jsonl(records, out_path)

    print(f"✅ Successfully wrote {len(records)} embeddings → {out_path}")


if __name__ == "__main__":
    main()
