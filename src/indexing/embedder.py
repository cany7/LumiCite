'''
this file embedds the cleaned and chunked text into vector embeddings

wraps the embedding model that is used to convert chunked text into vectors
'''

# embedding_generator.py
import json
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

def load_json(json_path: Path) -> List[Dict]:
    """Load list of chunks from JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def save_json(data: List[Dict], save_path: Path):
    """Save list of chunks (with embeddings) to JSON."""
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def generate_embeddings_from_json(
    json_path: Path,
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    save_path: Path | None = None,
):
    """
    Generate embeddings for 'content' in a JSON file and append results.

    Args:
        json_path: Path to input JSON (with 'content' field)
        model_id: Model name from Sentence Transformers
        save_path: Optional output JSON path with embeddings

    Returns:
        List[Dict]: Original data with added 'embedding' field
    """
    print(f"📂 Loading JSON file: {json_path}")
    data = load_json(json_path)
    print(f"Loaded {len(data)} text chunks")

    # --- Extract text list ---
    texts = [item["content"] for item in data if "content" in item]

    # --- Encode using SentenceTransformer ---
    print(f"🔍 Loading embedding model: {model_id}")
    model = SentenceTransformer(model_id)

    print(f"🧠 Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)

    # --- Attach embeddings back to JSON data ---
    for item, emb in zip(data, embeddings):
        item["embedding"] = emb.tolist()

    print(f"✅ Embeddings generated for {len(data)} chunks")

    # --- Optional save ---
    if save_path:
        save_json(data, save_path)
        print(f"💾 Saved to: {save_path}")

    return data
