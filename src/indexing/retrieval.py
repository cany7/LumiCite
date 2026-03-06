from pathlib import Path
import json
import faiss
import numpy as np
import pickle
import time
from sentence_transformers import SentenceTransformer
from typing import Dict, Any
from src.config.settings import get_settings
from src.core.logging import get_logger, timed
from src.core.paths import find_project_root

logger = get_logger(__name__)


@timed("retrieve")
def get_chunks(question: str, num_chunks: int = 3) -> Dict[int, Dict[str, Any]]:
    """
    High-level helper that:
      1. Finds the project root
      2. Loads (or builds) the FAISS index + text metadata
      3. Runs a similarity search for `question`
      4. Returns a dictionary-of-dicts with chunk text, paper id, and rank.

    Returns
    -------
    {
      1: {"chunk": "...", "paper": "jeff2020", "rank": 1},
      2: {"chunk": "...", "paper": "jeff2020", "rank": 2},
      ...
    }
    """
    # ----------------------------
    # Lazy one-time initialization
    # ----------------------------
    settings = get_settings()
    if not hasattr(get_chunks, "_initialized"):
        # --- locate project root ---
        try:
            root_dir = find_project_root()
        except FileNotFoundError as e:
            logger.info(f"Fatal Error: {e}")
            logger.info("Falling back to the current working directory as a last resort.")
            root_dir = Path.cwd()
            if not root_dir.exists():
                raise FileNotFoundError(
                    f"Error: Hardcoded fallback ROOT_DIR does not exist: {root_dir}"
                )

        # cache root dir on the function
        get_chunks._root_dir = root_dir

        # --- set up paths (inlined from your main block + locate_embeddings_file) ---
        data_dir = root_dir / "data"
        index_file = data_dir / "my_faiss.index"
        text_data_file = data_dir / "text_data.pkl"

        json_dir = root_dir / "data" / "JSON"
        json_dir.mkdir(parents=True, exist_ok=True)
        embeddings_file = json_dir / "embeddings.jsonl"

        get_chunks._data_dir = data_dir
        get_chunks._index_file = index_file
        get_chunks._text_data_file = text_data_file
        get_chunks._embeddings_file = embeddings_file

        # --- load SentenceTransformer model (same as in FaissSearch.__init__) ---
        model_name = settings.embedding_model
        logger.info(f"Loading Sentence Transformer model ({model_name})...")
        try:
            model = SentenceTransformer(model_name)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.info(f"Error: Failed to load model '{model_name}': {e}")
            raise
        get_chunks._model = model

        # --- load or build FAISS index + text_data (inlined from FaissSearch methods) ---
        index = None
        text_data = []

        if index_file.exists() and text_data_file.exists():
            logger.info(f"Loading existing index from {index_file.parent}...")
            try:
                index = faiss.read_index(str(index_file))
                with open(text_data_file, "rb") as f:
                    text_data = pickle.load(f)
                logger.info("FAISS index and text data loaded successfully.")
            except Exception as e:
                logger.info(f"Error loading index or text data: {e}")
                index = None
                text_data = []

        if index is None or not text_data:
            logger.info("Existing index not found or failed to load; building a new one...")
            if not embeddings_file.exists():
                raise FileNotFoundError(
                    f"Fatal Error: Cannot build index because {embeddings_file} is missing."
                )

            logger.info(f"Loading embeddings from {embeddings_file} to build FAISS index...")
            embeddings_list = []
            text_data = []

            with open(embeddings_file, "r", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    embeddings_list.append(record["embedding"])
                    text_data.append(
                        {
                            "id": record["id"],
                            "text": record["text"],
                            "metadata": record["metadata"],
                        }
                    )

            if not embeddings_list:
                raise ValueError("No embeddings found in the input file.")

            d = len(embeddings_list[0])
            embeddings = np.array(embeddings_list).astype("float32")

            index = faiss.IndexFlatL2(d)
            index.add(embeddings)
            logger.info(f"FAISS index built (contains {index.ntotal} vectors).")

            # save for future runs
            try:
                data_dir.mkdir(parents=True, exist_ok=True)
                faiss.write_index(index, str(index_file))
                with open(text_data_file, "wb") as f_out:
                    pickle.dump(text_data, f_out)
                logger.info(f"FAISS index saved to: {index_file}")
                logger.info(f"Text data saved to: {text_data_file}")
            except Exception as e:
                logger.info(f"Error saving index or text data: {e}")

        get_chunks._index = index
        get_chunks._text_data = text_data
        get_chunks._initialized = True

    # -----------------
    # Actual search step
    # -----------------
    model = get_chunks._model
    index = get_chunks._index
    text_data = get_chunks._text_data

    if index is None or model is None or not text_data:
        raise RuntimeError("Index, model, or text data not properly initialized.")

    if num_chunks <= 0:
        return {}

    logger.info(f"Searching FAISS for: '{question[:50]}...'")
    start_time = time.time()

    query_embedding = model.encode([question], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_embedding, num_chunks)

    logger.info("\nSquared L2 distances for top results:")
    for rank in range(min(num_chunks, len(indices[0]))):
        logger.info(f"  Rank {rank + 1}: {distances[0][rank]}")

    # --- CUT OFF: Reject if top distance is too large ---
    best_distance_threshold = settings.distance_threshold
    best_dist = float(distances[0][0])

    if best_dist > best_distance_threshold:
        logger.info(
            f"\nBest distance {best_dist:.4f} exceeds threshold "
            f"{best_distance_threshold}. Returning None."
        )
        return None

    # -----------------
    # Build return value
    # -----------------
    results: Dict[int, Dict[str, Any]] = {}
    for rank, idx in enumerate(indices[0][:num_chunks], start=1):
        if 0 <= idx < len(text_data):
            rec = text_data[idx]
            meta = rec.get("metadata", {}) or {}

            # Try a few reasonable keys for the paper identifier
            paper_id = (
                meta.get("paper_id")
                or meta.get("source_id")
                or meta.get("paper")
                or meta.get("id")
                or rec.get("id")
                or "unknown"
            )

            results[rank] = {
                "chunk": rec.get("text", ""),
                "paper": paper_id,
                "rank": rank,
            }

    logger.info(f"\nRetrieved {len(results)} chunks in {time.time() - start_time:.4f} seconds.")
    return results

import requests
from typing import Dict, Any, Optional


def rag_gemma3_answer(
    question: str,
    chunks_dict: Optional[Dict[int, Dict[str, Any]]],
    model: str = "gemma2:2b",
    max_context_chars: int = 8000,
    temperature: float = 0.2,
) -> str:
    """
    RAG-style generator that:
      - Takes a question and a dict-of-dicts of retrieved chunks:
            {
                1: { "chunk": "...", "paper": "...", "rank": 1 },
                2: { "chunk": "...", "paper": "...", "rank": 2 },
                3: { "chunk": "...", "paper": "...", "rank": 3 },
            }
      - Builds a context string from these chunks (trimmed to max_context_chars)
      - Calls a local Gemma3 model via Ollama's /api/generate endpoint
      - Returns the model's answer as a string.
    """

    # 1. Fallback if we don't trust the retrieval stage
    if not chunks_dict:
        fallback = "Unable to sufficiently answer query"
        logger.info(fallback)
        return fallback

    # 2. Build context from chunks, sorted by rank
    #    (assumes keys are ranks 1..k, but we sort to be safe)
    ordered_items = sorted(chunks_dict.items(), key=lambda kv: kv[0])

    context_blocks = []
    for rank, info in ordered_items:
        chunk_text = info.get("chunk", "")
        paper = info.get("paper", "unknown")
        context_blocks.append(
            f"[Rank {rank} | Paper: {paper}]\n{chunk_text}".strip()
        )

    full_context = "\n\n".join(context_blocks)

    # 3. Enforce a rough context-window limit by characters
    #    (simple but effective for a small number of chunks)
    if len(full_context) > max_context_chars:
        full_context = full_context[:max_context_chars] + "\n\n[Context truncated]\n"

    # 4. Build a clear RAG-style prompt for Gemma3
    prompt = f"""You are an assistant using retrieved context to answer the user's question.

You will be given:
1) Some context chunks from academic papers or documents.
2) A user question.

Rules:
- Base your answer primarily on the provided context.
- If the context is insufficient or does not cover the answer, say: "Unable to sufficiently answer query".
- Be concise, accurate, and explicit when something is not stated in the context.

--------------------
CONTEXT START
--------------------
{full_context}
--------------------
CONTEXT END
--------------------

Question: {question}

Answer:"""

    # 5. Call Gemma3 on Ollama
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                },
            },
            timeout=120,
        )
        response.raise_for_status()
    except requests.RequestException as e:
        # Network/model error fallback
        logger.info(f"Error calling Gemma3 via Ollama: {e}")
        return "Error calling Gemma3 via Ollama."

    data = response.json()

    # Ollama returns a JSON payload with "response" containing the text
    answer = data.get("response", "").strip()

    # If Gemma still gives an empty answer, fallback
    if not answer:
        return "Unable to sufficiently answer query"

    return answer


if __name__ == "__main__":
    question = "What were the net CO2e emissions from training the GShard-600B model?"
    chunks = get_chunks(question, num_chunks=3)
    logger.info(chunks)

    model_answer = rag_gemma3_answer(question, chunks)
    logger.info(f"\nModel answer: {model_answer}")

    #
    # for rank, info in chunks.items():
    #     logger.info(f"\nRank {rank}")
    #     logger.info("Paper:", info["paper"])
    #     logger.info("Chunk:", info["chunk"][:200], "...")
