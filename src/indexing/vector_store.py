

"""
Build or load FAISS index and perform similarity search with SentenceTransformer.
"""

import json
import faiss
import numpy as np
import pickle
import time
import sys
from pathlib import Path

from huggingface_hub import paper_info
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
from src.config.settings import get_settings
from src.core.logging import get_logger
from src.core.paths import find_project_root

logger = get_logger(__name__)


def locate_embeddings_file(root_dir: Path) -> Path:
    """
    Locate and return the path to 'embeddings.jsonl' relative to the project root.
    Ensures the parent directory ('data/JSON') exists.

    Args:
        root_dir (Path): The project's root directory.

    Returns:
        Path: The full path to the 'embeddings.jsonl' file.
    """
    json_dir = root_dir / "data" / "JSON"

    # Ensure the directory exists, create it if not
    json_dir.mkdir(parents=True, exist_ok=True)

    embeddings_file = json_dir / "embeddings.jsonl"

    return embeddings_file


# --- 2. FaissSearch Class ---

class FaissSearch:
    """
    A class to build, load, and search a FAISS index
    using SentenceTransformer for text encoding.
    """

    def __init__(
        self,
        index_path: Path,
        text_data_path: Path,
        model_name: Optional[str] = None,
    ):
        """
        Initializes the searcher.

        Args:
            index_path (Path): The save/load path for the .index file.
            text_data_path (Path): The save/load path for the .pkl file storing original text.
            model_name (str): The name of the SentenceTransformer model to use.
        """
        self.index_path = index_path
        self.text_data_path = text_data_path
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model

        self.index: Optional[faiss.Index] = None
        self.text_data: List[Dict[str, Any]] = []

        logger.info(f"Loading Sentence Transformer model ({self.model_name})...")
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.info(f"Error: Failed to load model '{self.model_name}': {e}")
            self.model = None
            raise

    def load_index(self) -> bool:
        """
        Attempts to load the FAISS index and text data from files.

        Returns:
            bool: True if loading was successful, False otherwise.
        """
        if self.index_path.exists() and self.text_data_path.exists():
            logger.info(f"Loading existing index from {self.index_path.parent}...")
            try:
                self.index = faiss.read_index(str(self.index_path))
                with open(self.text_data_path, "rb") as f:
                    self.text_data = pickle.load(f)
                logger.info(f"FAISS index and text data loaded successfully.")
                return True
            except Exception as e:
                logger.info(f"Error loading index or text data: {e}")
                self.index = None
                self.text_data = []
                return False
        logger.info("Existing index files not found.")
        return False

    def build_index(self, jsonl_file: Path):
        """
        Builds the FAISS index from a .jsonl file and saves it to disk.

        Args:
            jsonl_file (Path): Path to the .jsonl file containing 'embedding', 'id', 'text', 'metadata'.
        """
        if not jsonl_file.exists():
            raise FileNotFoundError(f"Input file not found: {jsonl_file}")

        if self.model is None:
            raise RuntimeError("SentenceTransformer model is not loaded, cannot build index.")

        logger.info(f"Loading embeddings from {jsonl_file} to build FAISS index...")
        embeddings_list = []
        self.text_data = []

        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                embeddings_list.append(record["embedding"])
                self.text_data.append({
                    "id": record["id"],
                    "text": record["text"],
                    "metadata": record["metadata"]
                })

        if not embeddings_list:
            raise ValueError("No embeddings found in the input file.")

        # Determine embedding dimension and create FAISS index
        d = len(embeddings_list[0])
        embeddings = np.array(embeddings_list).astype('float32')

        # We use IndexFlatL2 for exact, brute-force search (Euclidean distance)
        self.index = faiss.IndexFlatL2(d)
        self.index.add(embeddings)
        logger.info(f"FAISS index built (contains {self.index.ntotal} vectors).")

        # Save index and text data
        try:
            # Ensure the parent directory exists
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(self.index_path))
            with open(self.text_data_path, "wb") as f_out:
                pickle.dump(self.text_data, f_out)
            logger.info(f"FAISS index saved to: {self.index_path}")
            logger.info(f"Text data saved to: {self.text_data_path}")
        except Exception as e:
            logger.info(f"Error saving index or text data: {e}")

    def search(self, question: str, k: int = 3):
        if self.index is None or self.model is None or not self.text_data:
            raise RuntimeError("Index, model, or text data not loaded.")

        logger.info(f"Searching FAISS for: '{question[:50]}...'")
        start_time = time.time()

        question_embedding = self.model.encode([question], convert_to_numpy=True).astype('float32')

        distances, indices = self.index.search(question_embedding, k)

        logger.info("\nSquared L2 distances for top results:")
        for rank in range(k):
            logger.info(f"  Rank {rank + 1}: {distances[0][rank]}")

        results = []
        for i in indices[0]:
            if 0 <= i < len(self.text_data):
                results.append(self.text_data[i])

        logger.info(f"\nRetrieved {len(results)} chunks in {time.time() - start_time:.4f} seconds.")
        return results


# --- 3. Main Execution Block ---

if __name__ == "__main__":

    # --- 3.1. Set up Paths ---
    try:
        # 1. Find the project root directory
        ROOT_DIR = find_project_root()

    except FileNotFoundError as e:
        logger.info(f"Fatal Error: {e}")
        logger.info("Falling back to the current working directory as a last resort.")
        ROOT_DIR = Path.cwd()

        if not ROOT_DIR.exists():
            logger.info(f"Error: Hardcoded fallback ROOT_DIR does not exist: {ROOT_DIR}")
            sys.exit(1)  # Exit if the fallback also fails

    # 2. Define all other paths relative to the root
    DATA_DIR = ROOT_DIR / "data"
    INDEX_FILE = DATA_DIR / "my_faiss.index"
    TEXT_DATA_FILE = DATA_DIR / "text_data.pkl"

    # 3. Use the helper function to get the input file path
    INPUT_FILE = locate_embeddings_file(ROOT_DIR)

    # --- 3.2. Run the Searcher ---

    # 💡 新增控制开关：设置为 True 强制重建索引
    FORCE_REBUILD = True

    try:
        # Initialize the searcher class
        searcher = FaissSearch(index_path=INDEX_FILE, text_data_path=TEXT_DATA_FILE)

        # 检查是否需要强制重建，或者加载失败
        if FORCE_REBUILD or not searcher.load_index():  # <--- 修改了这一行

            if FORCE_REBUILD:
                logger.info("\n--- ⚠️ 强制重建模式已开启 ---")
                # 如果文件存在，可以在这里添加删除旧文件的代码（可选）
                if INDEX_FILE.exists():
                    INDEX_FILE.unlink()
                if TEXT_DATA_FILE.exists():
                    TEXT_DATA_FILE.unlink()

            else:  # 只有在 FORCE_REBUILD=False 且加载失败时才打印这个
                logger.info("Failed to load existing index, building a new one...")

            # Safety check: ensure the source file exists before building
            if not INPUT_FILE.exists():
                logger.info(f"Fatal Error: Cannot build index because {INPUT_FILE} is missing.")
                sys.exit(1)

            # 构建新的索引，它会覆盖旧文件
            searcher.build_index(jsonl_file=INPUT_FILE)


        # --- 3.3. Perform an Example Search ---
        logger.info("\n" + "=" * 20)
        question = "what is your favorite color?"  # <--- Change your query here
        results = searcher.search(question, k=3)

        logger.info(f"\n--- Search Results for '{question}' ---")
        if results:
            for i, res in enumerate(results):
                logger.info(f"\nResult {i + 1}:")
                logger.info(f"  ID: {res.get('id')}")
                logger.info(f"  Text: {res.get('text')[:]}...")  # Print first 150 chars
        else:
            logger.info("No results found.")

    except Exception as e:
        logger.info(f"\nAn unexpected error occurred during main execution: {e}")
        # For detailed debugging, uncomment the lines below
        # import traceback
#       # traceback.print_exc()
#
# def get_chunks(question, num_chunks = 3)
#
# # output
# # {
# #     {
#         Chunk: text
#         paper: jeff2020
#         rank: (1 2 or 3)
#     }
#     {
#         Chunk: text
#         paper: jeff2020
#         rank: (1 2 or 3)
#     }
# }
