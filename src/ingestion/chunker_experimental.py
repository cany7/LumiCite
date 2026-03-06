from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from llama_parse import LlamaParse
import os
from src.core.logging import get_logger

logger = get_logger(__name__)

def extract(pdf_path: str) -> str:
    # Get API key from environment
    api_key = os.getenv("LLAMA_INDEX_API_KEY") or os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key:
        raise ValueError("LLAMA_INDEX_API_KEY or LLAMA_CLOUD_API_KEY must be set in environment")
    
    parser = LlamaParse(
        api_key=api_key,
        result_type="text",  # "markdown" and "text" are available
        num_workers=4,  # if multiple files passed, split in `num_workers` API calls
        verbose=True,
    )

    documents = parser.load_data(pdf_path)

    # Extract text
    text = ""
    for document in documents:
        text += document.text + "**NEW PAGE**"
    return text

def chunk(text: str) -> list[str]:
    # Embedding model for semantic chunking
    embeddings = HuggingFaceEmbeddings(
        model_name="Qwen/Qwen3-Embedding-4B",
        model_kwargs={'device': 'cpu'},  # or 'cuda' if you have GPU
        encode_kwargs={'normalize_embeddings': False}
    )

    splitter = SemanticChunker(embeddings)

    # Chunk
    chunks = splitter.split_text(text)
    logger.info(f"Created {len(chunks)} chunks.")
    return chunks

extracted_text = extract("data/pdf/chen2024.pdf")
chunk(extracted_text[:300])
