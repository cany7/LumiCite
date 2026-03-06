from pathlib import Path
import json
import uuid
from .load_files import load_metadata_df, get_PDF_paths
from .chunker import extract_pdf_chunks
from src.core.logging import get_logger
from src.core.paths import find_project_root

logger = get_logger(__name__)

def access_json():
    """Locate and load the JSON file if it exists, otherwise return an empty dict."""
    path = find_project_root()

    JSON_dir = path / "data" / "JSON"
    JSON_dir.mkdir(parents=True, exist_ok=True)
    JSON_file = JSON_dir / "chunks.json"

    if JSON_file.exists():
        logger.info(f"{JSON_file.name} exists. Loading...")
        with open(JSON_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info("JSON loaded successfully!")
        return data, JSON_file
    else:
        logger.info(f"{JSON_file.name} does not exist, creating new one...")
        return {}, JSON_file


def save_json_safely(data, path):
    """Write JSON atomically (safe even if interrupted mid-write)."""
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    tmp_path.replace(path)  # replaces old file atomically


def build_json():
    df_meta = load_metadata_df()
    pdf_dict = get_PDF_paths()
    chunks_json, JSON_file = access_json()

    for pdf_id in df_meta["id"]:
        pdf_path = pdf_dict.get(pdf_id)
        if not pdf_path:
            logger.info(f"Metadata ID {pdf_id} not found in downloaded PDFs, skipping...")
            continue

        if pdf_id in chunks_json:
            logger.info(f"ID {pdf_id} already in JSON, skipping...")
            continue

        logger.info(f"\n📄 Extracting chunks for {pdf_id}...")

        try:
            chunks_list = extract_pdf_chunks(pdf_path)
        except Exception as e:
            logger.info(f"⚠️ Error extracting {pdf_id}: {e}")
            continue

        pdf_chunks = []
        for chunk in chunks_list:
            chunk_dict = chunk.export_json_dict()
            chunk_entry = {
                "chunk_id": f"{pdf_id}_{uuid.uuid4().hex[:8]}",  # unique
                "text": chunk_dict["text"],
                "headings": chunk_dict["meta"].get("headings", []),
                "page_number": chunk_dict["meta"]["doc_items"][0]["prov"][0]["page_no"],
                "source_file": chunk_dict["meta"]["origin"]["filename"]
            }
            pdf_chunks.append(chunk_entry)

        chunks_json[pdf_id] = pdf_chunks

        # 💾 Save progress immediately after each PDF
        save_json_safely(chunks_json, JSON_file)
        logger.info(f"✅ Saved progress after {pdf_id}")

    logger.info(f"\n🎉 All available PDFs processed. Final JSON at: {JSON_file}")


if __name__ == "__main__":
    build_json()
