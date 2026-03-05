from pathlib import Path
import json
import uuid
from .load_files import load_metadata_df, get_PDF_paths
from .chunker import extract_pdf_chunks

def access_json():
    """Locate and load the JSON file if it exists, otherwise return an empty dict."""
    path = Path(__file__).resolve()

    while not ((path / "src").exists() and (path / "data").exists()):
        if path.parent == path:
            raise RuntimeError("Project root directory not found")
        path = path.parent

    JSON_dir = path / "data" / "JSON"
    JSON_dir.mkdir(parents=True, exist_ok=True)
    JSON_file = JSON_dir / "chunks.json"

    if JSON_file.exists():
        print(f"{JSON_file.name} exists. Loading...")
        with open(JSON_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        print("JSON loaded successfully!")
        return data, JSON_file
    else:
        print(f"{JSON_file.name} does not exist, creating new one...")
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
            print(f"Metadata ID {pdf_id} not found in downloaded PDFs, skipping...")
            continue

        if pdf_id in chunks_json:
            print(f"ID {pdf_id} already in JSON, skipping...")
            continue

        print(f"\n📄 Extracting chunks for {pdf_id}...")

        try:
            chunks_list = extract_pdf_chunks(pdf_path)
        except Exception as e:
            print(f"⚠️ Error extracting {pdf_id}: {e}")
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
        print(f"✅ Saved progress after {pdf_id}")

    print(f"\n🎉 All available PDFs processed. Final JSON at: {JSON_file}")


if __name__ == "__main__":
    build_json()
