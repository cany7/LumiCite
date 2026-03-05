"""Generate submission.csv from questions using Ollama RAG with resume capability."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

import pandas as pd

# Add src to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indexing.vector_store import find_project_root
from src.indexing.retrieval import get_chunks
from src.indexing.ollama_generator import rag_ollama_answer


# Output schema (column order)
REQUIRED_COLS: List[str] = [
    "id",
    "question",
    "answer",
    "answer_value",
    "answer_unit",
    "ref_id",
    "ref_url",
    "supporting_materials",
    "explanation",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create submission.csv from questions using Ollama")
    p.add_argument("--input", type=str, default=str(find_project_root() / "data" / "test_qa.csv"),
                   help="Input questions CSV path")
    p.add_argument("--output", type=str, default=str(find_project_root() / "submission.csv"),
                   help="Output submission CSV path")
    p.add_argument("--top-k", type=int, default=4,
                   help="Number of chunks to retrieve per question")
    p.add_argument("--model", type=str, default=None,
                   help="Ollama model name (default: auto-detect, uses mistral if available)")
    p.add_argument("--resume", action="store_true", default=True,
                   help="Resume from existing output file (default: True)")
    p.add_argument("--force-restart", action="store_true", default=False,
                   help="Force restart from beginning, ignore existing output")
    return p.parse_args()


def load_metadata_map(root: Path) -> dict:
    """Load metadata.csv and return mapping of ref_id -> url."""
    import csv

    meta_path = root / "data" / "metadata" / "metadata.csv"
    if not meta_path.exists():
        print(f"Warning: metadata file not found at {meta_path}")
        return {}

    m = {}
    with meta_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row.get("id")
            if rid:
                m[rid] = row.get("url", "")
    return m


def format_list_for_csv(items: list) -> str:
    """Format list as Python-style string for CSV: ['item1','item2']"""
    if not items:
        return "is_blank"
    safe = [str(x).strip() for x in items if x]
    if not safe:
        return "is_blank"
    inner = ",".join([f"'{x}'" for x in safe])
    return f"[{inner}]"


def load_existing_progress(output_path: Path) -> tuple[pd.DataFrame | None, set[str]]:
    """Load existing output file and return DataFrame and set of completed question IDs."""
    if not output_path.exists():
        return None, set()

    try:
        df_existing = pd.read_csv(output_path)
        completed_ids = set(df_existing["id"].astype(str).tolist())
        print(f"📂 Found existing output with {len(completed_ids)} completed questions")
        return df_existing, completed_ids
    except Exception as e:
        print(f"⚠️  Error loading existing output: {e}")
        return None, set()


def save_progress(rows: list, output_path: Path) -> None:
    """Save current progress to CSV file."""
    if not rows:
        return

    df_out = pd.DataFrame(rows)

    # Ensure all required columns exist
    for c in REQUIRED_COLS:
        if c not in df_out.columns:
            df_out[c] = "is_blank"

    # Reorder columns
    df_out = df_out[REQUIRED_COLS]

    # Save to file
    df_out.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    root = find_project_root()

    inp = Path(args.input)
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input: {inp}")
    if not inp.exists():
        raise FileNotFoundError(f"Input file not found: {inp}")

    df_in = pd.read_csv(inp)
    if "id" not in df_in.columns or "question" not in df_in.columns:
        raise ValueError("Input CSV must contain 'id' and 'question' columns")

    print(f"Loading metadata...")
    meta_map = load_metadata_map(root)

    # Load existing progress
    rows = []
    completed_ids = set()

    if args.resume and not args.force_restart:
        df_existing, completed_ids = load_existing_progress(outp)
        if df_existing is not None:
            # Convert existing DataFrame to list of dicts
            rows = df_existing.to_dict('records')
            print(f"✅ Resuming from question {len(completed_ids) + 1}/{len(df_in)}")
        else:
            print(f"🆕 Starting fresh - no existing output found")
    else:
        if args.force_restart:
            print(f"🔄 Force restart - ignoring any existing output")
        print(f"🆕 Starting from beginning")

    total_questions = len(df_in)
    remaining = total_questions - len(completed_ids)

    print(f"Processing {remaining} remaining questions out of {total_questions} total")
    print(f"Config: top_k={args.top_k}, model={args.model or 'auto-detect'}")
    print("=" * 80)

    processed_count = 0

    for idx, row in df_in.iterrows():
        qid = str(row["id"]) if pd.notna(row["id"]) else ""
        question = str(row["question"]) if pd.notna(row["question"]) else ""

        if not qid or not question:
            print(f"⚠️  Skipping row {idx} with missing id/question")
            continue

        # Skip if already completed
        if qid in completed_ids:
            continue

        processed_count += 1
        progress = len(completed_ids) + processed_count
        print(f"\n[{progress}/{total_questions}] Q {qid}: {question[:70]}{'...' if len(question)>70 else ''}")

        # Step 1: Retrieve chunks
        try:
            chunks = get_chunks(question, num_chunks=args.top_k)
            if chunks:
                print(f"  ✓ Retrieved {len(chunks)} chunks")
            else:
                print(f"  ⚠️  No chunks retrieved - will return fallback answer")
        except Exception as e:
            print(f"  ❌ Error retrieving chunks: {e}")
            chunks = None

        # Step 2: Generate answer with Ollama
        try:
            result = rag_ollama_answer(question, chunks, model=args.model)
        except Exception as e:
            print(f"  ❌ Error generating answer: {e}")
            result = {
                "answer": "Unable to answer with confidence based on the provided documents.",
                "answer_value": "is_blank",
                "answer_unit": "is_blank",
                "ref_id": [],
                "supporting_materials": "is_blank",
                "explanation": "is_blank",
            }

        # Step 3: Format for CSV
        ref_ids = result.get("ref_id", [])
        if isinstance(ref_ids, str):
            ref_ids = [ref_ids] if ref_ids else []

        # Get URLs from metadata
        ref_urls = [meta_map.get(rid, "") for rid in ref_ids]
        ref_urls = [url for url in ref_urls if url]  # Remove empty URLs

        # Format lists for CSV
        ref_id_str = format_list_for_csv(ref_ids)
        ref_url_str = format_list_for_csv(ref_urls)

        record = {
            "id": qid,
            "question": question,
            "answer": result.get("answer", "is_blank"),
            "answer_value": str(result.get("answer_value", "is_blank")),
            "answer_unit": result.get("answer_unit", "is_blank"),
            "ref_id": ref_id_str,
            "ref_url": ref_url_str,
            "supporting_materials": result.get("supporting_materials", "is_blank"),
            "explanation": result.get("explanation", "is_blank"),
        }

        # Add to rows and mark as completed
        rows.append(record)
        completed_ids.add(qid)

        # Print answer preview
        answer_preview = result.get('answer', '')[:70]
        print(f"  ✓ Answer: {answer_preview}{'...' if len(result.get('answer', ''))>70 else ''}")

        # Save progress after every question
        try:
            save_progress(rows, outp)
            print(f"  💾 Progress saved ({progress}/{total_questions} complete)")
        except Exception as e:
            print(f"  ⚠️  Warning: Could not save progress: {e}")

    if not rows:
        raise RuntimeError("No answers generated; aborting without writing submission.")

    # Final save
    print("\n" + "=" * 80)
    print(f"✅ Complete! Processed {processed_count} new questions")
    print(f"📊 Total answers in submission: {len(rows)}/{total_questions}")
    print(f"💾 Output saved to: {outp}")
    print("=" * 80)


if __name__ == "__main__":
    main()
