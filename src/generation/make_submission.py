"""Generate submission_gemini.csv from questions using the RAG pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Set, Tuple

import pandas as pd

from src.generation.rag_pipeline import RAGPipeline, RAGConfig
from src.indexing.vector_store import find_project_root
from src.core.logging import get_logger


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

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create submission_gemini.csv from questions")
    p.add_argument("--input", type=str, default=str(find_project_root() / "data" / "test_qa.csv"), help="Input questions CSV path")
    p.add_argument("--output", type=str, default=str(find_project_root() / "submission_gemini.csv"), help="Output submission CSV path")
    p.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve per question")
    p.add_argument("--force-restart", action="store_true", default=False, help="Ignore any existing submission_gemini.csv and recompute all questions")
    args, _ = p.parse_known_args()
    return args


def load_existing_progress(output_path: Path) -> Tuple[List[dict], Set[str]]:
    """Return rows + completed ids from an existing submission file."""
    if not output_path.exists():
        return [], set()

    try:
        df_existing = pd.read_csv(output_path)
    except Exception as exc:
        logger.info(f"Warning: Could not read existing output at {output_path}: {exc}")
        return [], set()

    rows = df_existing.to_dict("records")
    completed_ids: Set[str] = set()
    for rec in rows:
        qid = rec.get("id")
        if pd.notna(qid):
            completed_ids.add(str(qid))
    if completed_ids:
        logger.info(f"Found existing output with {len(completed_ids)} completed questions.")
    return rows, completed_ids


def save_progress(rows: List[dict], output_path: Path) -> None:
    """Write current progress to CSV in the required column order."""
    if not rows:
        return
    df_out = pd.DataFrame(rows)
    for c in REQUIRED_COLS:
        if c not in df_out.columns:
            df_out[c] = ""
    df_out = df_out[REQUIRED_COLS]
    df_out.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()

    inp = Path(args.input)
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Input: {inp}")
    if not inp.exists():
        raise FileNotFoundError(f"Input CSV not found: {inp}")

    df_in = pd.read_csv(inp)
    total_questions = len(df_in)
    if "id" not in df_in.columns or "question" not in df_in.columns:
        raise ValueError("Input CSV must contain 'id' and 'question' columns")

    logger.info(f"Init pipeline (top_k={args.top_k})")
    rag_config = RAGConfig(top_k=args.top_k)
    pipe = RAGPipeline(config=rag_config)

    # Load previous progress unless explicitly restarting
    rows: List[dict]
    completed_ids: Set[str]
    if args.force_restart:
        logger.info("Force restart enabled - ignoring any previous submission file.")
        rows, completed_ids = [], set()
    else:
        rows, completed_ids = load_existing_progress(outp)
        if not rows:
            logger.info("No previous submission found - starting from the first question.")

    new_answers = 0

    try:
        for i, row in df_in.iterrows():
            qid = str(row["id"]) if pd.notna(row["id"]) else ""
            question = str(row["question"]) if pd.notna(row["question"]) else ""
            if not qid or not question:
                logger.info(f"Skipping row with missing id/question: {row}")
                continue

            if qid in completed_ids:
                continue
            
            progress_idx = len(completed_ids) + 1
            logger.info(f"\n--- Question {progress_idx}/{total_questions} (ID: {qid}) ---")
            logger.info(f"Q: {question[:120]}{'...' if len(question)>120 else ''}")
            
            rec = pipe.answer(qid, question)
            
            # Display the answer in the progress log
            ans = rec.get("answer", "N/A")
            logger.info(f"A: {ans}")
            
            rows.append(rec)
            completed_ids.add(qid)
            new_answers += 1
            
            try:
                save_progress(rows, outp)
                logger.info(f"Progress saved ({len(completed_ids)}/{total_questions} answered).")
            except Exception as exc:
                logger.info(f"Warning: Could not save progress: {exc}")
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user. Partial progress saved; exiting early.")

    if not rows:
        raise RuntimeError("No answers generated; aborting without writing submission.")

    logger.info(f"\nFinished. Answered {len(completed_ids)}/{total_questions} questions "
          f"(+{new_answers} this run).")
    logger.info(f"Output saved incrementally to: {outp}")


if __name__ == "__main__":
    main()
