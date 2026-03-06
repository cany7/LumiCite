# -*- coding: utf-8 -*-
# Batch visual alt-text generator for extracted figures/diagrams from academic PDFs
# Vision LLM: Google Gemini 2.5 Flash
# Input image filenames (PNG): "<ref_id>_<pic_name>.png"  e.g., "2104.10350v3_figure1.png"
# Metadata CSV: export_summary.csv with columns: ref_id, file_name[, fig_title, page_number]
# Output:
#   1) CSV: chunk_id, text, pic_name, ref_id, source_file
#   2) JSON: grouped by pdf_id (ref_id) with fields:
#       chunk_id, text, headings([pic_name]), page_number, source_file("<ref_id>.pdf")

import os
import uuid
import json
import time
import traceback
import pandas as pd
from PIL import Image
import google.generativeai as genai
from pathlib import Path
from src.core.logging import get_logger

logger = get_logger(__name__)

def generate_image_chunk_ids(pdf_id, image_count):
    # Generate image chunk IDs to avoid conflict with text chunks
    # Format: f"{pdf_id}_img_{uuid.uuid4().hex[:8]}"
    ids = []
    for _ in range(int(image_count)):
        ids.append(f"{pdf_id}_img_{uuid.uuid4().hex[:8]}")
    return ids


def generate_single_image_chunk_id(pdf_id):
    # Generate a single image chunk ID
    return f"{pdf_id}_img_{uuid.uuid4().hex[:8]}"


def truncate_to_char_limit(text, max_chars=2048):
    # Roughly limit text length to about 512 tokens (≈2048 characters)
    if text and len(text) > max_chars:
        return text[:max_chars].rstrip() + "..."
    return text


def build_prompt_for_alt_text(pic_name):
    # Build an English prompt for Gemini 2.5 Flash describing what to extract from the figure
    return f"""
You are an expert data-extraction analyst. Your task is to analyze the provided image and produce a single, dense, data-rich English summary (alt-text) for a Retrieval-Augmented Generation (RAG) workflow.

CRITICAL CONSTRAINTS:
Hard Limit: The output MUST NOT EXCEED 2048 characters. This is a technical limit. You must be concise, even if it means omitting low-priority information.
Data-Only Focus: Actively ignore all visual aesthetics. Do not describe colors, line styles (e.g., dashed/solid), shapes (e.g., circles/squares), or font.
Category Identification: When categories are present, refer to them by their name or label (e.g., "Group A," "Sample B"). Do not use their visual properties as identifiers (e.g., DO NOT say "the red line" or "the blue bar").
Output Format: A single block of continuous paragraph prose. No bullet points.
Required Information Coverage (integrate these points into the prose):
Chart type: Name the chart (e.g., "This is a line chart...").
Axes: Identify the variables, units, and scale (e.g., "The x-axis represents Time in years... The y-axis represents Population in millions on a log scale.").
Legend and Categories: Do not describe the legend itself. Simply state the names of the primary categories, samples, or groups being compared (e.g., "The chart compares data for 'Group A', 'Group B', and 'Control'.").
Key Values: State significant labeled data points: start/end values, intersections, labeled peaks, or values on reference lines.
Trends and Relationships: (HIGHEST PRIORITY). This is the most important section. Describe the core patterns, correlations, and comparisons (e.g., "Group A shows a rapid increase, plateauing at 50 units, while Group B maintains a steady decline," or "There is a strong inverse correlation between X and Y.").
Extrema: Identify the absolute minimum and maximum data values in the chart and state which category/sample they belong to.
Annotations: Describe any data-relevant elements like threshold lines (e.g., "a significance threshold at p=0.05"), confidence bands (e.g., "the 95% confidence interval for Group A..."), or regression lines and their equations, if visible.

Constraints:
- Be factual and grounded on visible information.
- Include exact labels if readable; if not, state 'unreadable'.
- Write coherent natural prose.
- The figure name in the paper is '{pic_name}'.
    """.strip()


def init_gemini_model():
    # Initialize Gemini 2.5 Flash model
    # API key here
    API_KEY = ""
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")
    return model


def call_gemini_on_image(model, image_path, pic_name, target_char_cap=2048, max_retries=3, retry_sleep=2.0):
    # Send the image and prompt to Gemini 2.5 Flash and return generated text
    # Includes retry logic and character truncation (~512 tokens)
    prompt = build_prompt_for_alt_text(pic_name)

    with Image.open(image_path) as img:
        image_for_model = img.copy()

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = model.generate_content([prompt, image_for_model])
            text = resp.text.strip() if hasattr(resp, "text") and resp.text else ""
            return truncate_to_char_limit(text, target_char_cap)
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(retry_sleep)
            else:
                fallback = f"(Generation failed after {max_retries} attempts. Error: {repr(e)})"
                return truncate_to_char_limit(fallback, target_char_cap)
    return truncate_to_char_limit(f"(Unexpected error: {repr(last_err)})", target_char_cap)


def parse_file_name(file_name_no_suffix):
    # Parse file name of the form '<ref_id>_<pic_name>'
    # Example: '2302.08476v1_table2' -> ('2302.08476v1', 'table2')
    parts = file_name_no_suffix.split("_", 1)
    if len(parts) != 2:
        ref_id = parts[0]
        pic_name = "figure"
    else:
        ref_id, pic_name = parts[0], parts[1]
    return ref_id, pic_name


def ensure_dir(path):
    # Create a directory if it does not exist
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_metadata_csv(csv_path):
    """
    Load export_summary.csv with minimally required columns (ref_id, file_name).
    If the CSV lacks a page_number column, create one filled with NA; otherwise coerce to numeric.
    """
    df = pd.read_csv(csv_path)
    required_cols = {"ref_id", "file_name"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    if "page_number" not in df.columns:
        df["page_number"] = pd.NA
    else:
        df["page_number"] = pd.to_numeric(df["page_number"], errors="coerce")

    return df


def sanitize_ref_id(ref_id):
    # Remove surrounding square brackets if present and trim whitespace
    if isinstance(ref_id, str):
        ref_id = ref_id.strip()
        if ref_id.startswith("[") and ref_id.endswith("]"):
            return ref_id[1:-1].strip()
    return ref_id


def build_outputs(rows):
    # Build CSV and JSON outputs from processed rows
    csv_records = []
    json_grouped = {}

    for r in rows:
        clean_ref_id = sanitize_ref_id(r["ref_id"])
        csv_records.append(
            {
                "chunk_id": r["chunk_id"],
                "text": r["text"],
                "pic_name": r["pic_name"],
                "ref_id": r["ref_id"],
                "source_file": r["source_file"],
            }
        )

        pdf_id = clean_ref_id
        entry = {
            "chunk_id": r["chunk_id"],
            "text": r["text"],
            "headings": [r.get("heading_name", r["pic_name"])],
            "page_number": int(r["page_number"]) if pd.notna(r["page_number"]) else None,
            "source_file": f"{clean_ref_id}.pdf",
        }
        json_grouped.setdefault(pdf_id, []).append(entry)

    df_out = pd.DataFrame(csv_records, columns=["chunk_id", "text", "pic_name", "ref_id", "source_file"])
    return df_out, json_grouped


def save_csv_and_json(df_out, json_grouped, out_csv_path, out_json_path):
    # Save CSV and JSON outputs to disk
    df_out.to_csv(out_csv_path, index=False, encoding="utf-8")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(json_grouped, f, ensure_ascii=False, indent=2)


def batch_generate_alt_text():
    # Main batch process:
    # 1. Load metadata CSV
    # 2. Generate IDs per PDF
    # 3. Call Gemini 2.5 Flash to summarize figures
    # 4. Save CSV and JSON outputs

    # ---------- Resolve project root and build paths (no hardcoded relatives) ----------
    # start from the current file
    root = Path(__file__).resolve()
    # walk up until folder name is the project root
    while not ((root / "src").exists() and (root / "data").exists()):
        if root.parent == root:  # safety in case we reach filesystem root
            raise RuntimeError("Project root directory not found")
        root = root.parent

    # build paths from project root
    IMG_DIR = root / "data" / "figures"
    CSV_PATH = IMG_DIR / "export_summary.csv"
    OUTCSV_PATH = root / "data" / "figures" / "alt_text.csv"
    OUTJSON_PATH = root / "data" / "JSON" / "alt_text.json"

    ensure_dir(OUTCSV_PATH.parent)
    ensure_dir(OUTJSON_PATH.parent)

    model = init_gemini_model()
    df = load_metadata_csv(CSV_PATH)

    if "file_name" in df.columns:
        df = df.sort_values(["ref_id", "file_name"]).reset_index(drop=True)
    else:
        df = df.sort_values(["ref_id"]).reset_index(drop=True)

    processed_rows = []
    total_pdfs = df["ref_id"].nunique()
    pdf_counter = 0

    for ref_id, group in df.groupby("ref_id", sort=False):
        pdf_counter += 1
        ids_for_pdf = generate_image_chunk_ids(ref_id, len(group))
        id_iter = iter(ids_for_pdf)

        for _, row in group.iterrows():
            file_name_no_suffix = str(row["file_name"])
            this_ref, pic_name = parse_file_name(file_name_no_suffix)
            if this_ref != ref_id:
                this_ref = ref_id

            image_path = os.path.join(IMG_DIR, f"{file_name_no_suffix}.png")
            source_file_pdf = f"{this_ref}.pdf"

            try:
                text_summary = call_gemini_on_image(
                    model=model,
                    image_path=image_path,
                    pic_name=pic_name,
                    target_char_cap=2048
                )
            except Exception:
                text_summary = f"(Unhandled exception while processing '{image_path}'. Traceback: {traceback.format_exc(limit=1)})"

            try:
                chunk_id = next(id_iter)
            except StopIteration:
                chunk_id = generate_single_image_chunk_id(this_ref)

            processed_rows.append(
                {
                    "chunk_id": chunk_id,
                    "text": text_summary,
                    "pic_name": pic_name,
                    "ref_id": this_ref,
                    "source_file": source_file_pdf,
                    "page_number": row.get("page_number", pd.NA),
                    "heading_name": file_name_no_suffix,
                }
            )
        logger.info(f"[Progress] {pdf_counter}/{total_pdfs} PDFs processed — {ref_id} ({len(group)} figures)")

    df_out, json_grouped = build_outputs(processed_rows)
    save_csv_and_json(df_out, json_grouped, OUTCSV_PATH, OUTJSON_PATH)

    logger.info(f"[Done] CSV saved to: {OUTCSV_PATH}")
    logger.info(f"[Done] JSON saved to: {OUTJSON_PATH}")


if __name__ == "__main__":
    batch_generate_alt_text()
