# Extract every figure (PictureItem) and table (TableItem) from one or more PDF
# documents using Docling, then export them as PNG images.
#   1) Place PDF files in ./pdfs
#   2) Run: python image_extract.py  (by default runs process_image_extract_abs)
#   3) Images appear in ./data/figures at the project root (<pdfstem>_figureN.png / <pdfstem>_tableN.png)
#   python image_extract.py docs/report1.pdf --out_dir ./export
from __future__ import annotations

import argparse
import csv
import glob
import logging
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

from docling_core.types.doc import PictureItem, TableItem
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

# Parse CLI options; ignore unknown IDE flags.
def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract figures/tables from PDF(s) with Docling."
    )
    parser.add_argument(
        "pdf_files",
        type=Path,
        nargs="*",
        default=[],
        help="PDF path(s). Leave empty to scan ./pdfs/*.pdf",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("./pics"),
        help="Directory to save exported PNGs (default: ./pics)",
    )
    # parse_known_args → ignore extra flags like --mode --host inserted by IDEs
    args, _ = parser.parse_known_args()
    return args

# Configure a global logging format and level.
def init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

# Return a list of PDFs to process; auto‑scan ./pdfs if none supplied.
def discover_pdfs(user_supplied: List[Path]) -> List[Path]:
    if user_supplied:
        return user_supplied

    # Unified path resolution (multi-base like process_image_extract_abs)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    default_candidates = [
        os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir, "pdfs")),  # project root
        os.path.abspath(os.path.join(script_dir, os.pardir, "pdfs")),
        os.path.abspath(os.path.join(script_dir, "pdfs")),
        os.path.abspath(os.path.join(cwd, "pdfs")),
    ]
    resolved_dir = None
    for cand in default_candidates:
        if os.path.isdir(cand):
            resolved_dir = cand
            break
    if not resolved_dir:
        raise FileNotFoundError(f"No PDFs provided and none found in any of: {', '.join(default_candidates)}")
    pdfs = sorted(glob.glob(os.path.join(resolved_dir, "*.pdf"))) + sorted(glob.glob(os.path.join(resolved_dir, "*.PDF")))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in resolved directory: {resolved_dir}")
    return [Path(p) for p in pdfs]

# Build and return a Docling DocumentConverter with PDF pipeline opts.
def build_converter(scale: float) -> DocumentConverter:
    opts = PdfPipelineOptions(
        images_scale=scale,
        generate_picture_images=True,
        generate_table_images=True,
    )
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )

# Convert a single PDF, export all figures/tables, collect CSV rows.
def _extract_page_number(element: Any) -> Optional[int]:
    """Best-effort fetch of the first provenance page number from a Docling element."""
    prov = getattr(element, "prov", None)
    if not prov:
        return None
    first = prov[0]
    if hasattr(first, "page_no"):
        return getattr(first, "page_no")
    if isinstance(first, dict):
        return first.get("page_no")
    first_dict = getattr(first, "__dict__", None)
    if isinstance(first_dict, dict):
        return first_dict.get("page_no")
    return None


def extract_from_pdf(
    pdf_path: Path, converter: DocumentConverter, out_dir: Path
) -> Tuple[int, int, List[Tuple[str, str, Optional[int]]]]:
    if not pdf_path.exists():
        logging.warning("%s not found – skipped.", pdf_path)
        return 0, 0, []

    # logging.info("Parsing %s ...", pdf_path.name)
    result = converter.convert(pdf_path)
    doc = result.document

    fig_count = tbl_count = 0
    rows: List[Tuple[str, str, Optional[int]]] = []
    # Walk every item in the document tree.
    for element, _ in doc.iterate_items():
        # Is the current item a figure or a table?
        if isinstance(element, PictureItem):
            fig_count += 1
            img = element.get_image(doc)
            img.save(out_dir / f"{pdf_path.stem}_figure{fig_count}.png")
            page_no = _extract_page_number(element)
            rows.append((pdf_path.stem, f"{pdf_path.stem}_figure{fig_count}", page_no))
        elif isinstance(element, TableItem):
            tbl_count += 1
            img = element.get_image(doc)
            img.save(out_dir / f"{pdf_path.stem}_table{tbl_count}.png")
            page_no = _extract_page_number(element)
            rows.append((pdf_path.stem, f"{pdf_path.stem}_table{tbl_count}", page_no))

    logging.info(
        "%s: exported %d figures, %d tables → %s",
        pdf_path.name,
        fig_count,
        tbl_count,
        out_dir,
    )
    return fig_count, tbl_count, rows

# Orchestrate batch extraction and write summary CSV.
def main() -> None:
    args = parse_cli_args()
    init_logging()
    logging.info("Initializing extraction ...")

    # Resolve out_dir anchored to project root (prefer script_dir/../../ as root)
    import os
    from pathlib import Path
    cwd = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_raw = str(args.out_dir)
    if os.path.isabs(out_raw):
        out_dir = Path(out_raw)
    else:
        base_candidates = [
            os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir)),  # project root two levels up
            os.path.abspath(os.path.join(script_dir, os.pardir)),
            cwd,
            script_dir,
        ]
        resolved_output = None
        for base in base_candidates:
            out_cand = os.path.abspath(os.path.join(base, out_raw))
            try:
                os.makedirs(out_cand, exist_ok=True)
                resolved_output = out_cand
                break
            except Exception:
                continue
        if not resolved_output:
            raise OSError(f"Failed to create out_dir at any candidate base for {out_raw}.")
        out_dir = Path(resolved_output)
    logging.info("Resolved OUTPUT_DIR (project-root anchored): %s", out_dir)
    args.out_dir = out_dir

    pdf_files = discover_pdfs(args.pdf_files)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    converter = build_converter(3.0)

    total_figs = total_tables = 0
    all_rows: List[Tuple[str, str, Optional[int]]] = []
    # Process each PDF and aggregate exported counts.
    for pdf in pdf_files:
        figs, tables, rows = extract_from_pdf(pdf, converter, args.out_dir)
        total_figs += figs
        total_tables += tables
        all_rows.extend(rows)

    csv_path = args.out_dir / "export_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ref_id", "file_name", "page_number"])
        writer.writerows(
            (ref_id, file_name, page if page is not None else "")
            for ref_id, file_name, page in all_rows
        )
    logging.info("CSV summary saved → %s", csv_path)

    logging.info("Done. Total exported: %d figures, %d tables. CSV saved: %s", total_figs, total_tables, csv_path)



# Batch processes all PDFs in a directory.
def process_image_extract(dir_path: Path, out_dir: Path = Path("./pics")) -> None:
    """
    Process all PDF files in a directory, extract figures/tables, and export as PNGs.
    Args:
        dir_path: Path to a directory containing PDF files.
        out_dir: Output directory for images.
    """
    if not dir_path.is_dir():
        raise ValueError("`dir_path` must be a directory containing PDF files.")
    pdf_files = sorted(dir_path.glob("*.pdf"))
    out_dir.mkdir(parents=True, exist_ok=True)
    converter = build_converter(3.0)  # fixed scale
    total_figs = total_tables = 0
    all_rows: List[Tuple[str, str, Optional[int]]] = []
    for pdf in pdf_files:
        figs, tables, rows = extract_from_pdf(pdf, converter, out_dir)
        total_figs += figs
        total_tables += tables
        all_rows.extend(rows)
    csv_path = out_dir / "export_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ref_id", "file_name", "page_number"])
        writer.writerows(
            (ref_id, file_name, page if page is not None else "")
            for ref_id, file_name, page in all_rows
        )
    logging.info("CSV summary saved → %s", csv_path)
    logging.info("Done. Total exported: %d figures, %d tables. CSV saved: %s directory", total_figs, total_tables, csv_path)


# Batch processes all PDFs using absolute paths defined inside this function (no arguments required).
def process_image_extract_abs() -> None:
    """
    Batch extract figures/tables from all PDFs in an absolute input directory and
    save outputs to an absolute output directory. Edit the two constants below
    to your own absolute paths. No Path objects are required when calling.
    """
    # === EDIT THESE TWO LINES TO YOUR ABSOLUTE DIRECTORIES ===
    INPUT_DIR = "data/pdf"
    OUTPUT_DIR = "data/figures"
    # ========================================================

    # --- Resolve directories: allow absolute or relative; try multiple bases ---
    cwd = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    _input_raw = INPUT_DIR
    _output_raw = OUTPUT_DIR

    # Candidate search order for INPUT_DIR when given relatively:
    # 1) Relative to current working directory
    # 2) Relative to this script's folder
    # 3) Relative to the parent of this script's folder (project root style)
    # 4) Relative to the grandparent of this script's folder (project root two levels up)
    input_candidates = []
    if os.path.isabs(_input_raw):
        input_candidates.append(_input_raw)
    else:
        # 1) Relative to current working directory
        input_candidates.append(os.path.abspath(os.path.expanduser(_input_raw)))
        # 2) Relative to this script's folder
        input_candidates.append(os.path.abspath(os.path.join(script_dir, _input_raw)))
        # 3) Relative to the parent of this script's folder (e.g., src/ → project root style one level up)
        input_candidates.append(os.path.abspath(os.path.join(script_dir, os.pardir, _input_raw)))
        # 4) Relative to the grandparent of this script's folder (e.g., src/ingestion/ → project root two levels up)
        input_candidates.append(os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir, _input_raw)))

    resolved_input = None
    for cand in input_candidates:
        if os.path.isdir(cand):
            resolved_input = cand
            break

    logging.info("CWD: %s", cwd)
    logging.info("Tried INPUT_DIR candidates: %s", " | ".join(input_candidates))

    if not resolved_input:
        # Show a peek of each parent dir for diagnostics
        for cand in input_candidates:
            parent = os.path.dirname(cand) or "/"
            try:
                listing = ", ".join(os.listdir(parent)[:20])
            except Exception:
                listing = "<unreadable>"
            logging.error("INPUT candidate missing: %s | Parent(%s) contains: %s", cand, parent, listing)
        raise ValueError("INPUT_DIR must be an existing directory containing PDF files.")

    INPUT_DIR = resolved_input

    # Resolve OUTPUT_DIR anchored to project root (prefer script_dir/../../ as root)
    if os.path.isabs(_output_raw):
        OUTPUT_DIR = _output_raw
    else:
        base_candidates = [
            os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir)),  # project root two levels up
            os.path.abspath(os.path.join(script_dir, os.pardir)),             # parent of script dir
            cwd,                                                              # current working directory
            script_dir,                                                       # script dir itself
        ]
        resolved_output = None
        for base in base_candidates:
            out_cand = os.path.abspath(os.path.join(base, _output_raw))
            # Accept the first base where the parent is writable/exists; create dirs below
            parent = os.path.dirname(out_cand)
            try:
                os.makedirs(out_cand, exist_ok=True)
                resolved_output = out_cand
                break
            except Exception:
                # Try next base
                continue
        if not resolved_output:
            raise OSError("Failed to create OUTPUT_DIR at any candidate base.")
        OUTPUT_DIR = resolved_output

    logging.info("Resolved INPUT_DIR: %s", INPUT_DIR)
    logging.info("Resolved OUTPUT_DIR (project-root anchored): %s", OUTPUT_DIR)

    # Collect PDFs (case-insensitive)
    pdf_files = []
    pdf_files.extend(glob.glob(os.path.join(INPUT_DIR, "*.pdf")))
    pdf_files.extend(glob.glob(os.path.join(INPUT_DIR, "*.PDF")))
    pdf_files = sorted(pdf_files)
    if not pdf_files:
        try:
            listing = ", ".join(os.listdir(INPUT_DIR)[:50])
        except Exception:
            listing = "<unreadable>"
        logging.error("No PDF files found in INPUT_DIR. Files present: %s", listing)
        raise FileNotFoundError(f"No PDF files found in INPUT_DIR: {INPUT_DIR}")

    # Build converter with fixed scale (3.0)
    converter = build_converter(3.0)

    total_figs = 0
    total_tables = 0
    all_rows: List[Tuple[str, str, Optional[int]]] = []

    # Process each PDF
    for pdf in pdf_files:
        # Internally convert to Path to interoperate with existing helpers
        figs, tables, rows = extract_from_pdf(Path(pdf), converter, Path(OUTPUT_DIR))
        total_figs += figs
        total_tables += tables
        all_rows.extend(rows)

    # Write CSV summary using absolute path strings
    csv_path = os.path.join(OUTPUT_DIR, "export_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ref_id", "file_name", "page_number"])
        writer.writerows(
            (ref_id, file_name, page if page is not None else "")
            for ref_id, file_name, page in all_rows
        )

    logging.info("CSV summary saved → %s", csv_path)
    logging.info(
        "Done. Total exported: %d figures, %d tables. CSV saved: %s",
        total_figs,
        total_tables,
        csv_path,
    )


if __name__ == "__main__":
    # Directly run the absolute-path batch processor (no CLI / no Path args)
    process_image_extract_abs()
