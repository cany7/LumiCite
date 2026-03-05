"""Prompt builder for structured answers with citations (JSON output)."""
from __future__ import annotations

from typing import List, Dict, Any


FALLBACK_ANSWER = "Unable to answer with confidence based on the provided documents."


def build_prompt(question: str, contexts: List[Dict], candidate_ref_ids: List[str]) -> str:
    """Create the instruction + context block for the model.

    contexts: items with 'ref_id', 'text', optional 'page'/'headings'.
    """
    instructions = f"""
Answer using ONLY the context below. If it's not enough, use the fallback.

Keep it short and factual. Include:
- ref_id list from metadata.csv
- supporting_materials (short quote/table/figure refs from context)
- explanation (1–2 lines linking the quote to the answer)

Normalization:
- Numeric: put the number in answer_value; unit in answer_unit.
- Ranges: answer_value as [low,high].
- TRUE/FALSE: answer in caps; answer_value 1/0; unit is_blank.
- Terms: answer_value = term; unit is_blank.
- If not answerable from context: answer = "{FALLBACK_ANSWER}", value/unit = is_blank.

Return ONLY JSON with keys:
answer, answer_value, answer_unit, ref_id, supporting_materials, explanation

Question:
{question}

Candidate reference IDs (from retrieval): {candidate_ref_ids}

Context snippets:
"""
    # Compose context block
    ctx_lines: List[str] = []
    for i, c in enumerate(contexts, start=1):
        ref_id = c.get("ref_id")
        page = c.get("page")
        
        # Safely handle the 'headings' field
        headings_val: Any = c.get("headings")
        heading_str = ""
        if isinstance(headings_val, list):
            heading_str = "; ".join(str(h) for h in headings_val if h)
        elif headings_val:
            heading_str = str(headings_val)

        header_bits = [f"ref_id={ref_id}"]
        if page is not None:
            header_bits.append(f"page={page}")
        if heading_str:
            header_bits.append(f"heading={heading_str}")
            
        header = ", ".join(header_bits)
        text = c.get("text", "").strip()
        ctx_lines.append(f"[{i}] ({header})\n{text}\n")

    ctx_block = "\n".join(ctx_lines)

    closing = "\n---\nOutput: JSON only."

    return instructions + ctx_block + closing
