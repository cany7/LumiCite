"""Prompt builder for structured answers with evidence-span citations."""
from __future__ import annotations

from typing import Any

from src.core.constants import FALLBACK_ANSWER


def build_prompt(question: str, contexts: list[dict[str, Any]], candidate_ref_ids: list[str]) -> str:
    """Create the instruction + context block for the model."""
    instructions = f"""
Answer using ONLY the context below. If it's not enough, use the fallback.

Keep it short and factual. Include:
- ref_id list from metadata.csv
- supporting_materials (short quote/table/figure refs from context)
- explanation (1–2 lines linking the quote to the answer)
- citations: a JSON array of objects with keys ref_id, page, evidence_text, evidence_type

Normalization:
- Numeric: put the number in answer_value; unit in answer_unit.
- Ranges: answer_value as [low,high].
- TRUE/FALSE: answer in caps; answer_value 1/0; unit is_blank.
- Terms: answer_value = term; unit is_blank.
- If not answerable from context: answer = "{FALLBACK_ANSWER}", value/unit = is_blank.

Citation rules:
- Every citation ref_id must come from the candidate reference IDs or visible context headers.
- evidence_text must be an exact supporting span copied from a context snippet.
- page must be the page number shown in the snippet header when present, else null.
- evidence_type must be one of text, table, figure.
- If multiple snippets support the answer, include multiple citations.

Return ONLY JSON with keys:
answer, answer_value, answer_unit, ref_id, supporting_materials, explanation, citations

Question:
{question}

Candidate reference IDs (from retrieval): {candidate_ref_ids}

Context snippets:
"""
    ctx_lines: list[str] = []
    for i, c in enumerate(contexts, start=1):
        ref_id = c.get("ref_id")
        page = c.get("page")
        headings_val: Any = c.get("headings")
        heading_str = ""
        if isinstance(headings_val, list):
            heading_str = "; ".join(str(h) for h in headings_val if h)
        elif headings_val:
            heading_str = str(headings_val)
        chunk_type = str(c.get("chunk_type", "text") or "text")

        header_bits = [f"ref_id={ref_id}", f"type={chunk_type}"]
        if page is not None:
            header_bits.append(f"page={page}")
        if heading_str:
            header_bits.append(f"heading={heading_str}")

        header = ", ".join(header_bits)
        text = c.get("text", "").strip()
        ctx_lines.append(f"[{i}] ({header})\n{text}\n")

    ctx_block = "\n".join(ctx_lines)

    closing = """
---
Output JSON shape:
{
  "answer": "string",
  "answer_value": "string",
  "answer_unit": "string",
  "ref_id": ["paper_id"],
  "supporting_materials": "string",
  "explanation": "string",
  "citations": [
    {
      "ref_id": "paper_id",
      "page": 3,
      "evidence_text": "exact supporting span",
      "evidence_type": "text"
    }
  ]
}
JSON only.
"""

    return instructions + ctx_block + closing
