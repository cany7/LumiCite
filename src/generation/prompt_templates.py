"""Prompt builder for structured answers with chunk-level citations."""
from __future__ import annotations

from typing import Any

from src.core.constants import FALLBACK_ANSWER


def build_prompt(
    question: str,
    contexts: list[dict[str, Any]],
    candidate_chunk_ids: list[str] | None = None,
) -> str:
    instructions = f"""
Answer using ONLY the context below. If the question cannot be answered directly and confidently from the provided context, return the fallback.

Return ONLY the following plain-text fields:
ANSWER:
SUPPORTING_MATERIALS:
EXPLANATION:
CITED_CHUNK_IDS:

Rules:
- First determine whether the provided context is directly relevant to the question.
- Topical similarity alone is not enough; the cited evidence must directly answer the question.
- If the context is irrelevant, loosely related, insufficient, or does not directly support an answer, return the fallback.
- Do not use outside knowledge.
- Keep the answer short, factual, and grounded in the cited evidence.
- Use chunk_id values exactly as provided in the context.
- Do not invent document IDs, page numbers, captions, asset paths, or any other missing details.
- CITED_CHUNK_IDS must include ONLY the chunk_id values directly needed to support the answer.
- Do not list candidate chunks that were not actually used.
- Keep CITED_CHUNK_IDS empty if the fallback is used.
- supporting_materials should be a brief evidence summary grounded only in the cited chunks.
- explanation should briefly state why the answer is supported by the cited evidence, or why the fallback was used.
- If fallback is used, set:
  - ANSWER to "{FALLBACK_ANSWER}"
  - SUPPORTING_MATERIALS to ""
  - EXPLANATION to "The provided context is irrelevant or insufficient to answer the question."
  - CITED_CHUNK_IDS to an empty list
- Do not output JSON.

Question:
{question}

Candidate chunk IDs:
{candidate_chunk_ids or [context.get("chunk_id") for context in contexts if context.get("chunk_id")]}

Context snippets:
"""

    snippet_lines: list[str] = []
    for index, context in enumerate(contexts, start=1):
        chunk_id = str(context.get("chunk_id", "") or "").strip()
        doc_id = str(context.get("doc_id", "") or "").strip()
        chunk_type = str(context.get("chunk_type", "text") or "text").strip()
        page_number = context.get("page_number")
        headings = context.get("headings") or []
        caption = str(context.get("caption", "") or "").strip()
        text = str(context.get("text", "") or "").strip()

        header_bits = [f"chunk_id={chunk_id}", f"doc_id={doc_id}", f"type={chunk_type}"]
        if page_number is not None:
            header_bits.append(f"page={page_number}")
        if headings:
            heading_text = " > ".join(str(item).strip() for item in headings if str(item).strip())
            if heading_text:
                header_bits.append(f"headings={heading_text}")
        if caption:
            header_bits.append(f"caption={caption}")

        snippet_lines.append(f"[{index}] ({', '.join(header_bits)})\n{text}\n")

    closing = """
---
Output format:
ANSWER: string
SUPPORTING_MATERIALS: string
EXPLANATION: string
CITED_CHUNK_IDS:
- doc_chunk_id
- another_chunk_id
Fields only.
"""

    return instructions + "\n".join(snippet_lines) + closing
