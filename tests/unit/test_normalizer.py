from __future__ import annotations

from src.core.constants import FALLBACK_ANSWER
from src.generation.llm_client import normalize_generation_payload, parse_generation_response, parse_json_response
from src.generation.rag_pipeline import _normalize_answer_payload


def test_parse_json_response_strips_markdown_fences() -> None:
    payload = parse_json_response(
        """```json
{"answer":"552 tCO2e","supporting_materials":"evidence","explanation":"because","citations":[]}
```"""
    )

    assert payload is not None
    assert payload["answer"] == "552 tCO2e"


def test_parse_json_response_repairs_invalid_backslash_escapes() -> None:
    payload = parse_json_response(
        """{
  "answer": "55%",
  "supporting_materials": "The context states that 75% of CVPR papers target accuracy and 20% of CVPR papers argue for a new efficiency result.",
  "explanation": "Chunk 2 explicitly provides the percentages for CVPR papers targeting accuracy (75%) and efficiency (20%), allowing for the calculation of the difference.",
  "citations": [
    {
      "chunk_id": "1907.10597_c15723a6",
      "evidence_text": "$7 5 \\%$ of CVPR papers",
      "evidence_type": "text"
    }
  ]
}"""
    )

    assert payload is not None
    assert payload["answer"] == "55%"
    assert payload["citations"][0]["evidence_text"] == "$7 5 \\%$ of CVPR papers"


def test_parse_generation_response_reads_sectioned_text() -> None:
    payload = parse_generation_response(
        """ANSWER: 55%
SUPPORTING_MATERIALS: CVPR papers target accuracy at 75% and efficiency at 20%.
EXPLANATION: The difference is 55 percentage points.
CITED_CHUNK_IDS:
- 1907.10597_c15723a6
- 1907.10597_tab_deadbeef
"""
    )

    assert payload is not None
    assert payload["answer"] == "55%"
    assert payload["cited_chunk_ids"] == ["1907.10597_c15723a6", "1907.10597_tab_deadbeef"]


def test_normalize_generation_payload_preserves_current_fields() -> None:
    payload = normalize_generation_payload(
        {
            "answer": "TRUE",
            "supporting_materials": "evidence",
            "explanation": "because",
            "citations": [],
            "cited_chunk_ids": ["paper1_deadbeef"],
        }
    )

    assert payload["answer"] == "TRUE"
    assert payload["citations"] == []
    assert payload["cited_chunk_ids"] == ["paper1_deadbeef"]


def test_normalize_answer_payload_builds_citations_from_selected_chunk_ids() -> None:
    normalized = _normalize_answer_payload(
        {
            "answer": "552 tCO2e",
            "supporting_materials": "",
            "explanation": "",
            "cited_chunk_ids": ["patterson2021_aaaabbbb"],
        },
        contexts=[
            {
                "doc_id": "patterson2021",
                "chunk_id": "patterson2021_aaaabbbb",
                "text": "Training GPT-3 resulted in 552 tCO2e according to Patterson et al.",
                "page_number": 8,
                "chunk_type": "text",
                "headings": ["4 Results"],
                "caption": "",
                "asset_path": "",
            }
        ],
    )

    assert normalized["answer"] == "552 tCO2e"
    assert normalized["supporting_materials"] == "Training GPT-3 resulted in 552 tCO2e according to Patterson et al."
    assert normalized["citations"][0].chunk_id == "patterson2021_aaaabbbb"
    assert normalized["citations"][0].doc_id == "patterson2021"


def test_normalize_answer_payload_does_not_backfill_unselected_contexts() -> None:
    normalized = _normalize_answer_payload(
        {
            "answer": "552 tCO2e",
            "supporting_materials": "alpha evidence",
            "explanation": "because",
            "cited_chunk_ids": [],
        },
        contexts=[
            {
                "doc_id": "patterson2021",
                "chunk_id": "patterson2021_aaaabbbb",
                "text": "Training GPT-3 resulted in 552 tCO2e according to Patterson et al.",
                "page_number": 8,
                "chunk_type": "text",
                "headings": ["4 Results"],
                "caption": "",
                "asset_path": "",
            }
        ],
    )

    assert normalized["answer"] == "552 tCO2e"
    assert normalized["citations"] == []


def test_normalize_answer_payload_returns_fallback_for_blank_answer() -> None:
    normalized = _normalize_answer_payload({"answer": " ", "citations": []}, contexts=[])

    assert normalized["answer"] == FALLBACK_ANSWER
    assert normalized["citations"] == []
