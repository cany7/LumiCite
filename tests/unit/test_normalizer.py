from __future__ import annotations

from src.core.constants import FALLBACK_ANSWER
from src.generation.llm_client import normalize_generation_payload, parse_json_response
from src.generation.rag_pipeline import _normalize_answer_payload


def test_parse_json_response_strips_markdown_fences() -> None:
    payload = parse_json_response(
        """```json
{"answer":"552 tCO2e","supporting_materials":"evidence","explanation":"because","citations":[]}
```"""
    )

    assert payload is not None
    assert payload["answer"] == "552 tCO2e"


def test_normalize_generation_payload_preserves_current_fields() -> None:
    payload = normalize_generation_payload(
        {
            "answer": "TRUE",
            "supporting_materials": "evidence",
            "explanation": "because",
            "citations": [],
        }
    )

    assert payload["answer"] == "TRUE"
    assert payload["citations"] == []


def test_normalize_answer_payload_backfills_grounded_fields() -> None:
    normalized = _normalize_answer_payload(
        {
            "answer": "552 tCO2e",
            "supporting_materials": "",
            "explanation": "",
            "citations": [{"chunk_id": "patterson2021_aaaabbbb"}],
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


def test_normalize_answer_payload_returns_fallback_for_blank_answer() -> None:
    normalized = _normalize_answer_payload({"answer": " ", "citations": []}, contexts=[])

    assert normalized["answer"] == FALLBACK_ANSWER
    assert normalized["citations"] == []
