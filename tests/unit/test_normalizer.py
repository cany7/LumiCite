from __future__ import annotations

from src.core.constants import FALLBACK_ANSWER
from src.generation.llm_client import normalize_generation_payload, parse_json_response
from src.generation.rag_pipeline import _normalize_answer_payload


def test_parse_json_response_strips_markdown_fences():
    payload = parse_json_response(
        """```json
{"answer":"552 tCO2e","answer_value":"552","answer_unit":"tCO2e","ref_id":"patterson2021","supporting_materials":"evidence","explanation":"because","citations":[]}
```"""
    )

    assert payload is not None
    assert payload["answer"] == "552 tCO2e"


def test_normalize_generation_payload_coerces_ref_id_to_list():
    payload = normalize_generation_payload(
        {
            "answer": "TRUE",
            "answer_value": "1",
            "answer_unit": "",
            "ref_id": "paper1",
            "supporting_materials": "evidence",
            "explanation": "because",
            "citations": [],
        }
    )

    assert payload["ref_id"] == ["paper1"]
    assert payload["citations"] == []


def test_normalize_answer_payload_backfills_grounded_fields():
    normalized = _normalize_answer_payload(
        {
            "answer": "552 tCO2e",
            "answer_value": "",
            "answer_unit": "tCO2e",
            "ref_id": [],
            "supporting_materials": "",
            "explanation": "",
            "citations": [],
        },
        contexts=[
            {
                "ref_id": "patterson2021",
                "text": "Training GPT-3 resulted in 552 tCO2e according to Patterson et al.",
                "page": 8,
                "chunk_type": "text",
            }
        ],
    )

    assert normalized["answer"] == "552 tCO2e"
    assert normalized["answer_value"] == "552 tCO2e"
    assert normalized["ref_id"] == ["patterson2021"]
    assert normalized["supporting_materials"] == "is_blank"
    assert normalized["citations"][0].ref_id == "patterson2021"


def test_normalize_answer_payload_returns_fallback_for_blank_answer():
    normalized = _normalize_answer_payload({"answer": " ", "citations": []}, contexts=[])

    assert normalized["answer"] == FALLBACK_ANSWER
    assert normalized["answer_value"] == "is_blank"
    assert normalized["ref_id"] == []
