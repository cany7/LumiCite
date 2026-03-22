from __future__ import annotations

from src.core.constants import FALLBACK_ANSWER
from src.core.schemas import Citation, ChunkType, RAGAnswer
from src.generation.verifier import Verifier


def _contexts() -> list[dict]:
    return [
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
    ]


def test_verifier_passes_grounded_answer() -> None:
    answer = RAGAnswer(
        answer="552 tCO2e",
        supporting_materials="Training GPT-3 resulted in 552 tCO2e according to Patterson et al.",
        explanation="The cited evidence gives the total emissions.",
        citations=[
            Citation(
                doc_id="patterson2021",
                chunk_id="patterson2021_aaaabbbb",
                page_number=8,
                evidence_text="Training GPT-3 resulted in 552 tCO2e according to Patterson et al.",
                evidence_type=ChunkType.TEXT,
            )
        ],
    )

    result = Verifier().verify(answer, _contexts())

    assert result.passed is True
    assert result.corrected_output is None
    assert result.confidence > 0.8


def test_verifier_falls_back_for_unsupported_citations() -> None:
    answer = RAGAnswer(
        answer="999 tCO2e",
        supporting_materials="A fabricated citation.",
        explanation="Unsupported claim.",
        citations=[
            Citation(
                doc_id="patterson2021",
                chunk_id="patterson2021_aaaabbbb",
                page_number=8,
                evidence_text="A fabricated citation.",
                evidence_type=ChunkType.TEXT,
            )
        ],
    )

    result = Verifier().verify(answer, _contexts())

    assert result.passed is False
    assert result.corrected_output is not None
    assert result.corrected_output["answer"] == FALLBACK_ANSWER


def test_verifier_backfills_supporting_materials_from_supported_citation() -> None:
    answer = RAGAnswer(
        answer="552 tCO2e",
        supporting_materials="is_blank",
        explanation="The cited evidence gives the total emissions.",
        citations=[
            Citation(
                doc_id="patterson2021",
                chunk_id="patterson2021_aaaabbbb",
                page_number=8,
                evidence_text="Training GPT-3 resulted in 552 tCO2e according to Patterson et al.",
                evidence_type=ChunkType.TEXT,
            )
        ],
    )

    result = Verifier().verify(answer, _contexts())

    assert result.passed is True
    assert result.corrected_output is not None
    assert "552 tCO2e" in result.corrected_output["supporting_materials"]
