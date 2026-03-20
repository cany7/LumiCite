from __future__ import annotations

from src.core.constants import FALLBACK_ANSWER
from src.core.schemas import Citation, RAGAnswer
from src.generation.verifier import Verifier


def _contexts() -> list[dict]:
    return [
        {
            "ref_id": "patterson2021",
            "text": "Training GPT-3 resulted in 552 tCO2e according to Patterson et al.",
            "page": 8,
            "chunk_type": "text",
        }
    ]


def test_verifier_passes_grounded_answer():
    answer = RAGAnswer(
        answer="552 tCO2e",
        answer_value="552",
        answer_unit="tCO2e",
        ref_id=["patterson2021"],
        supporting_materials="Training GPT-3 resulted in 552 tCO2e according to Patterson et al.",
        explanation="The cited evidence gives the total emissions.",
        citations=[
            Citation(
                ref_id="patterson2021",
                page=8,
                evidence_text="Training GPT-3 resulted in 552 tCO2e according to Patterson et al.",
                evidence_type="text",
            )
        ],
    )

    result = Verifier().verify(answer, _contexts())

    assert result.passed is True
    assert result.corrected_output is None
    assert result.confidence > 0.8


def test_verifier_corrects_missing_ref_ids_from_citations():
    answer = RAGAnswer(
        answer="552 tCO2e",
        answer_value="552",
        answer_unit="tCO2e",
        ref_id=[],
        supporting_materials="Training GPT-3 resulted in 552 tCO2e according to Patterson et al.",
        explanation="The cited evidence gives the total emissions.",
        citations=[
            Citation(
                ref_id="patterson2021",
                page=8,
                evidence_text="Training GPT-3 resulted in 552 tCO2e according to Patterson et al.",
                evidence_type="text",
            )
        ],
    )

    result = Verifier().verify(answer, _contexts())

    assert result.passed is True
    assert result.corrected_output is not None
    assert result.corrected_output["ref_id"] == ["patterson2021"]


def test_verifier_falls_back_for_unsupported_citations():
    answer = RAGAnswer(
        answer="999 tCO2e",
        answer_value="999",
        answer_unit="tCO2e",
        ref_id=["patterson2021"],
        supporting_materials="A fabricated citation.",
        explanation="Unsupported claim.",
        citations=[
            Citation(
                ref_id="patterson2021",
                page=8,
                evidence_text="A fabricated citation.",
                evidence_type="text",
            )
        ],
    )

    result = Verifier().verify(answer, _contexts())

    assert result.passed is False
    assert result.corrected_output is not None
    assert result.corrected_output["answer"] == FALLBACK_ANSWER


def test_verifier_falls_back_when_no_contexts_are_available():
    answer = RAGAnswer(
        answer="552 tCO2e",
        answer_value="552",
        answer_unit="tCO2e",
        ref_id=["patterson2021"],
        supporting_materials="Training GPT-3 resulted in 552 tCO2e according to Patterson et al.",
        explanation="The cited evidence gives the total emissions.",
        citations=[
            Citation(
                ref_id="patterson2021",
                page=8,
                evidence_text="Training GPT-3 resulted in 552 tCO2e according to Patterson et al.",
                evidence_type="text",
            )
        ],
    )

    result = Verifier().verify(answer, contexts=[])

    assert result.passed is False
    assert "No retrieved context above threshold." in result.warnings
    assert result.corrected_output is not None
    assert result.corrected_output["answer"] == FALLBACK_ANSWER


def test_verifier_backfills_supporting_materials_from_supported_citation():
    answer = RAGAnswer(
        answer="552 tCO2e",
        answer_value="552",
        answer_unit="tCO2e",
        ref_id=["patterson2021"],
        supporting_materials="is_blank",
        explanation="The cited evidence gives the total emissions.",
        citations=[
            Citation(
                ref_id="patterson2021",
                page=8,
                evidence_text="Training GPT-3 resulted in 552 tCO2e according to Patterson et al.",
                evidence_type="text",
            )
        ],
    )

    result = Verifier().verify(answer, _contexts())

    assert result.passed is True
    assert result.corrected_output is not None
    assert (
        result.corrected_output["supporting_materials"]
        == "Training GPT-3 resulted in 552 tCO2e according to Patterson et al."
    )
