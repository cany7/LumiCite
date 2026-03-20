from __future__ import annotations

from src.core.constants import FALLBACK_ANSWER
from src.generation.prompt_templates import build_prompt


def test_build_prompt_includes_question_context_and_schema_guidance():
    prompt = build_prompt(
        question="What were the CO2 emissions from training GPT-3?",
        contexts=[
            {
                "ref_id": "patterson2021",
                "text": "Training GPT-3 resulted in 552 tCO2e.",
                "page": 8,
                "headings": ["Results"],
                "chunk_type": "text",
            }
        ],
        candidate_ref_ids=["patterson2021"],
    )

    assert "What were the CO2 emissions from training GPT-3?" in prompt
    assert "Candidate reference IDs (from retrieval): ['patterson2021']" in prompt
    assert "ref_id=patterson2021, type=text, page=8, heading=Results" in prompt
    assert "Training GPT-3 resulted in 552 tCO2e." in prompt
    assert FALLBACK_ANSWER in prompt
    assert '"citations": [' in prompt
