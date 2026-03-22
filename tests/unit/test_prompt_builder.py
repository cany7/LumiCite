from __future__ import annotations

from src.core.constants import FALLBACK_ANSWER
from src.generation.prompt_templates import build_prompt


def test_build_prompt_includes_current_context_fields_and_excludes_asset_path() -> None:
    prompt = build_prompt(
        question="What were the CO2 emissions from training GPT-3?",
        contexts=[
            {
                "chunk_id": "patterson2021_aaaabbbb",
                "doc_id": "patterson2021",
                "text": "Training GPT-3 resulted in 552 tCO2e.",
                "page_number": 8,
                "headings": ["4 Results"],
                "chunk_type": "text",
                "caption": "Figure 3. Emissions by model size",
                "asset_path": "data/assets/patterson2021/figure.png",
            }
        ],
        candidate_chunk_ids=["patterson2021_aaaabbbb"],
    )

    assert "What were the CO2 emissions from training GPT-3?" in prompt
    assert "chunk_id=patterson2021_aaaabbbb" in prompt
    assert "doc_id=patterson2021" in prompt
    assert "page=8" in prompt
    assert "headings=4 Results" in prompt
    assert "caption=Figure 3. Emissions by model size" in prompt
    assert "data/assets" not in prompt
    assert FALLBACK_ANSWER in prompt
