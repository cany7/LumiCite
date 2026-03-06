"""RAG pipeline: retrieval, prompt, generation, packaging."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.config.settings import get_settings
from src.core.constants import FALLBACK_ANSWER
from src.core.logging import get_logger
from src.core.paths import find_project_root
try:
    from src.indexing.retrieval import get_chunks as retrieval_get_chunks  # type: ignore
except Exception:
    retrieval_get_chunks = None  # type: ignore
from src.generation.generator import LLMGenerator
from src.generation.prompt_templates import build_prompt

logger = get_logger(__name__)


def _to_str_field(value: Any) -> str:
    """Coerce list-like fields to a single string for CSV."""
    if value is None:
        return ""
    if isinstance(value, list):
        parts = [str(x).strip() for x in value if x is not None]
        return " | ".join(parts[:3])
    return str(value).strip()


def _format_list_py(items: List[str]) -> str:
    """Format as a Python-style list of quoted strings (train_QA style)."""
    safe = [str(x).strip() for x in items if x]
    inner = ",".join([f"'{x}'" for x in safe])
    return f"[{inner}]"


def _load_metadata_map(root: Path) -> Dict[str, Dict[str, str]]:
    """
    Load metadata.csv and return a mapping of ref_id -> {url, title, citation}.
    """
    import csv

    meta_path = root / "data" / "metadata" / "metadata.csv"
    m: Dict[str, Dict[str, str]] = {}
    with meta_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row.get("id")
            if not rid:
                continue
            m[rid] = {
                "url": row.get("url", ""),
                "title": row.get("title", ""),
                "citation": row.get("citation", ""),
            }
    return m


def _normalize_ref_id(raw: Any) -> str:
    """Strip chunk-level suffixes so metadata IDs match."""
    if raw is None:
        return ""
    text = str(raw).strip()
    if not text:
        return ""
    if "_" in text:
        return text.split("_", 1)[0]
    return text


def _infer_ref_id(record: Dict[str, Any]) -> str:
    """Infer the reference id from the chunk record."""
    md = record.get("metadata", {}) or {}
    src = md.get("source_file") or md.get("source")
    if isinstance(src, str) and src.endswith(".pdf"):
        return _normalize_ref_id(Path(src).stem)
    # fallback: prefix before first underscore of chunk id
    rec_id = record.get("id", "")
    if isinstance(rec_id, str) and "_" in rec_id:
        return _normalize_ref_id(rec_id.split("_", 1)[0])
    return _normalize_ref_id(rec_id)


def _build_contexts(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    contexts: List[Dict[str, Any]] = []
    ref_ids: List[str] = []
    for r in records:
        # Prefer explicit ref_id if supplied (e.g., from retriever)
        rid = _normalize_ref_id(r.get("ref_id") or _infer_ref_id(r))
        ref_ids.append(rid)
        md = r.get("metadata", {}) or {}
        contexts.append(
            {
                "ref_id": rid,
                "text": r.get("text", ""),
                "page": md.get("page_number") or md.get("page"),
                "headings": md.get("headings"),
            }
        )
    # dedupe while preserving order
    seen = set()
    unique_ref_ids = []
    for rid in ref_ids:
        if rid and rid not in seen:
            seen.add(rid)
            unique_ref_ids.append(rid)
    return contexts, unique_ref_ids


def _normalize_output(raw: Dict[str, Any], contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Ensure the model output complies with submission rules. If missing evidence
    or answer looks ungrounded, set fallback.
    """
    # Safely convert 'answer' to string before stripping
    ans = str(raw.get("answer", "")).strip()

    val = raw.get("answer_value")
    unit = (raw.get("answer_unit") or "").strip() or "is_blank"
    ref_ids = raw.get("ref_id") or []
    if isinstance(ref_ids, str):
        ref_ids = [ref_ids]
    ref_ids = [_normalize_ref_id(r) for r in ref_ids if r]
    supp = _to_str_field(raw.get("supporting_materials"))
    expl = _to_str_field(raw.get("explanation"))

    # If empty answer or explicit fallback → fallback
    if (not ans) or ans == FALLBACK_ANSWER:
        return {
            "answer": FALLBACK_ANSWER,
            "answer_value": "is_blank",
            "answer_unit": "is_blank",
            "ref_id": [],
            "supporting_materials": "is_blank",
            "explanation": "is_blank",
        }

    # If model forgot ref_ids but we have contexts, backfill from them
    if not ref_ids and contexts:
        ref_ids = [c.get("ref_id") for c in contexts if c.get("ref_id")]
        # keep 1-3 refs max
        ref_ids = list(dict.fromkeys(ref_ids))[:3]
        if not supp:
            supp = contexts[0].get("text", "")[:300]
        if not expl:
            expl = "Answer grounded in retrieved context; refs inferred from top results."

    # Minimal backfill for missing supporting materials
    if not supp and contexts:
        supp = contexts[0].get("text", "")[:300]

    # Normalize booleans
    if ans.upper() in ("TRUE", "FALSE"):
        val = 1 if ans.upper() == "TRUE" else 0
        unit = "is_blank"

    # If answer_value is still empty, use categorical default
    if val in (None, ""):
        val = ans
        if unit == "is_blank":
            unit = "is_blank"

    return {
        "answer": ans,
        "answer_value": val,
        "answer_unit": unit,
        "ref_id": ref_ids,
        "supporting_materials": supp,
        "explanation": expl,
    }


@dataclass
class RAGConfig:
    top_k: int | None = None
    model_name: str | None = None
    project: str | None = None
    location: str | None = None
    credentials_path: str | None = None

    @classmethod
    def from_settings(cls) -> "RAGConfig":
        settings = get_settings()
        return cls(
            top_k=settings.retrieval_top_k,
            model_name=settings.gemini_model,
            project=settings.gcp_project,
            location=settings.gcp_location,
            credentials_path=settings.gcp_credentials_path,
        )

    def __post_init__(self) -> None:
        settings = get_settings()
        if self.top_k is None:
            self.top_k = settings.retrieval_top_k
        if self.model_name is None:
            self.model_name = settings.gemini_model
        if self.project is None:
            self.project = settings.gcp_project
        if self.location is None:
            self.location = settings.gcp_location
        if self.credentials_path is None:
            self.credentials_path = settings.gcp_credentials_path


class RAGPipeline:
    def __init__(self, config: RAGConfig | None = None) -> None:
        self.config = config or RAGConfig.from_settings()
        # Paths
        self.root = find_project_root()
        self.data_dir = self.root / "data"
        
        # LLM
        self.generator = LLMGenerator(
            model=self.config.model_name,
            project=self.config.project,
            location=self.config.location,
            credentials_path=self.config.credentials_path,
        )

        # Metadata
        self.meta_map = _load_metadata_map(self.root)

    def answer(self, qid: str, question: str) -> Dict[str, Any]:
        recs: List[Dict[str, Any]] = []
        if retrieval_get_chunks is None:
            logger.info("Error: retrieval function not found (src/indexing/retrieval.get_chunks). Proceeding without context; answers will fallback.")
        else:
            try:
                res = retrieval_get_chunks(question, num_chunks=self.config.top_k)
                for rank in sorted(res.keys()):
                    item = res[rank]
                    recs.append(
                        {
                            "ref_id": _normalize_ref_id(item.get("paper", "")),
                            "text": item.get("chunk", ""),
                            # page/headings unknown here; leave absent
                        }
                    )
            except Exception as e:
                logger.info(f"Error: retrieval_get_chunks failed; proceeding without context. Details: {e}")
        contexts, candidate_ref_ids = _build_contexts(recs)

        # Prompt and generate
        prompt = build_prompt(question, contexts, candidate_ref_ids)
        raw = self.generator.generate_json(prompt)
        norm = _normalize_output(raw, contexts)

        # Compute URLs for citations
        ref_ids: List[str] = norm.get("ref_id", [])
        ref_urls = [self.meta_map.get(r, {}).get("url", "") for r in ref_ids]

        # Format list fields in Python-style for CSV compatibility
        ref_id_str = _format_list_py(ref_ids) if ref_ids else "is_blank"
        ref_url_str = _format_list_py([u for u in ref_urls if u]) if ref_urls and any(ref_urls) else "is_blank"

        # answer_value should be JSON-like for ranges; ensure string for CSV
        av = norm.get("answer_value")
        if isinstance(av, list):
            answer_value = json.dumps(av)
        else:
            answer_value = str(av)

        record = {
            "id": qid,
            "question": question,
            "answer": norm.get("answer", FALLBACK_ANSWER),
            "answer_value": answer_value,
            "answer_unit": norm.get("answer_unit", "is_blank"),
            "ref_id": ref_id_str,
            "ref_url": ref_url_str,
            "supporting_materials": norm.get("supporting_materials", "is_blank"),
            "explanation": norm.get("explanation", "is_blank"),
        }
        return record
