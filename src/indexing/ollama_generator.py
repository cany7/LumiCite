"""RAG generator using Ollama (free local models) with the local retrieval function."""

from typing import Dict, Any, Optional
import json
import requests
from src.core.logging import get_logger

logger = get_logger(__name__)


def rag_ollama_answer(
        question: str,
        chunks_dict: Optional[Dict[int, Dict[str, Any]]],
        model: str = None,
        max_context_chars: int = 12000,
        temperature: float = 0.1,
        ollama_url: str = "http://localhost:11434"
) -> Dict[str, Any]:
    """
    RAG-style generator using Ollama (free local LLM):
      - Takes a question and dict of retrieved chunks from get_chunks()
      - Builds context from chunks
      - Calls Ollama API for structured JSON response
      - Returns parsed answer with all required fields

    Args:
        question: The user's question
        chunks_dict: Output from get_chunks(), format:
            {
                1: {"chunk": "...", "paper": "...", "rank": 1},
                2: {"chunk": "...", "paper": "...", "rank": 2},
                ...
            }
        model: Ollama model name (llama2, mistral, llama3, etc.)
        max_context_chars: Maximum context length
        temperature: Model temperature (0.0-1.0)
        ollama_url: Base URL for Ollama API

    Returns:
        Dictionary with keys: answer, answer_value, answer_unit, ref_id,
        supporting_materials, explanation

    Requirements:
        - Ollama installed: https://ollama.ai/download
        - Model pulled: ollama pull llama2 (or mistral, llama3, etc.)
        - Ollama running: ollama serve (usually runs automatically)
    """

    # Fallback response structure
    FALLBACK = {
        "answer": "Unable to answer with confidence based on the provided documents.",
        "answer_value": "is_blank",
        "answer_unit": "is_blank",
        "ref_id": [],
        "supporting_materials": "is_blank",
        "explanation": "is_blank",
    }

    # Auto-detect model if not specified
    if model is None:
        try:
            resp = requests.get(f"{ollama_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                if models:
                    model = models[0].get("name", "llama2")
                    logger.info(f"Auto-detected model: {model}")
                else:
                    logger.info("No Ollama models found. Please run: ollama pull llama2")
                    return FALLBACK
            else:
                model = "llama2"
        except:
            model = "llama2"

    # Handle no chunks / failed retrieval
    if not chunks_dict:
        logger.info("No chunks retrieved - returning fallback")
        return FALLBACK

    # Build context from chunks
    ordered_items = sorted(chunks_dict.items(), key=lambda kv: kv[0])

    context_blocks = []
    paper_ids = []

    for rank, info in ordered_items:
        chunk_text = info.get("chunk", "").strip()
        paper = info.get("paper", "unknown")

        if paper and paper != "unknown":
            paper_ids.append(paper)

        context_blocks.append(
            f"[{rank}] (ref_id={paper})\n{chunk_text}"
        )

    full_context = "\n\n".join(context_blocks)

    # Trim context if too long
    if len(full_context) > max_context_chars:
        full_context = full_context[:max_context_chars] + "\n[Context truncated]"

    # Get unique paper IDs for reference
    unique_papers = list(dict.fromkeys([p for p in paper_ids if p]))

    # Build structured prompt (simplified for smaller models)
    prompt = f"""Answer this question using the context below. Output ONLY valid JSON.

Question: {question}

Context:
{full_context}

Instructions:
- Extract the specific answer from the context
- If the answer has a number, put just the number in "answer_value" and the unit in "answer_unit"
- Include reference paper IDs in "ref_id" as a list
- Quote relevant text in "supporting_materials"

Response Rules:
Use double quotes for all keys and all string values.
For lists (like ref_id), use JSON arrays, e.g. "ref_id": ["id1", "id2"].
Do not use any single quotes ' anywhere in the JSON.
Do not add comments, markdown, or text outside the JSON object.
The JSON object must have the keys: answer, answer_value, answer_unit, ref_id, supporting_materials, explanation.

Output this exact JSON format:
{{
  "answer": "full sentence answer",
  "answer_value": "number or value",
  "answer_unit": "unit (like tCO2e, kg, etc) or is_blank",
  "ref_id": {unique_papers},
  "supporting_materials": "quote from context",
  "explanation": "brief explanation"
}}

Here is an example of a valid JSON response:

{{
  "answer": "string",
  "answer_value": "string or number as string",
  "answer_unit": "string",
  "ref_id": ["paper_id1", "paper_id2"],
  "supporting_materials": "short explanation",
  "explanation": "longer explanation"
}}

JSON only, no other text:"""

    # Call Ollama API
    try:
        logger.info(f"Calling Ollama with model: {model}")

        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": 500  # Limit response length
                }
            },
            timeout=120  # 2 minute timeout for local generation
        )

        if response.status_code != 200:
            logger.info(f"Ollama API error: {response.status_code} - {response.text}")
            return FALLBACK

        # Parse Ollama response
        ollama_result = response.json()
        content = ollama_result.get("response", "").strip()

        logger.info(f"Raw Ollama response: {content[:200]}...")

        # Try to parse JSON from response
        try:
            # First try direct parse
            result = json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                json_str = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                json_str = content[start:end].strip()
            else:
                # Try to find JSON object boundaries
                start = content.find("{")
                end = content.rfind("}")
                if start == -1 or end == -1:
                    logger.info(f"Could not find JSON in response: {content[:300]}")
                    return FALLBACK
                json_str = content[start:end + 1]

            try:
                result = json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.info(f"Failed to parse JSON: {e}")
                logger.info(f"Attempted to parse: {json_str[:300]}")
                return FALLBACK

        # Validate and normalize output
        answer = result.get("answer", "").strip()

        # If answer is fallback or empty, return fallback structure
        if not answer or answer == FALLBACK["answer"]:
            return FALLBACK

        # Get ref_id as list
        ref_ids = result.get("ref_id", [])
        if isinstance(ref_ids, str):
            # Try to parse if it's a string representation of list
            if ref_ids.startswith("[") and ref_ids.endswith("]"):
                try:
                    ref_ids = json.loads(ref_ids)
                except:
                    ref_ids = [ref_ids.strip("[]").strip()]
            else:
                ref_ids = [ref_ids]
        if not isinstance(ref_ids, list):
            ref_ids = []

        # If model didn't provide ref_ids, use papers from retrieval
        if not ref_ids and unique_papers:
            ref_ids = unique_papers[:3]

        # Normalize answer_value
        answer_value = result.get("answer_value", "")
        answer_unit = result.get("answer_unit", "is_blank")

        # Handle TRUE/FALSE
        if answer.upper() in ("TRUE", "FALSE"):
            answer_value = 1 if answer.upper() == "TRUE" else 0
            answer_unit = "is_blank"

        # If answer_value empty, use answer text
        if not answer_value:
            answer_value = answer

        return {
            "answer": answer,
            "answer_value": answer_value if answer_value else "is_blank",
            "answer_unit": answer_unit if answer_unit else "is_blank",
            "ref_id": ref_ids,
            "supporting_materials": result.get("supporting_materials", "is_blank"),
            "explanation": result.get("explanation", "is_blank"),
        }

    except requests.exceptions.ConnectionError:
        logger.info("Error: Cannot connect to Ollama. Make sure:")
        logger.info("  1. Ollama is installed: https://ollama.ai/download")
        logger.info("  2. Ollama is running: 'ollama serve' or check if running")
        logger.info(f"  3. Ollama is accessible at: {ollama_url}")
        return FALLBACK
    except requests.exceptions.Timeout:
        logger.info("Error: Ollama request timed out. Model may be too slow or not loaded.")
        return FALLBACK
    except Exception as e:
        logger.info(f"Error calling Ollama API: {e}")
        return FALLBACK


# Example usage combining with the local retrieval function
if __name__ == "__main__":
    # Check if Ollama is available
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            logger.info("Available Ollama models:")
            for m in models:
                logger.info(f"  - {m.get('name', 'unknown')}")
            logger.info()
        else:
            logger.info("Warning: Ollama API accessible but returned unexpected response")
    except requests.exceptions.ConnectionError:
        logger.info("ERROR: Cannot connect to Ollama!")
        logger.info("\nPlease install and start Ollama:")
        logger.info("  1. Download from: https://ollama.ai/download")
        logger.info("  2. Install and restart terminal")
        logger.info("  3. Pull a model: ollama pull llama2")
        logger.info("  4. Run this script again\n")
        exit(1)

    from retrieval import get_chunks

    # Test question
    question = "What were the net CO2e emissions from training the GShard-600B model?"

    logger.info(f"Question: {question}\n")

    # Step 1: Retrieve chunks
    logger.info("Step 1: Retrieving relevant chunks...")
    chunks = get_chunks(question, num_chunks=3)

    if chunks:
        logger.info(f"Retrieved {len(chunks)} chunks\n")
        for rank, info in chunks.items():
            logger.info(f"Rank {rank} - Paper: {info['paper']}")
            logger.info(f"Preview: {info['chunk'][:150]}...\n")
    else:
        logger.info("No chunks retrieved\n")

    # Step 2: Generate answer
    logger.info("Step 2: Generating answer with Ollama...")
    logger.info("(This may take 10-30 seconds for local generation)\n")

    # Auto-detect available model (will use gemma2:2b if available)
    result = rag_ollama_answer(question, chunks, model=None)

    # Step 3: Display results
    logger.info("\n" + "=" * 60)
    logger.info("FINAL ANSWER")
    logger.info("=" * 60)
    logger.info(f"Answer: {result['answer']}")
    logger.info(f"Answer Value: {result['answer_value']}")
    logger.info(f"Answer Unit: {result['answer_unit']}")
    logger.info(f"Reference IDs: {result['ref_id']}")
    logger.info(f"Supporting Materials: {result['supporting_materials'][:200]}...")
    logger.info(f"Explanation: {result['explanation']}")
