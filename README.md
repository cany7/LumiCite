[🇺🇸 English](./README.md) | [🇨🇳 简体中文](./README_ZH.md)

# Multimodal RAG System for Academic Papers

This project is an end-to-end multimodal RAG system for academic paper PDFs. It parses **main text, figures, and tables** into a unified searchable corpus, then combines **hybrid retrieval, reranking, and citation-aware answer generation** to support accurate evidence retrieval, strong recall, and reliable answer synthesis for academic use cases.

## Highlights

- **Background:** Academic papers often distribute essential evidence across figures and tables. Conventional RAG systems that depend solely on plain-text parsing can therefore miss important information and produce incomplete answers.
- **Limitations of Image Embedding Models:** Multimodal embedding models that jointly encode text and images can partially address this issue, but they still have limited ability to represent comparisons, trends, and structural relationships in charts and tables, which makes precise answering difficult.
- **Unified Multimodal Corpus:** This project represents main text, figures, and tables as a unified searchable corpus under a unified chunk schema, allowing different evidence types to participate in the same indexing and retrieval pipeline and improving both retrieval accuracy and answer generation.
- **Hybrid Retrieval Mechanism:** Academic retrieval requires both semantic recall and exact matching for key terms, metrics, and domain-specific terminology. To support both, the system combines a `FAISS` dense index with a `BM25` sparse index and provides multiple retrieval modes.
- **Dual-Path Retrieval and Reranking Optimization:** To better handle dense academic writing and semantically similar concepts, the system adopts dual-path retrieval, merges candidates with `RRF`, and applies a cross-encoder reranker for final ranking, improving overall retrieval quality and precision.
- **Multi-Interface Support:** To support different deployment scenarios, the project provides both a CLI and an HTTP API, covering local use as well as service-oriented deployment.
- **Offline Deployment Environment Support:** By default, answer generation uses a standard LLM API. The system can also switch to containerized `ollama` as a local LLM backend for restricted environments such as offline or internal-network deployments. This switch applies to both query generation and the visual summary step during ingestion.
- **Multiple Input Source Support:** The system supports three input sources: `local_dir`, `url_csv`, and `url_list`, covering both local PDF ingestion and batch import from URLs.

## System Architecture Overview

```text
Source Layer
  -> local_dir / url_csv / url_list

Ingestion Layer
  -> PDF acquisition + authoritative snapshot
  -> local structured parsing
  -> multimodal normalization
  -> TextChunk / FigureChunk / TableChunk

Indexing Layer
  -> embedding generation
  -> FAISS vector index
  -> BM25 sparse index

Retrieval Layer
  -> dense / sparse / hybrid retrieval
  -> RRF fusion
  -> cross-encoder reranking

Generation Layer
  -> answer synthesis
  -> citation enrichment

Interface Layer
  -> rag parse / rag search / rag query
  -> HTTP API
```

## Requirements and Installation

For dependencies and version constraints, see `pyproject.toml`, `uv.lock`, and the related configuration files.

Install dependencies:

```bash
uv sync
cp .env.example .env
```

It is recommended to configure `.env` in the project root based on `.env.example`. Common environment variables include:

```env
RAG_API_BASE_URL=
RAG_API_KEY=
RAG_API_MODEL=

RAG_VISUAL_API_BASE_URL=
RAG_VISUAL_API_KEY=
RAG_VISUAL_API_MODEL=
```

Configuration notes:

- `RAG_API_*` can be used as the general model API configuration for both the visual summary step during ingestion and answer generation in the generation stage.
- `RAG_VISUAL_API_*` can be used to configure a separate model API specifically for image understanding.
- `RAG_OLLAMA_MODEL` specifies the model name for the local containerized `ollama` backend. The default is `qwen3.5:4b`.
- The default model for the generation stage is `qwen/qwen3.5-27b`, which can be changed via `RAG_API_MODEL` or `rag query --model`.
- The default model for the ingestion stage is `qwen/qwen2.5-vl-7b-instruct`, which can be changed via `RAG_VISUAL_API_MODEL`.
- When `ollama` is started with `docker compose`, it automatically pulls the model specified by `RAG_OLLAMA_MODEL`. When the CLI switches to `--llm ollama`, it waits until the model is ready before sending requests.
- The PDF parsing module is invoked directly by `rag parse`. The default compute device is `cpu`, and it can be changed with `rag parse --device`.

## Quick Start

For the first run, it is recommended to initialize the system in the following order:

1. Put PDF files into `data/pdfs/`
2. Run `rag parse` to download the required models, perform ingestion, generate embeddings, and build indexes
3. Use `rag search` or `rag query` for retrieval and QA

Examples:

```bash
uv run rag parse --source local_dir --path data/pdfs/
uv run rag search "energy consumption of GPT-3" --top-k 5
uv run rag query "What were the CO2 emissions from training GPT-3?"
```

## CLI Usage

### `rag parse`

`rag parse` synchronizes the paper knowledge base from the current input source, then completes parsing, chunk normalization, embedding generation, and index updates. It is intended for the initial knowledge base build, or for resynchronization after documents are added, replaced, or removed.

On the first run, `rag parse` automatically checks for the required core dependencies, downloads them when needed, and caches them in `data/model_cache/`.

Documents are processed sequentially. PDF parsing and chunk normalization are completed first, and once all documents have been processed, the system writes `chunks.jsonl`, generates embeddings, and updates the indexes in a single batch.

If the process is interrupted unexpectedly, the current run may be incomplete. In that case, rerunning `rag parse` is recommended.

Default options:

- `--source local_dir`
- `--device cpu`
- `--llm api`

Common arguments:

- `--source`
  - Options: `local_dir`, `url_csv`, `url_list`
- `--path`
  - Local input path such as `data/pdfs/`, `data/pdfs/papers.csv`, or `data/pdfs/papers.txt`
- `--device`
  - Parsing device, such as `cpu`, `cuda`, `cuda:0`, `mps`, or `npu`
- `--llm`
  - Options: `api`, `ollama`; controls the backend used for figure visual summaries
- `--rebuild-index`
  - Force rebuild of the `FAISS` and `BM25` indexes
- `--retry-failed`
  - Retry only documents that failed in previous runs
- `--dry-run`
  - Print the execution plan without actually running it

Examples:

```bash
uv run rag parse --source local_dir --path data/pdfs/
uv run rag parse --source local_dir --path data/pdfs/ --llm ollama
```

### `rag search`

`rag search` performs retrieval only and does not call a generation model. It is useful for inspecting recall results, evaluating retrieval quality, or comparing different retrieval strategies. By default, it uses `hybrid` retrieval and returns results in JSON format.

Default options:

- `--top-k 10`
- `--retrieval-mode hybrid`
- `--no-rerank`
- `--output-format json`

Common arguments:

- `--top-k`
  - A positive integer specifying how many top results to return
- `--retrieval-mode`
  - Options: `dense`, `sparse`, `hybrid`
- `--rerank` / `--no-rerank`
  - Whether to enable reranking; enabling it improves result quality but reduces speed
- `--output-format`
  - Options: `json`, `table`

Example:

```bash
uv run rag search "energy consumption of PaLM 540B" --top-k 10
```

### `rag query`

`rag query` generates the final answer from retrieval results and returns a citation-grounded response. It is intended for paper QA, evidence localization, and result export. By default, it uses `hybrid` retrieval with reranking enabled.

Default options:

- `--top-k 5`
- `--retrieval-mode hybrid`
- `--rerank`
- `--llm api`

Common arguments:

- `--top-k`
  - A positive integer specifying how many retrieved results are used for answer generation
- `--retrieval-mode`
  - Options: `dense`, `sparse`, `hybrid`
- `--rerank` / `--no-rerank`
  - Whether to enable reranking
- `--llm`
  - Options: `api`, `ollama`
- `--model`
  - Model name; only effective when `--llm api` is used
- `--output`
  - Output file path such as `answer.json`
- `ollama` model
  - By default, the system automatically deploys and pulls the containerized `qwen3.5:4b` model. You can switch to a different model via `RAG_OLLAMA_MODEL`

Examples:

```bash
uv run rag query "What were the CO2 emissions from training GPT-3?"
uv run rag query "What were the CO2 emissions from training GPT-3?" --llm ollama
```

### `rag serve`

`rag serve` starts the project's HTTP API. It is suitable for local debugging, API integration, and service deployment. By default, it listens on `0.0.0.0:8000`.

Default options:

- `--host 0.0.0.0`
- `--port 8000`

Common arguments:

- `--host`
  - Host address such as `0.0.0.0` or `127.0.0.1`
- `--port`
  - Port number such as `8000`
- `--reload`
  - Enable hot reload for development

Example:

```bash
uv run rag serve --host 0.0.0.0 --port 8000
```

### `rag benchmark`

`rag benchmark` runs the project evaluation pipeline. At the current stage, it primarily evaluates retrieval quality and retrieval performance by comparing recall and ranking behavior across different retrieval modes, top-k settings, and rerank configurations, while also reporting retrieval latency.

Evaluation results are recorded at the question level, including core metrics such as `recall_at_k`, `mrr`, `ndcg_at_k`, `mean_retrieval_latency_ms`, and `p95_retrieval_latency_ms`. By default, benchmarking runs with `hybrid` retrieval and reranking enabled.

Default options:

- `--dataset data/train_QA.csv`
- `--retrieval-mode hybrid`
- `--top-k 5`
- `--rerank`
- `--output-dir data/benchmark_results/`
- `--tag run`

Common arguments:

- `--dataset`
  - Evaluation dataset path such as `data/benchmark.csv`
- `--retrieval-mode`
  - Options: `dense`, `sparse`, `hybrid`
- `--top-k`
  - A positive integer specifying how many results are retrieved during evaluation
- `--rerank` / `--no-rerank`
  - Whether to enable reranking
- `--output-dir`
  - Output directory such as `data/benchmark_results/`
- `--tag`
  - Run identifier such as `smoke`

Required fields in the benchmark dataset:

- `question_id`: unique question identifier
- `question`: evaluation question text
- `ref_doc_id`: the `doc_id` of the reference document for the ground-truth answer

Example:

```bash
uv run rag benchmark --dataset data/benchmark_QA.csv --tag smoke
```

## Data and Pipeline Schema

During runtime, the project produces a fixed set of key directories and intermediate artifacts:

- `data/pdfs/`: location for source PDF files and default URL input files
  - `papers.csv`: URL CSV input file; required field: `url`
  - `papers.txt`: TXT URL list input file, one URL per line
- `data/intermediate/mineru/`: raw intermediate outputs generated during parsing
- `data/assets/`: canonical asset directory for figures and tables
- `data/metadata/`: canonical chunks, embeddings, and retrieval indexes
- `rag.log`: unified error log in the project root

- The project uses a unified chunk schema across ingestion, retrieval, and answering
  - Every chunk includes `chunk_id`, `doc_id`, `text`, `chunk_type`, `page_number`, and `headings` as base fields
  - Figure and table chunks additionally include fields such as `caption`, `footnotes`, and `asset_path` for localization and supplementary context
- The retrieval stage uses a unified `SearchResult` schema for matched results, while the QA stage uses `Citation` for evidence references and returns final outputs through `RAGAnswer`
  - Together, this schema design covers the full data path from chunk normalization, embedding generation, and index construction to retrieval outputs and final answer generation

## Usage Notes

- When the knowledge base changes through document additions, replacements, removals, or modifications to `papers.csv` / `papers.txt`, you need to rerun `rag parse` before performing retrieval or QA.
- If the knowledge base has not changed, there is no need to rerun `rag parse`; you can use `rag search` or `rag query` directly.
- `rag search` is useful for inspecting recall results, examining matched content, and debugging retrieval strategies. `rag query` is intended for producing final answers with citations on top of retrieval.
- The generation stage uses an external model API by default. If a local backend is required, you can explicitly switch both `rag parse` and `rag query` to `ollama`.
- This project is particularly well suited to QA over academic papers where main text, figures, and tables must be used together, because the system brings all evidence types into a unified retrieval, reranking, and answer generation workflow.

## Troubleshooting

### `rag parse` failed

Check the following first:

- Whether the local parsing runtime is correctly installed and callable in the current environment
- Whether the input path is correct
- Whether the visual model API configuration is available
- Whether the input directory or URL file content matches the expected format

### `rag query --llm api` failed

Check the following first:

- `RAG_API_BASE_URL`
- `RAG_API_KEY`
- `RAG_API_MODEL`
- Whether the current model API is accessible through the OpenAI SDK

### `rag query --llm ollama` failed

Check the following first:

- Whether Docker is running properly
- Whether the local `ollama` container can be started automatically
- Whether the model specified by `RAG_OLLAMA_MODEL` has already been pulled or can be pulled successfully
- Whether the current environment allows the local model backend to initialize correctly

All runtime errors are logged to `rag.log` in the project root.

## Tests

The current `tests/` directory includes both unit tests and integration tests, mainly covering the following areas:

- Retrieval pipeline: dense, sparse, and hybrid retrieval, RRF fusion, and reranker behavior
- QA pipeline: prompt construction, generation output normalization, citation completion, answer validation, and fallback behavior
- Parsing and ingestion pipeline: text chunking, MinerU output mapping, figure/table asset handling, embedding generation, index construction, and persistence consistency
- Data sources and incremental processing: discovery and ingestion for `local_dir`, `url_csv`, and `url_list`, manifest-based incremental processing, failed-document retry, and stale document pruning
- Interface and runtime contracts: default CLI and FastAPI parameters, response structures, health checks, and error response formats
- Engineering constraints: configuration loading, logging and error types, Docker / CI configuration constraints, and schema field validation

## Example Data

The repository includes a sample paper CSV list at `data/pdfs/sample_ai_impacts.csv`, containing 30 papers related to AI environmental impact. It can be used as a sample knowledge base input.

The benchmark dataset `data/benchmark_QA.csv` is constructed from the sample papers above and is used to evaluate recall, ranking, and retrieval latency in the benchmark pipeline.

## Next Steps

- Add evaluation modules focused on generation quality, using either manually annotated answers or LLM-as-a-judge methods to further assess correctness, completeness, relevance, and clarity
- Introduce RAG evaluation frameworks such as RAGAs to evaluate answer-evidence consistency across dimensions such as faithfulness, answer relevancy, context precision, and context recall, and further improve generation quality

## References and Dependencies

This project is built on the following open-source frameworks and components:

- [cross-encoder/ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2): default reranker model
- [Docker](https://www.docker.com/): used to run local `ollama` containers and related services
- [FAISS](https://github.com/facebookresearch/faiss): used for vector indexing and similarity search
- [FastAPI](https://fastapi.tiangolo.com/): used to build the HTTP API
- [MinerU](https://github.com/opendatalab/MinerU): used for structured parsing of academic paper PDFs and figure asset export
- [OpenAI SDK](https://platform.openai.com/docs/overview): used for model API integration and multimodal / text generation calls
- [Ollama](https://ollama.com/): optional local generation backend
- [rank_bm25](https://github.com/dorianbrown/rank_bm25): used for sparse retrieval and keyword matching
- [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2): default embedding model
- [Typer](https://typer.tiangolo.com/): used to build the CLI
- [Uvicorn](https://www.uvicorn.org/): used to run the HTTP service
- [uv](https://docs.astral.sh/uv/): used for Python dependency management and command execution

## License

This project is released under the AGPL-3.0 license, consistent with the license requirements of related dependencies used in the project, including MinerU.