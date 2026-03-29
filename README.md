[🇺🇸 English](./README.md) | [🇨🇳 简体中文](./README_ZH.md)

# LumiCite

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](./pyproject.toml)
[![License](https://img.shields.io/badge/License-AGPL--3.0-green)](./LICENSE)
[![CI](https://github.com/cany7/LumiCite/actions/workflows/ci.yml/badge.svg)](https://github.com/cany7/LumiCite/actions/workflows/ci.yml)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)

LumiCite is an end-to-end, multimodal RAG system designed for academic research. It unifies **main text, images, and tables** from academic papers into a searchable corpus, combining **hybrid retrieval, reranking, and citation-aware answering** to provide accurate evidence retrieval, high recall, and reliable answer generation for academic applications.

## Highlights

- **Multimodal Evidence Modeling:** Converts main text, images, and tables into unified searchable evidence, addressing information loss in text-only RAG and the inability of pure image embeddings to capture deep structural details like comparisons, trends, and relationships.
- **Hybrid Retrieval Mechanism:** Combines `FAISS` vector indexing with `BM25` sparse indexing to balance semantic recall with keyword matching, covering both semantic search and exact matching for key terms or metrics in academic scenarios.
- **Query Explanation Enhanced Retrieval:** Performs query explanation and expansion before retrieval to handle implicit conditions, key value conversions, and missing baselines, improving recall for complex academic questions.
- **Dual-Path Retrieval and Reranking Optimization:** Uses dual-path retrieval and `RRF` fusion, followed by cross-encoder reranking to improve precision and reduce noise caused by semantically similar or dense text features.
- **Multi-Interface Support:** Provides both CLI and HTTP API interfaces to adapt to different application scenarios, supporting local usage and cloud-based service deployment.
- **Switchable LLM Backend:** Supports standard LLM APIs and can switch to a containerized `ollama` local backend to adapt to restricted deployment environments such as offline or internal networks.
- **Multi-Input Source Support:** Supports three input sources: `local_dir`, `url_csv`, and `url_list`, adapting to different scenarios for local PDF ingestion and batch URL import.

*Planned Feature: Introduce an Agentic Router Layer with LangGraph for query-aware policy selection, controlled retry, and adaptive retrieval/generation parameter tuning.

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

## Evaluation Results

### QA Generation Quality Evaluation

The system was evaluated using the [Example Data](#example-data) benchmark dataset under default parameter settings. The results can be found in [benchmark_QA_default_query_results.csv](./tests/benchmark_QA_default_query_results.csv).

Based on a "semantic equivalence" criterion, the system achieved 33 correct answers, resulting in an overall accuracy of 82.5%. The average retrieval latency was 891 ms, and the average generation latency was 1756 ms.

Retrieval and generation latencies are subject to factors such as cloud LLM API service status and the computing performance of the test platform, so these results are for reference only and may vary across different environments.

For some complex questions that failed under default parameters, targeted supplementary tests were conducted by adjusting reasoning_effort, rerank, and top_k. After these supplementary tests, only a small number of highly difficult questions remained reliably unanswered.

Current bottlenecks are mainly concentrated on a few complex question patterns, such as those requiring multi-table or image evidence synthesis, cross-document evidence integration and reasoning, and complex computational inference. Answering these questions depends not only on the completeness of retrieval results but also heavily on the comprehensive capabilities of the LLM used for generation. Switching to a stronger model in the future could provide further improvements.

At the same time, when the system could not accurately answer these complex questions, it correctly returned fallback results without forcing an answer. Overall, the system is able to stably generate accurate, high-quality, and evidence-constrained answers in academic knowledge base scenarios, effectively avoiding AI hallucinations.

### System Retrieval & Recall Performance Evaluation

We compared retrieval performance with Query Explanation enabled (default baseline) versus disabled to evaluate the system's overall retrieval, recall latency, and performance:

| Metric           | Query Explanation **ON** (Default) | Query Explanation **OFF** | Change |
|:-----------------|:-----------------------------------|:--------------------------|:-------|
| **Recall@10**    | 0.8780                             | 0.8780                    | = 0%   |
| **MRR**          | 0.8171                             | 0.7642                    | ▼ 6.5% |
| **NDCG@10**      | 0.8321                             | 0.7933                    | ▼ 4.7% |
| **Mean Latency** | 994.64 ms                          | 65.28 ms                  | ▼ 93%  |
| **P95 Latency**  | 1667.82 ms                         | 39.07 ms                  | ▼ 97%  |

**Analysis Conclusion**:

- **Ranking Quality**: On this dataset, enabling Query Explanation improved **MRR by 6.5%**. The optimized queries more accurately hit the semantic core, pushing the most relevant evidence significantly higher in the candidate list (usually directly to Top 1-2), providing more effective and precise evidence for answer generation.
- **Response Performance**: Disabling this function eliminates the LLM inference overhead, reducing retrieval latency by **93%** and achieving millisecond-level distinct response times.
- **Scenario Suggestions**: In practice, tradeoffs can be made flexibly based on requirements:
  - In **complex academic QA** scenarios, it is recommended to default to **enabled** to utilize the LLM to mine deep semantics and implicit conditions, trading time for accuracy to ensure the best answer quality.
  - In **latency-sensitive** (e.g., real-time completion) or highly specific keyword scenarios, it is recommended to disable it to pursue ultimate response speed.

## Environment Configuration & Installation

For project dependencies and version constraints, please refer to `pyproject.toml`, `uv.lock`, and related configuration files.

Install dependencies:

```bash
uv sync
cp .env.example .env
```

It is recommended to refer to `.env.example` to configure `.env` in the project root directory. Common environment variables are as follows:

```env
RAG_API_BASE_URL=
RAG_API_KEY=
RAG_API_MODEL=

RAG_VISUAL_API_BASE_URL=
RAG_VISUAL_API_KEY=
RAG_VISUAL_API_MODEL=
```

Configuration description:

- `RAG_API_*` uses OpenAI SDK common API specifications and is used for both ingest and generation stages.
- `RAG_VISUAL_API_*` can specify model API configuration separately for the ingest stage.
- `RAG_OLLAMA_MODEL` specifies the model name for the local containerized `ollama` backend, defaulting to `qwen3.5:4b`.
- The default model for the generation stage is `qwen/qwen3.5-27b`, which can be adjusted via `RAG_API_MODEL` or `rag query --model`.
- The default model for the ingest stage is `qwen/qwen2.5-vl-7b-instruct`, which can be adjusted via `RAG_VISUAL_API_MODEL`.
- When `docker compose` starts the `ollama` service, it will automatically pull the model specified by `RAG_OLLAMA_MODEL`; the CLI will wait for the model to be ready before making requests when switching to `--llm ollama`.
- The PDF parsing module is directly called by the `rag parse` command, with the default computing device being `cpu`, which can be adjusted via `rag parse --device`.

## Quick Start

For the first run, it is recommended to initialize in the following order:

1. Place PDF files into `data/pdfs/`
2. Execute `rag parse` to complete model download, ingestion, embedding, and index construction
3. Use `rag search` or `rag query` for retrieval and Q&A

## CLI Usage Instructions

### `rag parse`

`rag parse` is used to synchronize the paper knowledge base based on the current input source, completing parsing, chunk normalization, embedding writing, and index updates; this command applies to the initial construction of the knowledge base or re-synchronization after documents are added, replaced, or deleted.
On the first run, `rag parse` automatically checks and downloads the required core dependency modules and caches them to `data/model_cache/`.
Document parsing processes documents sequentially, completing PDF parsing and chunk normalization first, and then uniformly writing to `chunks.jsonl`, generating embeddings, and updating the index after all documents are processed.
If interrupted unexpectedly, the results of the current round may be incomplete, and it is recommended to re-execute `rag parse`.

Default options:

- `--source local_dir`
- `--device cpu`
- `--llm api`

Common parameters:

- `--source`
  - Options: `local_dir`, `url_csv`, `url_list`
- `--path`
  - Fill in the local input path, such as `data/pdfs/`, `data/pdfs/papers.csv`, `data/pdfs/papers.txt`
- `--device`
  - Options: `cpu`, `cuda`, `cuda:0`, `mps`, `npu` as the parsing computing device
- `--llm`
  - Options: `api`, `ollama`, used to control the backend used for figure visual summary
- `--rebuild-index`
  - Force rebuild of `FAISS` and `BM25` indexes
- `--retry-failed`
  - Retry only previously failed documents
- `--dry-run`
  - Only output the plan, do not actually execute

Example:

```bash
uv run rag parse --source local_dir --path data/pdfs/
```

Result example:

```json
{
  "event": "parse_summary",
  "status": "ok",
  "source": "local_dir",
  "device": "cpu",
  "processed_pdfs": 0,
  "failed_pdfs": 0,
  "skipped_pdfs": 30,
  "pruned_pdfs": 0,
  "chunks_written": 3113,
  "embeddings_written": 3113,
  "rebuild_index": false,
  "indices_rebuilt": false
}
```

### `rag search`

`rag search` performs only retrieval without calling the generation model, suitable for viewing recall results, checking retrieval quality, or comparing hit situations under different retrieval strategies; it uses `hybrid` retrieval by default and outputs JSON results.

Default options:

- `--top-k 10` (or `RAG_RETRIEVAL_TOP_K`)
- `--retrieval-mode hybrid`
- `--no-rerank`
- `--query-explanation`
- `--output-format json`

Common parameters:

- `--top-k`
  - Fill in a positive integer indicating the selection of the top k recall results; if not passed, the CLI reads `RAG_RETRIEVAL_TOP_K`, default `10`
- `--retrieval-mode`
  - Options: `dense`, `sparse`, `hybrid` retrieval modes
- `--rerank` / `--no-rerank`
  - Control whether to enable rerank; if enabled, `top_k * 5` candidates are taken for re-ranking first, and then the final `top_k` are retained, which provides better results but is slower
- `--query-explanation` / `--no-query-explanation`
  - Optionally enable query explanation enhanced retrieval, suitable for complex questions, enabled by default
  - Query explanation reasoning does not use `--reasoning-effort` and is only controlled by `RAG_QUERY_EXPLANATION_REASONING_EFFORT` in `.env`, default `none`
- `--output-format`
  - Options: `json`, `table`

Example:

```bash
uv run rag search "energy consumption of PaLM 540B"
```

Result example:

- The command returned a JSON payload with `total_results: 10`, `retrieval_mode: "hybrid"`, and `retrieval_latency_ms: 3263.582`.
- The top hits mixed text and figure chunks, including passages about real-world AI service energy consumption and figures mentioning PaLM (540B), GPT-3, Gemini, and hardware or energy cost trends.

### `rag query`

`rag query` generates the final answer based on the retrieval results and returns a response with citation information, suitable for paper Q&A, evidence positioning, and result export; it defaults to `hybrid` fusion retrieval, enables query explanation, and disables rerank; final citations will only come from retrieval chunks actually selected in the generation stage, and the CLI outputs human-readable format by default.

Default options:

- `--top-k 10` (or `RAG_RETRIEVAL_TOP_K`)
- `--retrieval-mode hybrid`
- `--no-rerank`
- `--query-explanation`
- `--llm api`
- `--reasoning-effort none`

Common parameters:

- `--top-k`
  - Fill in a positive integer indicating the number of retrieval candidates provided for the generation stage; if not passed, the CLI reads `RAG_RETRIEVAL_TOP_K`, default `10`
- `--retrieval-mode`
  - Options: `dense`, `sparse`, `hybrid`
- `--rerank` / `--no-rerank`
  - Control whether to enable rerank; if enabled, `top_k * 5` candidates are taken for re-ranking first, and then the final `top_k` results are sent to generation
- `--query-explanation` / `--no-query-explanation`
  - Optionally enable query explanation enhanced retrieval, suitable for complex questions, enabled by default
- `--llm`
  - Options: `api`, `ollama`
- `--model`
  - Fill in the model name, only effective when `--llm api`
- `--reasoning-effort`
  - Options: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, only effective when `--llm api`, default `none`
  - This parameter only affects final answer generation and does not affect query explanation
- `--output`
  - Fill in the output file path, such as `answer.txt` or `answer.json`
- `--output-format`
  - Options: `text`, `json`, default `text`
- `--json`
  - Optional `--output-format json`, returning unprocessed JSON structured text for testing usage
- `ollama` model
  - Defaults to using `qwen3.5:4b` to automatically deploy and pull the containerized Ollama model; different models can be switched via `RAG_OLLAMA_MODEL`

Example:

```bash
uv run rag query "What were the CO2 emissions from training GPT-3?"
```

Result example:

```text
Question
What were the CO2 emissions from training GPT-3?

Answer
The provided context does not state a single specific CO2 emission value for training GPT-3, but it cites several estimates, including approximately 0.5 million kg of CO2e for GPT-3 training.

Evidence
- Chunk [2] estimates GPT-3 training time at over 3.5 million hours.
- Chunk [3] states GPT-3's carbon footprint for training is approximately 0.5 million kg.
```

### `rag serve`

`rag serve` is used to start the project HTTP API, suitable for local debugging, interface integration, or service operation; default listens on `0.0.0.0:8000`.

The HTTP API's `POST /api/v1/search` and `POST /api/v1/query` requests also support `query_explanation: true` to enable retrieval-oriented query expansion consistent with the CLI.

Default options:

- `--host 0.0.0.0`
- `--port 8000`

Common parameters:

- `--host`
  - Fill in listening address, such as `0.0.0.0` or `127.0.0.1`
- `--port`
  - Fill in port number, such as `8000`
- `--reload`
  - Enable development hot reload

Example:

```bash
uv run rag serve --host 0.0.0.0 --port 8000
```

### `rag benchmark`

`rag benchmark` is used to run the project evaluation pipeline, currently mainly evaluating retrieval layer effects and retrieval performance, comparing recall and ranking performance under different retrieval modes, top-k, and rerank configurations, and calculating retrieval latency.
Evaluation results record core metrics at the question granularity, including `recall_at_k`, `mrr`, `ndcg_at_k`, `mean_retrieval_latency_ms`, and `p95_retrieval_latency_ms`; defaults to executing evaluation based on `hybrid` retrieval, enabling query explanation, and disabling rerank.

Default options:

- `--dataset data/train_QA.csv`
- `--retrieval-mode hybrid`
- `--top-k 10` (or `RAG_RETRIEVAL_TOP_K`)
- `--no-rerank`
- `--query-explanation`
- `--output-dir data/benchmark_results/`
- `--tag run`

Common parameters:

- `--dataset`
  - Fill in evaluation dataset path, such as `data/benchmark.csv`
- `--retrieval-mode`
  - Options: `dense`, `sparse`, `hybrid`
- `--top-k`
  - Fill in a positive integer indicating the number of retrievals during evaluation; if not passed, the CLI reads `RAG_RETRIEVAL_TOP_K`, default `10`
- `--rerank` / `--no-rerank`
  - Control whether to enable rerank; if enabled, `top_k * 5` candidates are taken for re-ranking first, and then the final `top_k` are retained
- `--query-explanation` / `--no-query-explanation`
  - Optionally rewrite the original question into a retrieval-oriented expanded query before each benchmark question retrieval; the expanded query is only used to enhance retrieval and is merged with the original query's candidate set before rerank, and does not participate in answer generation
  - Query explanation reasoning does not use command line parameters and is only controlled by `RAG_QUERY_EXPLANATION_REASONING_EFFORT` in `.env`, default `none`
- `--output-dir`
  - Fill in result output directory, such as `data/benchmark_results/`
- `--tag`
  - Fill in the identifier name for this run, such as `smoke`

Test dataset required fields:

- `question_id`: Unique question identifier
- `question`: Evaluation question text
- `ref_doc_id`: Document `doc_id` corresponding to the standard answer

Example:

```bash
uv run rag benchmark --dataset data/benchmark_QA.csv --tag smoke
```

Result example:

```json
{
  "report": "data/benchmark_results/smoke_20260322_175009_report.json",
  "summary": "data/benchmark_results/smoke_20260322_175009_summary.csv"
}
```

## Usage Suggestions

- If you make changes to the knowledge base (additions, replacements, deletions, or modifications to `papers.csv` or `papers.txt`), you must manually execute `rag parse` before performing retrieval or Q&A.
- If the knowledge base has not changed, there is no need to repeat `rag parse`; you can proceed directly to `rag search` or `rag query`.
- `rag search` is suitable for checking recall results, inspecting hits, and debugging retrieval strategies; `rag query` is intended for generating final answers based on retrieval results and providing citation information.
- By default, the generation stage uses an external model API. To use a local backend, you must explicitly switch to `ollama` for both `rag parse` and `rag query`.
- This project is optimized for academic Q&A scenarios that require combining main text, images, and tables, as the system unifies these evidence types throughout the retrieval, reranking, and answer generation process.

## Project Testing

Currently, the `tests/` directory contains unit and integration tests, primarily covering the following functional areas:

- **Retrieval Pipeline:** Dense, sparse, and hybrid retrieval, RRF fusion, and reranker logic.
- **Q&A Pipeline:** Prompt construction, generation normalization, citation completion, answer validation, and fallback behavior.
- **Parsing and Ingestion Pipeline:** Text chunking, MinerU output mapping, image/table asset processing, embedding generation, index construction, and persistence consistency.
- **Data Source and Incremental Processing:** Discovery and crawling for `local_dir`, `url_csv`, and `url_list`; manifest-based incremental processing, retry logic, and stale document pruning.
- **Interface and Runtime Contracts:** CLI and FastAPI parameter defaults, response structures, health checks, and error response formats.
- **Engineering Constraints:** Configuration loading, logging and error types, Docker/CI configuration constraints, and field validation for core schemas.

## Data and Pipeline Schema

The project generates a fixed set of key directories and intermediate results during operation:

- `data/pdfs/`: Location for original PDF documents and default URL input files.
  - `papers.csv`: URL CSV input file (required field: `url`).
  - `papers.txt`: TXT URL list input file (one URL per line).
- `data/intermediate/mineru/`: Raw intermediate outputs from the parsing stage.
- `data/assets/`: Canonical asset directory for images and tables.
- `data/metadata/`: Canonical directory for chunks, embeddings, and retrieval indexes.
- `rag.log`: Unified error log located in the project root directory.

- The project uses a unified chunk schema to connect the ingestion, retrieval, and answering stages.
  - Each chunk contains basic fields: `chunk_id`, `doc_id`, `text`, `chunk_type`, `page_number`, and `headings`.
  - Image and table chunks include additional positioning and supplementary information, such as `caption`, `footnotes`, and `asset_path`.
- The retrieval stage uses a unified `SearchResult` schema to organize hits, while the Q&A stage uses `Citation` to denote references and returns the final answer with citations via `RAGAnswer`.
  - This overall schema design covers the complete data flow from chunk normalization, embedding, and index construction to retrieval and answer synthesis.

## Troubleshooting

### `rag parse` failed

Check the following:

- Is the local parsing runtime correctly installed and callable in the current environment?
- Is the input path correct?
- Is the visual model API configuration available?
- Does the content in the input directory or URL file match the expected format?

### `rag query --llm api` failed

Check the following:

- `RAG_API_BASE_URL`
- `RAG_API_KEY`
- `RAG_API_MODEL`
- Is the current model API accessible via the OpenAI SDK?

### `rag query --llm ollama` failed

Check the following:

- Is Docker running normally?
- Can the local `ollama` container be started automatically?
- Has the model specified by `RAG_OLLAMA_MODEL` been pulled, or can it be pulled successfully?
- Does the current environment allow the local model backend to initialize correctly?

All runtime errors are uniformly logged to `rag.log` in the project root directory.

## Example Data
The repository provides a sample paper CSV list at `data/pdfs/sample_ai_impacts.csv`, containing 30 papers related to the environmental impact of AI, which can be used as a sample knowledge base.

A benchmark dataset `data/benchmark_QA.csv`, constructed from these sample papers, is also provided. It is used to assess recall, ranking, and retrieval latency in the benchmark pipeline and contains 40 test questions.

The sample data is adapted from publicly available datasets and is intended solely for non-commercial research and evaluation purposes within this project. The underlying data remains subject to its original license.

## Future Steps

- **Add a generation quality evaluation module:** Use manually annotated answers or LLM-as-a-judge to further assess the correctness, completeness, relevance, and clarity of generated answers.
- **Introduce RAG evaluation frameworks:** Adopt frameworks like RAGAs to evaluate answer-evidence consistency (faithfulness, answer relevancy, context precision, context recall) and further optimize generation performance.
- **Refine context assembly (local):** Implement a strategy to expand the local neighborhood of high-relevance chunks by supplementing them with adjacent text, charts from the same page, and related captions/footnotes to improve evidence completeness.
- **Refine context assembly (filtering):** Enhance filtering strategies to reduce interference from similar but irrelevant noise across different documents.

## References and Dependencies

This project is built upon the following open-source frameworks and modules:

- [cross-encoder/ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2): Default reranker model.
- [Docker](https://www.docker.com/): Manages local `ollama` containers and related services.
- [FAISS](https://github.com/facebookresearch/faiss): Vector indexing and similarity search.
- [FastAPI](https://fastapi.tiangolo.com/): HTTP API framework.
- [MinerU](https://github.com/opendatalab/MinerU): Structured PDF parsing and image asset export.
- [OpenAI SDK](https://platform.openai.com/docs/overview): Model API access and multimodal/text generation.
- [Ollama](https://ollama.com/): Optional local generation backend.
- [rank_bm25](https://github.com/dorianbrown/rank_bm25): Sparse retrieval and keyword matching.
- [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2): Default embedding model.
- [Typer](https://typer.tiangolo.com/): CLI builder.
- [Uvicorn](https://www.uvicorn.org/): HTTP service runner.
- [uv](https://docs.astral.sh/uv/): Python dependency management and command execution.
- [WattBot 25](https://www.kaggle.com/competitions/WattBot2025/overview): Source of the paper list and benchmark dataset in the example data.

## License
This project is released under the **AGPL-3.0** license, consistent with its dependencies (MinerU). This license does not apply to the papers and benchmark datasets in the example data, which remain subject to their original license, **CC BY-NC 4.0**.


## Agentic Router Layer (Planned Feature)

To better adapt retrieval and generation strategies to different academic question patterns, the system plans to introduce a **Router / Policy Layer** before the current Retrieval Layer and Generation Layer.

### Background and Objective

Evaluation experiments showed that different academic domains and question types are sensitive to different prompts and parameter settings across the retrieval and recall pipeline. A single default configuration cannot reliably balance answer quality and system performance across all scenarios.

The current system already supports multiple retrieval, recall, and generation parameters, but these still need to be adjusted and tested manually by the user, which introduces additional usage overhead and trial-and-error cost.

To address this, we plan to introduce an Agentic router layer using LangGraph. At the beginning of each query, the layer will use an LLM to perform preliminary query analysis and assessment, dynamically select the most suitable retrieval path, and evaluate intermediate results for controlled retry when necessary. The goal is to further improve Accuracy and Recall while reducing user-side tuning cost and increasing overall adaptability.

### Functional Scope

The Router Layer is responsible for the following tasks:

- Analyze the current query and identify its question pattern, such as definition, comparison, quantitative, causal, multi-hop, or figure/table-dependent questions
- Infer the approximate academic domain or subtopic of the query, in order to account for domain-specific terminology and evidence distribution
- Decide whether query explanation is needed and select an appropriate explanation prompt strategy
- Dynamically select the retrieval policy, including retrieval mode, top_k, fetch_k, and whether reranking should be enabled
- Dynamically select the generation policy, including prompt variant, reasoning_effort, and whether stricter citation or fallback behavior is needed
- Trigger a limited strategy switch or controlled retry when the initial retrieval result is clearly insufficient

The Router Layer only outputs policy decisions. The actual retrieval, reranking, and answer synthesis remain handled by the existing modules.

### Node Definition

The Router Layer can be abstracted into the following core nodes:

**1. Query Analysis**
Performs structured analysis of the original query and outputs labels such as question type, academic domain, estimated complexity, and whether the question likely depends on tables, figures, or cross-document evidence.

**2. Policy Selection**
Maps the analysis result to a concrete execution strategy, including:

- whether query explanation should be enabled
- which explanation prompt variant should be used
- retrieval mode
- top_k / fetch_k
- whether reranking should be enabled
- generation prompt variant
- reasoning_effort
- whether additional verification or a more conservative fallback policy is required

**3. Retrieval Execution**
Invokes the existing retrieval pipeline to perform explanation, recall, fusion, and reranking under the selected policy.

**4. Result Assessment**
Performs lightweight quality checks on the retrieval result, such as relevance scores, evidence coverage, figure/table hit rate, or concentration of high-confidence results, in order to determine whether the current result set is sufficient.

**5. Retry / Fallback**
If the result is inadequate, the system may perform one controlled strategy switch, such as changing the explanation prompt, increasing top_k, or enabling reranking. If evidence remains insufficient, the system falls back to the existing conservative fallback path instead of retrying indefinitely.

### Decision Flow

The typical execution flow of the Router Layer is:

**original query → Query Analysis → Policy Selection → Retrieval Execution → Result Assessment → generation / limited retry / fallback**

In practice:

- simple and well-matched queries can directly use a lightweight policy to reduce latency
- complex queries with implicit conditions or multimodal evidence requirements can use a stronger explanation and retrieval configuration
- if the first retrieval result is weak, the system can switch strategy once within a controlled range instead of relying on a single fixed parameter set
- if sufficient evidence still cannot be obtained, the system preserves the current conservative answering behavior and returns fallback rather than forcing an unsupported answer

### Relationship to the Existing Architecture

The Router Layer is an additional upper-level control module and does not replace the current Ingestion, Indexing, Retrieval, or Generation components. The following parts of the system can remain unchanged:

- PDF parsing and multimodal normalization
- TextChunk / FigureChunk / TableChunk schema
- embedding generation and index construction
- dense / sparse / hybrid retrieval
- RRF fusion and cross-encoder reranking
- answer synthesis, citation enrichment, and fallback

With the new layer, the overall architecture becomes:

**Source Layer → Ingestion Layer → Indexing Layer → Router / Policy Layer → Retrieval Layer → Generation Layer → Interface Layer**