[🇺🇸 English](./README.md) | [🇨🇳 简体中文](./README_ZH.md)

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](./pyproject.toml)
[![License](https://img.shields.io/badge/License-AGPL--3.0-green)](./LICENSE)
[![CI](https://github.com/cany7/LumiCite/actions/workflows/ci.yml/badge.svg)](https://github.com/cany7/LumiCite/actions/workflows/ci.yml)

# LumiCite

LumiCite 是一个针对学术科研场景的端到端、多模态 RAG 系统，通过将学术论文中的**正文、图像与表格**统一解析为可检索语料，结合 **hybrid retrieval、reranking 与 citation-aware answering**，为学术应用提供精准的证据检索、召回与答案生成

## 项目特点

- **多模态证据建模：** 将论文中的正文、图像与表格转化为统一的可检索证据，解决纯文本 RAG 遗漏大量信息，及纯图像 embedding 编码难以充分理解对比、趋势变化、结构关系等深层信息等问题
- **混合检索机制：** 结合 `FAISS` 向量索引与 `BM25` 稀疏索引，兼顾语义召回与关键词匹配能力，覆盖学术场景中基于语义召回，及关键术语、指标名称等关键词匹配的混合场景
- **Query explanation 增强检索：** 在检索前对隐式条件、关键数值换算、基准量缺失等复杂问题进行 query explanation / expansion，提升复杂学术场景下的召回能力
- **双路召回与精排优化：** 采用双路召回及 `RRF` 融合，通过 cross-encoder reranker 精排提升检索质量与准确性，优化概念相近、表述稠密等文本特征导致的检索噪声问题
- **多接口支持：** 为适配不同应用场景，项目同时提供 CLI 与 HTTP API 接口，支持本地调用与云端服务化部署
- **可切换 LLM 后端：** 支持标准化 LLM API，并可切换为容器化 `ollama` 本地推理后端，以适配离线内网等受限部署场景
- **多输入源支持：** 支持 `local_dir`、`url_csv`、`url_list` 三类输入源，适配本地 PDF 文档与 URL 批量导入的不同场景

## 系统架构概览

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

## 环境配置&安装

项目依赖与版本约束请参考 `pyproject.toml`、`uv.lock` 及相关配置文件

安装依赖：

```bash
uv sync
cp .env.example .env
```

建议参考 `.env.example` 在项目根目录配置 `.env`，常见环境变量如下：

```env
RAG_API_BASE_URL=
RAG_API_KEY=
RAG_API_MODEL=

RAG_VISUAL_API_BASE_URL=
RAG_VISUAL_API_KEY=
RAG_VISUAL_API_MODEL=
```

配置说明：

- `RAG_API_*` 使用OpenAI SDK 通用 API 规范配置，供 ingest 与 generation 阶段共同使用
- `RAG_VISUAL_API_*` 可以为 ingest 阶段单独指定模型 API 配置
- `RAG_OLLAMA_MODEL` 指定本地容器化 `ollama` 后端的模型名称，默认使用 `qwen3.5:4b`
- generation 阶段默认模型为 `qwen/qwen3.5-27b`，可通过 `RAG_API_MODEL` 或 `rag query --model` 调整
- ingest 阶段默认模型为 `qwen/qwen2.5-vl-7b-instruct`，可通过 `RAG_VISUAL_API_MODEL` 调整
- `docker compose` 启动 `ollama` 服务时会自动拉取 `RAG_OLLAMA_MODEL` 指定的模型；CLI 在切换到 `--llm ollama` 时会等待该模型就绪后再发起请求
- PDF解析模块由 `rag parse` 命令直接调用，默认计算设备为 `cpu`，可通过parse --device` 进行调整

## 快速开始

首次运行时，推荐按照以下顺序进行初始化：

1. 将 PDF 文件放入 `data/pdfs/`
2. 执行 `rag parse` 完成模型下载、ingest、embedding 与索引构建
3. 使用 `rag search` 或 `rag query` 进行检索与问答

## CLI 使用说明

### `rag parse`

`rag parse` 用于根据当前输入源同步论文知识库，并完成解析、chunk 归一化、embedding 写入与索引更新；该命令适用于首次构建知识库，或在文档发生新增、替换、删除后重新同步知识库
首次运行时，`rag parse` 会自动检查并下载所需核心依赖模块，并缓存至 `data/model_cache/`
文档解析会顺序逐篇处理文档，先完成 PDF 解析与 chunk 归一化，待全部文档处理结束后统一写入 `chunks.jsonl`、生成 embedding 并更新索引
若中途意外中断，本轮结果可能不完整，建议重新执行一次 `rag parse`

默认选项：

- `--source local_dir`
- `--device cpu`
- `--llm api`

常用参数：

- `--source`
  - 可选 `local_dir`、`url_csv`、`url_list`
- `--path`
  - 填写本地输入路径，如 `data/pdfs/`、`data/pdfs/papers.csv`、`data/pdfs/papers.txt`
- `--device`
  - 可选 `cpu`、`cuda`、`cuda:0`、`mps`、`npu` 作为解析计算设备
- `--llm`
  - 可选 `api`、`ollama`，用于控制 figure visual summary 使用的后端
- `--rebuild-index`
  - 强制重建 `FAISS` 与 `BM25` 索引
- `--retry-failed`
  - 仅重试此前失败的文档
- `--dry-run`
  - 仅输出计划，不实际执行

示例：

```bash
uv run rag parse --source local_dir --path data/pdfs/
```

结果示例：

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

`rag search` 仅执行检索，不调用生成模型，适合用于查看召回结果、检查检索质量或对比不同检索策略下的命中情况；默认采用 `hybrid` 检索并输出 JSON 结果

默认选项：

- `--top-k 10`（或 `RAG_RETRIEVAL_TOP_K`）
- `--retrieval-mode hybrid`
- `--no-rerank`
- `--query-explanation`
- `--output-format json`

常用参数：

- `--top-k`
  - 填写正整数，表示选取前 k 个召回结果；如果不传，CLI 会读取 `RAG_RETRIEVAL_TOP_K`，默认 `10`
- `--retrieval-mode`
  - 可选 `dense`、`sparse`、`hybrid` 三种检索模式
- `--rerank` / `--no-rerank`
  - 控制是否启用 rerank，启用后会先取 `top_k * 5` 个候选做重排，再保留最终 `top_k`，效果更好但速度更慢
- `--query-explanation` / `--no-query-explanation`
  - 可选是否开启 query explanation 增强检索，适用于复杂问题，默认开启
  - query explanation 的 reasoning 不走 `--reasoning-effort`，只通过 `.env` 中的 `RAG_QUERY_EXPLANATION_REASONING_EFFORT` 控制，默认 `none`
- `--output-format`
  - 可选 `json`、`table`

示例：

```bash
uv run rag search "energy consumption of PaLM 540B"
```

结果示例：

- The command returned a JSON payload with `total_results: 10`, `retrieval_mode: "hybrid"`, and `retrieval_latency_ms: 3263.582`.
- The top hits mixed text and figure chunks, including passages about real-world AI service energy consumption and figures mentioning PaLM (540B), GPT-3, Gemini, and hardware or energy cost trends.

### `rag query`

`rag query` 在检索结果基础上生成最终答案，并返回带引用信息的响应，适合用于论文问答、证据定位与结果导出；默认采用 `hybrid` 融合检索，开启 query explanation，并关闭 rerank；最终引用只会来自 generation 阶段实际选中的检索 chunk，CLI 默认输出人类可读格式

默认选项：

- `--top-k 10`（或 `RAG_RETRIEVAL_TOP_K`）
- `--retrieval-mode hybrid`
- `--no-rerank`
- `--query-explanation`
- `--llm api`
- `--reasoning-effort none`

常用参数：

- `--top-k`
  - 填写正整数，表示提供给 generation 阶段作为候选证据的检索数量；如果不传，CLI 会读取 `RAG_RETRIEVAL_TOP_K`，默认 `10`
- `--retrieval-mode`
  - 可选 `dense`、`sparse`、`hybrid`
- `--rerank` / `--no-rerank`
  - 控制是否启用 rerank；启用后会先取 `top_k * 5` 个候选做重排，再把最终 `top_k` 条结果送入 generation
- `--query-explanation` / `--no-query-explanation`
  - 可选是否开启 query explanation 增强检索，适用于复杂问题，默认开启
- `--llm`
  - 可选 `api`、`ollama`
- `--model`
  - 填写模型名称，仅在 `--llm api` 时生效
- `--reasoning-effort`
  - 可选 `none`、`minimal`、`low`、`medium`、`high`、`xhigh`，仅在 `--llm api` 时生效，默认 `none`
  - 这个参数只作用于最终答案生成，不影响 query explanation
- `--output`
  - 填写输出文件路径，如 `answer.txt` 或 `answer.json`
- `--output-format`
  - 可选 `text`、`json`，默认 `text`
- `--json`
  - 可选`--output-format json`，返回未处理的 JSON 结构化文本，供测试使用 
- `ollama` 模型
  - 默认使用 `qwen3.5:4b` 自动部署并拉取容器化 Ollama 模型，可通过 `RAG_OLLAMA_MODEL` 切换不同模型

示例：

```bash
uv run rag query "What were the CO2 emissions from training GPT-3?"
```

结果示例：

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

`rag serve` 用于启动项目 HTTP API，适合本地调试、接口联调或服务化运行；默认监听 `0.0.0.0:8000`

HTTP API 的 `POST /api/v1/search` 与 `POST /api/v1/query` 请求同样支持 `query_explanation: true`，可启用与 CLI 一致的 retrieval-oriented query expansion

默认选项：

- `--host 0.0.0.0`
- `--port 8000`

常用参数：

- `--host`
  - 填写监听地址，如 `0.0.0.0` 或 `127.0.0.1`
- `--port`
  - 填写端口号，如 `8000`
- `--reload`
  - 启用开发热重载

示例：

```bash
uv run rag serve --host 0.0.0.0 --port 8000
```

### `rag benchmark`

`rag benchmark` 用于运行项目评测流程，当前主要评估 retrieval 层效果与检索性能，比较不同 retrieval mode、top-k 与 rerank 配置下的召回与排序表现，并统计检索延迟
评测结果会按问题粒度记录核心指标，包括 `recall_at_k`、`mrr`、`ndcg_at_k`、`mean_retrieval_latency_ms` 与 `p95_retrieval_latency_ms`；默认基于 `hybrid` 检索，开启 query explanation，并关闭 rerank 执行评测

默认选项：

- `--dataset data/train_QA.csv`
- `--retrieval-mode hybrid`
- `--top-k 10`（或 `RAG_RETRIEVAL_TOP_K`）
- `--no-rerank`
- `--query-explanation`
- `--output-dir data/benchmark_results/`
- `--tag run`

常用参数：

- `--dataset`
  - 填写评测数据集路径，如 `data/benchmark.csv`
- `--retrieval-mode`
  - 可选 `dense`、`sparse`、`hybrid`
- `--top-k`
  - 填写正整数，表示评测时的检索数量；如果不传，CLI 会读取 `RAG_RETRIEVAL_TOP_K`，默认 `10`
- `--rerank` / `--no-rerank`
  - 控制是否启用 rerank；启用后会先取 `top_k * 5` 个候选做重排，再保留最终 `top_k`
- `--query-explanation` / `--no-query-explanation`
  - 可选地在每个 benchmark 问题检索前将原问题改写为面向证据召回的 expanded query；expanded query 只用于增强检索，与原 query 的候选集合并后再做 rerank，不参与答案生成
  - query explanation 的 reasoning 不走命令行参数，只通过 `.env` 中的 `RAG_QUERY_EXPLANATION_REASONING_EFFORT` 控制，默认 `none`
- `--output-dir`
  - 填写结果输出目录，如 `data/benchmark_results/`
- `--tag`
  - 填写本次运行的标识名称，如 `smoke`

测试数据集必需字段：

- `question_id`：问题唯一标识
- `question`：评测问题文本
- `ref_doc_id`：标准答案对应的文档 `doc_id`

示例：

```bash
uv run rag benchmark --dataset data/benchmark_QA.csv --tag smoke
```

结果示例：

```json
{
  "report": "data/benchmark_results/smoke_20260322_175009_report.json",
  "summary": "data/benchmark_results/smoke_20260322_175009_summary.csv"
}
```

The generated report in this run recorded:

- `num_questions: 41`
- `recall_at_k: 0.8537`
- `mrr: 0.8049`
- `ndcg_at_k: 0.8177`
- `mean_retrieval_latency_ms: 473.535`

## 数据与 Pipeline Schema

项目运行过程中会产出一组固定的关键目录与中间结果：

- `data/pdfs/`：原始 PDF 文档与默认 URL 输入文件位置
  - `papers.csv`：URL CSV 输入文件，必需字段为 `url`
  - `papers.txt`：TXT URL list 输入文件，每行一个 URL
- `data/intermediate/mineru/`：解析阶段的原始中间产物
- `data/assets/`：图像与表格的 canonical 资产目录
- `data/metadata/`：canonical chunks、embeddings 与检索索引目录
- `rag.log`：项目根目录下的统一错误日志

- 本项目使用统一的 chunk schema 串联 ingest、retrieval 与 answer 阶段
  - 每条 chunk 均以 `chunk_id`、`doc_id`、`text`、`chunk_type`、`page_number`、`headings` 作为基础字段
  - 图片和表格类 chunk 额外包含 `caption`、`footnotes`、`asset_path` 等定位与补充信息
- 检索阶段使用统一的 `SearchResult` 组织命中结果，问答阶段使用 `Citation` 表达引用定位，并通过 `RAGAnswer` 返回最终答案与引用信息
  - 整体 schema 设计覆盖了从 chunk 归一化、embedding、索引构建，到检索返回与答案补全的完整数据链路

## 使用建议

- 当知识库发生新增、替换、删除，或 `papers.csv`、`papers.txt` 内容变更时，需手动执行 `rag parse`，再进行检索或问答
- 当知识库未变化时，无需重复执行 `rag parse`，可直接使用 `rag search` 或 `rag query`
- `rag search` 适合用于检查召回结果、观察命中内容和调试检索策略；`rag query` 适合在检索基础上生成最终答案并返回引用信息
- 生成阶段默认通过外部模型 API 执行；如需使用本地后端，可对 `rag parse` 与 `rag query` 显式切换到 `ollama`
- 本项目更适用于学术论文中的正文、图像与表格联合问答场景，因为系统会将多种证据类型统一纳入检索、重排与答案生成流程

## 故障排查

### `rag parse` 失败

可优先检查：

- 本地解析运行时是否已正确安装并可在当前环境中调用
- 输入路径是否正确
- 视觉模型API相关配置是否可用
- 输入目录或 URL 文件中的内容是否满足预期格式

### `rag query --llm api` 失败

可优先检查：

- `RAG_API_BASE_URL`
- `RAG_API_KEY`
- `RAG_API_MODEL`
- 当前模型 API 是否可通过 OpenAI SDK 正常访问

### `rag query --llm ollama` 失败

可优先检查：

- Docker 是否正常运行
- 本地 `ollama` 容器是否可以被自动拉起
- `RAG_OLLAMA_MODEL` 对应模型是否已拉取完成，或是否被成功自动拉取
- 当前环境是否允许本地模型后端正常初始化

所有运行时错误会统一记录到项目根目录下的 `rag.log`

## 项目测试

当前 `tests/` 目录包含 unit tests 与 integration tests，主要覆盖以下功能测试：

- 检索链路：dense、sparse、hybrid 检索，RRF 融合，以及 reranker 改排逻辑
- 问答链路：prompt 构造、生成结果归一化、引用补全、答案校验与 fallback 行为
- 解析与入库链路：文本切块、MinerU 输出映射、图像/表格资产处理、embedding 生成、索引构建与持久化一致性
- 数据源与增量处理：`local_dir`、`url_csv`、`url_list` 的发现与抓取，manifest 增量处理、失败重试与 stale 文档剪枝
- 接口与运行契约：CLI 与 FastAPI 的参数默认值、返回结构、健康检查与错误响应格式
- 工程约束：配置加载、日志与错误类型、Docker / CI 配置约束，以及核心 schema 的字段校验

## 示例数据
仓库提供了一份示例论文 CSV 列表data/pdfs/sample_ai_impacts.csv，选取了30 篇与 AI 环境影响相关的论文，可作为示例知识库输入使用

同时提供评测数据集data/benchmark_QA.csv基于上述示例论文构建，用于 benchmark 流程中的召回、排序与检索延迟评估，总共包含40道测试问题

示例数据基于公开发布的数据集适配，仅用于本项目的非商业研究与评测用途，对应数据仍受其原始数据许可证约束

## 评测结果
使用该评测数据集，在默认参数设置下对系统进行了完整测试，结果见[benchmark_QA_default_query_results.csv](./tests/benchmark_QA_default_query_results.csv)

按“语义一致”口径统计，共得到 33 个正确答案，整体正确率为 82.5%。平均 retrieval latency 为 891 ms，平均 generation latency 为 1756 ms

召回及生成过程中，会受到云端 LLM API 服务状态、测试平台计算性能等因素影响，因此延时结果仅供参考，不同运行环境可能存在较大差异

对于部分在默认参数下回答失败的复杂问题，又进一步调整了 reasoning_effort、rerank 和 top_k，进行了定向补充测试。补充测试后，仅剩少量高难度问题仍无法稳定得到正确答案

当前瓶颈主要集中在几类复杂问题模式上，例如：需要多表格或图像证据拼接、跨文档证据整合及推理、以及复杂计算型推理。这类问题的回答效果不仅受检索结果完整性影响，也较依赖所使用 LLM 生成模型的综合能力，后续若切换到更强的模型，仍有进一步提升空间

同时，当前系统在无法准确回答这类复杂问题时，均正确返回了 fallback 结果，没有出现强行作答的情况。整体来看，系统在学术知识库场景下能够稳定地生成准确、质量较高且具备证据约束的回答，并有效规避了 AI 幻觉问题

## 下一步方向

- 增加面向生成质量的评估模块，基于人工标注答案或使用 LLM-as-a-judge 进一步评估生成回答的 correctness、completeness、relevance 与 clarity
- 引入 RAGAs 等 RAG 评估框架， faithfulness、answer relevancy、context precision、context recall 等维度评估答案与证据的一致性，并进一步优化生成效果
- 在检索与生成之间增加更精细的 context assembly 策略，在保留高相关 chunk 的基础上，引入同文档局部邻域扩展，补充相邻正文、同页图表及相关 caption/footnote，提升证据完整性，减少“主证据命中但补充证据缺失”的情况
- 在检索与生成之间增加更精细的 context assembly 与 context filtering 策略，补充同文档局部邻域证据，减少跨文档相似噪声对生成结果的干扰
- 加入对 query 的自动识别与分流，根据问题类型自动选择 `top_k`、`reasoning_effort`、`rerank` 等检索与生成参数

## 引用与依赖

本项目基于以下开源框架、模块构建：

- [cross-encoder/ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)：用作默认 reranker 模型
- [Docker](https://www.docker.com/)：用于本地 `ollama` 容器运行与相关服务管理
- [FAISS](https://github.com/facebookresearch/faiss)：用于向量索引与相似度检索
- [FastAPI](https://fastapi.tiangolo.com/)：用于构建 HTTP API
- [MinerU](https://github.com/opendatalab/MinerU)：用于学术论文 PDF 的结构化解析与图像资产导出
- [OpenAI SDK](https://platform.openai.com/docs/overview)：用于模型 API 接入与多模态/文本生成调用
- [Ollama](https://ollama.com/)：用于可选的本地生成后端
- [rank_bm25](https://github.com/dorianbrown/rank_bm25)：用于稀疏检索与关键词匹配
- [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)：用作默认 embedding 模型
- [Typer](https://typer.tiangolo.com/)：用于构建 CLI
- [Uvicorn](https://www.uvicorn.org/)：用于运行 HTTP 服务
- [uv](https://docs.astral.sh/uv/)：用于 Python 依赖管理与命令执行
- [WattBot 25](https://www.kaggle.com/competitions/WattBot2025/overview)：示例数据中的文列表与评测数据集来源

## 许可证说明
本项目采用 **AGPL-3.0** 协议发布，与项目中使用的相关依赖模块（MinerU）保持一致性；不适用于示例数据中的论文与评测数据集，相关数据仍遵循原始数据许可证 **CC BY-NC 4.0**
