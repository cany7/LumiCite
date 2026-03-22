[🇺🇸 English](./README.md) | [🇨🇳 简体中文](./README_ZH.md)

# 学术论文多模态 RAG 系统

本项目是一个针对学术论文 PDF 的端到端、多模态 RAG 系统，通过将论文中的**正文、图像与表格**统一解析为可检索语料，结合 **hybrid retrieval、reranking 与 citation-aware answering**，为学术应用提供精准的证据检索、召回与答案生成。

## 项目特点

- **问题背景：** 学术论文中存在大量关键信息分散于图像、表格中，传统RAG系统仅依赖纯文本解析，容易遗漏关键证据，导致生成回答不完整
- **图像 embedding 模型局限性：** 采用图文混合的 embedding 模型可解决部分问题，但仍难以完整理解图表中的数据对比、变化趋势、结构关系等信息，无法提供准确答案
- **统一多模态检索语料：** 本项目通过将论文中的正文、图像与表格统一建模为可检索语料，采用统一的 chunk schema，让多种信息能够在同一索引、检索链路中协同工作，为学术应用提供更精准的检索与答案生成
- **混合检索机制：** 学术检索既依赖语义召回，又依赖关键术语、指标名称等关键词的精准命中，本项目结合 `FAISS` 向量索引与 `BM25` 稀疏索引，兼顾语义召回与关键词匹配能力，并支持不同模式切换
- **双路召回与精排优化：** 针对论文中概念相近、表述稠密等文本特征，项目采用双路召回，并通过 `RRF` 融合候选结果，结合 cross-encoder reranker 精排提升检索质量与准确性
- **多接口支持：** 为适配不同应用场景，项目同时提供 CLI 与 HTTP API 接口，支持本地调用与云端服务化部署
- **适配离线部署环境：** 默认采用标准 LLM API 进行生成，并可按需切换为容器化 `ollama` 作为本地 LLM 后端，以适配离线内网等受限部署场景；该切换同时适用于 query generation 与 ingest 阶段的 visual summary
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

项目依赖与版本约束请参考 `pyproject.toml`、`uv.lock` 及相关配置文件。

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

- `RAG_API_*` 可作为通用模型 API 配置，供 ingest 阶段的视觉摘要与 generation 阶段的问答生成共同使用
- `RAG_VISUAL_API_*` 可用于为图像信息提取环节单独指定模型 API 配置
- `RAG_OLLAMA_MODEL` 可用于指定本地容器化 `ollama` 后端的模型名称，默认使用 `qwen3.5:4b`
- generation 阶段默认模型为 `qwen/qwen3.5-27b`，可通过 `RAG_API_MODEL` 或 `rag query --model` 调整
- ingest 阶段默认模型为 `qwen/qwen2.5-vl-7b-instruct`，可通过 `RAG_VISUAL_API_MODEL` 调整
- `docker compose` 启动 `ollama` 服务时会自动拉取 `RAG_OLLAMA_MODEL` 指定的模型；CLI 在切换到 `--llm ollama` 时会等待该模型就绪后再发起请求
- PDF解析模块由 `rag parse` 命令直接调用，默认计算设备为 `cpu`，可通过parse --device` 进行调整

## 快速开始

首次运行时，推荐按照以下顺序进行初始化：

1. 将 PDF 文件放入 `data/pdfs/`
2. 执行 `rag parse` 完成模型下载、ingest、embedding 与索引构建
3. 使用 `rag search` 或 `rag query` 进行检索与问答

示例：

```bash
uv run rag parse --source local_dir --path data/pdfs/
uv run rag search "energy consumption of GPT-3" --top-k 5
uv run rag query "What were the CO2 emissions from training GPT-3?"
```

## CLI 使用说明

### `rag parse`

`rag parse` 用于根据当前输入源同步论文知识库，并完成解析、chunk 归一化、embedding 写入与索引更新。该命令适用于首次构建知识库，或在文档发生新增、替换、删除后重新同步知识库。
首次运行时，`rag parse` 会自动检查并下载所需核心依赖模块，并缓存至 `data/model_cache/`。
文档解析会顺序逐篇处理文档，先完成 PDF 解析与 chunk 归一化，待全部文档处理结束后统一写入 `chunks.jsonl`、生成 embedding 并更新索引。
若中途意外中断，本轮结果可能不完整，建议重新执行一次 `rag parse`。

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
uv run rag parse --source local_dir --path data/pdfs/ --llm ollama
```

### `rag search`

`rag search` 仅执行检索，不调用生成模型，适合用于查看召回结果、检查检索质量或对比不同检索策略下的命中情况。默认采用 `hybrid` 检索并输出 JSON 结果。

默认选项：

- `--top-k 10`
- `--retrieval-mode hybrid`
- `--no-rerank`
- `--output-format json`

常用参数：

- `--top-k`
  - 填写正整数，表示选取前 k 个召回结果
- `--retrieval-mode`
  - 可选 `dense`、`sparse`、`hybrid` 三种检索模式
- `--rerank` / `--no-rerank`
  - 控制是否启用 rerank，启用将提升召回质量，但会降低速度
- `--output-format`
  - 可选 `json`、`table` 将回答以为文件形式输出

示例：

```bash
uv run rag search "energy consumption of PaLM 540B" --top-k 10
```

### `rag query`

`rag query` 在检索结果基础上生成最终答案，并返回带引用信息的响应，适合用于论文问答、证据定位与结果导出。默认采用 `hybrid` 融合检索，并启用 rerank 重排。

默认选项：

- `--top-k 5`
- `--retrieval-mode hybrid`
- `--rerank`
- `--llm api`

常用参数：

- `--top-k`
  - 填写正整数，表示参与回答的检索数量
- `--retrieval-mode`
  - 可选 `dense`、`sparse`、`hybrid`
- `--rerank` / `--no-rerank`
  - 控制是否启用 rerank
- `--llm`
  - 可选 `api`、`ollama`
- `--model`
  - 填写模型名称，仅在 `--llm api` 时生效
- `--output`
  - 填写输出文件路径，如 `answer.json`
- `ollama` 模型
  - 默认使用 `qwen3.5:4b` 自动部署并拉取容器化 Ollama 模型，可通过 `RAG_OLLAMA_MODEL` 切换不同模型

示例：

```bash
uv run rag query "What were the CO2 emissions from training GPT-3?"
uv run rag query "What were the CO2 emissions from training GPT-3?" --llm ollama
```

### `rag serve`

`rag serve` 用于启动项目 HTTP API，适合本地调试、接口联调或服务化运行。默认监听 `0.0.0.0:8000`。

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

### `rag benchmark`

### `rag benchmark`

`rag benchmark` 用于运行项目评测流程，当前主要评估 retrieval 层效果与检索性能，比较不同 retrieval mode、top-k 与 rerank 配置下的召回与排序表现，并统计检索延迟。
评测结果会按问题粒度记录核心指标，包括 `recall_at_k`、`mrr`、`ndcg_at_k`、`mean_retrieval_latency_ms` 与 `p95_retrieval_latency_ms`。默认基于 `hybrid` 检索且开启 rerank 执行评测。

默认选项：

- `--dataset data/train_QA.csv`
- `--retrieval-mode hybrid`
- `--top-k 5`
- `--rerank`
- `--output-dir data/benchmark_results/`
- `--tag run`

常用参数：

- `--dataset`
  - 填写评测数据集路径，如 `data/benchmark.csv`
- `--retrieval-mode`
  - 可选 `dense`、`sparse`、`hybrid`
- `--top-k`
  - 填写正整数，表示评测时的检索数量
- `--rerank` / `--no-rerank`
  - 控制是否启用 rerank
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

所有运行时错误会统一记录到项目根目录下的 `rag.log`。

## 项目测试

当前 `tests/` 目录包含 unit tests 与 integration tests，主要覆盖以下功能测试：

- 检索链路：dense、sparse、hybrid 检索，RRF 融合，以及 reranker 改排逻辑
- 问答链路：prompt 构造、生成结果归一化、引用补全、答案校验与 fallback 行为
- 解析与入库链路：文本切块、MinerU 输出映射、图像/表格资产处理、embedding 生成、索引构建与持久化一致性
- 数据源与增量处理：`local_dir`、`url_csv`、`url_list` 的发现与抓取，manifest 增量处理、失败重试与 stale 文档剪枝
- 接口与运行契约：CLI 与 FastAPI 的参数默认值、返回结构、健康检查与错误响应格式
- 工程约束：配置加载、日志与错误类型、Docker / CI 配置约束，以及核心 schema 的字段校验

## 示例数据
仓库提供了一份示例论文 CSV 列表`data/pdfs/sample_ai_impacts.csv`，选取了30 篇与 AI 环境影响相关的论文，可作为示例知识库输入使用。
评测数据集`data/benchmark_QA.csv`基于上述示例论文构建，用于 benchmark 流程中的召回、排序与检索延迟评估。

## 下一步方向

- 增加面向生成质量的评估模块，基于人工标注答案或使用 LLM-as-a-judge 进一步评估生成回答的 correctness、completeness、relevance 与 clarity
- 引入 RAGAs 等 RAG 评估框架， faithfulness、answer relevancy、context precision、context recall 等维度评估答案与证据的一致性，并进一步优化生成效果

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

## 许可证说明
本项目采用 AGPL-3.0 协议发布，与项目中使用的相关依赖模块（MinerU）保持一致性。
