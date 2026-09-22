# 后端服务

本目录是独立的 uv Python 项目，包含 FastAPI 应用、解析器、Chunk 流水线、Milvus 证据索引、数据库模型和自动化测试

## 主要模块

| 模块 | 用途 |
|---|---|
| `app/chunking.py` | Markdown 层级切分与递归兜底切分 |
| `app/chunk_service.py` | 解析结果对齐、策略自动判断与 Parent-Child 持久化 |
| `app/document_parsers.py` | DOCX、PPTX、Markdown 和音频转录的归一化文本解析 |
| `app/milvus_store.py` | Milvus Collection Schema、索引、Upsert、过滤和检索 |
| `app/evidence_index_service.py` | Chunk 向量化、索引任务执行与证据记录映射 |
| `app/model_provider.py` | 百炼 Chat 与 Embedding 模型配置 |
| `app/talent_evaluation_dispatch.py` | 动态维度、候选人 × 维度分发、要求级并发 Branch Worker 与 Requirement Evidence 协议 |
| `app/talent_evaluation_runtime.py` | 装配第 15 课分发图与第 16 课报告图，分别导出 `graph` 和 `report_graph` |
| `app/talent_evaluation_report.py` | 12 路单维度评分、候选人级一致性检查、短引用映射、候选人聚合、排序和报告校验渲染 |
| `scripts/verify_talent_evaluation_report.py` | 读取第 15 课分发结果并调用正式 `report_graph` 验证评分、聚合和报告链路 |

第 16 课报告图默认最多并发执行 12 个候选人 × 维度评分任务，并按候选人并发执行跨维度一致性检查。模型上下文使用 `E1`、`E2` 形式的短证据编号，图 State 保留短编号到完整 Chunk ID 的映射，模型返回后恢复完整 ID 再执行白名单和候选人归属校验。报告正文显示短编号，引用链接仍使用完整 Chunk ID 与原文偏移

## 环境初始化

```bash
uv sync --dev
```

## 运行测试

```bash
uv run pytest tests -q
```

运行需要外部 Chat Model 的真实 Evidence Pack 抽取测试：

```bash
RUN_LIVE_MODEL_TESTS=1 uv run --no-sync pytest \
  tests/test_talent_evaluation_branch_agent_live.py -q -s
```

运行第 15 课真实分发链路并输出结构化诊断日志：

```bash
uv run --no-sync python -m scripts.verify_talent_evaluation_dispatch \
  --log-level INFO
```

`--max-concurrency` 默认值为 6，用于控制 LangGraph 候选人 × 维度分支的并发上限。每个分支内部并发执行当前维度的全部 `evidence_requirements`，并按原始要求顺序聚合结果。

日志使用 `task_id`、`call_id`、`requirement_id` 和 `chunk_id` 连接图节点、工具调用、混合检索、原文回表、事实抽取和来源校验，不记录履历原文或模型 Prompt

## 启动 API

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 18080
```
