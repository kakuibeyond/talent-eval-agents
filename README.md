# 多 Agent 人才评估与推荐系统

课程贯穿项目的统一代码根目录

## 当前目录

| 路径 | 用途 |
|---|---|
| `backend/` | FastAPI 后端、uv 环境、依赖锁、解析器、数据模型与测试 |
| `database/init.sql` | PostgreSQL 完整建表 SQL |
| `database/migrations/005_chunking.sql` | 第 4 课到第 5 课的唯一幂等增量迁移 |
| `database/migrations/007_milvus_evidence_index.sql` | 第 6 课 Milvus 索引任务表的幂等增量迁移 |
| `database/migrations/012_talent_tools.sql` | 第 12 课岗位 JD 与人才工具调用审计表的幂等增量迁移 |
| `frontend/` | React 前端 |
| `docker-compose.yml` | 默认启动 PostgreSQL、MinIO、Redis；`app` profile 启动 backend、worker、frontend |

## 产品定位

前端是多 Agent 人才评估与推荐系统的统一应用壳层，人才档案模块包含员工花名册和档案资料库。花名册维护结构化员工数据，提供新建员工和每页 10 人分页。档案资料库通过知识库下拉框切换文件集合，文件表按每页 10 条分页，上传后自动排入解析、切片和 Milvus 索引链路，同时支持批量删除材料与关联数据。文件详情展示原文件、解析结果、索引记录、切片预览与基础元数据。档案资料库页面内置证据召回测试，可输入查询并附带候选人、材料类型和权限范围过滤项，直接查看匹配 Chunk、相似度分数和来源材料

## 运行模式

本项目只维护一套中间件环境：PostgreSQL、Redis、MinIO 始终由 Docker Compose 托管。开发模式和发布模式都共用这三类基础服务。

所有 Docker Compose 命令统一使用项目名参数 `-p talent-eval-agents-course`。

### 开发模式

开发模式的目标是保留前后端热更新，同时继续复用 Compose 内的中间件。

1. 启动基础设施

```bash
docker compose -p talent-eval-agents-course up -d
```

2. 启动 MinerU 本地服务

```bash
uv sync --project backend/mineru-runtime
uv run --project backend/mineru-runtime mineru-api --host 127.0.0.1 --port 18001
```

3. 启动后端 API

```bash
cd backend
uv sync --dev
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 18080
```

4. 启动本地 worker

```bash
cd backend
uv run python -m app.worker
```

5. 启动前端开发服务器

```bash
cd frontend
npm install
npm run dev
```

开发模式下的更新规则：

- 改前端页面：Vite 自动热更新，不需要重启 Docker
- 改后端 API：`uvicorn --reload` 自动重载，不需要重建镜像
- 改 worker、解析流水线或队列消费逻辑：手动重启本地 `uv run python -m app.worker`
- 改数据库结构：对已有数据库执行 migration；只有新建数据库卷时才会读取 `database/init.sql`

### 发布模式

发布模式用于测试环境、演示环境和生产口径的容器化运行。backend、worker、frontend 会和中间件一起由 Compose 托管。

```bash
docker compose -p talent-eval-agents-course --profile app up -d --build
```

发布模式下的更新规则：

- 改前端代码：`docker compose -p talent-eval-agents-course --profile app up -d --build frontend`
- 改后端代码：`docker compose -p talent-eval-agents-course --profile app up -d --build backend worker`
- 前后端都改了：`docker compose -p talent-eval-agents-course --profile app up -d --build`
- 只改了环境变量、挂载文件或容器运行参数，没有改镜像内容：`docker compose -p talent-eval-agents-course --profile app restart backend worker frontend`

`docker restart` 或 `docker compose restart` 只会重启旧容器，不会重新构建镜像，也不会把你本地新改的源码带进去。所以发布模式下只要改了代码，就必须用 `up -d --build`，不能只用 `restart`。


## 人才档案接口

| 接口 | 用途 |
|---|---|
| `GET /api/employees` | 查询员工花名册及关联材料数量 |
| `POST /api/employees` | 新建结构化员工档案 |
| `GET /api/knowledge-bases` | 查询档案知识库及文件数量 |
| `POST /api/knowledge-bases` | 创建档案知识库 |
| `GET /api/documents` | 查询知识库文件 |
| `POST /api/documents` | 上传文件、关联员工与知识库，并自动创建解析任务 |
| `DELETE /api/documents` | 批量删除文档，同时删除原文件、解析产物、切片和 Milvus 向量 |
| `GET /api/documents/{id}` | 查询原文件、基础元数据、解析任务、索引任务与 Markdown 产物 |
| `POST /api/documents/{id}/parse` | 创建异步解析任务 |
| `GET /api/documents/{id}/chunks` | 查询最新一次成功切片及 Parent-Child 关系 |
| `POST /api/documents/{id}/chunks` | 按材料结构自动执行 Markdown 结构化切分或纯文本递归切分 |
| `POST /api/documents/{id}/evidence-index` | 为当前文档版本手动创建异步 Milvus 索引任务 |
| `POST /api/evidence/search` | 按租户、权限范围和业务条件检索人才证据，并返回来源材料与候选人上下文（纯向量） |
| `POST /api/evidence/hybrid-search` | 稠密 + BM25 双路召回、RRF 融合、Rerank 精排的证据检索 |
| `POST /api/talent-search/plan` | 把自然语言人才需求编译为受控 Query Plan |
| `POST /api/talent-search/candidates` | 校验 Query Plan，并用确定性 SQL Builder 返回候选人编号集合 |
| `POST /api/talent-search` | 输入自然语言查询，依次执行 Query Plan、SQL 候选筛选、混合检索和可选查询优化，返回 Chunk 列表 |
| `POST /api/index-jobs/{id}/retry` | 为失败的 Milvus 索引任务创建新的幂等重试任务 |

## 第 8 课复合人才检索

| 路径 | 用途 |
|---|---|
| `backend/app/query_plan.py` | Query Plan、Filter DSL 注册表、自然语言结构化输出、SQLAlchemy 查询构造与查询优化计划 |
| `backend/tests/test_query_plan.py` | DSL 提示词、字段白名单、参数化 SQL、年龄边界、待澄清条件、空候选集和优化策略测试 |
| `backend/tests/test_talent_search_api.py` | 从 HTTP 请求到 Chunk 响应的端到端接口测试，覆盖标准检索与首轮为空后的自动优化检索 |

查询计划把结构化硬条件、语义条件、偏好和待澄清条件分开。Pydantic Schema 提供输出结构，`FILTER_FIELD_REGISTRY` 同时维护字段说明、允许操作符和 SQL 构造函数，并生成提示词中的 Filter DSL。租户从请求头注入，模型不能生成权限条件或原始 SQL。`/talent-search/candidates` 返回 `employee_no` 作为后续 Milvus 检索的 `candidate_ids`。`execute_composite_search()` 在候选集为空时直接结束，不调用 Milvus，避免空列表退化为全库搜索；多个语义要求按 `requirement_id` 分别保存证据。`search_with_optimization()` 在首轮无有效命中时选择 Rewrite、Multi Query 或 Decompose，生成最多 3 条受约束查询，再次检索并按 `chunk_id` 去重合并

## 第 9 课可信证据输出

`POST /api/talent-search` 增加可选参数 `include_evidence_pack`。默认值为 `false`，原有调用继续返回 Query Plan、候选集、检索记录和 Chunk 列表。参数设为 `true` 时，接口从 PostgreSQL 重新读取并鉴权检索命中的 Chunk，再按候选人和 `requirement_id` 输出 `Candidate Evidence Pack`

| 路径 | 用途 |
|---|---|
| `backend/app/evidence_pack.py` | Pydantic 抽取协议、事实来源校验、重复事实合并、yes/no 冲突推导和证据状态计算 |
| `backend/tests/test_evidence_pack.py` | Chunk 去重、来源合并、伪造引用拦截、冲突范围、时间格式和失败降级测试 |
| `backend/scripts/verify_evidence_pack.py` | 以隔离合成材料验证查询、事实合并、冲突输出和材料缺失 |
| `backend/samples/lesson09/` | C901、C902 虚构材料及数据性质说明 |

证据包协议版本为 `2.0`。事实字段为 `event`、`period`、`claim`、`answer` 和 `sources`，其中 `answer` 只允许 yes 或 no。每个 source 包含 `citation_id`、`chunk_id`、`quote`、`quote_start` 和 `quote_end`，坐标是 Chunk 文本内的半开区间 `[quote_start, quote_end)`。模型只抽取事实，程序校验来源后按 `event + period + claim + answer` 合并重复事实，再根据同一 `event + period + claim` 下的 yes/no 推导冲突。证据状态分为 `sufficient`、`partial`、`missing` 和 `conflicting`

模型返回的每条来源必须引用输入中的 `chunk_id`，`quote` 必须是对应 Chunk 的连续原文。`quote_start` 和 `quote_end` 由后端在校验通过后根据 PostgreSQL 重读的 Chunk 计算，不接受模型生成的位置。校验失败时，系统返回 `partial + extraction_failed`，保留原始引用，不返回未经核验的事实

隔离演示默认使用内存 SQLite、固定检索结果和固定抽取结果，不读取现有员工数据库或 Milvus

```bash
cd backend
uv run --no-sync python -m scripts.verify_evidence_pack
```

使用项目配置的模型验证时，只有 `backend/samples/lesson09/` 下明确标注为虚构的三份材料会发送到模型 API

```bash
uv run --no-sync python -m scripts.verify_evidence_pack --live-model
```

## 第 10 课检索链路与引用展示

| 路径 | 用途 |
|---|---|
| `backend/app/evidence_citations.py` | 从 PostgreSQL 重读 Chunk、绑定材料版本并按当前租户和权限解析引用 |
| `backend/tests/test_evidence_citations.py` | 版本绑定、租户与权限检查、历史版本和引用接口测试 |
| `frontend/src/EvidencePackPanel.tsx` | 展示证据状态、事实来源、待核查冲突与引用原文抽屉 |

引用接口为 `GET /api/evidence/citations/{chunk_id}?quote_start={start}&quote_end={end}`。事实来源打开引用时同时传递 quote 坐标；未指定坐标时仍可查看整个 Chunk。接口校验坐标成对出现且不超出 Chunk，并根据当前请求上下文重新检查 Chunk、材料版本、文档、资料库、候选人、租户和权限范围。历史版本仍可访问时返回 `version_state=historical`，权限撤销或材料不可用时统一返回 404。前端抽屉展示 `heading_path`、Chunk 起止位置和 quote 起止位置，并在完整 Chunk 中高亮 quote

第 10 课把当前界面定位为档案知识库中的检索验收台。正式的人才评估与推荐 Chat 页面留到后续课程接入 LangGraph、SSE 和会话状态

第 10 课相关回归命令：

```bash
cd backend
uv run --no-sync pytest -q \
  tests/test_query_plan.py \
  tests/test_hybrid_search_service.py \
  tests/test_talent_search_api.py \
  tests/test_evidence_pack.py \
  tests/test_evidence_citations.py
```

结果输出：

```text
43 passed, 3 warnings
```

结果说明：这组测试使用数据库、Milvus 和模型替身，只用于代码分支回归，不能替代开发模式联调

第 10 课开发模式联调沿用 Compose 中已经启动的 PostgreSQL、Redis、MinIO、Milvus 和 etcd，并分别启动本地 backend、worker 和 frontend

运行材料导入脚本：

```bash
cd backend
uv run python scripts/import_markdown_reviews.py
```

结果输出：

```text
C001  C001_林晓岚_项目复盘  chunk=markdown  index=succeeded
C002  C002_陈泽宇_项目复盘  chunk=markdown  index=succeeded
C003  C003_周雨桐_项目复盘  chunk=markdown  index=succeeded
C004  C004_赵明远_项目复盘  chunk=markdown  index=succeeded
C005  C005_郭思远_项目复盘  chunk=markdown  index=succeeded
```

结果说明：

- PostgreSQL 新增 5 份材料，每份 20 个 Parent/Child Chunk，共 100 个 Chunk
- Milvus `talent_evidence_v2` 写入 50 个可召回 Child Chunk，每名候选人 10 个
- 查询 `筛选在上海且有企业知识库和大模型应用项目经验的候选人` 生成地区过滤和 1 项语义要求
- 浏览器返回 C001 的 `sufficient` 状态、5 条事实和 16 条原始证据
- 引用抽屉返回当前版本、标题路径、Markdown 起始位置和 PostgreSQL 原文
- 使用 `company_internal` 权限读取同一 `hr_private` 引用返回 `404`

模型抽取结果具有不确定性。同一材料的另一轮请求返回 `partial`，并把缺失项目时间记录为 `period`。演示和评测需要保留每次运行输出，不能把一次输出写成固定结果

## 第 11 课 LangGraph 状态模型与人才决策主图

| 路径 | 用途 |
|---|---|
| `backend/app/talent_decision_graph.py` | 定义人才决策输入、内部 State、输出、Runtime Context、节点、路由和可运行主图 |
| `backend/tests/test_talent_decision_graph.py` | 验证正常完成、空候选结束、可信 Context 和并行 Reducer 行为 |
| `backend/scripts/verify_talent_decision_graph.py` | 输出逐节点 State 增量、空候选结果和 Reducer 前后对照 |

`TalentDecisionState` 将消息、人才要求、候选人范围、证据引用、评估结果、报告和运行状态分开保存。租户与权限通过只读 `DecisionContext` 注入，不接受用户消息或模型输出提供的同名值。主图当前使用注入式候选人 Provider 和占位评估结果，后续课程在保持 State Contract 的前提下接入真实工具、任务理解、并行评估和人机协同

运行本节独立验证：

```bash
cd backend
uv run --no-sync python -m scripts.verify_talent_decision_graph
```

运行本节测试：

```bash
cd backend
uv run --no-sync pytest -q tests/test_talent_decision_graph.py
```

结果输出为 `6 passed in 0.16s`

## 第 12 课人才数据工具与统一证据协议

| 路径 | 用途 |
|---|---|
| `backend/app/talent_tools.py` | 四类只读人才工具、Runtime Context、统一返回协议、重试、熔断和审计执行器 |
| `backend/tests/test_talent_tools.py` | 验证工具 Schema、租户隔离、Evidence Pack 版本、重试、熔断、审计与主图适配 |
| `backend/scripts/verify_talent_tools.py` | 使用隔离 SQLite 数据输出四类工具与故障链路结果 |
| `backend/samples/lesson12/job_descriptions.json` | 第 12 课合成岗位 JD 数据 |
| `database/migrations/012_talent_tools.sql` | 新增 `job_descriptions` 与 `tool_call_audits` |

工具清单为 `lookup_job_descriptions`、`filter_candidates`、`search_candidate_evidence` 和 `get_candidate_profiles`。模型可见 Schema 只包含业务参数，租户、权限、身份和 Run 编号通过 `ToolRuntime` 注入。结构化筛选复用第 8 课 Query Plan 与 SQL Builder，证据检索继续返回第 9 课 Candidate Evidence Pack `2.0`

`ToolExecutor` 统一返回 `ok`、`data`、`error` 和 `meta`，仅对瞬时依赖错误与超时执行受控重试。`policy` 设置默认超时与重试参数，`tool_policies` 可以按工具名称覆盖。连续失败达到阈值后打开熔断器。`SqlAuditSink` 只保存参数键名与调用摘要，不保存查询原文、候选人信息或证据正文

在代码仓库根目录启动 PostgreSQL 并执行第 12 课幂等迁移：

```bash
docker compose -p talent-eval-agents-course up -d postgres
docker compose -p talent-eval-agents-course exec -T postgres \
  psql -U talent -d talent_docs -v ON_ERROR_STOP=1 -f /dev/stdin \
  < database/migrations/012_talent_tools.sql
```

迁移使用 `CREATE TABLE IF NOT EXISTS` 和 `CREATE INDEX IF NOT EXISTS`，可以重复执行。全新环境由 `database/init.sql` 直接创建最新结构

运行本节独立验证：

```bash
cd backend
uv run --no-sync python -m scripts.verify_talent_tools
```

运行本节与第 11 课主图测试：

```bash
uv run --no-sync pytest -q tests/test_talent_tools.py tests/test_talent_decision_graph.py
```

结果输出为 `23 passed in 0.65s`。后端完整回归结果为 `151 passed, 3 warnings in 4.85s`

## 第 13 课 MCP 人才能力服务

| 路径 | 用途 |
|---|---|
| `backend/app/talent_mcp_server.py` | 人才 MCP Server、4 个 Tools、3 类 Resources、评估 Prompt 与 Access Token Context 映射 |
| `backend/app/talent_mcp_client.py` | 将 MCPAdapter 发现的 LangChain Tools 放入 LangGraph ToolNode |
| `backend/samples/lesson13/codestats_mcp_v2.py` | FastMCP v1 示例的 SDK v2 迁移版，包含 Tool、Resource、目录边界与 STDIO 入口 |
| `backend/tests/test_codestats_mcp_v2.py` | 验证 CodeStats Tool、Resource、`.venv` 排除和目录越界拦截 |
| `backend/tests/test_talent_mcp_server.py` | 验证能力发现、结构化结果、Resource、Prompt、scope、audience 和租户隔离 |
| `backend/tests/test_talent_mcp_client.py` | 验证 MCP 工具在 LangGraph ToolNode 中的异步执行 |
| `backend/scripts/verify_talent_mcp.py` | 使用隔离 SQLite 数据启动 Streamable HTTP，验证协议、鉴权和 LangGraph 链路 |

MCP 适配层直接复用第 12 课的 `TalentToolService` 和 `ToolExecutor`。`talent_tools.py` 只新增了按 `job_code` 读取单个岗位 JD 的方法，且 SQL 继续强制使用可信 `tenant_id`

Server 能力清单：

- Tools：`lookup_job_descriptions`、`filter_candidates`、`search_candidate_evidence`、`get_candidate_profiles`
- Resource Templates：`talent://jobs/{job_code}`、`talent://candidates/{candidate_id}/profile`
- 固定 Resource：`talent://policies/evaluation/current`
- Prompt：`talent_assessment`

Streamable HTTP 保护边界使用 MCP `AuthSettings` 与可注入 `TokenVerifier`。OAuth scope `talent:read` 决定客户端能否访问 MCP 服务，token claims 中的 `tenant_id` 和 `permission_scopes` 映射为 `TalentToolContext`。工具参数与 `clientInfo` 不参与可信身份判定

本节新增直接依赖 `mcp[cli]>=2.2,<3`、`langchain[mcp]>=1.4,<2` 和 `langchain-core>=1.2,<2`。`MCPAdapter` 在当前 LangChain 版本中仍为 Beta API，升级后需运行本节回归测试

SDK v1 的 `FastMCP` 在 SDK v2 中更名为 `MCPServer`，Tool 与 Resource 装饰器以及 `mcp dev` 的基本用法保持不变。`mcp dev` 会启动 Inspector v2，启动 URL 使用 `MCP_INSPECTOR_API_TOKEN`。Inspector 的 `6274` 端口是调试页面，不是业务 MCP endpoint

接受 `mcpServers` JSON 的 Host 可以用 `command`、`args` 和 `env` 启动 STDIO Server。Codex 当前使用 `~/.codex/config.toml` 或项目内 `.codex/config.toml`，远程服务的 `url` 应指向实际 Streamable HTTP 地址，例如 `http://127.0.0.1:8000/mcp`

启动 CodeStats 的可视化调试：

```bash
cd backend
CODE_STATS_ROOT=/Users/noora/projects \
uv run --no-sync mcp dev samples/lesson13/codestats_mcp_v2.py
```

运行协议、鉴权与 LangGraph 独立验证：

```bash
cd backend
uv run --no-sync python scripts/verify_talent_mcp.py
```

核心输出：

```text
[protocol] {"protocol": "2026-07-28", "tools": ["lookup_job_descriptions", "filter_candidates", "search_candidate_evidence", "get_candidate_profiles"], "resource_templates": ["talent://jobs/{job_code}", "talent://candidates/{candidate_id}/profile"], "resource_job_code": "JD-AI-001", "prompt_role": "user"}
[auth] {"missing_token": 401, "missing_scope": 403, "wrong_audience": 401}
[langgraph] {"discovered_tools": ["lookup_job_descriptions", "filter_candidates", "search_candidate_evidence", "get_candidate_profiles"], "candidate_ids": ["C001"], "tool_status": "success"}
```

运行本节定向测试：

```bash
uv run --no-sync pytest -q tests/test_codestats_mcp_v2.py tests/test_talent_mcp_client.py tests/test_talent_mcp_server.py tests/test_talent_tools.py
```

结果输出为 `33 passed in 1.76s`。后端完整回归结果为 `167 passed, 2 warnings in 9.19s`

## Chunk 模块

| 路径 | 用途 |
|---|---|
| `backend/app/chunking.py` | Markdown 层级切分与递归兜底切分 |
| `backend/app/chunk_service.py` | 解析结果对齐、策略自动判断与 Parent-Child 持久化 |
| `backend/app/document_parsers.py` | DOCX/PPTX/Markdown/音频转录的归一化文本解析 |

第 5 课的代码口径是先统一把材料归一化成可切分文本，再由后端根据结构自动选择切分方式，不再让前端指定策略

默认规则如下

- Markdown、DOCX、PPTX、MinerU 解析后的 PDF 和图片：优先走 Markdown 结构化切分，再建立 Parent-Child
- 语音转录：只保留纯文本和时间区间，直接走递归切分
- DOCX：使用 `python-docx` 读取 OOXML 样式后转成 Markdown
- PPTX：使用 OOXML 解析幻灯片文本并转成 Markdown
- PDF 和图片：统一走 MinerU，保留标题层级、页码和 bbox
- 前端不暴露策略选择，只允许调整 `chunk_size` 和 `chunk_overlap`

所有 Chunk 使用 `markdown_start` 和 `markdown_end` 定位归一化 Markdown

PDF、图片、PPTX 和音频的辅助定位写入 `document_chunks.source_locators`

Markdown 和 DOCX 的 `source_locators` 为空数组

## 第 5 课数据库变更

`005_chunking.sql` 集中包含第 5 课的全部数据库变更

- 新增 `chunking_status` 枚举
- 新增 `chunk_strategy` 枚举
- 新增 `chunking_runs` 切片运行表
- 新增 `document_chunks` Chunk 表
- 新增切片运行、文档版本和候选人查询索引
- `document_chunks` 保存 Parent-Child、前后 Chunk 和标题路径
- `document_chunks` 新增 Markdown 起止偏移
- `document_chunks` 新增原文件辅助定位集合

### 从第 4 课迁移到第 5 课

先确保 PostgreSQL 已经通过同一个项目名启动：

```bash
docker compose -p talent-eval-agents-course up -d postgres
```

然后在代码仓库根目录执行一次以下命令

```bash
docker compose -p talent-eval-agents-course exec -T postgres \
  psql -U talent -d talent_docs -v ON_ERROR_STOP=1 -f /dev/stdin \
 < database/migrations/005_chunking.sql
```
迁移脚本使用 `IF NOT EXISTS` 与重复对象处理，可以在已执行旧版 `005` 的开发环境中再次执行。

全新环境由 `database/init.sql` 直接创建最新结构，不需要再执行增量迁移

## Milvus 证据索引

Milvus 作为独立向量检索服务，PostgreSQL 继续保存文档、版本、Chunk 和索引任务状态

| 路径 | 用途 |
|---|---|
| `backend/app/milvus_store.py` | Collection Schema、HNSW、Upsert、标量过滤、搜索与按版本删除 |
| `backend/app/evidence_index_service.py` | Chunk 向量化、Evidence Record 映射、索引任务执行与状态更新 |
| `backend/app/document_pipeline.py` | 上传后的自动解析、自动切片、自动索引编排与处理状态汇总 |
| `backend/app/document_cleanup.py` | 文档删除时的对象存储、PostgreSQL 和 Milvus 级联清理 |
| `backend/scripts/verify_milvus.py` | 使用确定性向量验证 Collection、Upsert、权限过滤和 HNSW 搜索 |
| `backend/tests/test_milvus_store.py` | Milvus 存储适配器的行为回归测试 |
| `backend/tests/test_evidence_index_service.py` | Chunk 到 Evidence Record 的转换与索引编排测试 |
| `backend/tests/test_document_pipeline.py` | 自动处理编排与任务入队测试 |
| `backend/tests/test_document_cleanup.py` | 材料删除时的跨存储级联清理测试 |

开发环境使用 Milvus Standalone 2.6.17、etcd 和已有 MinIO

- 宿主机 Milvus gRPC 端口：`19531`
- 宿主机 Milvus 健康检查端口：`19091`
- 容器内 backend 与 worker 通过 `http://milvus:19530` 访问

先确保 PostgreSQL 已经通过同一个项目名启动：

```bash
docker compose -p talent-eval-agents-course up -d postgres
```

然后在代码仓库根目录执行：

```bash
docker compose -p talent-eval-agents-course exec -T postgres \
  psql -U talent -d talent_docs -v ON_ERROR_STOP=1 -f /dev/stdin \
  < database/migrations/007_milvus_evidence_index.sql
```

全新环境由 `database/init.sql` 直接创建最新结构，不需要再执行增量迁移

启动服务并执行 Milvus 独立验收：

```bash
docker compose -p talent-eval-agents-course up -d
cd backend
uv run python -m scripts.verify_milvus
```

验收脚本输出

```json
{"collection": "lesson6_verification_v1", "upserted": 2, "matched": 1, "top_candidate": "C001", "top_score": 1.0, "permission_scope": "hr_private"}
```

Collection 默认使用 1024 维向量、COSINE 距离和 HNSW 索引

- `M=16`

- `efConstruction=128`

- 查询默认 `ef=80`

- 常规查询默认使用 Bounded consistency

- 写后读验收使用 Strong consistency

## 混合检索与精排

Collection `talent_evidence_v2` 在稠密向量之外增加 BM25 稀疏向量，由 Milvus 原生 Function 在写入时从 `content` 自动生成，`content` 使用 `chinese`（jieba + cnalphanumonly）分析器分词

| 路径 | 用途 |
|---|---|
| `backend/app/reranker.py` | DashScope `gte-rerank-v2` 的 Cross-Encoder 精排封装 |
| `backend/app/hybrid_search_service.py` | 稠密 + BM25 召回、RRF 融合、Rerank 精排的编排 |
| `backend/scripts/verify_hybrid.py` | 对比纯向量、纯 BM25、混合召回和精排的召回与延迟 |
| `backend/scripts/import_markdown_reviews.py` | 上传 5 份项目复盘 Markdown 并走完解析、切片、索引 |
| `backend/tests/test_hybrid_search_service.py` | 混合检索编排与 Rerank 映射的回归测试 |

稠密召回使用 `text-embedding-v3` 生成 1024 维查询向量，BM25 召回直接传入查询原文，两路结果由 `RRFRanker(k=60)` 融合，最后用 `gte-rerank-v2` 对候选做 Cross-Encoder 精排

相关配置

- `RERANK_MODEL`：精排模型，默认 `gte-rerank-v2`

- `RERANK_TOP_N`：精排候选数，默认 20

- `MILVUS_COLLECTION`：证据集合，当前为 `talent_evidence_v2`

混合检索验收

```bash
cd backend
uv run python -m scripts.verify_hybrid "有 Flink 实时计算经验的数据平台工程师" 5
```

## 验证

```bash
cd backend
uv sync --dev
uv run pytest tests -q
```

当前回归结果以本次本地 `pytest` 结果为准

结果输出：`167 passed, 2 warnings in 9.19s`

## 服务日志

backend 与 worker 使用统一 Python 日志配置，控制台日志同时写入 `backend/logs/`

```text
backend/logs/backend-YYYY-MM-DD.log
backend/logs/backend-error-YYYY-MM-DD.log
backend/logs/worker-YYYY-MM-DD.log
backend/logs/worker-error-YYYY-MM-DD.log
```

普通日志记录 HTTP 请求、员工与知识库写入、文件上传、MinIO 读写、解析任务、产物和 Worker 消费过程。ERROR 及以上日志额外写入对应服务的独立 error 文件。服务跨越零点运行时自动切换到新的日期文件

本地运行默认使用 `backend/logs/`。发布模式下，Compose 为 backend 和 worker 配置 `LOG_DIR=/app/logs` 与 `TZ=Asia/Shanghai`，并将宿主机 `./backend/logs` 挂载到容器内

## 端口约定

- PostgreSQL：`15432`
- Redis：`16379`
- MinIO API：`19000`
- MinIO Console：`19001`
- Milvus gRPC：`19531`
- Milvus Health：`19091`
- Backend API：`18080`
- Frontend：`15173`
