import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  BookOpen,
  ChevronLeft,
  ChevronRight,
  FileSearch,
  FileText,
  FlaskConical,
  LayoutGrid,
  Plus,
  Search,
  Sparkles,
  Trash2,
  Upload,
  UserSquare2,
  X,
} from "lucide-react";
import { EvidencePackPanel } from "./EvidencePackPanel";
import "./style.css";
import "./refinements.css";

const API = "http://127.0.0.1:18080/api";
const TENANT_ID = "course-demo";
const PAGE_SIZE = 10;

type Employee = {
  id: string;
  employee_no: string;
  name: string;
  gender?: string;
  age?: number;
  region?: string;
  current_position?: string;
  job_level?: string;
  years_of_experience?: number;
  department?: string;
  material_count: number;
};

type KB = {
  id: string;
  name: string;
  description?: string;
  permission_scope: string;
  file_count: number;
};

type Doc = {
  id: string;
  material_no?: string;
  candidate_id: string;
  employee_name: string;
  knowledge_base_id?: string;
  title: string;
  document_type: string;
  status: string;
  pipeline_status?: string;
  parse_status?: string | null;
  chunk_status?: string | null;
  index_status?: string | null;
  created_at: string;
};

type Job = {
  id: string;
  parser_name: string;
  parser_version?: string;
  status: string;
  progress: number;
  error_message?: string;
  created_at: string;
};

type IndexJob = {
  id: string;
  status: string;
  embedding_model: string;
  collection_name: string;
  indexed_count: number;
  retry_count: number;
  error_message?: string;
  created_at: string;
};

type Artifact = {
  id: string;
  type: string;
  url: string;
  content?: string;
};

type Detail = Doc & {
  employee?: { id: string; employee_no: string; name: string };
  knowledge_base?: { id: string; name: string };
  permission_scope: string;
  updated_at: string;
  version?: { id: string; version_no: number; created_at: string };
  file?: {
    name: string;
    mime_type: string;
    size_bytes: number;
    bucket_name: string;
    object_key: string;
    preview_url: string;
  };
  jobs: Job[];
  index_jobs: IndexJob[];
  artifacts: Artifact[];
};

type Chunk = {
  id: string;
  stable_key: string;
  position: number;
  level: "parent" | "child";
  content: string;
  element_ids: string[];
  heading_path: string[];
  parent_chunk_id?: string | null;
  page_start?: number | null;
  page_end?: number | null;
  timestamp_start?: number | null;
  timestamp_end?: number | null;
  markdown_start?: number | null;
  markdown_end?: number | null;
  source_locators: Record<string, unknown>[];
};

type ChunkData = {
  run?: {
    id: string;
    strategy: string;
    status: string;
    chunk_size: number;
    chunk_overlap: number;
    chunker_version: string;
  };
  chunks: Chunk[];
};

type EvidenceHit = {
  chunk_id: string;
  candidate_id: string;
  candidate_name: string;
  document_id?: string | null;
  document_title?: string | null;
  material_no?: string | null;
  document_type?: string | null;
  permission_scope?: string | null;
  page_start?: number | null;
  page_end?: number | null;
  content: string;
  score: number;
  metadata: Record<string, unknown>;
};

const types: Record<string, string> = {
  resume: "个人简历",
  interview: "面试记录",
  project: "项目成果",
  certificate: "能力证明",
  hris: "HRIS 数据",
  performance: "绩效记录",
  performance_review: "绩效记录",
  review: "述职报告",
  promotion: "晋升材料",
};

const pipelineLabels: Record<string, string> = {
  uploaded: "仅上传",
  parsing: "解析中",
  parsed: "已解析",
  chunking: "切片中",
  chunked: "已切片",
  indexing: "索引中",
  ready: "可召回",
  failed: "失败",
};

const pipelineTone: Record<string, string> = {
  uploaded: "pending",
  parsing: "pending",
  parsed: "running",
  chunking: "running",
  chunked: "running",
  indexing: "running",
  ready: "succeeded",
  failed: "failed",
};

async function json(url: string, options?: RequestInit) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let message = "请求失败";
    try {
      const payload = await response.json();
      message = payload.detail || message;
    } catch {
      const text = await response.text();
      if (text) message = text;
    }
    throw new Error(message);
  }
  return response.json();
}

function size(n = 0) {
  if (n < 1024) return `${n} B`;
  if (n < 1048576) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1048576).toFixed(1)} MB`;
}

function formatPipeline(status?: string | null) {
  return pipelineLabels[status || "uploaded"] || status || "仅上传";
}

function statusClass(status?: string | null) {
  return pipelineTone[status || "uploaded"] || status || "pending";
}

function Pagination({
  page,
  total,
  onChange,
}: {
  page: number;
  total: number;
  onChange: (value: number) => void;
}) {
  const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  return (
    <div className="pagination">
      <span>
        共 {total} 条 · 第 {page}/{pages} 页
      </span>
      <div>
        <button disabled={page === 1} onClick={() => onChange(page - 1)}>
          <ChevronLeft />
          上一页
        </button>
        <button disabled={page === pages} onClick={() => onChange(page + 1)}>
          下一页
          <ChevronRight />
        </button>
      </div>
    </div>
  );
}

function ChunkPane({ document, onDocumentChanged }: { document: Detail; onDocumentChanged: (id: string) => Promise<void> }) {
  const [data, setData] = useState<ChunkData>({ chunks: [] });
  const [chunkBusy, setChunkBusy] = useState(false);
  const [indexBusy, setIndexBusy] = useState(false);
  const [error, setError] = useState("");
  const [chunkSize, setChunkSize] = useState(800);
  const [overlap, setOverlap] = useState(100);

  async function load() {
    setData(await json(`${API}/documents/${document.id}/chunks`));
  }

  useEffect(() => {
    void load();
  }, [document.id]);

  async function generate() {
    setChunkBusy(true);
    setError("");
    try {
      await json(`${API}/documents/${document.id}/chunks`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ chunk_size: chunkSize, chunk_overlap: overlap }),
      });
      await load();
      await onDocumentChanged(document.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "切片失败");
    } finally {
      setChunkBusy(false);
    }
  }

  async function createIndex() {
    setIndexBusy(true);
    setError("");
    try {
      await json(`${API}/documents/${document.id}/evidence-index`, { method: "POST" });
      await onDocumentChanged(document.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "索引失败");
    } finally {
      setIndexBusy(false);
    }
  }

  const children = data.chunks.filter((item) => item.level === "child");
  const parents = data.chunks.filter((item) => item.level === "parent");
  const canIndex = Boolean(data.run && children.length > 0);
  const latestIndexJob = document.index_jobs[0];

  return (
    <div className="pane chunkPane">
      <div className="chunkControls">
        <p className="chunkHint">上传后会自动走解析、切片和索引。若当前材料只完成了解析，可以先手动生成切片，再创建索引。</p>
        <label>
          Chunk Size
          <input type="number" value={chunkSize} onChange={(event) => setChunkSize(Number(event.target.value))} />
        </label>
        <label>
          Overlap
          <input type="number" value={overlap} onChange={(event) => setOverlap(Number(event.target.value))} />
        </label>
        <button className="primary" onClick={generate} disabled={chunkBusy}>
          {chunkBusy ? "处理中" : data.run ? "重建切片" : "生成切片"}
        </button>
        <button className="ghost" onClick={createIndex} disabled={indexBusy || !canIndex}>
          {indexBusy ? "提交中" : latestIndexJob ? "重新创建索引" : "创建索引"}
        </button>
      </div>
      {!canIndex && document.parse_status === "succeeded" && (
        <p className="chunkHint secondaryText">当前材料已解析完成，但还没有可用切片。先点击生成切片，再创建索引。</p>
      )}
      {latestIndexJob && (
        <div className="runSummary">
          <span>
            索引状态 <b>{latestIndexJob.status}</b>
          </span>
          <span>
            向量模型 <b>{latestIndexJob.embedding_model}</b>
          </span>
          <span>
            已写入 <b>{latestIndexJob.indexed_count}</b>
          </span>
        </div>
      )}
      {error && <p className="chunkError">{error}</p>}
      {data.run ? (
        <div className="runSummary">
          <span>
            策略 <b>{data.run.strategy}</b>
          </span>
          <span>
            版本 <b>{data.run.chunker_version}</b>
          </span>
          <span>
            子 Chunk <b>{children.length}</b>
          </span>
          <span>
            Parent <b>{parents.length}</b>
          </span>
        </div>
      ) : (
        <div className="empty compact">
          <BookOpen />
          <h3>尚未生成切片</h3>
          <p>当前文档还没有可用切片，等待自动处理完成后会在这里显示。</p>
        </div>
      )}
      {children.map((chunk, index) => (
        <article className="chunkCard" key={chunk.id}>
          <header>
            <span>Chunk {index + 1}</span>
            <em>{chunk.content.length} 字符</em>
            {chunk.page_start != null && (
              <em>
                第 {chunk.page_start}
                {chunk.page_end != null && chunk.page_end !== chunk.page_start ? `-${chunk.page_end}` : ""} 页
              </em>
            )}
            {chunk.timestamp_start != null && chunk.timestamp_end != null && (
              <em>
                {chunk.timestamp_start.toFixed(2)}s - {chunk.timestamp_end.toFixed(2)}s
              </em>
            )}
          </header>
          {chunk.heading_path.length > 0 && <small>{chunk.heading_path.join(" / ")}</small>}
          <p>{chunk.content}</p>
          <footer>
            <code>{chunk.element_ids.join(", ") || "未记录元素 ID"}</code>
            <span>{chunk.parent_chunk_id ? "已关联 Parent" : "独立证据单元"}</span>
          </footer>
        </article>
      ))}
    </div>
  );
}

function RecallLab({
  employees,
  onOpenDocument,
}: {
  employees: Employee[];
  onOpenDocument: (id: string) => void;
}) {
  const [query, setQuery] = useState("");
  const [candidateId, setCandidateId] = useState("");
  const [documentType, setDocumentType] = useState("");
  const [permissionScope, setPermissionScope] = useState("hr_private");
  const [limit, setLimit] = useState(5);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [results, setResults] = useState<EvidenceHit[]>([]);

  async function searchEvidence() {
    if (!query.trim()) {
      setError("请输入召回查询");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const payload: Record<string, unknown> = { query, limit, ef: 80 };
      if (candidateId) payload.candidate_ids = [candidateId];
      if (documentType) payload.document_types = [documentType];
      const response = await json(`${API}/evidence/search`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Tenant-Id": TENANT_ID,
          "X-Permission-Scopes": permissionScope,
        },
        body: JSON.stringify(payload),
      });
      setResults(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "召回失败");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="recallLab">
      <EvidencePackPanel api={API} tenant={TENANT_ID} />
      <div className="recallHead">
        <div>
          <h3>证据召回测试</h3>
          <p>直接验证向量检索结果，查看匹配分数、来源材料、候选人和过滤条件的作用。</p>
        </div>
        <button className="ghost" onClick={searchEvidence} disabled={busy}>
          <FileSearch />
          {busy ? "召回中" : "执行召回"}
        </button>
      </div>
      <div className="recallFilters">
        <label className="wide">
          Query
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="例如：负责推荐系统升级并带来显著收益" />
        </label>
        <label>
          候选人
          <select value={candidateId} onChange={(event) => setCandidateId(event.target.value)}>
            <option value="">全部候选人</option>
            {employees.map((item) => (
              <option value={item.employee_no} key={item.id}>
                {item.name} · {item.employee_no}
              </option>
            ))}
          </select>
        </label>
        <label>
          材料类型
          <select value={documentType} onChange={(event) => setDocumentType(event.target.value)}>
            <option value="">全部类型</option>
            {Object.entries(types).map(([value, label]) => (
              <option value={value} key={value}>
                {label}
              </option>
            ))}
          </select>
        </label>
        <label>
          权限范围
          <select value={permissionScope} onChange={(event) => setPermissionScope(event.target.value)}>
            <option value="hr_private">HR 专属</option>
            <option value="department">部门可见</option>
            <option value="internal">公司内部</option>
          </select>
        </label>
        <label>
          Top K
          <input type="number" min={1} max={20} value={limit} onChange={(event) => setLimit(Number(event.target.value) || 5)} />
        </label>
      </div>
      {error && <p className="chunkError">{error}</p>}
      {results.length === 0 ? (
        <div className="recallEmpty">
          <FlaskConical />
          <span>输入查询后，这里会展示召回到的 Chunk 与对应上下文。</span>
        </div>
      ) : (
        <div className="recallResults">
          {results.map((item, index) => (
            <article className="recallCard" key={item.chunk_id}>
              <header>
                <strong>#{index + 1}</strong>
                <span className={`status ${statusClass("ready")}`}>Score {item.score.toFixed(4)}</span>
              </header>
              <div className="recallMeta">
                <span>{item.candidate_name}</span>
                <span>{types[item.document_type || ""] || item.document_type || "-"}</span>
                <span>{item.document_title || "未关联材料标题"}</span>
                <span>{item.permission_scope || "-"}</span>
                <span>
                  {item.page_start != null ? `第 ${item.page_start}${item.page_end && item.page_end !== item.page_start ? `-${item.page_end}` : ""} 页` : "无页码"}
                </span>
              </div>
              <p>{item.content}</p>
              <footer>
                <code>{item.material_no || item.chunk_id}</code>
                {item.document_id && (
                  <button className="outline small" onClick={() => onOpenDocument(item.document_id || "")}>
                    查看材料
                  </button>
                )}
              </footer>
            </article>
          ))}
        </div>
      )}
    </section>
  );
}

function App() {
  const [page, setPage] = useState<"roster" | "library">("roster");
  const [employees, setEmployees] = useState<Employee[]>([]);
  const [kbs, setKbs] = useState<KB[]>([]);
  const [docs, setDocs] = useState<Doc[]>([]);
  const [kb, setKb] = useState("");
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<Detail | null>(null);
  const [detailTab, setDetailTab] = useState<"original" | "parsed" | "chunks" | "meta">("original");
  const [uploadOpen, setUploadOpen] = useState(false);
  const [employeeOpen, setEmployeeOpen] = useState(false);
  const [knowledgeBaseOpen, setKnowledgeBaseOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const [rosterPage, setRosterPage] = useState(1);
  const [docPage, setDocPage] = useState(1);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [selectedDocs, setSelectedDocs] = useState<string[]>([]);
  const [pipelineFilter, setPipelineFilter] = useState("");

  async function load(preferredKnowledgeBaseId = kb) {
    const [employeeRows, kbRows] = await Promise.all([
      json(`${API}/employees`),
      json(`${API}/knowledge-bases`),
    ]);
    const activeKnowledgeBaseId = kbRows.some((item: KB) => item.id === preferredKnowledgeBaseId)
      ? preferredKnowledgeBaseId
      : kbRows[0]?.id || "";
    const documentRows = await json(`${API}/documents${activeKnowledgeBaseId ? `?knowledge_base_id=${activeKnowledgeBaseId}` : ""}`);
    setEmployees(employeeRows);
    setKbs(kbRows);
    setDocs(documentRows);
    setKb(activeKnowledgeBaseId);
  }

  async function openDocument(id: string) {
    if (!id) return;
    setSelected(await json(`${API}/documents/${id}`));
  }

  useEffect(() => {
    void load();
  }, [kb]);

  useEffect(() => {
    const hasRunningDocs = docs.some((item) => ["parsing", "chunking", "indexing"].includes(item.pipeline_status || ""));
    if (!hasRunningDocs) return;
    const timer = window.setInterval(() => void load(), 2000);
    return () => window.clearInterval(timer);
  }, [docs, kb]);

  useEffect(() => {
    if (!selected || !["parsing", "chunking", "indexing"].includes(selected.pipeline_status || "")) return;
    const timer = window.setInterval(() => void openDocument(selected.id), 1500);
    return () => window.clearInterval(timer);
  }, [selected]);

  const filteredEmployees = useMemo(
    () =>
      employees.filter((item) =>
        `${item.employee_no}${item.name}${item.current_position || ""}${item.department || ""}`.toLowerCase().includes(query.toLowerCase()),
      ),
    [employees, query],
  );
  const filteredDocs = useMemo(
    () =>
      docs.filter((item) => {
        const matchedKeyword = `${item.title}${item.employee_name}${item.document_type}${item.material_no || ""}`
          .toLowerCase()
          .includes(query.toLowerCase());
        const matchedPipeline = !pipelineFilter || (item.pipeline_status || "uploaded") === pipelineFilter;
        return matchedKeyword && matchedPipeline;
      }),
    [docs, query, pipelineFilter],
  );
  const pagedEmployees = filteredEmployees.slice((rosterPage - 1) * PAGE_SIZE, rosterPage * PAGE_SIZE);
  const pagedDocs = filteredDocs.slice((docPage - 1) * PAGE_SIZE, docPage * PAGE_SIZE);
  const allVisibleSelected = pagedDocs.length > 0 && pagedDocs.every((item) => selectedDocs.includes(item.id));
  const markdown = selected?.artifacts.find((item) => item.type === "markdown")?.content;

  async function createEmployee(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    try {
      const values = Object.fromEntries(new FormData(event.currentTarget));
      await json(`${API}/employees`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...values,
          years_of_experience: Number(values.years_of_experience) || null,
        }),
      });
      setEmployeeOpen(false);
      await load();
      setNotice("员工已保存");
    } finally {
      setBusy(false);
    }
  }

  async function createKnowledgeBase(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    try {
      const values = Object.fromEntries(new FormData(event.currentTarget));
      const created = await json(`${API}/knowledge-bases`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(values),
      });
      setKb(created.id);
      setKnowledgeBaseOpen(false);
      await load(created.id);
      setNotice("知识库已创建");
    } finally {
      setBusy(false);
    }
  }

  async function deleteKnowledgeBase(item: KB) {
    if (item.file_count > 0) {
      setNotice(`请先删除或迁移知识库中的 ${item.file_count} 份材料`);
      return;
    }
    if (!window.confirm(`确认删除知识库“${item.name}”吗？`)) return;
    setBusy(true);
    try {
      await json(`${API}/knowledge-bases/${item.id}`, { method: "DELETE" });
      await load(kb === item.id ? "" : kb);
      setNotice("知识库已删除");
    } finally {
      setBusy(false);
    }
  }

  async function upload(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    try {
      const data = new FormData(event.currentTarget);
      await json(`${API}/documents`, { method: "POST", body: data });
      setUploadOpen(false);
      setNotice("材料已上传，解析、切片和索引已自动排队");
      await load();
    } finally {
      setBusy(false);
    }
  }

  async function parse() {
    if (!selected) return;
    setBusy(true);
    try {
      await json(`${API}/documents/${selected.id}/parse`, { method: "POST" });
      await openDocument(selected.id);
      await load();
    } finally {
      setBusy(false);
    }
  }

  async function deleteSelectedDocuments() {
    if (selectedDocs.length === 0) return;
    if (!window.confirm(`确认删除选中的 ${selectedDocs.length} 份材料吗？这会同时删除原文件、解析结果、切片和向量索引。`)) return;
    setBusy(true);
    try {
      await json(`${API}/documents`, {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ document_ids: selectedDocs }),
      });
      if (selected && selectedDocs.includes(selected.id)) setSelected(null);
      setSelectedDocs([]);
      await load();
      setNotice("材料及关联索引已删除");
    } finally {
      setBusy(false);
    }
  }

  function toggleDocumentSelection(documentId: string, checked: boolean) {
    setSelectedDocs((current) =>
      checked ? Array.from(new Set([...current, documentId])) : current.filter((item) => item !== documentId),
    );
  }

  function togglePageSelection(checked: boolean) {
    setSelectedDocs((current) => {
      const visibleIds = pagedDocs.map((item) => item.id);
      if (checked) return Array.from(new Set([...current, ...visibleIds]));
      return current.filter((item) => !visibleIds.includes(item));
    });
  }

  return (
    <div className={`app${sidebarCollapsed ? " sidebarCollapsed" : ""}`}>
      <aside className="side">
        <div className="brand">
          <img src="/talentos-favicon.png" alt="TalentOS" />
          <div>
            <b>智鉴 TalentOS</b>
            <small>多 AGENT 人才决策平台</small>
          </div>
        </div>
        <nav>
          <button>
            <LayoutGrid />
            <span>工作台</span>
          </button>
          <button className="active">
            <UserSquare2 />
            <span>人才档案</span>
          </button>
          <button>
            <Sparkles />
            <span>评估与推荐</span>
          </button>
          <button>
            <FlaskConical />
            <span>评测中心</span>
          </button>
        </nav>
        <div className="env">
          <div>
            <i />
            <b>测试环境</b>
          </div>
          <small>服务正常 · V1.0</small>
        </div>
        <button
          type="button"
          className="sideToggle"
          onClick={() => setSidebarCollapsed((value) => !value)}
          aria-label={sidebarCollapsed ? "展开导航栏" : "收起导航栏"}
          aria-expanded={!sidebarCollapsed}
        >
          {sidebarCollapsed ? <ChevronRight /> : <ChevronLeft />}
        </button>
      </aside>
      <main>
        <header>
          <div>
            <span>人才档案</span>
            <ChevronRight />
            <b>{page === "roster" ? "员工花名册" : "档案资料库"}</b>
          </div>
          <div className="currentUser">
            <b>Will</b>
            <span>W</span>
          </div>
        </header>
        <section className="content">
          <div className="moduleTabs">
            <button
              className={page === "roster" ? "active" : ""}
              onClick={() => {
                setPage("roster");
                setQuery("");
                setRosterPage(1);
              }}
            >
              员工花名册
            </button>
            <button
              className={page === "library" ? "active" : ""}
              onClick={() => {
                setPage("library");
                setQuery("");
                setDocPage(1);
              }}
            >
              档案资料库
            </button>
          </div>

          {page === "roster" ? (
            <>
              <div className="pageHead">
                <div>
                  <h1>员工花名册</h1>
                  <p>维护结构化员工信息，并查看每名员工关联的档案材料</p>
                </div>
                <div className="actions">
                  <button className="primary" onClick={() => setEmployeeOpen(true)}>
                    <Plus />
                    新建员工
                  </button>
                </div>
              </div>
              <div className="toolbar">
                <label>
                  <Search />
                  <input
                    value={query}
                    onChange={(event) => {
                      setQuery(event.target.value);
                      setRosterPage(1);
                    }}
                    placeholder="搜索工号、姓名、岗位或部门"
                  />
                </label>
                <span>{filteredEmployees.length} 名员工</span>
              </div>
              <div className="table roster">
                <div className="tr th">
                  <span>员工</span>
                  <span>性别</span>
                  <span>年龄</span>
                  <span>地区</span>
                  <span>当前岗位</span>
                  <span>职级</span>
                  <span>工作年限</span>
                  <span>所属部门</span>
                  <span>关联材料</span>
                </div>
                {pagedEmployees.map((item) => (
                  <div className="tr" key={item.id}>
                    <span className="person">
                      <i>{item.name.slice(-2)}</i>
                      <b>
                        {item.name}
                        <small>{item.employee_no}</small>
                      </b>
                    </span>
                    <span>{item.gender || "-"}</span>
                    <span>{item.age ?? "-"}</span>
                    <span>{item.region || "-"}</span>
                    <span>{item.current_position || "-"}</span>
                    <span>
                      <em>{item.job_level || "-"}</em>
                    </span>
                    <span>{item.years_of_experience ?? "-"} 年</span>
                    <span>{item.department || "-"}</span>
                    <span>
                      <strong>{item.material_count} 份</strong>
                    </span>
                  </div>
                ))}
              </div>
              <Pagination page={rosterPage} total={filteredEmployees.length} onChange={setRosterPage} />
            </>
          ) : (
            <>
              <div className="pageHead libraryHead">
                <div>
                  <h1>档案资料库</h1>
                  <div className="librarySubtitle">
                    <p>上传后自动完成解析、切片和证据索引，并支持直接做召回实验。</p>
                    <label>
                      当前知识库
                      <select
                        value={kb}
                        onChange={(event) => {
                          setKb(event.target.value);
                          setDocPage(1);
                        }}
                      >
                        {kbs.map((item) => (
                          <option value={item.id} key={item.id}>
                            {item.name} · {item.file_count} 个文件
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>
                </div>
                <div className="actions">
                  <button className="ghost" onClick={() => setKnowledgeBaseOpen(true)}>
                    <Plus />
                    新建知识库
                  </button>
                  <button className="ghost danger" onClick={deleteSelectedDocuments} disabled={busy || selectedDocs.length === 0}>
                    <Trash2 />
                    删除已选 {selectedDocs.length > 0 ? `(${selectedDocs.length})` : ""}
                  </button>
                  <button className="primary" onClick={() => setUploadOpen(true)} disabled={!kb}>
                    <Upload />
                    上传材料
                  </button>
                </div>
              </div>

              <section className="knowledgeBaseManager">
                <div className="managerHead">
                  <div>
                    <h2>知识库管理</h2>
                    <p>共 {kbs.length} 个知识库，删除前需要先清空其中的材料</p>
                  </div>
                </div>
                {kbs.length === 0 ? (
                  <div className="empty compact">暂无知识库，先新建知识库再上传材料</div>
                ) : (
                  <div className="knowledgeBaseGrid">
                    {kbs.map((item) => (
                      <article className={`knowledgeBaseCard ${kb === item.id ? "active" : ""}`} key={item.id}>
                        <button className="knowledgeBaseMain" onClick={() => {
                          setKb(item.id);
                          setDocPage(1);
                        }}>
                          <BookOpen />
                          <span>
                            <b>{item.name}</b>
                            <small>{item.description || "暂无描述"}</small>
                          </span>
                        </button>
                        <div className="knowledgeBaseMeta">
                          <span>{item.file_count} 份材料</span>
                          <span>{item.permission_scope === "hr_private" ? "HR 专属" : item.permission_scope === "department" ? "部门可见" : "公司内部"}</span>
                          <button
                            className="iconDanger"
                            title={item.file_count > 0 ? "请先清空知识库中的材料" : "删除知识库"}
                            disabled={busy || item.file_count > 0}
                            onClick={() => void deleteKnowledgeBase(item)}
                          >
                            <Trash2 />
                          </button>
                        </div>
                      </article>
                    ))}
                  </div>
                )}
              </section>

              <RecallLab employees={employees} onOpenDocument={(id) => void openDocument(id)} />

              <div className="documents full">
                <div className="toolbar">
                  <label>
                    <Search />
                    <input
                      value={query}
                      onChange={(event) => {
                        setQuery(event.target.value);
                        setDocPage(1);
                      }}
                      placeholder="搜索文件名、材料编号或员工"
                    />
                  </label>
                  <label className="filterSelect">
                    <span>处理状态</span>
                    <select value={pipelineFilter} onChange={(event) => {
                      setPipelineFilter(event.target.value);
                      setDocPage(1);
                    }}>
                      <option value="">全部状态</option>
                      {Object.entries(pipelineLabels).map(([value, label]) => (
                        <option value={value} key={value}>
                          {label}
                        </option>
                      ))}
                    </select>
                  </label>
                  <span>{filteredDocs.length} 份材料</span>
                </div>
                <div className="table docs">
                  <div className="tr th">
                    <span className="checkCell">
                      <input type="checkbox" checked={allVisibleSelected} onChange={(event) => togglePageSelection(event.target.checked)} />
                    </span>
                    <span>文件</span>
                    <span>员工</span>
                    <span>材料类型</span>
                    <span>处理状态</span>
                    <span>更新时间</span>
                    <span />
                  </div>
                  {pagedDocs.map((item) => (
                    <div className="tr docRow" key={item.id}>
                      <span className="checkCell">
                        <input
                          type="checkbox"
                          checked={selectedDocs.includes(item.id)}
                          onChange={(event) => toggleDocumentSelection(item.id, event.target.checked)}
                        />
                      </span>
                      <span className="file">
                        <FileText />
                        <b>
                          {item.title}
                          <small>{item.material_no}</small>
                        </b>
                      </span>
                      <span>{item.employee_name}</span>
                      <span>
                        <em>{types[item.document_type] || item.document_type}</em>
                      </span>
                      <span>
                        <strong className={`status ${statusClass(item.pipeline_status)}`}>{formatPipeline(item.pipeline_status)}</strong>
                      </span>
                      <span>{new Date(item.created_at).toLocaleString()}</span>
                      <span>
                        <button
                          className="outline small"
                          onClick={() => {
                            setDetailTab("original");
                            void openDocument(item.id);
                          }}
                        >
                          预览
                        </button>
                      </span>
                    </div>
                  ))}
                </div>
                <Pagination page={docPage} total={filteredDocs.length} onChange={setDocPage} />
              </div>
            </>
          )}
        </section>
      </main>

      {employeeOpen && (
        <div className="overlay">
          <form className="dialog" onSubmit={createEmployee}>
            <button type="button" className="close" onClick={() => setEmployeeOpen(false)}>
              <X />
            </button>
            <h2>新建员工</h2>
            <div className="formGrid">
              <label>
                员工工号
                <input name="employee_no" required />
              </label>
              <label>
                姓名
                <input name="name" required />
              </label>
              <label>
                性别
                <select name="gender">
                  <option>女</option>
                  <option>男</option>
                </select>
              </label>
              <label>
                出生日期
                <input name="birth_date" type="date" />
              </label>
              <label>
                所在地区
                <input name="region" />
              </label>
              <label>
                当前岗位
                <input name="current_position" />
              </label>
              <label>
                当前职级
                <select name="job_level">
                  {Array.from({ length: 10 }, (_, index) => (
                    <option key={index + 1}>L{index + 1}</option>
                  ))}
                </select>
              </label>
              <label>
                工作年限
                <input name="years_of_experience" type="number" step="0.5" />
              </label>
              <label className="wide">
                所属部门
                <input name="department" />
              </label>
            </div>
            <button className="primary" disabled={busy}>
              保存员工
            </button>
          </form>
        </div>
      )}

      {knowledgeBaseOpen && (
        <div className="overlay">
          <form className="dialog knowledgeBaseDialog" onSubmit={createKnowledgeBase}>
            <button type="button" className="close" onClick={() => setKnowledgeBaseOpen(false)}>
              <X />
            </button>
            <h2>新建知识库</h2>
            <p className="dialogHint">知识库用于组织不同范围的档案材料，创建后可直接上传文件。</p>
            <label>
              知识库名称
              <input name="name" placeholder="例如：研发中心人才档案" required autoFocus />
            </label>
            <label>
              知识库描述
              <textarea name="description" placeholder="说明该知识库收录的材料范围" rows={3} />
            </label>
            <label>
              权限范围
              <select name="permission_scope" defaultValue="hr_private">
                <option value="hr_private">HR 专属</option>
                <option value="department">部门可见</option>
                <option value="internal">公司内部</option>
              </select>
            </label>
            <button className="primary" disabled={busy}>
              创建知识库
            </button>
          </form>
        </div>
      )}

      {uploadOpen && (
        <div className="overlay">
          <form className="dialog" onSubmit={upload}>
            <button type="button" className="close" onClick={() => setUploadOpen(false)}>
              <X />
            </button>
            <h2>上传档案材料</h2>
            <label>
              所属知识库
              <select name="knowledge_base_id" value={kb} onChange={(event) => setKb(event.target.value)}>
                {kbs.map((item) => (
                  <option value={item.id} key={item.id}>
                    {item.name}
                  </option>
                ))}
              </select>
            </label>
            <div className="formGrid">
              <label>
                关联员工
                <select name="employee_id">
                  {employees.map((item) => (
                    <option value={item.id} key={item.id}>
                      {item.name} · {item.employee_no}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                材料类型
                <select name="document_type">
                  {Object.entries(types).map(([value, label]) => (
                    <option value={value} key={value}>
                      {label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="wide">
                材料标题
                <input name="title" required />
              </label>
              <label className="wide">
                权限范围
                <select name="permission_scope">
                  <option value="hr_private">HR 专属</option>
                  <option value="department">部门可见</option>
                  <option value="internal">公司内部</option>
                </select>
              </label>
              <label className="drop wide">
                <Upload />
                <b>选择 PDF、Office、图片、音频或文本文件</b>
                <input name="file" type="file" required />
              </label>
            </div>
            <button className="primary" disabled={busy}>
              上传并自动处理
            </button>
          </form>
        </div>
      )}

      {selected && (
        <div className="drawerWrap" onClick={() => setSelected(null)}>
          <article className="drawer" onClick={(event) => event.stopPropagation()}>
            <button className="close" onClick={() => setSelected(null)}>
              <X />
            </button>
            <div className="drawerHead">
              <small>{selected.material_no}</small>
              <h2>{selected.title}</h2>
              <p>
                {selected.employee?.name || selected.candidate_id} · {types[selected.document_type] || selected.document_type} · v
                {selected.version?.version_no || 1}
              </p>
              <div className="drawerStatusRow">
                <span className={`status ${statusClass(selected.pipeline_status)}`}>{formatPipeline(selected.pipeline_status)}</span>
                <span>解析 {selected.parse_status || "-"}</span>
                <span>切片 {selected.chunk_status || "-"}</span>
                <span>索引 {selected.index_status || "-"}</span>
              </div>
            </div>
            <div className="tabs">
              <button className={detailTab === "original" ? "active" : ""} onClick={() => setDetailTab("original")}>
                原始文件
              </button>
              <button className={detailTab === "parsed" ? "active" : ""} onClick={() => setDetailTab("parsed")}>
                解析结果
              </button>
              <button className={detailTab === "chunks" ? "active" : ""} onClick={() => setDetailTab("chunks")}>
                切片预览
              </button>
              <button className={detailTab === "meta" ? "active" : ""} onClick={() => setDetailTab("meta")}>
                文件元数据
              </button>
            </div>

            {detailTab === "original" && (
              <div className="pane">
                <div className="fileHero">
                  <FileText />
                  <div>
                    <b>{selected.file?.name}</b>
                    <span>
                      {selected.file?.mime_type} · {size(selected.file?.size_bytes)}
                    </span>
                  </div>
                </div>
                <a className="outline" href={selected.file?.preview_url} target="_blank" rel="noreferrer">
                  打开原始文件
                </a>
              </div>
            )}

            {detailTab === "parsed" && (
              <div className="pane">
                {markdown ? (
                  <pre className="markdown">{markdown}</pre>
                ) : (
                  <div className="empty">
                    <FileText />
                    <h3>尚无解析结果</h3>
                    <p>自动处理未完成时，这里会展示最新 Markdown 结果，也可以手动重试解析。</p>
                    <button className="primary" onClick={parse} disabled={busy}>
                      重新解析
                    </button>
                  </div>
                )}
                <h3>解析记录</h3>
                {selected.jobs.map((item) => (
                  <div className="job" key={item.id}>
                    <span className={`status ${statusClass(item.status)}`}>{item.status}</span>
                    <b>{item.parser_name}</b>
                    <small>{item.error_message || `进度 ${item.progress}%`}</small>
                  </div>
                ))}
                <h3>索引记录</h3>
                {selected.index_jobs.length === 0 ? (
                  <p className="secondaryText">当前版本还没有索引任务。</p>
                ) : (
                  selected.index_jobs.map((item) => (
                    <div className="job" key={item.id}>
                      <span className={`status ${statusClass(item.status)}`}>{item.status}</span>
                      <b>
                        {item.embedding_model} · {item.collection_name}
                      </b>
                      <small>{item.error_message || `已写入 ${item.indexed_count} 个 Chunk`}</small>
                    </div>
                  ))
                )}
              </div>
            )}

            {detailTab === "chunks" && <ChunkPane document={selected} onDocumentChanged={async (id) => {
              await openDocument(id);
              await load();
            }} />}

            {detailTab === "meta" && (
              <div className="pane meta">
                <dl>
                  <div>
                    <dt>材料 ID</dt>
                    <dd>{selected.material_no}</dd>
                  </div>
                  <div>
                    <dt>关联员工</dt>
                    <dd>
                      {selected.employee?.name} · {selected.employee?.employee_no}
                    </dd>
                  </div>
                  <div>
                    <dt>所属知识库</dt>
                    <dd>{selected.knowledge_base?.name}</dd>
                  </div>
                  <div>
                    <dt>材料类型</dt>
                    <dd>{types[selected.document_type] || selected.document_type}</dd>
                  </div>
                  <div>
                    <dt>文档版本</dt>
                    <dd>v{selected.version?.version_no || 1}</dd>
                  </div>
                  <div>
                    <dt>原始文件名</dt>
                    <dd>{selected.file?.name}</dd>
                  </div>
                  <div>
                    <dt>文件大小</dt>
                    <dd>{size(selected.file?.size_bytes)}</dd>
                  </div>
                  <div>
                    <dt>上传时间</dt>
                    <dd>{new Date(selected.created_at).toLocaleString()}</dd>
                  </div>
                  <div>
                    <dt>权限范围</dt>
                    <dd>{selected.permission_scope}</dd>
                  </div>
                  <div>
                    <dt>对象存储</dt>
                    <dd>
                      {selected.file?.bucket_name}/{selected.file?.object_key}
                    </dd>
                  </div>
                  <div>
                    <dt>Collection</dt>
                    <dd>{selected.index_jobs[0]?.collection_name || "-"}</dd>
                  </div>
                  <div>
                    <dt>Embedding 模型</dt>
                    <dd>{selected.index_jobs[0]?.embedding_model || "-"}</dd>
                  </div>
                </dl>
              </div>
            )}
          </article>
        </div>
      )}

      {notice && <div className="toast" onAnimationEnd={() => setNotice("")}>{notice}</div>}
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
