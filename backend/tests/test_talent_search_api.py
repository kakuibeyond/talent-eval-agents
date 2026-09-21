from types import SimpleNamespace

from fastapi.testclient import TestClient

from app import api
from app.database import get_db
from app.main import create_app
from app.milvus_store import EvidenceSearchResult
from app.query_plan import QueryOptimizationPlan, QueryPlan


class FakeScalarResult:
    def all(self):
        return ["C001", "C004"]


class FakeSession:
    def __init__(self):
        self.statements = []

    def scalars(self, statement):
        self.statements.append(statement)
        return FakeScalarResult()


class FakeStructuredModel:
    def __init__(self, schema):
        self.schema = schema

    def invoke(self, messages):
        if self.schema is QueryPlan:
            return QueryPlan.model_validate(
                {
                    "task_type": "find_talent",
                    "filters": [{"field": "age", "operator": "lt", "value": 35}],
                    "semantic_requirements": [
                        {"requirement_id": "S1", "query": "AI 项目工作经历", "required": True}
                    ],
                    "preferences": [],
                    "clarifications": [],
                }
            )
        raise AssertionError(f"unexpected schema: {self.schema}")


class FakeModel:
    def with_structured_output(self, schema):
        return FakeStructuredModel(schema)


class FakeEmbedder:
    def embed_query(self, query):
        assert query == "AI 项目工作经历"
        return [0.1, 0.2, 0.3]


class FakeStore:
    def __init__(self):
        self.filters = None

    def hybrid_search(self, query_text, query_vector, *, filters, limit, ef, rrf_k):
        self.filters = filters
        return [
            EvidenceSearchResult(
                chunk_id="chunk-001",
                candidate_id="C001",
                content="负责企业知识库与大模型应用项目",
                score=0.032,
                metadata={"document_type": "markdown", "permission_scope": "hr_private"},
            )
        ]


class FakeReranker:
    def rerank(self, *, query, documents, top_n):
        return [SimpleNamespace(index=0, score=0.91)]


def test_talent_search_endpoint_runs_query_plan_sql_and_hybrid_search(monkeypatch):
    app = create_app()
    db = FakeSession()
    store = FakeStore()
    app.dependency_overrides[get_db] = lambda: db
    monkeypatch.setattr(api, "get_chat_model", lambda temperature=0: FakeModel())
    monkeypatch.setattr(api, "get_embedding_model", lambda: FakeEmbedder())
    monkeypatch.setattr(api, "get_reranker", lambda: FakeReranker())
    monkeypatch.setattr(api, "get_evidence_store", lambda: store)

    response = TestClient(app).post(
        "/api/talent-search",
        headers={
            "X-Tenant-ID": "course-demo",
            "X-Permission-Scopes": "hr_private",
        },
        json={
            "query": "筛选 35 岁以下且有 AI 工作经历的候选人",
            "retrieval_mode": "auto_optimize",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["candidate_ids"] == ["C001", "C004"]
    assert body["query_plan"]["filters"] == [{"field": "age", "operator": "lt", "value": 35}]
    assert body["chunks"] == [
        {
            "chunk_id": "chunk-001",
            "candidate_id": "C001",
            "content": "负责企业知识库与大模型应用项目",
            "score": 0.91,
            "metadata": {"document_type": "markdown", "permission_scope": "hr_private"},
            "requirement_ids": ["S1"],
        }
    ]
    assert body["searches"] == [
        {"requirement_id": "S1", "strategy": None, "queries": ["AI 项目工作经历"], "chunk_count": 1}
    ]
    assert len(db.statements) == 1
    assert store.filters.candidate_ids == ["C001", "C004"]
    assert store.filters.tenant_id == "course-demo"
    assert store.filters.permission_scopes == ["hr_private"]


def test_talent_search_endpoint_retries_with_optimized_queries_when_first_search_is_empty(monkeypatch):
    class OptimizingStructuredModel(FakeStructuredModel):
        def invoke(self, messages):
            if self.schema is QueryOptimizationPlan:
                return QueryOptimizationPlan(
                    strategy="multi_query",
                    queries=["机器学习项目经历", "大模型项目经历"],
                    reason="首轮没有召回结果",
                )
            return super().invoke(messages)

    class OptimizingModel:
        def with_structured_output(self, schema):
            return OptimizingStructuredModel(schema)

    class AnyQueryEmbedder:
        def embed_query(self, query):
            return [0.1, 0.2, 0.3]

    class RetryStore(FakeStore):
        def __init__(self):
            super().__init__()
            self.queries = []

        def hybrid_search(self, query_text, query_vector, *, filters, limit, ef, rrf_k):
            self.filters = filters
            self.queries.append(query_text)
            if query_text == "AI 项目工作经历":
                return []
            return [
                EvidenceSearchResult(
                    chunk_id="chunk-optimized",
                    candidate_id="C004",
                    content="主导大模型应用项目落地",
                    score=0.03,
                    metadata={"document_type": "markdown", "permission_scope": "hr_private"},
                )
            ]

    app = create_app()
    store = RetryStore()
    app.dependency_overrides[get_db] = lambda: FakeSession()
    monkeypatch.setattr(api, "get_chat_model", lambda temperature=0: OptimizingModel())
    monkeypatch.setattr(api, "get_embedding_model", lambda: AnyQueryEmbedder())
    monkeypatch.setattr(api, "get_reranker", lambda: FakeReranker())
    monkeypatch.setattr(api, "get_evidence_store", lambda: store)

    response = TestClient(app).post(
        "/api/talent-search",
        headers={"X-Tenant-ID": "course-demo", "X-Permission-Scopes": "hr_private"},
        json={"query": "筛选 35 岁以下且有 AI 工作经历的候选人", "retrieval_mode": "auto_optimize"},
    )

    assert response.status_code == 200
    body = response.json()
    assert store.queries == ["AI 项目工作经历", "机器学习项目经历", "大模型项目经历"]
    assert body["searches"] == [
        {
            "requirement_id": "S1",
            "strategy": "multi_query",
            "queries": ["AI 项目工作经历", "机器学习项目经历", "大模型项目经历"],
            "chunk_count": 1,
        }
    ]
    assert [item["chunk_id"] for item in body["chunks"]] == ["chunk-optimized"]


def test_opt_in_evidence_pack_keeps_existing_chunks(monkeypatch):
    from app.evidence_pack import EvidenceExtraction
    class EvidenceModel:
        def with_structured_output(self, schema, **kwargs):
            if schema is EvidenceExtraction:
                assert kwargs == {'method': 'json_mode'}
                return SimpleNamespace(invoke=lambda _: EvidenceExtraction(
                    facts=[], fully_supported=False, missing_information=['材料未说明职责']))
            return FakeStructuredModel(schema)
    app = create_app()
    app.dependency_overrides[get_db] = lambda: FakeSession()
    monkeypatch.setattr(api, 'get_chat_model', lambda temperature=0: EvidenceModel())
    monkeypatch.setattr(api, 'get_embedding_model', lambda: FakeEmbedder())
    monkeypatch.setattr(api, 'get_reranker', lambda: FakeReranker())
    monkeypatch.setattr(api, 'get_evidence_store', lambda: FakeStore())
    # The API test substitutes storage only; grouping and model-output validation run normally.
    monkeypatch.setattr(api, 'load_pack_sources', lambda db, rows, **kw: [
        dict(rows[0], citation_id=rows[0]['chunk_id'])], raising=False)
    response = TestClient(app).post('/api/talent-search',
        headers={'X-Tenant-ID': 'course-demo', 'X-Permission-Scopes': 'hr_private'},
        json={'query': '有 AI 工作经历', 'include_evidence_pack': True})
    assert response.status_code == 200
    body = response.json()
    assert 'evidence_packs' in body
    assert len(body['chunks']) == 1
    assert body['evidence_packs'][0]['requirements'][0]['reason'] == 'no_relevant_evidence'
    assert body['evidence_packs'][1]['requirements'][0]['reason'] == 'no_accessible_hits'
