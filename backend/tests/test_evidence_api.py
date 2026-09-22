from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app import api
from app.api import EvidenceSearchInput
from app.models import Base, Document, DocumentVersion, EmployeeProfile, EvidenceIndexJob, FileObject, IndexStatus, KnowledgeBase


class FakeRedis:
    def __init__(self):
        self.items: list[tuple[str, str]] = []

    def rpush(self, queue: str, payload: str) -> None:
        self.items.append((queue, payload))


class FakeDb:
    def __init__(self, *, index_job: EvidenceIndexJob | None = None):
        self.index_job = index_job
        self.added: list[object] = []
        self.commits = 0

    def add(self, item: object) -> None:
        self.added.append(item)
        if isinstance(item, EvidenceIndexJob) and item.id is None:
            item.id = uuid4()

    def commit(self) -> None:
        self.commits += 1

    def get(self, model: object, value: UUID) -> object | None:
        if model is EvidenceIndexJob and self.index_job and self.index_job.id == value:
            return self.index_job
        return None


def test_create_evidence_index_queues_job_for_current_document_version(monkeypatch):
    fake_db = FakeDb()
    fake_redis = FakeRedis()
    version_id = uuid4()

    monkeypatch.setattr(api, "current_version", lambda db, document_id: SimpleNamespace(id=version_id))
    monkeypatch.setattr(api, "get_settings", lambda: SimpleNamespace(
        redis_url="redis://test",
        embedding_model="text-embedding-v3",
        milvus_collection="talent_evidence_v1",
    ))
    monkeypatch.setattr(api.Redis, "from_url", lambda *args, **kwargs: fake_redis)

    result = api.create_evidence_index(uuid4(), db=fake_db)

    assert result["document_version_id"] == version_id
    assert result["status"] == IndexStatus.PENDING
    assert fake_db.commits == 1
    assert fake_db.added
    assert fake_redis.items == [("talent:index:queue", f"{result['id']}:{version_id}")]


def test_search_evidence_builds_filter_from_headers_and_payload(monkeypatch):
    seen: dict[str, object] = {}

    class Embedder:
        def embed_query(self, text: str) -> list[float]:
            seen["query"] = text
            return [0.1, 0.2, 0.3]

    class Store:
        def search(self, query_vector, *, filters, limit, ef):
            seen["query_vector"] = query_vector
            seen["filters"] = filters
            seen["limit"] = limit
            seen["ef"] = ef
            return [
                SimpleNamespace(
                    chunk_id="chunk-1",
                    candidate_id="C001",
                    content="负责推荐系统升级",
                    score=0.97,
                    metadata={"permission_scope": "hr_private"},
                )
            ]

    monkeypatch.setattr(api, "get_embedding_model", lambda: Embedder())
    monkeypatch.setattr(api, "get_evidence_store", lambda: Store())

    result = api.search_evidence(
        EvidenceSearchInput(
            query="推荐系统负责人",
            candidate_ids=["C001"],
            document_types=["resume"],
            limit=5,
            ef=60,
        ),
        x_tenant_id="course-demo",
        x_permission_scopes="hr_private, manager_private",
        db=None,
    )

    assert result == [
        {
            "chunk_id": "chunk-1",
            "candidate_id": "C001",
            "candidate_name": "C001",
            "document_id": None,
            "document_title": None,
            "material_no": None,
            "document_type": None,
            "permission_scope": "hr_private",
            "page_start": None,
            "page_end": None,
            "content": "负责推荐系统升级",
            "score": 0.97,
            "metadata": {"permission_scope": "hr_private"},
        }
    ]
    assert seen["query"] == "推荐系统负责人"
    assert seen["query_vector"] == [0.1, 0.2, 0.3]
    assert seen["filters"].tenant_id == "course-demo"
    assert seen["filters"].permission_scopes == ["hr_private", "manager_private"]
    assert seen["filters"].candidate_ids == ["C001"]
    assert seen["filters"].document_types == ["resume"]
    assert seen["limit"] == 5
    assert seen["ef"] == 60


def test_search_evidence_rejects_empty_permission_scope_header():
    with pytest.raises(api.HTTPException, match="缺少可用的证据权限范围"):
        api.search_evidence(
            EvidenceSearchInput(query="推荐系统负责人"),
            x_tenant_id="course-demo",
            x_permission_scopes=" , ",
            db=None,
        )


def test_retry_index_job_requires_failed_status():
    job_id = uuid4()
    fake_db = FakeDb(
        index_job=EvidenceIndexJob(
            id=job_id,
            document_version_id=uuid4(),
            embedding_model="text-embedding-v3",
            collection_name="talent_evidence_v1",
            status=IndexStatus.SUCCEEDED,
        )
    )

    with pytest.raises(api.HTTPException, match="只能重试失败的索引任务"):
        api.retry_index_job(job_id, db=fake_db)


def test_delete_knowledge_base_rejects_non_empty_library():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    knowledge_base = KnowledgeBase(name="员工档案")
    session.add(knowledge_base)
    session.flush()
    session.add(
        Document(
            candidate_id="C001",
            knowledge_base_id=knowledge_base.id,
            tenant_id="course-demo",
            title="林晓岚简历",
            document_type="resume",
            permission_scope="hr_private",
        )
    )
    session.commit()

    with pytest.raises(api.HTTPException, match="请先删除或迁移材料"):
        api.delete_knowledge_base(knowledge_base.id, db=session)


def test_delete_empty_knowledge_base():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    knowledge_base = KnowledgeBase(name="待删除知识库")
    session.add(knowledge_base)
    session.commit()

    result = api.delete_knowledge_base(knowledge_base.id, db=session)

    assert result == {"id": knowledge_base.id, "name": "待删除知识库", "deleted": True}
    assert session.get(KnowledgeBase, knowledge_base.id) is None


def test_search_evidence_returns_document_and_candidate_context(monkeypatch):
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    employee = EmployeeProfile(employee_no="C001", name="林晓岚")
    session.add(employee)
    session.flush()
    document = Document(
        candidate_id="C001",
        employee_id=employee.id,
        tenant_id="course-demo",
        title="林晓岚简历",
        document_type="resume",
        permission_scope="hr_private",
    )
    session.add(document)
    session.flush()
    file_object = FileObject(
        bucket_name="talent-documents",
        object_key="documents/demo.pdf",
        original_name="demo.pdf",
        mime_type="application/pdf",
        size_bytes=128,
        sha256="1" * 64,
    )
    session.add(file_object)
    session.flush()
    version = DocumentVersion(document_id=document.id, file_object_id=file_object.id, version_no=1, is_current=True)
    session.add(version)
    session.commit()

    class Embedder:
        def embed_query(self, text: str) -> list[float]:
            return [0.1, 0.2, 0.3]

    class Store:
        def search(self, query_vector, *, filters, limit, ef):
            return [
                SimpleNamespace(
                    chunk_id="chunk-1",
                    candidate_id="C001",
                    content="负责推荐系统升级",
                    score=0.97,
                    metadata={
                        "document_version_id": str(version.id),
                        "document_type": "resume",
                        "permission_scope": "hr_private",
                        "page_start": 2,
                        "page_end": 3,
                    },
                )
            ]

    monkeypatch.setattr(api, "get_embedding_model", lambda: Embedder())
    monkeypatch.setattr(api, "get_evidence_store", lambda: Store())

    result = api.search_evidence(
        EvidenceSearchInput(query="推荐系统负责人"),
        x_tenant_id="course-demo",
        x_permission_scopes="hr_private",
        db=session,
    )

    assert result[0]["candidate_name"] == "林晓岚"
    assert result[0]["document_title"] == "林晓岚简历"
    assert result[0]["material_no"] is None
    assert result[0]["page_start"] == 2
    assert result[0]["page_end"] == 3
