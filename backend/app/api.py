import logging
from datetime import date
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Body, Depends, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field
from redis import Redis
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.chunk_service import (
    create_chunking_run,
    latest_chunks,
)
from app.config import get_settings
from app.database import get_db
from app.document_cleanup import delete_document_bundle
from app.document_pipeline import queue_index_job, queue_parse_job, summarize_pipeline_status
from app.evidence_index_service import get_evidence_store
from app.evidence_citations import load_pack_sources, resolve_citation
from app.evidence_pack import build_evidence_packs, model_extractor
from app.hybrid_search_service import hybrid_search_evidence as hybrid_search_evidence_service
from app.milvus_store import EvidenceFilter
from app.model_provider import get_embedding_model
from app.model_provider import get_chat_model
from app.query_plan import QueryPlan, compile_query_plan, merge_query_results, search_with_optimization, select_candidate_ids
from app.reranker import get_reranker
from app.models import (
    ChunkingRun,
    Document,
    DocumentVersion,
    EmployeeProfile,
    EvidenceIndexJob,
    FileObject,
    IndexStatus,
    KnowledgeBase,
    ParseArtifact,
    ParseJob,
)
from app.object_store import ObjectStore
from app.services import current_version, parse_version, upload_document

router = APIRouter(prefix="/api")
logger = logging.getLogger(__name__)


class EmployeeInput(BaseModel):
    employee_no: str
    name: str
    gender: str | None = None
    birth_date: date | None = None
    region: str | None = None
    current_position: str | None = None
    job_level: str | None = Field(default=None, pattern=r"^L\d+$")
    years_of_experience: float | None = None
    department: str | None = None


class KnowledgeBaseInput(BaseModel):
    name: str
    description: str | None = None
    permission_scope: str = "hr_private"


class ChunkingInput(BaseModel):
    chunk_size: int = Field(default=800, ge=100, le=8000)
    chunk_overlap: int = Field(default=100, ge=0, le=2000)


class EvidenceSearchInput(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    candidate_ids: list[str] | None = None
    document_types: list[str] | None = None
    limit: int = Field(default=10, ge=1, le=100)
    ef: int = Field(default=80, ge=10, le=1000)


class HybridSearchInput(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    candidate_ids: list[str] | None = None
    document_types: list[str] | None = None
    limit: int = Field(default=10, ge=1, le=100)
    ef: int = Field(default=80, ge=10, le=1000)
    rrf_k: int = Field(default=60, ge=1, le=1000)
    rerank_top_n: int = Field(default=20, ge=1, le=100)


class QueryPlanInput(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


class TalentSearchInput(BaseModel):
    include_evidence_pack: bool = False
    query: str = Field(min_length=1, max_length=2000)
    retrieval_mode: Literal["standard", "auto_optimize"] = "standard"
    limit: int = Field(default=10, ge=1, le=100)
    ef: int = Field(default=80, ge=10, le=1000)
    rrf_k: int = Field(default=60, ge=1, le=1000)
    rerank_top_n: int = Field(default=20, ge=1, le=100)


class DeleteDocumentsInput(BaseModel):
    document_ids: list[UUID] = Field(min_length=1, max_length=100)


def employee_json(item: EmployeeProfile, material_count: int = 0):
    today = date.today()
    age = None
    if item.birth_date:
        age = today.year - item.birth_date.year - ((today.month, today.day) < (item.birth_date.month, item.birth_date.day))
    return {"id": item.id, "employee_no": item.employee_no, "name": item.name, "gender": item.gender, "birth_date": item.birth_date, "age": age, "region": item.region, "current_position": item.current_position, "job_level": item.job_level, "years_of_experience": item.years_of_experience, "department": item.department, "employment_status": item.employment_status, "material_count": material_count}


def latest_parse_job(db: Session, version_id: UUID | None) -> ParseJob | None:
    if version_id is None:
        return None
    return db.scalar(
        select(ParseJob)
        .where(ParseJob.document_version_id == version_id)
        .order_by(ParseJob.created_at.desc())
        .limit(1)
    )


def latest_chunking_run(db: Session, version_id: UUID | None) -> ChunkingRun | None:
    if version_id is None:
        return None
    return db.scalar(
        select(ChunkingRun)
        .join(ParseJob, ParseJob.id == ChunkingRun.parse_job_id)
        .where(ParseJob.document_version_id == version_id)
        .order_by(ChunkingRun.created_at.desc())
        .limit(1)
    )


def latest_index_job(db: Session, version_id: UUID | None) -> EvidenceIndexJob | None:
    if version_id is None:
        return None
    return db.scalar(
        select(EvidenceIndexJob)
        .where(EvidenceIndexJob.document_version_id == version_id)
        .order_by(EvidenceIndexJob.created_at.desc())
        .limit(1)
    )


def build_document_pipeline_payload(db: Session, document_id: UUID) -> dict[str, str | None]:
    version = current_version(db, document_id)
    snapshot = summarize_pipeline_status(
        parse_job=latest_parse_job(db, version.id if version else None),
        chunk_run=latest_chunking_run(db, version.id if version else None),
        index_job=latest_index_job(db, version.id if version else None),
    )
    return {
        "pipeline_status": snapshot.pipeline_status,
        "parse_status": snapshot.parse_status,
        "chunk_status": snapshot.chunk_status,
        "index_status": snapshot.index_status,
    }


def enrich_evidence_context(db: Session, results):
    version_ids = [
        UUID(value)
        for item in results
        for value in [item.metadata.get("document_version_id")]
        if value
    ] if db else []
    versions = db.scalars(select(DocumentVersion).where(DocumentVersion.id.in_(version_ids))).all() if db and version_ids else []
    version_by_id = {str(item.id): item for item in versions}
    documents = db.scalars(select(Document).where(Document.id.in_([item.document_id for item in versions]))).all() if db and versions else []
    document_by_id = {item.id: item for item in documents}
    employees = db.scalars(select(EmployeeProfile).where(EmployeeProfile.id.in_([item.employee_id for item in documents if item.employee_id]))).all() if db and documents else []
    employee_by_id = {item.id: item for item in employees}
    return version_by_id, document_by_id, employee_by_id


def evidence_item_json(item, version_by_id, document_by_id, employee_by_id, *, score: float):
    version = version_by_id.get(str(item.metadata.get("document_version_id")))
    document = document_by_id.get(version.document_id) if version else None
    employee = employee_by_id.get(document.employee_id) if document and document.employee_id else None
    return {
        "chunk_id": item.chunk_id,
        "candidate_id": item.candidate_id,
        "candidate_name": employee.name if employee else item.candidate_id,
        "document_id": str(document.id) if document else None,
        "document_title": document.title if document else None,
        "material_no": document.material_no if document else None,
        "document_type": item.metadata.get("document_type"),
        "permission_scope": item.metadata.get("permission_scope"),
        "page_start": item.metadata.get("page_start"),
        "page_end": item.metadata.get("page_end"),
        "content": item.content,
        "score": score,
        "metadata": item.metadata,
    }


@router.get("/employees")
def list_employees(db: Session = Depends(get_db)):
    counts = dict(db.execute(select(Document.employee_id, func.count(Document.id)).where(Document.employee_id.is_not(None)).group_by(Document.employee_id)).all())
    return [employee_json(item, counts.get(item.id, 0)) for item in db.scalars(select(EmployeeProfile).order_by(EmployeeProfile.employee_no)).all()]


@router.post("/employees", status_code=201)
def create_employee(payload: EmployeeInput, db: Session = Depends(get_db)):
    if db.scalar(select(EmployeeProfile).where(EmployeeProfile.employee_no == payload.employee_no)):
        raise HTTPException(409, "员工工号已存在")
    item = EmployeeProfile(**payload.model_dump())
    db.add(item)
    db.commit()
    db.refresh(item)
    logger.info("employee_created employee_id=%s employee_no=%s name=%s", item.id, item.employee_no, item.name)
    return employee_json(item)


@router.get("/knowledge-bases")
def list_knowledge_bases(db: Session = Depends(get_db)):
    counts = dict(db.execute(select(Document.knowledge_base_id, func.count(Document.id)).where(Document.knowledge_base_id.is_not(None)).group_by(Document.knowledge_base_id)).all())
    items = db.scalars(select(KnowledgeBase).order_by(KnowledgeBase.created_at)).all()
    return [{"id": item.id, "name": item.name, "description": item.description, "permission_scope": item.permission_scope, "file_count": counts.get(item.id, 0)} for item in items]


@router.post("/knowledge-bases", status_code=201)
def create_knowledge_base(payload: KnowledgeBaseInput, db: Session = Depends(get_db)):
    item = KnowledgeBase(**payload.model_dump())
    db.add(item)
    db.commit()
    db.refresh(item)
    logger.info("knowledge_base_created knowledge_base_id=%s name=%s permission_scope=%s", item.id, item.name, item.permission_scope)
    return {"id": item.id, "name": item.name, "description": item.description, "permission_scope": item.permission_scope, "file_count": 0}


@router.delete("/knowledge-bases/{knowledge_base_id}")
def delete_knowledge_base(knowledge_base_id: UUID, db: Session = Depends(get_db)):
    item = db.get(KnowledgeBase, knowledge_base_id)
    if not item:
        raise HTTPException(404, "知识库不存在")
    file_count = db.scalar(select(func.count(Document.id)).where(Document.knowledge_base_id == knowledge_base_id)) or 0
    if file_count:
        raise HTTPException(409, f"知识库中还有 {file_count} 份材料，请先删除或迁移材料")
    db.delete(item)
    db.commit()
    logger.info("knowledge_base_deleted knowledge_base_id=%s name=%s", item.id, item.name)
    return {"id": item.id, "name": item.name, "deleted": True}


@router.get("/documents")
def list_documents(knowledge_base_id: UUID | None = None, db: Session = Depends(get_db)):
    query = select(Document).order_by(Document.created_at.desc())
    if knowledge_base_id:
        query = query.where(Document.knowledge_base_id == knowledge_base_id)
    items = db.scalars(query).all()
    result = []
    for item in items:
        pipeline = build_document_pipeline_payload(db, item.id)
        result.append(
            {
                "id": item.id,
                "material_no": item.material_no,
                "candidate_id": item.candidate_id,
                "employee_name": item.employee.name if item.employee else item.candidate_id,
                "knowledge_base_id": item.knowledge_base_id,
                "title": item.title,
                "document_type": item.document_type,
                "status": item.status,
                "pipeline_status": pipeline["pipeline_status"],
                "parse_status": pipeline["parse_status"],
                "chunk_status": pipeline["chunk_status"],
                "index_status": pipeline["index_status"],
                "created_at": item.created_at,
            }
        )
    return result


@router.post("/documents", status_code=201)
async def create_document(file: UploadFile = File(...), employee_id: UUID | None = Form(None), knowledge_base_id: UUID | None = Form(None), candidate_id: str = Form(""), title: str = Form(...), document_type: str = Form(...), permission_scope: str = Form("hr_private"), db: Session = Depends(get_db)):
    logger.info("document_upload_request filename=%s employee_id=%s knowledge_base_id=%s document_type=%s", file.filename, employee_id, knowledge_base_id, document_type)
    content = await file.read()
    if not content:
        raise HTTPException(400, "文件内容为空")
    if employee_id and not db.get(EmployeeProfile, employee_id):
        raise HTTPException(404, "员工不存在")
    if knowledge_base_id and not db.get(KnowledgeBase, knowledge_base_id):
        raise HTTPException(404, "知识库不存在")
    document = upload_document(db, ObjectStore(), filename=file.filename or "upload.bin", content=content, candidate_id=candidate_id, employee_id=employee_id, knowledge_base_id=knowledge_base_id, title=title, document_type=document_type, permission_scope=permission_scope)
    version = current_version(db, document.id)
    parse_job = queue_parse_job(
        db,
        Redis.from_url(get_settings().redis_url, decode_responses=True),
        version.id,
    )
    logger.info("document_upload_response document_id=%s material_no=%s", document.id, document.material_no)
    return {
        "id": document.id,
        "material_no": document.material_no,
        "candidate_id": document.candidate_id,
        "title": document.title,
        "pipeline_status": "parsing",
        "parse_job_id": parse_job.id,
    }


@router.get("/documents/{document_id}")
def get_document(document_id: UUID, db: Session = Depends(get_db)):
    document = db.get(Document, document_id)
    if not document:
        raise HTTPException(404, "文档不存在")
    version = current_version(db, document_id)
    file_object = db.get(FileObject, version.file_object_id) if version else None
    jobs = db.scalars(select(ParseJob).where(ParseJob.document_version_id == version.id).order_by(ParseJob.created_at.desc())).all() if version else []
    index_jobs = db.scalars(select(EvidenceIndexJob).where(EvidenceIndexJob.document_version_id == version.id).order_by(EvidenceIndexJob.created_at.desc())).all() if version else []
    latest = jobs[0] if jobs else None
    artifacts = db.scalars(select(ParseArtifact).where(ParseArtifact.parse_job_id == latest.id)).all() if latest else []
    store = ObjectStore()
    artifact_items = []
    for item in artifacts:
        content = None
        if item.artifact_type == "markdown":
            content = store.get_bytes(item.object_key).decode("utf-8", errors="replace")
        artifact_items.append({"id": item.id, "type": item.artifact_type, "content_type": item.content_type, "url": store.presigned_get(item.object_key), "content": content})
    pipeline = build_document_pipeline_payload(db, document.id)
    return {"id": document.id, "material_no": document.material_no, "candidate_id": document.candidate_id, "employee": {"id": document.employee.id, "employee_no": document.employee.employee_no, "name": document.employee.name} if document.employee else None, "knowledge_base": {"id": document.knowledge_base.id, "name": document.knowledge_base.name} if document.knowledge_base else None, "title": document.title, "document_type": document.document_type, "permission_scope": document.permission_scope, "created_at": document.created_at, "updated_at": document.updated_at, "version": {"id": version.id, "version_no": version.version_no, "created_at": version.created_at} if version else None, "file": {"name": file_object.original_name, "mime_type": file_object.mime_type, "size_bytes": file_object.size_bytes, "bucket_name": file_object.bucket_name, "object_key": file_object.object_key, "preview_url": store.presigned_get(file_object.object_key)} if file_object else None, "jobs": [{"id": job.id, "parser_name": job.parser_name, "parser_version": job.parser_version, "status": job.status, "progress": job.progress, "error_message": job.error_message, "created_at": job.created_at} for job in jobs], "index_jobs": [{"id": job.id, "status": job.status, "embedding_model": job.embedding_model, "collection_name": job.collection_name, "indexed_count": job.indexed_count, "retry_count": job.retry_count, "error_message": job.error_message, "created_at": job.created_at} for job in index_jobs], "pipeline_status": pipeline["pipeline_status"], "parse_status": pipeline["parse_status"], "chunk_status": pipeline["chunk_status"], "index_status": pipeline["index_status"], "artifacts": artifact_items}


@router.post("/documents/{document_id}/parse")
def parse_document_endpoint(document_id: UUID, async_mode: bool = True, db: Session = Depends(get_db)):
    logger.info("parse_request_received document_id=%s async_mode=%s", document_id, async_mode)
    version = current_version(db, document_id)
    if not version:
        raise HTTPException(404, "文档版本不存在")
    if async_mode:
        job = queue_parse_job(
            db,
            Redis.from_url(get_settings().redis_url, decode_responses=True),
            version.id,
        )
        logger.info("parse_job_queued document_id=%s version_id=%s job_id=%s queue=talent:parse:queue", document_id, version.id, job.id)
        return {"id": job.id, "status": job.status, "parser_name": job.parser_name, "error_message": None}
    job = parse_version(db, ObjectStore(), version.id)
    logger.info("parse_sync_completed document_id=%s version_id=%s job_id=%s status=%s", document_id, version.id, job.id, job.status)
    return {"id": job.id, "status": job.status, "parser_name": job.parser_name, "error_message": job.error_message}


@router.get("/documents/{document_id}/chunks")
def get_document_chunks(document_id: UUID, db: Session = Depends(get_db)):
    version = current_version(db, document_id)
    if not version:
        raise HTTPException(404, "文档版本不存在")
    run, chunks = latest_chunks(db, version.id)
    if not run:
        return {"run": None, "chunks": []}
    return {
        "run": {
            "id": run.id,
            "strategy": run.strategy.value,
            "status": run.status,
            "chunk_size": run.chunk_size,
            "chunk_overlap": run.chunk_overlap,
            "chunker_version": run.chunker_version,
            "created_at": run.created_at,
        },
        "chunks": [
            {
                "id": chunk.id,
                "stable_key": chunk.stable_key,
                "position": chunk.position,
                "level": chunk.chunk_level,
                "content": chunk.content,
                "element_ids": chunk.element_ids,
                "heading_path": chunk.heading_path,
                "parent_chunk_id": chunk.parent_chunk_id,
                "previous_chunk_id": chunk.previous_chunk_id,
                "next_chunk_id": chunk.next_chunk_id,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "timestamp_start": chunk.timestamp_start,
                "timestamp_end": chunk.timestamp_end,
                "markdown_start": chunk.markdown_start,
                "markdown_end": chunk.markdown_end,
                "source_locators": chunk.source_locators,
            }
            for chunk in chunks
        ],
    }


@router.post("/documents/{document_id}/chunks", status_code=201)
def create_document_chunks(document_id: UUID, payload: ChunkingInput, db: Session = Depends(get_db)):
    document = db.get(Document, document_id)
    if not document:
        raise HTTPException(404, "文档不存在")
    version = current_version(db, document_id)
    if not version:
        raise HTTPException(404, "文档版本不存在")
    if payload.chunk_overlap >= payload.chunk_size:
        raise HTTPException(422, "chunk_overlap 必须小于 chunk_size")
    try:
        run = create_chunking_run(
            db,
            ObjectStore(),
            document,
            version,
            chunk_size=payload.chunk_size,
            chunk_overlap=payload.chunk_overlap,
        )
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"id": run.id, "strategy": run.strategy.value, "status": run.status}


@router.post("/documents/{document_id}/evidence-index", status_code=202)
def create_evidence_index(document_id: UUID, db: Session = Depends(get_db)):
    version = current_version(db, document_id)
    if not version:
        raise HTTPException(404, "文档版本不存在")
    settings = get_settings()
    job = queue_index_job(
        db,
        Redis.from_url(settings.redis_url, decode_responses=True),
        version.id,
        settings=settings,
    )
    logger.info("evidence_index_job_queued job_id=%s version_id=%s", job.id, version.id)
    return {"id": job.id, "status": job.status, "document_version_id": version.id}


@router.delete("/documents")
def delete_documents(payload: DeleteDocumentsInput = Body(...), db: Session = Depends(get_db)):
    missing = [document_id for document_id in payload.document_ids if not db.get(Document, document_id)]
    if missing:
        raise HTTPException(404, f"文档不存在: {missing[0]}")
    store = ObjectStore()
    milvus_store = get_evidence_store()
    items = [delete_document_bundle(db, store, milvus_store, document_id) for document_id in payload.document_ids]
    return {"deleted_count": len(items), "items": items}


@router.post("/evidence/search")
def search_evidence(
    payload: EvidenceSearchInput,
    x_tenant_id: str = Header(...),
    x_permission_scopes: str = Header(...),
    db: Session | None = Depends(get_db),
):
    permission_scopes = [value.strip() for value in x_permission_scopes.split(",") if value.strip()]
    if not permission_scopes:
        raise HTTPException(403, "缺少可用的证据权限范围")
    embedder = get_embedding_model()
    if embedder is None:
        raise HTTPException(503, "Embedding 服务未配置")
    try:
        query_vector = embedder.embed_query(payload.query)
        results = get_evidence_store().search(
            list(query_vector),
            filters=EvidenceFilter(
                tenant_id=x_tenant_id,
                permission_scopes=permission_scopes,
                candidate_ids=payload.candidate_ids,
                document_types=payload.document_types,
            ),
            limit=payload.limit,
            ef=payload.ef,
        )
    except Exception as exc:
        logger.exception("evidence_search_failed tenant_id=%s", x_tenant_id)
        raise HTTPException(503, f"证据检索失败: {exc}") from exc
    version_ids = [
        UUID(value)
        for item in results
        for value in [item.metadata.get("document_version_id")]
        if value
    ] if db else []
    versions = db.scalars(select(DocumentVersion).where(DocumentVersion.id.in_(version_ids))).all() if db and version_ids else []
    version_by_id = {str(item.id): item for item in versions}
    documents = db.scalars(select(Document).where(Document.id.in_([item.document_id for item in versions]))).all() if db and versions else []
    document_by_id = {item.id: item for item in documents}
    employees = db.scalars(select(EmployeeProfile).where(EmployeeProfile.id.in_([item.employee_id for item in documents if item.employee_id]))).all() if db and documents else []
    employee_by_id = {item.id: item for item in employees}

    items = []
    for item in results:
        version = version_by_id.get(str(item.metadata.get("document_version_id")))
        document = document_by_id.get(version.document_id) if version else None
        employee = employee_by_id.get(document.employee_id) if document and document.employee_id else None
        items.append(
            {
                "chunk_id": item.chunk_id,
                "candidate_id": item.candidate_id,
                "candidate_name": employee.name if employee else item.candidate_id,
                "document_id": str(document.id) if document else None,
                "document_title": document.title if document else None,
                "material_no": document.material_no if document else None,
                "document_type": item.metadata.get("document_type"),
                "permission_scope": item.metadata.get("permission_scope"),
                "page_start": item.metadata.get("page_start"),
                "page_end": item.metadata.get("page_end"),
                "content": item.content,
                "score": item.score,
                "metadata": item.metadata,
            }
        )
    return items


@router.post("/evidence/hybrid-search")
def hybrid_search_evidence(
    payload: HybridSearchInput,
    x_tenant_id: str = Header(...),
    x_permission_scopes: str = Header(...),
    db: Session | None = Depends(get_db),
):
    permission_scopes = [value.strip() for value in x_permission_scopes.split(",") if value.strip()]
    if not permission_scopes:
        raise HTTPException(403, "缺少可用的证据权限范围")
    embedder = get_embedding_model()
    if embedder is None:
        raise HTTPException(503, "Embedding 服务未配置")
    reranker = get_reranker()
    if reranker is None:
        raise HTTPException(503, "Rerank 服务未配置")
    try:
        results = hybrid_search_evidence_service(
            query=payload.query,
            filters=EvidenceFilter(
                tenant_id=x_tenant_id,
                permission_scopes=permission_scopes,
                candidate_ids=payload.candidate_ids,
                document_types=payload.document_types,
            ),
            store=get_evidence_store(),
            embedder=embedder,
            reranker=reranker,
            limit=payload.limit,
            ef=payload.ef,
            rrf_k=payload.rrf_k,
            rerank_top_n=payload.rerank_top_n,
        )
    except Exception as exc:
        logger.exception("hybrid_search_failed tenant_id=%s", x_tenant_id)
        raise HTTPException(503, f"混合检索失败: {exc}") from exc
    version_by_id, document_by_id, employee_by_id = enrich_evidence_context(db, results)
    return [
        evidence_item_json(item, version_by_id, document_by_id, employee_by_id, score=item.rerank_score)
        for item in results
    ]


@router.post("/talent-search/plan")
def create_talent_query_plan(payload: QueryPlanInput):
    model = get_chat_model(temperature=0)
    if model is None:
        raise HTTPException(503, "查询计划模型未配置")
    try:
        return compile_query_plan(payload.query, model)
    except Exception as exc:
        logger.exception("query_plan_compile_failed")
        raise HTTPException(503, f"查询计划生成失败: {exc}") from exc


@router.post("/talent-search/candidates")
def query_talent_candidates(
    plan: QueryPlan,
    x_tenant_id: str = Header(...),
    db: Session = Depends(get_db),
):
    if not plan.executable:
        raise HTTPException(422, {"code": "clarification_required", "items": [item.model_dump() for item in plan.clarifications]})
    try:
        candidate_ids = select_candidate_ids(db, plan, tenant_id=x_tenant_id)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"candidate_ids": candidate_ids, "candidate_count": len(candidate_ids), "semantic_requirements": plan.semantic_requirements}


@router.post("/talent-search")
def search_talent(
    payload: TalentSearchInput,
    x_tenant_id: str = Header(...),
    x_permission_scopes: str = Header(...),
    db: Session = Depends(get_db),
):
    permission_scopes = [value.strip() for value in x_permission_scopes.split(",") if value.strip()]
    if not permission_scopes:
        raise HTTPException(403, "缺少可用的证据权限范围")

    model = get_chat_model(temperature=0)
    embedder = get_embedding_model()
    reranker = get_reranker()
    if model is None:
        raise HTTPException(503, "查询计划模型未配置")
    if embedder is None:
        raise HTTPException(503, "Embedding 服务未配置")
    if reranker is None:
        raise HTTPException(503, "Rerank 服务未配置")

    try:
        plan = compile_query_plan(payload.query, model)
        if not plan.executable:
            raise HTTPException(
                422,
                {
                    "code": "clarification_required",
                    "items": [item.model_dump() for item in plan.clarifications],
                },
            )
        candidate_ids = select_candidate_ids(db, plan, tenant_id=x_tenant_id)
        if not candidate_ids:
            return {"query_plan": plan, "candidate_ids": [], "searches": [], "chunks": [],
                    **({"evidence_packs": [], "evidence_status": "no_candidates"} if payload.include_evidence_pack else {})}

        store = get_evidence_store()

        def run_hybrid_search(query: str):
            return hybrid_search_evidence_service(
                query=query,
                filters=EvidenceFilter(
                    tenant_id=x_tenant_id,
                    permission_scopes=permission_scopes,
                    candidate_ids=candidate_ids,
                ),
                store=store,
                embedder=embedder,
                reranker=reranker,
                limit=payload.limit,
                ef=payload.ef,
                rrf_k=payload.rrf_k,
                rerank_top_n=payload.rerank_top_n,
            )

        searches = []
        result_sets = []
        requirement_ids_by_chunk: dict[str, list[str]] = {}
        for requirement in plan.semantic_requirements:
            if payload.retrieval_mode == "auto_optimize":
                outcome = search_with_optimization(requirement, search_query=run_hybrid_search, model=model)
            else:
                results = run_hybrid_search(requirement.query)
                outcome = {"strategy": None, "queries": [requirement.query], "results": results}
            result_sets.append(outcome["results"])
            for item in outcome["results"]:
                requirement_ids_by_chunk.setdefault(item.chunk_id, []).append(requirement.requirement_id)
            searches.append(
                {
                    "requirement_id": requirement.requirement_id,
                    "strategy": outcome["strategy"],
                    "queries": outcome["queries"],
                    "chunk_count": len(outcome["results"]),
                }
            )

        chunks = [
            {
                "chunk_id": item.chunk_id,
                "candidate_id": item.candidate_id,
                "content": item.content,
                "score": item.rerank_score,
                "metadata": item.metadata,
                "requirement_ids": requirement_ids_by_chunk[item.chunk_id],
            }
            for item in merge_query_results(result_sets)
        ]
        response = {
            "query_plan": plan,
            "candidate_ids": candidate_ids,
            "searches": searches,
            "chunks": chunks,
        }
        if payload.include_evidence_pack:
            sources = load_pack_sources(db, chunks, tenant_id=x_tenant_id,
                permission_scopes=permission_scopes)
            # Do not return stale/unauthorized index text alongside the validated pack.
            source_by_id = {row["chunk_id"]: row for row in sources}
            response["chunks"] = [dict(hit, content=source_by_id[hit["chunk_id"]]["content"])
                for hit in chunks if hit["chunk_id"] in source_by_id]
            response["evidence_packs"] = build_evidence_packs(
                candidate_ids=candidate_ids,
                requirements=[r.model_dump() for r in plan.semantic_requirements],
                sources=sources, extract=model_extractor(model))
            response["evidence_status"] = "reviewed" if plan.semantic_requirements else "not_requested"
        return response
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("talent_search_failed tenant_id=%s", x_tenant_id)
        raise HTTPException(503, f"人才检索失败: {exc}") from exc


@router.post("/jobs/{job_id}/retry")
def retry_job(job_id: UUID, db: Session = Depends(get_db)):
    old = db.get(ParseJob, job_id)
    if not old:
        raise HTTPException(404, "解析任务不存在")
    job = ParseJob(document_version_id=old.document_version_id, parser_name="queued", retry_count=old.retry_count + 1)
    db.add(job)
    db.commit()
    Redis.from_url(get_settings().redis_url, decode_responses=True).rpush("talent:parse:queue", f"{job.id}:{job.document_version_id}")
    logger.info("parse_job_retried previous_job_id=%s new_job_id=%s version_id=%s retry_count=%s", old.id, job.id, job.document_version_id, job.retry_count)
    return {"id": job.id, "status": job.status, "retry_count": job.retry_count}


@router.post("/index-jobs/{job_id}/retry", status_code=202)
def retry_index_job(job_id: UUID, db: Session = Depends(get_db)):
    old = db.get(EvidenceIndexJob, job_id)
    if not old:
        raise HTTPException(404, "索引任务不存在")
    if old.status != IndexStatus.FAILED:
        raise HTTPException(409, "只能重试失败的索引任务")
    job = EvidenceIndexJob(
        document_version_id=old.document_version_id,
        status=IndexStatus.PENDING,
        embedding_model=old.embedding_model,
        collection_name=old.collection_name,
        retry_count=old.retry_count + 1,
    )
    db.add(job)
    db.commit()
    Redis.from_url(get_settings().redis_url, decode_responses=True).rpush(
        "talent:index:queue", f"{job.id}:{job.document_version_id}"
    )
    logger.info(
        "evidence_index_job_retried previous_job_id=%s new_job_id=%s retry_count=%s",
        old.id,
        job.id,
        job.retry_count,
    )
    return {"id": job.id, "status": job.status, "retry_count": job.retry_count}


@router.get("/jobs/{job_id}/artifacts")
def list_artifacts(job_id: UUID, db: Session = Depends(get_db)):
    items = db.scalars(select(ParseArtifact).where(ParseArtifact.parse_job_id == job_id)).all()
    store = ObjectStore()
    return [{"id": item.id, "type": item.artifact_type, "content_type": item.content_type, "url": store.presigned_get(item.object_key)} for item in items]


@router.get("/evidence/citations/{chunk_id}")
def get_evidence_citation(chunk_id: UUID, x_tenant_id: str = Header(...),
    x_permission_scopes: str = Header(...), quote_start: int | None = None,
    quote_end: int | None = None, db: Session = Depends(get_db)):
    return resolve_citation(db, chunk_id, tenant_id=x_tenant_id,
        permission_scopes=[s.strip() for s in x_permission_scopes.split(",") if s.strip()],
        quote_start=quote_start, quote_end=quote_end)
