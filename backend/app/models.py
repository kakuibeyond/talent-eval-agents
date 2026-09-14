from datetime import date, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from sqlalchemy import BigInteger, Boolean, Date, DateTime, Enum, Float, ForeignKey, Integer, JSON, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class DocumentStatus(StrEnum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    DELETED = "deleted"


class ParseStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIALLY_SUCCEEDED = "partially_succeeded"
    FAILED = "failed"


class ChunkingStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class IndexStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ChunkStrategyName(StrEnum):
    RECURSIVE = "recursive"
    MARKDOWN = "markdown"


class EmployeeProfile(Base):
    __tablename__ = "employee_profiles"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    employee_no: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True, default="course-demo")
    name: Mapped[str] = mapped_column(String(128), index=True)
    gender: Mapped[str | None] = mapped_column(String(16))
    birth_date: Mapped[date | None] = mapped_column(Date)
    region: Mapped[str | None] = mapped_column(String(128))
    current_position: Mapped[str | None] = mapped_column(String(255))
    job_level: Mapped[str | None] = mapped_column(String(64))
    years_of_experience: Mapped[float | None] = mapped_column(Float)
    department: Mapped[str | None] = mapped_column(String(255))
    employment_status: Mapped[str] = mapped_column(String(32), default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class JobDescription(Base):
    __tablename__ = "job_descriptions"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    job_code: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str] = mapped_column(String(255), index=True)
    content: Mapped[str] = mapped_column(Text)
    version: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(32), default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class ToolCallAudit(Base):
    __tablename__ = "tool_call_audits"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    call_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    tool_name: Mapped[str] = mapped_column(String(128), index=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    actor_id: Mapped[str] = mapped_column(String(128))
    run_id: Mapped[str] = mapped_column(String(128), index=True)
    argument_keys: Mapped[list[str]] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(32))
    attempts: Mapped[int] = mapped_column(Integer)
    duration_ms: Mapped[float] = mapped_column(Float)
    error_code: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True, default="course-demo")
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text)
    permission_scope: Mapped[str] = mapped_column(String(64), default="hr_private")
    status: Mapped[str] = mapped_column(String(32), default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    candidate_id: Mapped[str] = mapped_column(String(64), index=True)
    material_no: Mapped[str | None] = mapped_column(String(64), unique=True)
    employee_id: Mapped[UUID | None] = mapped_column(ForeignKey("employee_profiles.id"), index=True)
    knowledge_base_id: Mapped[UUID | None] = mapped_column(ForeignKey("knowledge_bases.id"), index=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    title: Mapped[str] = mapped_column(String(255))
    document_type: Mapped[str] = mapped_column(String(64))
    source: Mapped[str] = mapped_column(String(64), default="manual_upload")
    permission_scope: Mapped[str] = mapped_column(String(64), default="hr_private")
    confidentiality_level: Mapped[str] = mapped_column(String(32), default="internal")
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(DocumentStatus, name="document_status", values_callable=lambda values: [item.value for item in values]),
        default=DocumentStatus.ACTIVE,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    versions: Mapped[list["DocumentVersion"]] = relationship(back_populates="document")
    employee: Mapped[EmployeeProfile | None] = relationship()
    knowledge_base: Mapped[KnowledgeBase | None] = relationship()


class FileObject(Base):
    __tablename__ = "file_objects"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    bucket_name: Mapped[str] = mapped_column(String(128))
    object_key: Mapped[str] = mapped_column(String(1024), unique=True)
    original_name: Mapped[str] = mapped_column(String(512))
    mime_type: Mapped[str] = mapped_column(String(255))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class DocumentVersion(Base):
    __tablename__ = "document_versions"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(ForeignKey("documents.id"))
    file_object_id: Mapped[UUID] = mapped_column(ForeignKey("file_objects.id"))
    version_no: Mapped[int] = mapped_column(Integer)
    is_current: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    document: Mapped[Document] = relationship(back_populates="versions")
    file_object: Mapped[FileObject] = relationship()


class ParseJob(Base):
    __tablename__ = "parse_jobs"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    document_version_id: Mapped[UUID] = mapped_column(ForeignKey("document_versions.id"))
    parser_name: Mapped[str] = mapped_column(String(64))
    parser_version: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[ParseStatus] = mapped_column(
        Enum(ParseStatus, name="parse_status", values_callable=lambda values: [item.value for item in values]),
        default=ParseStatus.PENDING,
    )
    progress: Mapped[int] = mapped_column(Integer, default=0)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ParseArtifact(Base):
    __tablename__ = "parse_artifacts"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    parse_job_id: Mapped[UUID] = mapped_column(ForeignKey("parse_jobs.id"))
    artifact_type: Mapped[str] = mapped_column(String(64))
    bucket_name: Mapped[str] = mapped_column(String(128))
    object_key: Mapped[str] = mapped_column(String(1024))
    content_type: Mapped[str] = mapped_column(String(255))
    size_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class DocumentMetadata(Base):
    __tablename__ = "document_metadata"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(ForeignKey("documents.id"))
    metadata_key: Mapped[str] = mapped_column(String(128))
    metadata_value: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class ChunkingRun(Base):
    __tablename__ = "chunking_runs"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    parse_job_id: Mapped[UUID] = mapped_column(ForeignKey("parse_jobs.id"), index=True)
    strategy: Mapped[ChunkStrategyName] = mapped_column(
        Enum(ChunkStrategyName, name="chunk_strategy", values_callable=lambda values: [item.value for item in values])
    )
    status: Mapped[ChunkingStatus] = mapped_column(
        Enum(ChunkingStatus, name="chunking_status", values_callable=lambda values: [item.value for item in values]),
        default=ChunkingStatus.PENDING,
    )
    chunk_size: Mapped[int] = mapped_column(Integer)
    chunk_overlap: Mapped[int] = mapped_column(Integer, default=0)
    chunker_version: Mapped[str] = mapped_column(String(64), default="lesson-5-v1")
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    chunking_run_id: Mapped[UUID] = mapped_column(ForeignKey("chunking_runs.id", ondelete="CASCADE"), index=True)
    document_version_id: Mapped[UUID] = mapped_column(ForeignKey("document_versions.id"), index=True)
    candidate_id: Mapped[str] = mapped_column(String(64), index=True)
    document_type: Mapped[str] = mapped_column(String(64))
    permission_scope: Mapped[str] = mapped_column(String(64))
    stable_key: Mapped[str] = mapped_column(String(64), index=True)
    position: Mapped[int] = mapped_column(Integer)
    chunk_level: Mapped[str] = mapped_column(String(16), default="child")
    content: Mapped[str] = mapped_column(Text)
    element_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    heading_path: Mapped[list[str]] = mapped_column(JSON, default=list)
    parent_chunk_id: Mapped[UUID | None] = mapped_column(ForeignKey("document_chunks.id"))
    previous_chunk_id: Mapped[UUID | None] = mapped_column(ForeignKey("document_chunks.id"))
    next_chunk_id: Mapped[UUID | None] = mapped_column(ForeignKey("document_chunks.id"))
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    timestamp_start: Mapped[float | None] = mapped_column(Float)
    timestamp_end: Mapped[float | None] = mapped_column(Float)
    markdown_start: Mapped[int | None] = mapped_column(Integer)
    markdown_end: Mapped[int | None] = mapped_column(Integer)
    source_locators: Mapped[list[dict]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EvidenceIndexJob(Base):
    __tablename__ = "evidence_index_jobs"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    document_version_id: Mapped[UUID] = mapped_column(ForeignKey("document_versions.id"), index=True)
    status: Mapped[IndexStatus] = mapped_column(
        Enum(IndexStatus, name="index_status", values_callable=lambda values: [item.value for item in values]),
        default=IndexStatus.PENDING,
    )
    embedding_model: Mapped[str] = mapped_column(String(128))
    collection_name: Mapped[str] = mapped_column(String(128))
    indexed_count: Mapped[int] = mapped_column(Integer, default=0)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
