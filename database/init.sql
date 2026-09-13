CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TYPE document_status AS ENUM ('active', 'inactive', 'deleted');
CREATE TYPE parse_status AS ENUM ('pending', 'running', 'succeeded', 'partially_succeeded', 'failed');
CREATE TYPE chunking_status AS ENUM ('pending', 'running', 'succeeded', 'failed');
CREATE TYPE chunk_strategy AS ENUM ('recursive', 'markdown');
CREATE TYPE index_status AS ENUM ('pending', 'running', 'succeeded', 'failed');

CREATE TABLE employee_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    employee_no VARCHAR(64) NOT NULL UNIQUE,
    tenant_id VARCHAR(64) NOT NULL DEFAULT 'course-demo',
    name VARCHAR(128) NOT NULL,
    gender VARCHAR(16),
    birth_date DATE,
    region VARCHAR(128),
    current_position VARCHAR(255),
    job_level VARCHAR(64) CHECK (job_level IS NULL OR job_level ~ '^L[0-9]+$'),
    years_of_experience DOUBLE PRECISION CHECK (years_of_experience >= 0),
    department VARCHAR(255),
    employment_status VARCHAR(32) NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE job_descriptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_code VARCHAR(64) NOT NULL UNIQUE,
    tenant_id VARCHAR(64) NOT NULL,
    name VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1 CHECK (version > 0),
    status VARCHAR(32) NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_job_descriptions_tenant_name ON job_descriptions (tenant_id, name);

CREATE TABLE tool_call_audits (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    call_id VARCHAR(64) NOT NULL UNIQUE,
    tool_name VARCHAR(128) NOT NULL,
    tenant_id VARCHAR(64) NOT NULL,
    actor_id VARCHAR(128) NOT NULL,
    run_id VARCHAR(128) NOT NULL,
    argument_keys JSONB NOT NULL DEFAULT '[]'::jsonb,
    status VARCHAR(32) NOT NULL,
    attempts INTEGER NOT NULL CHECK (attempts >= 0),
    duration_ms DOUBLE PRECISION NOT NULL CHECK (duration_ms >= 0),
    error_code VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_tool_call_audits_tenant_created ON tool_call_audits (tenant_id, created_at DESC);
CREATE INDEX idx_tool_call_audits_run ON tool_call_audits (run_id);

CREATE TABLE knowledge_bases (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(64) NOT NULL DEFAULT 'course-demo',
    name VARCHAR(255) NOT NULL,
    description TEXT,
    permission_scope VARCHAR(64) NOT NULL DEFAULT 'hr_private',
    status VARCHAR(32) NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (tenant_id, name)
);

CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    candidate_id VARCHAR(64) NOT NULL,
    material_no VARCHAR(64) UNIQUE,
    employee_id UUID REFERENCES employee_profiles(id),
    knowledge_base_id UUID REFERENCES knowledge_bases(id),
    tenant_id VARCHAR(64) NOT NULL,
    title VARCHAR(255) NOT NULL,
    document_type VARCHAR(64) NOT NULL,
    source VARCHAR(64) NOT NULL DEFAULT 'manual_upload',
    permission_scope VARCHAR(64) NOT NULL DEFAULT 'hr_private',
    confidentiality_level VARCHAR(32) NOT NULL DEFAULT 'internal',
    status document_status NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_documents_employee ON documents (employee_id);
CREATE INDEX idx_documents_knowledge_base ON documents (knowledge_base_id);

CREATE TABLE file_objects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    bucket_name VARCHAR(128) NOT NULL,
    object_key VARCHAR(1024) NOT NULL UNIQUE,
    original_name VARCHAR(512) NOT NULL,
    mime_type VARCHAR(255) NOT NULL,
    size_bytes BIGINT NOT NULL CHECK (size_bytes >= 0),
    sha256 CHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_file_objects_sha256 ON file_objects (sha256);

CREATE TABLE document_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id),
    file_object_id UUID NOT NULL REFERENCES file_objects(id),
    version_no INTEGER NOT NULL CHECK (version_no > 0),
    is_current BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (document_id, version_no)
);

CREATE UNIQUE INDEX uq_document_current_version
ON document_versions (document_id) WHERE is_current;

CREATE TABLE parse_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_version_id UUID NOT NULL REFERENCES document_versions(id),
    parser_name VARCHAR(64) NOT NULL,
    parser_version VARCHAR(64),
    status parse_status NOT NULL DEFAULT 'pending',
    progress SMALLINT NOT NULL DEFAULT 0 CHECK (progress BETWEEN 0 AND 100),
    retry_count SMALLINT NOT NULL DEFAULT 0 CHECK (retry_count >= 0),
    error_code VARCHAR(64),
    error_message TEXT,
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_parse_jobs_status_created ON parse_jobs (status, created_at);

CREATE TABLE parse_artifacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    parse_job_id UUID NOT NULL REFERENCES parse_jobs(id) ON DELETE CASCADE,
    artifact_type VARCHAR(64) NOT NULL,
    bucket_name VARCHAR(128) NOT NULL,
    object_key VARCHAR(1024) NOT NULL,
    content_type VARCHAR(255) NOT NULL,
    size_bytes BIGINT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (parse_job_id, artifact_type, object_key)
);

CREATE TABLE document_metadata (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    metadata_key VARCHAR(128) NOT NULL,
    metadata_value JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (document_id, metadata_key)
);

CREATE TABLE chunking_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    parse_job_id UUID NOT NULL REFERENCES parse_jobs(id),
    strategy chunk_strategy NOT NULL,
    status chunking_status NOT NULL DEFAULT 'pending',
    chunk_size INTEGER NOT NULL CHECK (chunk_size > 0),
    chunk_overlap INTEGER NOT NULL DEFAULT 0 CHECK (chunk_overlap >= 0),
    chunker_version VARCHAR(64) NOT NULL DEFAULT 'lesson-5-v1',
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ
);

CREATE INDEX idx_chunking_runs_parse_job ON chunking_runs (parse_job_id, created_at DESC);

CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunking_run_id UUID NOT NULL REFERENCES chunking_runs(id) ON DELETE CASCADE,
    document_version_id UUID NOT NULL REFERENCES document_versions(id),
    candidate_id VARCHAR(64) NOT NULL,
    document_type VARCHAR(64) NOT NULL,
    permission_scope VARCHAR(64) NOT NULL,
    stable_key VARCHAR(64) NOT NULL,
    position INTEGER NOT NULL CHECK (position >= 0),
    chunk_level VARCHAR(16) NOT NULL DEFAULT 'child' CHECK (chunk_level IN ('parent', 'child')),
    content TEXT NOT NULL,
    element_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    heading_path JSONB NOT NULL DEFAULT '[]'::jsonb,
    parent_chunk_id UUID REFERENCES document_chunks(id),
    previous_chunk_id UUID REFERENCES document_chunks(id),
    next_chunk_id UUID REFERENCES document_chunks(id),
    page_start INTEGER,
    page_end INTEGER,
    timestamp_start DOUBLE PRECISION,
    timestamp_end DOUBLE PRECISION,
    markdown_start INTEGER,
    markdown_end INTEGER,
    source_locators JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (chunking_run_id, stable_key)
);

CREATE INDEX idx_document_chunks_version ON document_chunks (document_version_id, position);
CREATE INDEX idx_document_chunks_candidate ON document_chunks (candidate_id, document_type);

CREATE TABLE evidence_index_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_version_id UUID NOT NULL REFERENCES document_versions(id),
    status index_status NOT NULL DEFAULT 'pending',
    embedding_model VARCHAR(128) NOT NULL,
    collection_name VARCHAR(128) NOT NULL,
    indexed_count INTEGER NOT NULL DEFAULT 0 CHECK (indexed_count >= 0),
    retry_count INTEGER NOT NULL DEFAULT 0 CHECK (retry_count >= 0),
    error_message TEXT,
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_evidence_index_jobs_version_created
ON evidence_index_jobs (document_version_id, created_at DESC);
COMMENT ON TABLE documents IS '业务层逻辑文档，同一文档可以有多个文件版本';
COMMENT ON TABLE employee_profiles IS '员工花名册结构化基础信息';
COMMENT ON TABLE job_descriptions IS '企业维护的岗位 JD，只读人才工具按租户检索';
COMMENT ON TABLE tool_call_audits IS '人才数据工具调用审计，只保存参数键与运行摘要';
COMMENT ON TABLE knowledge_bases IS '档案资料库中的知识库';
COMMENT ON TABLE file_objects IS '对象存储中的物理文件索引';
COMMENT ON TABLE document_versions IS '逻辑文档与物理文件之间的版本关系';
COMMENT ON TABLE parse_jobs IS '文件解析任务及状态变化';
COMMENT ON TABLE parse_artifacts IS 'Markdown、JSON、图片、转写等解析产物索引';
COMMENT ON TABLE chunking_runs IS '一次可复现的文档切片运行及其自动判定策略';
COMMENT ON TABLE document_chunks IS '带来源定位、父子关系、相邻关系和权限信息的人才证据单元';
COMMENT ON TABLE evidence_index_jobs IS 'PostgreSQL 与 Milvus 之间的证据索引任务及一致性状态';
