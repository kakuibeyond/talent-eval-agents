CREATE TABLE IF NOT EXISTS job_descriptions (
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

CREATE INDEX IF NOT EXISTS idx_job_descriptions_tenant_name
ON job_descriptions (tenant_id, name);

COMMENT ON TABLE job_descriptions IS '企业维护的岗位 JD，只读人才工具按租户检索';

CREATE TABLE IF NOT EXISTS tool_call_audits (
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

CREATE INDEX IF NOT EXISTS idx_tool_call_audits_tenant_created
ON tool_call_audits (tenant_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_tool_call_audits_run
ON tool_call_audits (run_id);

COMMENT ON TABLE tool_call_audits IS '人才数据工具调用审计，只保存参数键与运行摘要';
