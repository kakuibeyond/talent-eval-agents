from __future__ import annotations

from datetime import date
import time

import pytest
from langchain.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.models import Base, EmployeeProfile, JobDescription, ToolCallAudit
from app.query_plan import FilterCondition
from app.talent_decision_graph import DecisionContext, build_talent_decision_graph
from app.talent_tools import (
    CircuitBreaker,
    InMemoryAuditSink,
    SqlAuditSink,
    TalentToolContext,
    TalentToolService,
    ToolExecutor,
    ToolPolicy,
    TransientToolError,
    build_talent_tools,
    build_candidate_provider,
)


@pytest.fixture
def session_factory(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'tools.db'}")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        db.add_all(
            [
                JobDescription(
                    job_code="JD-AI-001",
                    tenant_id="tenant-a",
                    name="高级 AI 应用工程师",
                    content="负责企业知识库与 Agent 应用建设",
                    version=2,
                ),
                JobDescription(
                    job_code="JD-JAVA-001",
                    tenant_id="tenant-a",
                    name="Java 后端工程师",
                    content="负责交易系统后端开发",
                    version=1,
                ),
                JobDescription(
                    job_code="JD-AI-OTHER",
                    tenant_id="tenant-b",
                    name="高级 AI 应用工程师",
                    content="其他租户数据",
                    version=1,
                ),
                EmployeeProfile(
                    employee_no="C001",
                    tenant_id="tenant-a",
                    name="林晓岚",
                    birth_date=date(1993, 5, 1),
                    region="上海",
                    current_position="AI 应用工程师",
                    job_level="L3",
                    years_of_experience=7,
                    department="技术中心",
                ),
                EmployeeProfile(
                    employee_no="C002",
                    tenant_id="tenant-b",
                    name="越权数据",
                    region="上海",
                    job_level="L4",
                    years_of_experience=9,
                ),
            ]
        )
        db.commit()

    def factory():
        return Session(engine)

    return factory


@pytest.fixture
def context():
    return TalentToolContext(
        tenant_id="tenant-a",
        permission_scopes=("hr_private",),
        actor_id="teacher-demo",
        run_id="run-12",
    )


def test_job_lookup_returns_tenant_scoped_ranked_matches(session_factory, context):
    service = TalentToolService(session_factory=session_factory)

    result = service.lookup_job_descriptions("AI 应用工程师", context=context)

    assert result[0]["job_code"] == "JD-AI-001"
    assert result[0]["version"] == 2
    assert result[0]["match_score"] == 1.0
    assert all(item["content"] != "其他租户数据" for item in result)


def test_job_resource_lookup_uses_job_code_and_tenant_boundary(session_factory, context):
    service = TalentToolService(session_factory=session_factory)

    visible = service.get_job_description("JD-AI-001", context=context)
    hidden = service.get_job_description("JD-AI-OTHER", context=context)

    assert visible == {
        "job_code": "JD-AI-001",
        "name": "高级 AI 应用工程师",
        "version": 2,
        "content": "负责企业知识库与 Agent 应用建设",
    }
    assert hidden is None


def test_candidate_filter_reuses_query_plan_and_tenant_boundary(session_factory, context):
    service = TalentToolService(session_factory=session_factory)

    result = service.filter_candidates(
        [FilterCondition(field="region", operator="eq", value="上海")],
        context=context,
    )

    assert result == ["C001"]


def test_profile_lookup_returns_only_requested_authorized_candidate(session_factory, context):
    service = TalentToolService(session_factory=session_factory)

    result = service.get_candidate_profiles(["C001", "C002"], context=context)

    assert [item["candidate_id"] for item in result] == ["C001"]
    assert result[0]["name"] == "林晓岚"


def test_evidence_search_preserves_evidence_pack_v2(session_factory, context):
    def provider(**kwargs):
        assert kwargs["tenant_id"] == "tenant-a"
        assert kwargs["permission_scopes"] == ["hr_private"]
        return {
            "candidate_ids": ["C001"],
            "evidence_packs": [{"schema_version": "2.0", "candidate_id": "C001", "requirements": []}],
        }

    service = TalentToolService(session_factory=session_factory, evidence_provider=provider)

    result = service.search_candidate_evidence("企业知识库经验", ["C001"], context=context)

    assert result["evidence_packs"][0]["schema_version"] == "2.0"


def test_tool_schema_hides_runtime_context(session_factory):
    service = TalentToolService(session_factory=session_factory)
    tools = build_talent_tools(service, ToolExecutor())

    schemas = {item.name: item.tool_call_schema.model_json_schema() for item in tools}

    assert "context" not in str(schemas)
    assert "runtime" not in schemas["lookup_job_descriptions"]["properties"]
    assert set(schemas) == {
        "lookup_job_descriptions",
        "filter_candidates",
        "search_candidate_evidence",
        "get_candidate_profiles",
    }


def test_tool_node_injects_runtime_context(session_factory, context):
    service = TalentToolService(session_factory=session_factory)
    tools = build_talent_tools(service, ToolExecutor())
    builder = StateGraph(MessagesState, context_schema=TalentToolContext)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    graph = builder.compile()
    call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "filter_candidates",
                "args": {"filters": [{"field": "region", "operator": "eq", "value": "上海"}]},
                "id": "call-12",
                "type": "tool_call",
            }
        ],
    )

    result = graph.invoke({"messages": [call]}, context=context)

    assert '"ok": true' in result["messages"][-1].content
    assert "C001" in result["messages"][-1].content


def test_runtime_context_blocks_model_selected_tenant(session_factory, context):
    def insecure_lookup(candidate_id: str, tenant_id: str):
        with session_factory() as db:
            return db.scalar(
                select(EmployeeProfile.name).where(
                    EmployeeProfile.employee_no == candidate_id,
                    EmployeeProfile.tenant_id == tenant_id,
                )
            )

    assert insecure_lookup("C002", tenant_id="tenant-b") == "越权数据"

    service = TalentToolService(session_factory=session_factory)
    secure_result = service.get_candidate_profiles(["C002"], context=context)

    assert secure_result == []


def test_executor_retries_transient_error_and_records_audit(context):
    attempts = 0
    audit = InMemoryAuditSink()
    executor = ToolExecutor(
        policy=ToolPolicy(max_attempts=3, timeout_seconds=0.2, backoff_seconds=0),
        audit_sink=audit,
    )

    def operation():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise TransientToolError("dependency_unavailable")
        return {"candidate_ids": ["C001"]}

    result = executor.execute("filter_candidates", operation, context=context, arguments={"filters": 1})

    assert result["ok"] is True
    assert result["meta"]["attempts"] == 3
    assert audit.records[-1]["status"] == "succeeded"
    assert "filters" in audit.records[-1]["argument_keys"]


def test_retry_policy_changes_transient_failure_result(context):
    attempts = {"without": 0, "with": 0}

    def unstable(key):
        attempts[key] += 1
        if attempts[key] == 1:
            raise TransientToolError("temporary_failure")
        return "recovered"

    without_retry = ToolExecutor(
        policy=ToolPolicy(max_attempts=1, backoff_seconds=0)
    ).execute("evidence", lambda: unstable("without"), context=context, arguments={})
    with_retry = ToolExecutor(
        policy=ToolPolicy(max_attempts=2, backoff_seconds=0)
    ).execute("evidence", lambda: unstable("with"), context=context, arguments={})

    assert without_retry["ok"] is False
    assert attempts["without"] == 1
    assert with_retry["data"] == "recovered"
    assert attempts["with"] == 2


def test_each_tool_can_override_default_policy(context):
    executor = ToolExecutor(
        policy=ToolPolicy(max_attempts=1, backoff_seconds=0),
        tool_policies={
            "search_candidate_evidence": ToolPolicy(max_attempts=3, backoff_seconds=0),
        },
    )
    attempts = {"profile": 0, "evidence": 0}

    def always_fails(key):
        attempts[key] += 1
        raise TransientToolError("temporary_failure")

    executor.execute(
        "get_candidate_profiles", lambda: always_fails("profile"), context=context, arguments={}
    )
    executor.execute(
        "search_candidate_evidence", lambda: always_fails("evidence"), context=context, arguments={}
    )

    assert attempts == {"profile": 1, "evidence": 3}


def test_circuit_opens_after_consecutive_failures(context):
    breaker = CircuitBreaker(failure_threshold=2, recovery_seconds=60)
    executor = ToolExecutor(
        policy=ToolPolicy(max_attempts=1, timeout_seconds=0.2, backoff_seconds=0),
        circuit_breaker=breaker,
    )

    for _ in range(2):
        result = executor.execute(
            "search_candidate_evidence",
            lambda: (_ for _ in ()).throw(TransientToolError("milvus_unavailable")),
            context=context,
            arguments={},
        )
        assert result["error"]["code"] == "dependency_unavailable"

    blocked = executor.execute(
        "search_candidate_evidence",
        lambda: {"unexpected": True},
        context=context,
        arguments={},
    )

    assert blocked["ok"] is False
    assert blocked["error"]["code"] == "circuit_open"
    assert blocked["meta"]["degraded"] is True


def test_timeout_returns_without_waiting_for_slow_operation(context):
    executor = ToolExecutor(
        policy=ToolPolicy(max_attempts=1, timeout_seconds=0.01, backoff_seconds=0),
    )
    started = time.monotonic()

    result = executor.execute(
        "search_candidate_evidence",
        lambda: time.sleep(0.2),
        context=context,
        arguments={},
    )

    assert time.monotonic() - started < 0.1
    assert result["error"]["code"] == "dependency_unavailable"


def test_timeout_policy_changes_slow_operation_result(context):
    started = time.monotonic()
    direct_result = (time.sleep(0.08), "completed")[1]
    direct_elapsed = time.monotonic() - started

    started = time.monotonic()
    protected = ToolExecutor(
        policy=ToolPolicy(max_attempts=1, timeout_seconds=0.01, backoff_seconds=0)
    ).execute(
        "evidence", lambda: time.sleep(0.08), context=context, arguments={}
    )
    protected_elapsed = time.monotonic() - started

    assert direct_result == "completed"
    assert direct_elapsed >= 0.08
    assert protected["error"]["code"] == "dependency_unavailable"
    assert protected_elapsed < direct_elapsed


def test_circuit_breaker_stops_repeated_dependency_calls(context):
    calls = {"without": 0, "with": 0}

    def fail(key):
        calls[key] += 1
        raise TransientToolError("dependency_down")

    for _ in range(3):
        try:
            fail("without")
        except TransientToolError:
            pass

    executor = ToolExecutor(
        policy=ToolPolicy(max_attempts=1, backoff_seconds=0),
        circuit_breaker=CircuitBreaker(failure_threshold=2, recovery_seconds=60),
    )
    results = [
        executor.execute("evidence", lambda: fail("with"), context=context, arguments={})
        for _ in range(3)
    ]

    assert calls == {"without": 3, "with": 2}
    print(results[-1])
    assert results[-1]["error"]["code"] == "circuit_open"


def test_sql_audit_sink_persists_safe_summary(session_factory, context):
    executor = ToolExecutor(
        policy=ToolPolicy(max_attempts=1),
        audit_sink=SqlAuditSink(session_factory),
    )

    executor.execute(
        "get_candidate_profiles",
        lambda: [{"candidate_id": "C001"}],
        context=context,
        arguments={"candidate_ids": ["C001"]},
    )

    with session_factory() as db:
        row = db.scalar(select(ToolCallAudit))
    assert row.status == "succeeded"
    assert row.argument_keys == ["candidate_ids"]
    assert row.actor_id == "teacher-demo"


def test_graph_candidate_provider_adapts_decision_context(session_factory):
    service = TalentToolService(session_factory=session_factory)
    provider = build_candidate_provider(
        service,
        compile_filters=lambda _: [FilterCondition(field="region", operator="eq", value="上海")],
    )

    result = provider(
        {"original_text": "筛选上海候选人", "task_type": "evaluate_and_recommend"},
        type("Context", (), {"tenant_id": "tenant-a", "permission_scopes": ("hr_private",)})(),
    )

    assert result == ["C001"]


def test_candidate_provider_is_injected_when_graph_is_built():
    calls = []

    def provider(request, decision_context):
        calls.append((request["original_text"], decision_context.tenant_id))
        return ["C001"]

    graph = build_talent_decision_graph(provider)
    result = graph.invoke(
        {"messages": [], "request_text": "筛选上海候选人"},
        context=DecisionContext(tenant_id="tenant-a", permission_scopes=("hr_private",)),
    )

    assert calls == [("筛选上海候选人", "tenant-a")]
    assert result["candidate_ids"] == ["C001"]
