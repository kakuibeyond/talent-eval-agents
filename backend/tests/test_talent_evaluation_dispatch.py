from __future__ import annotations

from types import SimpleNamespace


def _dimension(
    dimension_id: str,
    *,
    weight_percent: int,
    source_requirement_ids: list[str],
):
    from app.talent_evaluation_dispatch import EvaluationDimension, ScoreAnchor

    return EvaluationDimension(
        dimension_id=dimension_id,
        name=dimension_id,
        definition=f"{dimension_id} 的可取证定义",
        weight_percent=weight_percent,
        evidence_requirements=["项目职责与交付结果"],
        score_anchors=[
            ScoreAnchor(score=0, description="无相关证据"),
            ScoreAnchor(score=3, description="存在部分可验证事实"),
            ScoreAnchor(score=5, description="存在完整且可验证的交付事实"),
        ],
        retrieval_hints=[f"{dimension_id} 项目经验", f"{dimension_id} 交付结果"],
        source_requirement_ids=source_requirement_ids,
    )


def test_dimension_validation_only_rejects_weight_total_other_than_100():
    from app.talent_evaluation_dispatch import EvaluationDimensionPlan, validate_dimension_plan

    plan = EvaluationDimensionPlan(
        dimensions=[
            _dimension("rag_delivery", weight_percent=60, source_requirement_ids=["S1"]),
            _dimension("agent_evaluation", weight_percent=30, source_requirement_ids=["preference:1"]),
        ]
    )

    issues = validate_dimension_plan(plan)

    assert [item.code for item in issues] == ["weight_total_invalid"]
    assert issues[0].message.endswith("当前为 90")


def test_graph_dispatches_every_candidate_dimension_pair_and_aggregates_results():
    from app.talent_decision_graph import DecisionContext
    from app.talent_evaluation_dispatch import (
        BranchEvidenceDraft,
        EvaluationDimensionPlan,
        build_talent_evaluation_dispatch_graph,
    )

    dimensions = EvaluationDimensionPlan(
        dimensions=[
            _dimension("rag_delivery", weight_percent=60, source_requirement_ids=["S1"]),
            _dimension("agent_evaluation", weight_percent=40, source_requirement_ids=["preference:1"]),
        ]
    )

    def dimension_generator(talent_request, query_plan):
        assert talent_request["original_text"] == "招聘 AI 应用工程师"
        assert query_plan["task_type"] == "find_talent"
        return dimensions

    def candidate_provider(query_plan, context):
        assert query_plan["filters"][0]["field"] == "region"
        assert context.tenant_id == "tenant-a"
        return ["C001", "C002"]

    def branch_worker(work_item, context):
        return BranchEvidenceDraft(
            task_id=work_item.task_id,
            candidate_id=work_item.candidate_id,
            dimension_id=work_item.dimension.dimension_id,
            status="succeeded",
            evidence_refs=[f"{work_item.candidate_id}:{work_item.dimension.dimension_id}:evidence"],
            missing_items=[],
            tool_call_ids=[f"call-{work_item.task_id}"],
        )

    graph = build_talent_evaluation_dispatch_graph(
        dimension_generator=dimension_generator,
        candidate_provider=candidate_provider,
        branch_worker=branch_worker,
    )
    result = graph.invoke(
        {
            "talent_request": {
                "original_text": "招聘 AI 应用工程师",
                "semantic_conditions": [{"requirement_id": "S1", "query": "RAG 项目经验"}],
                "evaluation_preferences": ["Agent 评测经验优先"],
            },
            "query_plan": {
                "task_type": "find_talent",
                "filters": [{"field": "region", "operator": "eq", "value": "上海"}],
            },
        },
        context=DecisionContext(tenant_id="tenant-a", permission_scopes=("hr_private",)),
    )

    assert result["status"] == "branches_ready"
    assert len(result["work_items"]) == 4
    assert len(result["branch_results"]) == 4
    assert {
        (item["candidate_id"], item["dimension_id"])
        for item in result["branch_results"]
    } == {
        ("C001", "rag_delivery"),
        ("C001", "agent_evaluation"),
        ("C002", "rag_delivery"),
        ("C002", "agent_evaluation"),
    }


def test_graph_stops_when_dimension_weights_are_invalid():
    from app.talent_decision_graph import DecisionContext
    from app.talent_evaluation_dispatch import (
        EvaluationDimensionPlan,
        build_talent_evaluation_dispatch_graph,
    )

    plan = EvaluationDimensionPlan(
        dimensions=[_dimension("rag_delivery", weight_percent=90, source_requirement_ids=["S1"])]
    )
    graph = build_talent_evaluation_dispatch_graph(
        dimension_generator=lambda request, query_plan: plan,
        candidate_provider=lambda query_plan, context: (_ for _ in ()).throw(
            AssertionError("权重错误时不应查询候选人")
        ),
        branch_worker=lambda work_item, context: (_ for _ in ()).throw(
            AssertionError("权重错误时不应启动分支")
        ),
    )

    result = graph.invoke(
        {
            "talent_request": {"original_text": "招聘 AI 应用工程师"},
            "query_plan": {"task_type": "find_talent", "filters": []},
        },
        context=DecisionContext(tenant_id="tenant-a", permission_scopes=("hr_private",)),
    )

    assert result["status"] == "dimension_invalid"
    assert result["errors"] == ["维度权重合计必须为 100，当前为 90"]


def test_graph_records_one_branch_failure_without_losing_other_results():
    from app.talent_decision_graph import DecisionContext
    from app.talent_evaluation_dispatch import (
        BranchEvidenceDraft,
        EvaluationDimensionPlan,
        build_talent_evaluation_dispatch_graph,
    )

    plan = EvaluationDimensionPlan(
        dimensions=[_dimension("rag_delivery", weight_percent=100, source_requirement_ids=["S1"])]
    )

    def branch_worker(work_item, context):
        if work_item.candidate_id == "C002":
            raise TimeoutError("证据检索超时")
        return BranchEvidenceDraft(
            task_id=work_item.task_id,
            candidate_id=work_item.candidate_id,
            dimension_id=work_item.dimension.dimension_id,
            status="succeeded",
            evidence_refs=["chunk-1"],
        )

    graph = build_talent_evaluation_dispatch_graph(
        dimension_generator=lambda request, query_plan: plan,
        candidate_provider=lambda query_plan, context: ["C001", "C002"],
        branch_worker=branch_worker,
    )
    result = graph.invoke(
        {
            "talent_request": {"original_text": "招聘 AI 应用工程师"},
            "query_plan": {"task_type": "find_talent", "filters": []},
        },
        context=DecisionContext(tenant_id="tenant-a", permission_scopes=("hr_private",)),
    )

    results = {item["candidate_id"]: item for item in result["branch_results"]}
    assert result["status"] == "branches_ready_with_failures"
    assert results["C001"]["status"] == "succeeded"
    assert results["C002"]["status"] == "failed"
    assert results["C002"]["error_code"] == "branch_timeout"


def test_graph_stops_before_fanout_when_work_item_budget_is_exceeded():
    from app.talent_decision_graph import DecisionContext
    from app.talent_evaluation_dispatch import (
        EvaluationDimensionPlan,
        build_talent_evaluation_dispatch_graph,
    )

    plan = EvaluationDimensionPlan(
        dimensions=[
            _dimension("rag_delivery", weight_percent=60, source_requirement_ids=["S1"]),
            _dimension("agent_evaluation", weight_percent=40, source_requirement_ids=["S1"]),
        ]
    )
    graph = build_talent_evaluation_dispatch_graph(
        dimension_generator=lambda request, query_plan: plan,
        candidate_provider=lambda query_plan, context: ["C001", "C002", "C003"],
        branch_worker=lambda work_item, context: (_ for _ in ()).throw(
            AssertionError("超出任务上限时不应启动分支")
        ),
        max_work_items=4,
    )

    result = graph.invoke(
        {
            "talent_request": {"original_text": "招聘 AI 应用工程师"},
            "query_plan": {"task_type": "find_talent", "filters": []},
        },
        context=DecisionContext(tenant_id="tenant-a", permission_scopes=("hr_private",)),
    )

    assert result["status"] == "capacity_exceeded"
    assert result["required_work_items"] == 6
    assert result["work_item_limit"] == 4
    assert result["branch_results"] == []


def test_agent_branch_worker_passes_dimension_hints_and_trusted_context():
    from app.talent_decision_graph import DecisionContext
    from app.talent_evaluation_dispatch import (
        AssessmentWorkItem,
        BranchAgentFinding,
        build_agent_branch_worker,
    )

    class RecordingAgent:
        def __init__(self):
            self.input = None
            self.config = None
            self.context = None

        def invoke(self, input_value, config=None, *, context=None):
            self.input = input_value
            self.config = config
            self.context = context
            return {
                "structured_response": BranchAgentFinding(
                    status="succeeded",
                    evidence_refs=["chunk-1"],
                    missing_items=["缺少量化指标"],
                    tool_call_ids=["call-1"],
                )
            }

    agent = RecordingAgent()
    worker = build_agent_branch_worker(agent)
    work_item = AssessmentWorkItem(
        task_id="C001:rag_delivery",
        candidate_id="C001",
        dimension=_dimension("rag_delivery", weight_percent=100, source_requirement_ids=["S1"]),
    )

    result = worker(
        work_item,
        DecisionContext(tenant_id="tenant-a", permission_scopes=("hr_private",)),
    )

    prompt = agent.input["messages"][0]["content"]
    assert "C001" in prompt
    assert "rag_delivery 项目经验" in prompt
    assert "项目职责与交付结果" in prompt
    assert agent.context.tenant_id == "tenant-a"
    assert agent.context.permission_scopes == ("hr_private",)
    assert agent.context.run_id == "C001:rag_delivery"
    assert agent.config == {"recursion_limit": 8}
    assert result.evidence_refs == ["chunk-1"]


def test_structured_dimension_generator_sends_request_and_query_plan_to_model():
    from app.talent_evaluation_dispatch import (
        EvaluationDimensionPlan,
        build_structured_dimension_generator,
    )

    plan = EvaluationDimensionPlan(
        dimensions=[_dimension("rag_delivery", weight_percent=100, source_requirement_ids=["S1"])]
    )

    class StructuredModel:
        def __init__(self):
            self.messages = None

        def with_structured_output(self, schema):
            assert schema is EvaluationDimensionPlan
            return self

        def invoke(self, messages):
            self.messages = messages
            return plan

    model = StructuredModel()
    generator = build_structured_dimension_generator(lambda: model)
    result = generator(
        {
            "semantic_conditions": [{"requirement_id": "S1", "query": "RAG 项目经验"}],
            "evaluation_preferences": [],
        },
        {"task_type": "find_talent"},
    )

    assert result == plan
    assert "semantic_conditions" in model.messages[1][1]
    assert "query_plan" in model.messages[1][1]
    assert "review_feedback" not in model.messages[1][1]
    assert "hard_conditions" in model.messages[0][1]


def test_query_plan_candidate_provider_uses_only_trusted_runtime_context():
    from app.talent_decision_graph import DecisionContext
    from app.talent_evaluation_dispatch import build_query_plan_candidate_provider

    class RecordingService:
        def __init__(self):
            self.filters = None
            self.context = None

        def filter_candidates(self, filters, *, context):
            self.filters = filters
            self.context = context
            return ["C001"]

    service = RecordingService()
    provider = build_query_plan_candidate_provider(service)
    result = provider(
        {
            "filters": [{"field": "region", "operator": "eq", "value": "上海"}],
            "tenant_id": "tenant-from-model",
        },
        DecisionContext(tenant_id="tenant-a", permission_scopes=("hr_private",)),
    )

    assert result == ["C001"]
    assert service.filters[0].field == "region"
    assert service.context.tenant_id == "tenant-a"
    assert service.context.permission_scopes == ("hr_private",)


def test_evaluation_branch_agent_exposes_only_profile_and_evidence_tools(monkeypatch):
    import app.talent_evaluation_dispatch as dispatch

    captured = {}

    def fake_create_agent(**kwargs):
        captured.update(kwargs)
        return "compiled-agent"

    monkeypatch.setattr(dispatch, "create_agent", fake_create_agent)
    tools = [
        SimpleNamespace(name="lookup_job_descriptions"),
        SimpleNamespace(name="filter_candidates"),
        SimpleNamespace(name="search_candidate_evidence"),
        SimpleNamespace(name="get_candidate_profiles"),
    ]

    result = dispatch.build_evaluation_branch_agent("model", tools)

    assert result == "compiled-agent"
    assert [item.name for item in captured["tools"]] == [
        "search_candidate_evidence",
        "get_candidate_profiles",
    ]
    assert captured["response_format"] is dispatch.BranchAgentFinding
    assert captured["context_schema"] is dispatch.TalentToolContext
    assert captured["name"] == "dimension_evidence_agent"
    assert "retrieval_hints" in captured["system_prompt"]
    assert "证据缺失" in captured["system_prompt"]


def test_sync_work_item_send_leaves_timeout_to_tool_executor():
    from app.talent_evaluation_dispatch import build_work_item_sends

    sends = build_work_item_sends(
        [
            {
                "task_id": "C001:rag_delivery",
                "candidate_id": "C001",
                "dimension": _dimension(
                    "rag_delivery",
                    weight_percent=100,
                    source_requirement_ids=["S1"],
                ).model_dump(mode="json"),
            }
        ]
    )

    assert sends[0].node == "run_assessment_branch"
    assert sends[0].arg["work_item"]["task_id"] == "C001:rag_delivery"
    assert sends[0].timeout is None
