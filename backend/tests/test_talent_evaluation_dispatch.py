from __future__ import annotations

from types import SimpleNamespace


def _dimension(
    dimension_name: str,
    *,
    weight_percent: int,
    source_requirement_ids: list[str],
):
    from app.talent_evaluation_dispatch import EvaluationDimension, ScoreAnchor

    return EvaluationDimension(
        name=dimension_name,
        definition=f"{dimension_name} 的可取证定义",
        weight_percent=weight_percent,
        evidence_requirements=["项目职责与交付结果"],
        score_anchors=[
            ScoreAnchor(score=0, description="无相关证据"),
            ScoreAnchor(score=3, description="存在部分可验证事实"),
            ScoreAnchor(score=5, description="存在完整且可验证的交付事实"),
        ],
        retrieval_hints=[f"{dimension_name} 项目经验", f"{dimension_name} 交付结果"],
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
            dimension_number=work_item.dimension_number,
            execution_status="succeeded",
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
    assert all("dimension_number" not in item for item in result["dimensions"])
    assert [item["dimension_number"] for item in result["work_items"]] == [1, 2, 1, 2]
    assert len(result["work_items"]) == 4
    assert len(result["branch_results"]) == 4
    assert {
        (item["candidate_id"], item["dimension_number"])
        for item in result["branch_results"]
    } == {
        ("C001", 1),
        ("C001", 2),
        ("C002", 1),
        ("C002", 2),
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
            dimension_number=work_item.dimension_number,
            execution_status="succeeded",
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
    assert results["C001"]["execution_status"] == "succeeded"
    assert results["C002"]["execution_status"] == "failed"
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
    assert "dimension_number" not in EvaluationDimensionPlan.model_json_schema()[
        "$defs"
    ]["EvaluationDimension"]["properties"]
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


def test_sync_work_item_send_leaves_timeout_to_tool_executor():
    from app.talent_evaluation_dispatch import build_work_item_sends

    sends = build_work_item_sends(
        [
            {
                "task_id": "C001:1",
                "candidate_id": "C001",
                "dimension_number": 1,
                "dimension": _dimension(
                    "rag_delivery",
                    weight_percent=100,
                    source_requirement_ids=["S1"],
                ).model_dump(mode="json"),
            }
        ]
    )

    assert sends[0].node == "run_assessment_branch"
    assert sends[0].arg["work_item"]["task_id"] == "C001:1"
    assert sends[0].timeout is None


def test_verification_script_reuses_registered_runtime_graph(monkeypatch):
    import importlib
    import sys
    from types import ModuleType

    runtime = ModuleType("app.talent_evaluation_runtime")
    runtime.graph = object()
    monkeypatch.setitem(sys.modules, "app.talent_evaluation_runtime", runtime)
    sys.modules.pop("scripts.verify_talent_evaluation_dispatch", None)

    verifier = importlib.import_module("scripts.verify_talent_evaluation_dispatch")

    assert verifier.graph is runtime.graph
    assert not hasattr(verifier, "build_graph")


def test_verification_script_defaults_to_six_way_concurrency(monkeypatch):
    import importlib
    import sys
    from types import ModuleType

    runtime = ModuleType("app.talent_evaluation_runtime")
    runtime.graph = object()
    monkeypatch.setitem(sys.modules, "app.talent_evaluation_runtime", runtime)
    sys.modules.pop("scripts.verify_talent_evaluation_dispatch", None)

    verifier = importlib.import_module("scripts.verify_talent_evaluation_dispatch")

    assert verifier._build_parser().parse_args([]).max_concurrency == 6


def test_evidence_branch_worker_searches_each_requirement_and_keeps_evidence_pack(caplog):
    from app.talent_decision_graph import DecisionContext
    from app.talent_evaluation_dispatch import (
        AssessmentWorkItem,
        build_evidence_branch_worker,
    )
    from app.talent_tools import ToolExecutor, ToolPolicy

    class RecordingService:
        def __init__(self):
            self.queries = []

        def search_candidate_evidence(self, query, candidate_ids, *, context):
            self.queries.append((query, candidate_ids, context))
            requirement = next(
                item
                for item in work_item.dimension.evidence_requirements
                if item in query
            )
            has_evidence = requirement == "项目职责"
            return {
                "candidate_ids": candidate_ids,
                "evidence_packs": [
                    {
                        "schema_version": "2.0",
                        "candidate_id": "C001",
                        "requirements": [
                            {
                                "requirement_id": "branch_requirement",
                                "query": query,
                                "status": "sufficient" if has_evidence else "missing",
                                "reason": "evidence_review" if has_evidence else "no_accessible_hits",
                                "extraction_status": "succeeded" if has_evidence else "not_run",
                                "facts": [
                                    {
                                        "event": "星河项目",
                                        "period": "2025",
                                        "claim": requirement,
                                        "answer": "yes",
                                        "sources": [
                                            {
                                                "citation_id": "citation-1",
                                                "chunk_id": "chunk-1",
                                                "quote": "主导星河项目上线",
                                                "quote_start": 0,
                                                "quote_end": 8,
                                                "source_label": "项目复盘",
                                            }
                                        ],
                                    }
                                ] if has_evidence else [],
                                "conflicts": [],
                                "missing_information": [] if has_evidence else [requirement],
                                "citations": [],
                            }
                        ],
                    }
                ],
            }

    work_item = AssessmentWorkItem(
        task_id="C001:1",
        candidate_id="C001",
        dimension_number=1,
        dimension=_dimension(
            "rag_delivery",
            weight_percent=100,
            source_requirement_ids=["S1"],
        ).model_copy(
            update={
                "evidence_requirements": ["项目职责", "量化结果"],
                "retrieval_hints": ["RAG", "上线"],
            }
        ),
    )
    service = RecordingService()
    worker = build_evidence_branch_worker(
        service,
        ToolExecutor(policy=ToolPolicy(max_attempts=1, backoff_seconds=0)),
    )

    with caplog.at_level("INFO", logger="app.talent_evaluation_dispatch"):
        result = worker(
            work_item,
            DecisionContext(tenant_id="tenant-a", permission_scopes=("hr_private",)),
        )

    assert len(service.queries) == 2
    assert all(item[1] == ["C001"] for item in service.queries)
    assert all("RAG" in item[0] for item in service.queries)
    assert any("项目职责" in item[0] for item in service.queries)
    assert all(item[2].tenant_id == "tenant-a" for item in service.queries)
    assert result.execution_status == "succeeded"
    assert [item.requirement_id for item in result.requirements] == [
        "1:1",
        "1:2",
    ]
    assert result.requirements[0].facts[0].sources[0].chunk_id == "chunk-1"
    assert result.requirements[1].missing_information == ["量化结果"]
    assert len(result.tool_call_ids) == 2
    assert not hasattr(result, "missing_items")
    messages = "\n".join(caplog.messages)
    assert "event=branch_worker_started" in messages
    assert "event=branch_requirement_started" in messages
    assert "event=branch_requirement_completed" in messages
    assert "event=branch_worker_completed" in messages
    assert "task_id=C001:1" in messages
    assert "candidate_id=C001" in messages


def test_evidence_branch_worker_runs_requirements_concurrently():
    from threading import Condition

    from app.talent_decision_graph import DecisionContext
    from app.talent_evaluation_dispatch import (
        AssessmentWorkItem,
        build_evidence_branch_worker,
    )
    from app.talent_tools import ToolExecutor, ToolPolicy

    class ConcurrentService:
        def __init__(self):
            self.condition = Condition()
            self.started = 0
            self.active = 0
            self.max_active = 0

        def search_candidate_evidence(self, query, candidate_ids, *, context):
            del context
            with self.condition:
                self.started += 1
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                self.condition.notify_all()
                self.condition.wait_for(lambda: self.started >= 2, timeout=0.5)
                self.active -= 1
            return {
                "candidate_ids": candidate_ids,
                "evidence_packs": [
                    {
                        "schema_version": "2.0",
                        "candidate_id": candidate_ids[0],
                        "requirements": [
                            {
                                "requirement_id": "branch_requirement",
                                "query": query,
                                "status": "missing",
                                "reason": "no_relevant_evidence",
                                "extraction_status": "succeeded",
                                "facts": [],
                                "conflicts": [],
                                "missing_information": ["材料未覆盖当前要求"],
                                "citations": [],
                            }
                        ],
                    }
                ],
            }

    work_item = AssessmentWorkItem(
        task_id="C001:1",
        candidate_id="C001",
        dimension_number=1,
        dimension=_dimension(
            "rag_delivery",
            weight_percent=100,
            source_requirement_ids=["S1"],
        ).model_copy(
            update={"evidence_requirements": ["项目职责", "量化结果"]}
        ),
    )
    service = ConcurrentService()
    worker = build_evidence_branch_worker(
        service,
        ToolExecutor(
            policy=ToolPolicy(
                max_attempts=1,
                timeout_seconds=1,
                backoff_seconds=0,
            )
        ),
    )

    result = worker(
        work_item,
        DecisionContext(tenant_id="tenant-a", permission_scopes=("hr_private",)),
    )

    assert service.max_active == 2
    assert [item.requirement_id for item in result.requirements] == [
        "1:1",
        "1:2",
    ]


def test_branch_evidence_provider_builds_reviewed_pack_from_trusted_sources():
    from types import SimpleNamespace

    from app.talent_evaluation_dispatch import build_branch_evidence_provider

    def search(**kwargs):
        return [
            SimpleNamespace(
                chunk_id="chunk-1",
                candidate_id="C001",
                content="index-copy",
                rerank_score=0.9,
                metadata={},
            )
        ]

    def load_sources(chunks, *, tenant_id, permission_scopes):
        return [
            {
                "chunk_id": "chunk-1",
                "citation_id": "citation-1",
                "candidate_id": "C001",
                "content": "主导星河项目上线",
                "document_title": "项目复盘",
                "requirement_ids": ["branch_requirement"],
            }
        ]

    provider = build_branch_evidence_provider(
        search=search,
        load_sources=load_sources,
        extract=lambda requirement, sources: {
            "facts": [
                {
                    "event": "星河项目",
                    "period": "2025",
                    "claim": "主导项目上线",
                    "answer": "yes",
                    "sources": [
                        {"chunk_id": "chunk-1", "quote": "主导星河项目上线"}
                    ],
                }
            ],
            "fully_supported": True,
            "missing_information": [],
        },
    )

    result = provider(
        query="RAG 项目职责",
        candidate_ids=["C001"],
        tenant_id="tenant-a",
        permission_scopes=["hr_private"],
        include_evidence_pack=True,
    )

    requirement = result["evidence_packs"][0]["requirements"][0]
    assert requirement["status"] == "sufficient"
    assert requirement["facts"][0]["sources"][0]["chunk_id"] == "chunk-1"
    assert requirement["facts"][0]["sources"][0]["source_label"] == "项目复盘"


def test_evidence_extraction_failure_does_not_relabel_successful_branch_execution():
    from app.talent_decision_graph import DecisionContext
    from app.talent_evaluation_dispatch import (
        AssessmentWorkItem,
        build_evidence_branch_worker,
    )
    from app.talent_tools import ToolExecutor, ToolPolicy

    class Service:
        def search_candidate_evidence(self, query, candidate_ids, *, context):
            del query, context
            return {
                "candidate_ids": candidate_ids,
                "evidence_packs": [
                    {
                        "schema_version": "2.0",
                        "candidate_id": "C001",
                        "requirements": [
                            {
                                "requirement_id": "branch_requirement",
                                "query": "RAG 项目职责",
                                "status": "partial",
                                "reason": "extraction_failed",
                                "extraction_status": "failed",
                                "facts": [],
                                "conflicts": [],
                                "missing_information": [],
                                "citations": [
                                    {
                                        "citation_id": "citation-1",
                                        "chunk_id": "chunk-1",
                                        "document_title": "项目复盘",
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }

    worker = build_evidence_branch_worker(
        Service(),
        ToolExecutor(policy=ToolPolicy(max_attempts=1, backoff_seconds=0)),
    )
    result = worker(
        AssessmentWorkItem(
            task_id="C001:1",
            candidate_id="C001",
            dimension_number=1,
            dimension=_dimension(
                "rag_delivery",
                weight_percent=100,
                source_requirement_ids=["S1"],
            ),
        ),
        DecisionContext(tenant_id="tenant-a", permission_scopes=("hr_private",)),
    )

    assert result.execution_status == "succeeded"
    assert result.requirements[0].extraction_status == "failed"
    assert result.requirements[0].reason == "extraction_failed"


def test_branch_draft_rejects_removed_parallel_summary_fields():
    import pytest
    from pydantic import ValidationError

    from app.talent_evaluation_dispatch import BranchEvidenceDraft

    with pytest.raises(ValidationError):
        BranchEvidenceDraft(
            task_id="C001:1",
            candidate_id="C001",
            dimension_number=1,
            execution_status="succeeded",
            evidence_refs=["chunk-1"],
            missing_items=["缺少量化结果"],
        )
