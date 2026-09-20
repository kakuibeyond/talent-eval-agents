from __future__ import annotations

import operator
import json
from collections.abc import Callable
from typing import Annotated, Any, Literal

from langchain.agents import create_agent
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Send
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from app.query_plan import FilterCondition
from app.talent_decision_graph import DecisionContext
from app.talent_tools import TalentToolContext


class ScoreAnchor(BaseModel):
    score: Literal[0, 3, 5]
    description: str = Field(min_length=1, max_length=300)


class EvaluationDimension(BaseModel):
    dimension_id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    name: str = Field(min_length=1, max_length=80)
    definition: str = Field(min_length=1, max_length=500)
    weight_percent: int = Field(ge=1, le=100)
    evidence_requirements: list[str] = Field(min_length=1, max_length=8)
    score_anchors: list[ScoreAnchor] = Field(min_length=3, max_length=3)
    retrieval_hints: list[str] = Field(min_length=1, max_length=8)
    source_requirement_ids: list[str] = Field(min_length=1, max_length=8)


class EvaluationDimensionPlan(BaseModel):
    dimensions: list[EvaluationDimension] = Field(min_length=1, max_length=6)


class DimensionValidationIssue(BaseModel):
    code: str
    message: str
    dimension_ids: list[str] = Field(default_factory=list)


class AssessmentWorkItem(BaseModel):
    task_id: str
    candidate_id: str
    dimension: EvaluationDimension


class BranchEvidenceDraft(BaseModel):
    task_id: str
    candidate_id: str
    dimension_id: str
    status: Literal["succeeded", "degraded", "failed"]
    evidence_refs: list[str] = Field(default_factory=list)
    missing_items: list[str] = Field(default_factory=list)
    tool_call_ids: list[str] = Field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None


class BranchAgentFinding(BaseModel):
    status: Literal["succeeded", "degraded"]
    evidence_refs: list[str] = Field(default_factory=list)
    missing_items: list[str] = Field(default_factory=list)
    tool_call_ids: list[str] = Field(default_factory=list)


class TalentEvaluationDispatchState(TypedDict, total=False):
    talent_request: dict[str, Any]
    query_plan: dict[str, Any]
    dimensions: list[dict[str, Any]]
    candidate_ids: list[str]
    work_items: list[dict[str, Any]]
    branch_results: Annotated[list[dict[str, Any]], operator.add]
    validation_issues: list[dict[str, Any]]
    required_work_items: int
    work_item_limit: int
    status: str
    errors: list[str]


DimensionGenerator = Callable[
    [dict[str, Any], dict[str, Any]],
    EvaluationDimensionPlan,
]
CandidateProvider = Callable[[dict[str, Any], DecisionContext], list[str]]
BranchWorker = Callable[[AssessmentWorkItem, DecisionContext], BranchEvidenceDraft]
ModelProvider = Callable[[], Any]


DIMENSION_GENERATOR_PROMPT = """你负责把已编译的人才要求转成可取证的动态评估维度。
只允许根据 semantic_conditions 和 evaluation_preferences 生成维度。
hard_conditions 已用于候选人过滤，不得再生成评分维度。
语义要求的来源使用原 requirement_id，偏好按顺序使用 preference:1、preference:2。
维度之间不得重复，必须能通过候选人档案或原始材料取证。
权重使用整数百分比且合计为 100。评分锚点固定为 0、3、5 分。
retrieval_hints 应给出可直接用于候选人证据检索的关键词组合。
""".strip()

BRANCH_AGENT_PROMPT = """你是候选人单维度证据分析 Agent。
每次只处理输入中的一名候选人和一个评估维度。
先读取候选人基础档案，再将 retrieval_hints 与 evidence_requirements 组合成检索查询，逐项调用证据检索工具。
只能返回工具结果中存在的 chunk_id 或 citation_id，不得生成新的证据标识。
evidence_refs 保存与该维度相关的可观察事实，包括支持事实和明确冲突事实。
没有检索到足够材料的要求写入 missing_items，不得把证据缺失解释成负面事实。工具降级但仍能返回部分结果时使用 degraded。
本节不计算分数、置信度、排名或最终结论。
""".strip()


class BranchExecutionError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def build_structured_dimension_generator(
    model_provider: ModelProvider,
) -> DimensionGenerator:
    def generate(
        talent_request: dict[str, Any],
        query_plan: dict[str, Any],
    ) -> EvaluationDimensionPlan:
        model = model_provider()
        if model is None:
            raise RuntimeError("评估维度生成模型未配置")
        payload = {
            "talent_request": talent_request,
            "query_plan": query_plan,
        }
        return model.with_structured_output(EvaluationDimensionPlan).invoke(
            [
                ("system", DIMENSION_GENERATOR_PROMPT),
                ("user", json.dumps(payload, ensure_ascii=False)),
            ]
        )

    return generate


def build_evaluation_branch_agent(model: Any, tools: list[Any]):
    allowed_names = {"search_candidate_evidence", "get_candidate_profiles"}
    branch_tools = [tool for tool in tools if tool.name in allowed_names]
    return create_agent(
        model=model,
        tools=branch_tools,
        system_prompt=BRANCH_AGENT_PROMPT,
        response_format=BranchAgentFinding,
        context_schema=TalentToolContext,
        name="dimension_evidence_agent",
    )


def build_agent_branch_worker(agent: Any) -> BranchWorker:
    def branch_worker(
        work_item: AssessmentWorkItem,
        context: DecisionContext,
    ) -> BranchEvidenceDraft:
        task_payload = {
            "candidate_id": work_item.candidate_id,
            "dimension": work_item.dimension.model_dump(mode="json"),
        }
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": json.dumps(task_payload, ensure_ascii=False),
                    }
                ]
            },
            {"recursion_limit": 8},
            context=TalentToolContext(
                tenant_id=context.tenant_id,
                permission_scopes=tuple(context.permission_scopes),
                actor_id="evaluation-subagent",
                run_id=work_item.task_id,
            ),
        )
        structured_response = result.get("structured_response")
        if structured_response is None:
            raise BranchExecutionError(
                "structured_output_missing",
                "评估分支未返回结构化结果",
            )
        finding = BranchAgentFinding.model_validate(structured_response)
        return BranchEvidenceDraft(
            task_id=work_item.task_id,
            candidate_id=work_item.candidate_id,
            dimension_id=work_item.dimension.dimension_id,
            **finding.model_dump(mode="json"),
        )

    return branch_worker


def build_query_plan_candidate_provider(service: Any) -> CandidateProvider:
    def candidate_provider(
        query_plan: dict[str, Any],
        context: DecisionContext,
    ) -> list[str]:
        filters = [
            FilterCondition.model_validate(item)
            for item in query_plan.get("filters", [])
        ]
        return service.filter_candidates(
            filters,
            context=TalentToolContext(
                tenant_id=context.tenant_id,
                permission_scopes=tuple(context.permission_scopes),
                actor_id="evaluation-dispatch",
                run_id="lesson-15",
            ),
        )

    return candidate_provider


def validate_dimension_plan(
    plan: EvaluationDimensionPlan,
) -> list[DimensionValidationIssue]:
    total = sum(item.weight_percent for item in plan.dimensions)
    if total == 100:
        return []
    return [
        DimensionValidationIssue(
            code="weight_total_invalid",
            message=f"维度权重合计必须为 100，当前为 {total}",
        )
    ]


def _receive_input(
    state: TalentEvaluationDispatchState,
    runtime: Runtime[DecisionContext],
) -> dict[str, Any]:
    if not runtime.context.tenant_id.strip():
        raise ValueError("Runtime Context 中的 tenant_id 不能为空")
    if not runtime.context.permission_scopes:
        raise ValueError("Runtime Context 中的 permission_scopes 不能为空")
    if not state.get("talent_request"):
        raise ValueError("talent_request 不能为空")
    if not state.get("query_plan"):
        raise ValueError("query_plan 不能为空")
    return {
        "dimensions": [],
        "candidate_ids": [],
        "work_items": [],
        "branch_results": [],
        "validation_issues": [],
        "required_work_items": 0,
        "work_item_limit": 0,
        "status": "received",
        "errors": [],
    }


def _generate_dimensions(dimension_generator: DimensionGenerator):
    def generate_dimensions(state: TalentEvaluationDispatchState) -> dict[str, Any]:
        plan = dimension_generator(
            state["talent_request"],
            state["query_plan"],
        )
        return {
            "dimensions": [item.model_dump(mode="json") for item in plan.dimensions],
            "status": "dimensions_generated",
        }

    return generate_dimensions


def _validate_dimensions(state: TalentEvaluationDispatchState) -> dict[str, Any]:
    plan = EvaluationDimensionPlan(dimensions=state["dimensions"])
    issues = validate_dimension_plan(plan)
    return {
        "validation_issues": [item.model_dump(mode="json") for item in issues],
        "status": "dimensions_valid" if not issues else "dimension_invalid",
    }


def _route_dimension_validation(
    state: TalentEvaluationDispatchState,
) -> Literal["retrieve_candidates", "dimension_invalid"]:
    if state.get("validation_issues"):
        return "dimension_invalid"
    return "retrieve_candidates"


def _retrieve_candidates(candidate_provider: CandidateProvider):
    def retrieve_candidates(
        state: TalentEvaluationDispatchState,
        runtime: Runtime[DecisionContext],
    ) -> dict[str, Any]:
        candidate_ids = candidate_provider(state["query_plan"], runtime.context)
        return {
            "candidate_ids": candidate_ids,
            "status": "candidates_ready" if candidate_ids else "no_candidates",
        }

    return retrieve_candidates


def _prepare_work_items(max_work_items: int):
    def prepare_work_items(state: TalentEvaluationDispatchState) -> dict[str, Any]:
        dimensions = [
            EvaluationDimension.model_validate(item) for item in state["dimensions"]
        ]
        required_work_items = len(state["candidate_ids"]) * len(dimensions)
        if required_work_items > max_work_items:
            return {
                "work_items": [],
                "required_work_items": required_work_items,
                "work_item_limit": max_work_items,
                "status": "capacity_exceeded",
            }
        work_items = [
            AssessmentWorkItem(
                task_id=f"{candidate_id}:{dimension.dimension_id}",
                candidate_id=candidate_id,
                dimension=dimension,
            ).model_dump(mode="json")
            for candidate_id in state["candidate_ids"]
            for dimension in dimensions
        ]
        return {
            "work_items": work_items,
            "required_work_items": required_work_items,
            "work_item_limit": max_work_items,
            "status": "work_items_ready" if work_items else "no_candidates",
        }

    return prepare_work_items


def build_work_item_sends(
    work_items: list[dict[str, Any]],
) -> list[Send]:
    return [
        Send("run_assessment_branch", {"work_item": item})
        for item in work_items
    ]


def _dispatch_work_items(state: TalentEvaluationDispatchState):
    if state.get("status") == "capacity_exceeded":
        return "capacity_exceeded"
    if not state.get("work_items"):
        return "no_candidates"
    return build_work_item_sends(state["work_items"])


def _run_assessment_branch(branch_worker: BranchWorker):
    def run_assessment_branch(
        state: dict[str, Any],
        runtime: Runtime[DecisionContext],
    ) -> dict[str, Any]:
        work_item = AssessmentWorkItem.model_validate(state["work_item"])
        try:
            result = branch_worker(work_item, runtime.context)
        except TimeoutError as exc:
            result = BranchEvidenceDraft(
                task_id=work_item.task_id,
                candidate_id=work_item.candidate_id,
                dimension_id=work_item.dimension.dimension_id,
                status="failed",
                error_code="branch_timeout",
                error_message=str(exc),
            )
        except BranchExecutionError as exc:
            result = BranchEvidenceDraft(
                task_id=work_item.task_id,
                candidate_id=work_item.candidate_id,
                dimension_id=work_item.dimension.dimension_id,
                status="failed",
                error_code=exc.code,
                error_message=str(exc),
            )
        return {"branch_results": [result.model_dump(mode="json")]}

    return run_assessment_branch


def _finalize(state: TalentEvaluationDispatchState) -> dict[str, Any]:
    has_failures = any(
        item["status"] == "failed" for item in state.get("branch_results", [])
    )
    return {
        "status": "branches_ready_with_failures" if has_failures else "branches_ready"
    }


def _dimension_invalid(state: TalentEvaluationDispatchState) -> dict[str, Any]:
    return {
        "status": "dimension_invalid",
        "errors": [item["message"] for item in state.get("validation_issues", [])],
    }


def _no_candidates(_: TalentEvaluationDispatchState) -> dict[str, Any]:
    return {"status": "no_candidates", "work_items": [], "branch_results": []}


def _capacity_exceeded(state: TalentEvaluationDispatchState) -> dict[str, Any]:
    return {
        "status": "capacity_exceeded",
        "errors": [
            f"评估任务数 {state['required_work_items']} 超过本次上限 {state['work_item_limit']}"
        ],
    }


def build_talent_evaluation_dispatch_graph(
    *,
    dimension_generator: DimensionGenerator,
    candidate_provider: CandidateProvider,
    branch_worker: BranchWorker,
    max_work_items: int = 24,
):
    if max_work_items < 1:
        raise ValueError("max_work_items 必须大于等于 1")
    builder = StateGraph(
        TalentEvaluationDispatchState,
        context_schema=DecisionContext,
    )
    builder.add_node("receive_input", _receive_input)
    builder.add_node("generate_dimensions", _generate_dimensions(dimension_generator))
    builder.add_node("validate_dimensions", _validate_dimensions)
    builder.add_node("retrieve_candidates", _retrieve_candidates(candidate_provider))
    builder.add_node("prepare_work_items", _prepare_work_items(max_work_items))
    builder.add_node("run_assessment_branch", _run_assessment_branch(branch_worker))
    builder.add_node("finalize", _finalize)
    builder.add_node("dimension_invalid", _dimension_invalid)
    builder.add_node("no_candidates", _no_candidates)
    builder.add_node("capacity_exceeded", _capacity_exceeded)
    builder.add_edge(START, "receive_input")
    builder.add_edge("receive_input", "generate_dimensions")
    builder.add_edge("generate_dimensions", "validate_dimensions")
    builder.add_conditional_edges(
        "validate_dimensions",
        _route_dimension_validation,
    )
    builder.add_edge("retrieve_candidates", "prepare_work_items")
    builder.add_conditional_edges(
        "prepare_work_items",
        _dispatch_work_items,
        ["run_assessment_branch", "no_candidates", "capacity_exceeded"],
    )
    builder.add_edge("run_assessment_branch", "finalize")
    builder.add_edge("finalize", END)
    builder.add_edge("dimension_invalid", END)
    builder.add_edge("no_candidates", END)
    builder.add_edge("capacity_exceeded", END)
    return builder.compile()
