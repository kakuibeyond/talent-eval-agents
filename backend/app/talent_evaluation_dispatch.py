from __future__ import annotations

import json
import logging
import operator
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Send
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from typing_extensions import TypedDict

from app.evidence_pack import build_evidence_packs
from app.query_plan import FilterCondition
from app.talent_decision_graph import DecisionContext
from app.talent_tools import TalentToolContext, ToolExecutor

logger = logging.getLogger(__name__)


class ScoreAnchor(BaseModel):
    score: Literal[0, 3, 5]
    description: str = Field(min_length=1, max_length=300)


class EvaluationDimension(BaseModel):
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
    dimension_numbers: list[int] = Field(default_factory=list)


class AssessmentWorkItem(BaseModel):
    task_id: str
    candidate_id: str
    dimension_number: int = Field(ge=1)
    dimension: EvaluationDimension


class EvidenceSourceRef(BaseModel):
    citation_id: str
    chunk_id: str
    quote: str = Field(min_length=1)
    quote_start: int = Field(ge=0)
    quote_end: int = Field(gt=0)
    source_label: str = Field(min_length=1)


class BranchEvidenceFact(BaseModel):
    event: str
    period: str
    claim: str
    answer: Literal["yes", "no"]
    sources: list[EvidenceSourceRef] = Field(default_factory=list)


class EvidenceCitationRef(BaseModel):
    citation_id: str
    chunk_id: str
    source_label: str = Field(min_length=1)


class RequirementEvidence(BaseModel):
    requirement_id: str
    query: str
    status: Literal["sufficient", "partial", "missing", "conflicting"]
    reason: str
    extraction_status: Literal["not_run", "succeeded", "failed"]
    facts: list[BranchEvidenceFact] = Field(default_factory=list)
    conflicts: list[list[int]] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    citations: list[EvidenceCitationRef] = Field(default_factory=list)


class BranchEvidenceDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    candidate_id: str
    dimension_number: int = Field(ge=1)
    execution_status: Literal["succeeded", "degraded", "failed"] = Field(
        validation_alias=AliasChoices("execution_status", "status")
    )
    requirements: list[RequirementEvidence] = Field(default_factory=list)
    tool_call_ids: list[str] = Field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None


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
EvidenceSearch = Callable[..., list[Any]]
EvidenceSourceLoader = Callable[..., list[dict[str, Any]]]


DIMENSION_GENERATOR_PROMPT = """你负责把已编译的人才要求转成可取证的动态评估维度。
只允许根据 semantic_conditions 和 evaluation_preferences 生成维度。
hard_conditions 已用于候选人过滤，不得再生成评分维度。
语义要求的来源使用原 requirement_id，偏好按顺序使用 preference:1、preference:2。
维度之间不得重复，必须能通过候选人档案或原始材料取证。
权重使用整数百分比且合计为 100。评分锚点固定为 0、3、5 分。
retrieval_hints 应给出可直接用于候选人证据检索的关键词组合。
""".strip()

class BranchExecutionError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def build_branch_evidence_provider(
    *,
    search: EvidenceSearch,
    load_sources: EvidenceSourceLoader,
    extract: Callable[[dict[str, Any], list[dict[str, Any]]], Any],
):
    """Return authorized chunks together with Candidate Evidence Pack 2.0."""

    def evidence_provider(
        *,
        query: str,
        candidate_ids: list[str],
        tenant_id: str,
        permission_scopes: list[str],
        include_evidence_pack: bool,
    ) -> dict[str, Any]:
        logger.info(
            "event=evidence_provider_started function=evidence_provider "
            "candidate_ids=%s query_chars=%s permission_scope_count=%s "
            "include_evidence_pack=%s",
            candidate_ids, len(query), len(permission_scopes), include_evidence_pack,
        )
        ranked = search(
            query=query,
            candidate_ids=candidate_ids,
            tenant_id=tenant_id,
            permission_scopes=permission_scopes,
        )
        logger.info(
            "event=evidence_search_completed function=evidence_provider "
            "candidate_ids=%s hit_count=%s chunk_ids=%s",
            candidate_ids, len(ranked), [item.chunk_id for item in ranked],
        )
        hits = [
            {
                "chunk_id": item.chunk_id,
                "candidate_id": item.candidate_id,
                "content": item.content,
                "score": item.rerank_score,
                "metadata": item.metadata,
                "requirement_ids": ["branch_requirement"],
            }
            for item in ranked
        ]
        trusted_sources = load_sources(
            hits,
            tenant_id=tenant_id,
            permission_scopes=permission_scopes,
        )
        logger.info(
            "event=evidence_sources_loaded function=evidence_provider "
            "candidate_ids=%s requested_chunk_count=%s trusted_source_count=%s "
            "trusted_chunk_ids=%s",
            candidate_ids, len(hits), len(trusted_sources),
            [item["chunk_id"] for item in trusted_sources],
        )
        evidence_packs = (
            build_evidence_packs(
                candidate_ids=candidate_ids,
                requirements=[
                    {
                        "requirement_id": "branch_requirement",
                        "query": query,
                    }
                ],
                sources=trusted_sources,
                extract=extract,
            )
            if include_evidence_pack
            else []
        )
        logger.info(
            "event=evidence_provider_completed function=evidence_provider "
            "candidate_ids=%s trusted_source_count=%s evidence_pack_count=%s",
            candidate_ids, len(trusted_sources), len(evidence_packs),
        )
        return {
            "candidate_ids": candidate_ids,
            "chunks": trusted_sources,
            "evidence_packs": evidence_packs,
        }

    return evidence_provider


def build_structured_dimension_generator(
    model_provider: ModelProvider,
) -> DimensionGenerator:
    def generate(
        talent_request: dict[str, Any],
        query_plan: dict[str, Any],
    ) -> EvaluationDimensionPlan:
        logger.info(
            "event=dimension_generation_started "
            "function=build_structured_dimension_generator.generate "
            "semantic_condition_count=%s preference_count=%s filter_count=%s",
            len(talent_request.get("semantic_conditions", [])),
            len(talent_request.get("evaluation_preferences", [])),
            len(query_plan.get("filters", [])),
        )
        model = model_provider()
        if model is None:
            raise RuntimeError("评估维度生成模型未配置")
        payload = {
            "talent_request": talent_request,
            "query_plan": query_plan,
        }
        result = model.with_structured_output(EvaluationDimensionPlan).invoke(
            [
                ("system", DIMENSION_GENERATOR_PROMPT),
                ("user", json.dumps(payload, ensure_ascii=False)),
            ]
        )
        logger.info(
            "event=dimension_generation_completed "
            "function=build_structured_dimension_generator.generate "
            "dimension_count=%s dimension_names=%s weight_total=%s",
            len(result.dimensions),
            [item.name for item in result.dimensions],
            sum(item.weight_percent for item in result.dimensions),
        )
        return result

    return generate


def _branch_query(dimension: EvaluationDimension, requirement: str) -> str:
    terms = list(dict.fromkeys([*dimension.retrieval_hints, requirement]))
    return " ".join(terms)[:500]


def _citation_refs(raw_citations: list[Any]) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in raw_citations:
        if not isinstance(raw, dict):
            continue
        chunk_id = raw.get("chunk_id")
        citation_id = raw.get("citation_id") or chunk_id
        if not isinstance(chunk_id, str) or not chunk_id or chunk_id in seen:
            continue
        seen.add(chunk_id)
        refs.append(
            {
                "citation_id": str(citation_id),
                "chunk_id": chunk_id,
                "source_label": str(
                    raw.get("document_title")
                    or raw.get("source_label")
                    or chunk_id
                ),
            }
        )
    return refs


def _missing_requirement(
    *,
    requirement_id: str,
    query: str,
    requirement: str,
    reason: str,
) -> RequirementEvidence:
    return RequirementEvidence(
        requirement_id=requirement_id,
        query=query,
        status="missing",
        reason=reason,
        extraction_status="not_run",
        missing_information=[requirement],
    )


def build_evidence_branch_worker(service: Any, executor: ToolExecutor) -> BranchWorker:
    """Prepare evaluation-ready evidence without an additional ReAct loop."""

    def branch_worker(
        work_item: AssessmentWorkItem,
        context: DecisionContext,
    ) -> BranchEvidenceDraft:
        logger.info(
            "event=branch_worker_started function=branch_worker task_id=%s "
            "candidate_id=%s dimension_number=%s requirement_count=%s",
            work_item.task_id, work_item.candidate_id,
            work_item.dimension_number,
            len(work_item.dimension.evidence_requirements),
        )
        tool_context = TalentToolContext(
            tenant_id=context.tenant_id,
            permission_scopes=tuple(context.permission_scopes),
            actor_id="evaluation-branch-worker",
            run_id=work_item.task_id,
        )
        requirement_results: list[RequirementEvidence] = []
        call_ids: list[str] = []
        degraded = False

        def evaluate_requirement(
            index: int,
            requirement: str,
        ) -> tuple[RequirementEvidence, str | None, bool]:
            requirement_id = f"{work_item.dimension_number}:{index}"
            query = _branch_query(work_item.dimension, requirement)
            logger.info(
                "event=branch_requirement_started function=branch_worker "
                "task_id=%s candidate_id=%s dimension_number=%s "
                "requirement_id=%s query_chars=%s",
                work_item.task_id, work_item.candidate_id,
                work_item.dimension_number, requirement_id, len(query),
            )
            envelope = executor.execute(
                "search_candidate_evidence",
                lambda query=query: service.search_candidate_evidence(
                    query,
                    [work_item.candidate_id],
                    context=tool_context,
                ),
                context=tool_context,
                arguments={
                    "query": query,
                    "candidate_ids": [work_item.candidate_id],
                },
            )
            meta = envelope.get("meta") or {}
            call_id = meta.get("call_id")
            normalized_call_id = call_id if isinstance(call_id, str) and call_id else None
            if not envelope.get("ok"):
                error = envelope.get("error") or {}
                logger.error(
                    "event=branch_requirement_failed function=branch_worker "
                    "task_id=%s candidate_id=%s dimension_number=%s "
                    "requirement_id=%s call_id=%s error_code=%s degraded=%s",
                    work_item.task_id, work_item.candidate_id,
                    work_item.dimension_number, requirement_id, call_id,
                    error.get("code") or "tool_failed", meta.get("degraded"),
                )
                return (
                    _missing_requirement(
                        requirement_id=requirement_id,
                        query=query,
                        requirement=requirement,
                        reason=str(error.get("code") or "tool_failed"),
                    ),
                    normalized_call_id,
                    True,
                )

            data = envelope.get("data") or {}
            pack = next(
                (
                    item
                    for item in data.get("evidence_packs", [])
                    if item.get("candidate_id") == work_item.candidate_id
                ),
                None,
            )
            raw_requirement = (
                pack.get("requirements", [None])[0]
                if isinstance(pack, dict) and pack.get("requirements")
                else None
            )
            if not isinstance(raw_requirement, dict):
                logger.error(
                    "event=branch_requirement_failed function=branch_worker "
                    "task_id=%s candidate_id=%s dimension_number=%s "
                    "requirement_id=%s call_id=%s error_code=evidence_pack_missing",
                    work_item.task_id, work_item.candidate_id,
                    work_item.dimension_number, requirement_id, call_id,
                )
                return (
                    _missing_requirement(
                        requirement_id=requirement_id,
                        query=query,
                        requirement=requirement,
                        reason="evidence_pack_missing",
                    ),
                    normalized_call_id,
                    True,
                )
            result = RequirementEvidence.model_validate(
                {
                    **raw_requirement,
                    "requirement_id": requirement_id,
                    "query": query,
                    "citations": _citation_refs(
                        raw_requirement.get("citations", [])
                    ),
                }
            )
            logger.info(
                "event=branch_requirement_completed function=branch_worker "
                "task_id=%s candidate_id=%s dimension_number=%s "
                "requirement_id=%s call_id=%s status=%s reason=%s "
                "extraction_status=%s fact_count=%s citation_count=%s",
                work_item.task_id, work_item.candidate_id,
                work_item.dimension_number, requirement_id, call_id,
                result.status, result.reason, result.extraction_status,
                len(result.facts), len(result.citations),
            )
            return result, normalized_call_id, bool(meta.get("degraded"))

        indexed_requirements = list(
            enumerate(work_item.dimension.evidence_requirements, start=1)
        )
        with ThreadPoolExecutor(
            max_workers=len(indexed_requirements),
            thread_name_prefix="evaluation-requirement",
        ) as pool:
            futures = [
                pool.submit(evaluate_requirement, index, requirement)
                for index, requirement in indexed_requirements
            ]
            outcomes = [future.result() for future in futures]

        for result, call_id, requirement_degraded in outcomes:
            requirement_results.append(result)
            if call_id is not None:
                call_ids.append(call_id)
            degraded = degraded or requirement_degraded

        branch_result = BranchEvidenceDraft(
            task_id=work_item.task_id,
            candidate_id=work_item.candidate_id,
            dimension_number=work_item.dimension_number,
            execution_status="degraded" if degraded else "succeeded",
            requirements=requirement_results,
            tool_call_ids=call_ids,
        )
        logger.info(
            "event=branch_worker_completed function=branch_worker task_id=%s "
            "candidate_id=%s dimension_number=%s execution_status=%s "
            "requirement_count=%s tool_call_count=%s",
            work_item.task_id, work_item.candidate_id,
            work_item.dimension_number, branch_result.execution_status,
            len(branch_result.requirements), len(branch_result.tool_call_ids),
        )
        return branch_result

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
    logger.info(
        "event=graph_node_started function=_receive_input node=receive_input "
        "tenant_id=%s permission_scope_count=%s has_talent_request=%s "
        "has_query_plan=%s",
        runtime.context.tenant_id, len(runtime.context.permission_scopes),
        bool(state.get("talent_request")), bool(state.get("query_plan")),
    )
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
        logger.info(
            "event=graph_node_completed function=generate_dimensions "
            "node=generate_dimensions dimension_count=%s dimension_names=%s",
            len(plan.dimensions), [item.name for item in plan.dimensions],
        )
        return {
            "dimensions": [item.model_dump(mode="json") for item in plan.dimensions],
            "status": "dimensions_generated",
        }

    return generate_dimensions


def _validate_dimensions(state: TalentEvaluationDispatchState) -> dict[str, Any]:
    plan = EvaluationDimensionPlan(dimensions=state["dimensions"])
    issues = validate_dimension_plan(plan)
    logger.info(
        "event=graph_node_completed function=_validate_dimensions "
        "node=validate_dimensions issue_count=%s weight_total=%s",
        len(issues), sum(item.weight_percent for item in plan.dimensions),
    )
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
        logger.info(
            "event=graph_node_completed function=retrieve_candidates "
            "node=retrieve_candidates candidate_count=%s candidate_ids=%s",
            len(candidate_ids), candidate_ids,
        )
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
            logger.error(
                "event=graph_capacity_exceeded function=prepare_work_items "
                "node=prepare_work_items candidate_count=%s dimension_count=%s "
                "required_work_items=%s work_item_limit=%s",
                len(state["candidate_ids"]), len(dimensions),
                required_work_items, max_work_items,
            )
            return {
                "work_items": [],
                "required_work_items": required_work_items,
                "work_item_limit": max_work_items,
                "status": "capacity_exceeded",
            }
        numbered_dimensions = list(enumerate(dimensions, start=1))
        work_items = [
            AssessmentWorkItem(
                task_id=f"{candidate_id}:{dimension_number}",
                candidate_id=candidate_id,
                dimension_number=dimension_number,
                dimension=dimension,
            ).model_dump(mode="json")
            for candidate_id in state["candidate_ids"]
            for dimension_number, dimension in numbered_dimensions
        ]
        logger.info(
            "event=graph_node_completed function=prepare_work_items "
            "node=prepare_work_items candidate_count=%s dimension_count=%s "
            "work_item_count=%s task_ids=%s",
            len(state["candidate_ids"]), len(dimensions), len(work_items),
            [item["task_id"] for item in work_items],
        )
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
    logger.info(
        "event=work_items_dispatched function=build_work_item_sends "
        "work_item_count=%s task_ids=%s",
        len(work_items), [item["task_id"] for item in work_items],
    )
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
        logger.info(
            "event=graph_branch_started function=run_assessment_branch "
            "node=run_assessment_branch task_id=%s candidate_id=%s dimension_number=%s",
            work_item.task_id, work_item.candidate_id,
            work_item.dimension_number,
        )
        try:
            result = branch_worker(work_item, runtime.context)
        except TimeoutError as exc:
            logger.error(
                "event=graph_branch_failed function=run_assessment_branch "
                "node=run_assessment_branch task_id=%s candidate_id=%s "
                "dimension_number=%s error_code=branch_timeout error_type=%s",
                work_item.task_id, work_item.candidate_id,
                work_item.dimension_number, type(exc).__name__,
            )
            result = BranchEvidenceDraft(
                task_id=work_item.task_id,
                candidate_id=work_item.candidate_id,
                dimension_number=work_item.dimension_number,
                execution_status="failed",
                error_code="branch_timeout",
                error_message=str(exc),
            )
        except BranchExecutionError as exc:
            logger.error(
                "event=graph_branch_failed function=run_assessment_branch "
                "node=run_assessment_branch task_id=%s candidate_id=%s "
                "dimension_number=%s error_code=%s error_type=%s",
                work_item.task_id, work_item.candidate_id,
                work_item.dimension_number, exc.code, type(exc).__name__,
            )
            result = BranchEvidenceDraft(
                task_id=work_item.task_id,
                candidate_id=work_item.candidate_id,
                dimension_number=work_item.dimension_number,
                execution_status="failed",
                error_code=exc.code,
                error_message=str(exc),
            )
        except Exception as exc:
            logger.exception(
                "event=graph_branch_crashed function=run_assessment_branch "
                "node=run_assessment_branch task_id=%s candidate_id=%s "
                "dimension_number=%s error_type=%s",
                work_item.task_id, work_item.candidate_id,
                work_item.dimension_number, type(exc).__name__,
            )
            raise
        logger.info(
            "event=graph_branch_completed function=run_assessment_branch "
            "node=run_assessment_branch task_id=%s execution_status=%s "
            "requirement_count=%s",
            work_item.task_id, result.execution_status, len(result.requirements),
        )
        return {"branch_results": [result.model_dump(mode="json")]}

    return run_assessment_branch


def _finalize(state: TalentEvaluationDispatchState) -> dict[str, Any]:
    has_failures = any(
        item["execution_status"] == "failed"
        for item in state.get("branch_results", [])
    )
    status = "branches_ready_with_failures" if has_failures else "branches_ready"
    status_counts: dict[str, int] = {}
    for item in state.get("branch_results", []):
        execution_status = item["execution_status"]
        status_counts[execution_status] = status_counts.get(execution_status, 0) + 1
    logger.info(
        "event=graph_node_completed function=_finalize node=finalize "
        "status=%s branch_count=%s status_counts=%s",
        status, len(state.get("branch_results", [])), status_counts,
    )
    return {"status": status}


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
