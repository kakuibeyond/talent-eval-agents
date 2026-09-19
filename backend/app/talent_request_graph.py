from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import interrupt
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from app.database import SessionLocal
from app.model_provider import get_chat_model
from app.query_plan import Clarification, FilterCondition, QueryPlan, SemanticRequirement, TaskType
from app.talent_decision_graph import DecisionContext
from app.talent_tools import TalentToolContext, TalentToolService


RequestInputMode = Literal["detailed_requirement", "job_name"]
RequestStatus = Literal[
    "received",
    "request_understood",
    "jobs_found",
    "job_confirmed",
    "plan_ready",
    "clarification_required",
]


class JobMatch(TypedDict):
    job_code: str
    name: str
    match_score: float
    match_type: Literal["exact", "contains", "fuzzy"]
    version: int
    content: str


class TargetJob(TypedDict):
    job_code: str
    name: str
    version: int


class TalentRequestDraft(BaseModel):
    input_mode: RequestInputMode
    job_query: str | None = None
    filters: list[FilterCondition] = Field(default_factory=list)
    semantic_requirements: list[SemanticRequirement] = Field(default_factory=list)
    evaluation_preferences: list[str] = Field(default_factory=list)
    clarifications: list[Clarification] = Field(default_factory=list)


class TalentRequestInput(TypedDict):
    request_text: str


class TalentRequestOutput(TypedDict):
    status: RequestStatus
    input_mode: RequestInputMode
    job_matches: list[JobMatch]
    selected_job: JobMatch | None
    talent_request: dict[str, Any]
    query_plan: dict[str, Any]
    clarifications: list[dict[str, str]]
    errors: list[str]


class TalentRequestState(TypedDict, total=False):
    request_text: str
    input_mode: RequestInputMode
    draft: dict[str, Any]
    job_matches: list[JobMatch]
    selected_job: JobMatch | None
    talent_request: dict[str, Any]
    query_plan: dict[str, Any]
    clarifications: list[dict[str, str]]
    status: RequestStatus
    errors: list[str]


RequestInterpreter = Callable[[str, JobMatch | None], TalentRequestDraft]
JobLookup = Callable[[str, DecisionContext], list[JobMatch]]


def _receive_request(state: TalentRequestState, runtime: Runtime[DecisionContext]) -> dict[str, Any]:
    if not runtime.context.tenant_id.strip():
        raise ValueError("Runtime Context 中的 tenant_id 不能为空")
    if not runtime.context.permission_scopes:
        raise ValueError("Runtime Context 中的 permission_scopes 不能为空")
    request_text = state["request_text"].strip()
    if not request_text:
        raise ValueError("request_text 不能为空")
    return {
        "request_text": request_text,
        "job_matches": [],
        "selected_job": None,
        "talent_request": {},
        "query_plan": {},
        "clarifications": [],
        "status": "received",
        "errors": [],
    }


def _interpret_input(request_interpreter: RequestInterpreter):
    def interpret_input(state: TalentRequestState) -> dict[str, Any]:
        draft = request_interpreter(state["request_text"], None)
        return {
            "draft": draft.model_dump(mode="json"),
            "input_mode": draft.input_mode,
            "status": "request_understood",
        }

    return interpret_input


def _route_input(state: TalentRequestState) -> Literal["lookup_jobs", "build_plan"]:
    return "lookup_jobs" if state["input_mode"] == "job_name" else "build_plan"


def _lookup_jobs(job_lookup: JobLookup):
    def lookup_jobs(state: TalentRequestState, runtime: Runtime[DecisionContext]) -> dict[str, Any]:
        job_query = str(state["draft"].get("job_query") or state["request_text"])
        matches = job_lookup(job_query, runtime.context)
        return {"job_matches": matches, "status": "jobs_found"}

    return lookup_jobs


def _route_job_matches(
    state: TalentRequestState,
) -> Literal["select_exact_job", "confirm_job", "no_job_match"]:
    exact_matches = [item for item in state["job_matches"] if item["match_type"] == "exact"]
    if len(exact_matches) == 1:
        return "select_exact_job"
    if state["job_matches"]:
        return "confirm_job"
    return "no_job_match"


def _select_exact_job(state: TalentRequestState) -> dict[str, Any]:
    selected = next(item for item in state["job_matches"] if item["match_type"] == "exact")
    return {"selected_job": selected, "status": "job_confirmed"}


def _confirm_job(state: TalentRequestState) -> dict[str, Any]:
    selection = interrupt(
        {
            "type": "job_selection",
            "question": "请选择本次人才评估使用的岗位 JD",
            "request_text": state["request_text"],
            "options": [
                {
                    "job_code": item["job_code"],
                    "name": item["name"],
                    "match_score": item["match_score"],
                    "match_type": item["match_type"],
                    "version": item["version"],
                }
                for item in state["job_matches"]
            ],
        }
    )
    if not isinstance(selection, dict) or selection.get("action") != "select":
        raise ValueError("恢复数据必须包含 action=select")
    selected_code = selection.get("job_code")
    selected = next(
        (item for item in state["job_matches"] if item["job_code"] == selected_code),
        None,
    )
    if selected is None:
        raise ValueError("恢复数据中的 job_code 不在待确认岗位列表中")
    return {"selected_job": selected, "status": "job_confirmed"}


def _compile_selected_job(request_interpreter: RequestInterpreter):
    def compile_selected_job(state: TalentRequestState) -> dict[str, Any]:
        draft = request_interpreter(state["request_text"], state["selected_job"])
        return {
            "draft": draft.model_dump(mode="json"),
            "input_mode": "job_name",
        }

    return compile_selected_job


def _build_plan(state: TalentRequestState) -> dict[str, Any]:
    draft = TalentRequestDraft.model_validate(state["draft"])
    selected_job = state.get("selected_job")
    target_job: TargetJob | None = None
    if selected_job is not None:
        target_job = {
            "job_code": selected_job["job_code"],
            "name": selected_job["name"],
            "version": selected_job["version"],
        }

    talent_request = {
        "original_text": state["request_text"],
        "task_type": "evaluate_and_recommend",
        "source": state["input_mode"],
        "target_job": target_job,
        "hard_conditions": [item.model_dump(mode="json") for item in draft.filters],
        "semantic_conditions": [
            item.model_dump(mode="json") for item in draft.semantic_requirements
        ],
        "evaluation_preferences": draft.evaluation_preferences,
    }
    query_plan = QueryPlan(
        task_type=TaskType.FIND_TALENT,
        filters=draft.filters,
        semantic_requirements=draft.semantic_requirements,
        preferences=draft.evaluation_preferences,
        clarifications=draft.clarifications,
    )
    status: RequestStatus = "plan_ready" if query_plan.executable else "clarification_required"
    return {
        "talent_request": talent_request,
        "query_plan": query_plan.model_dump(mode="json"),
        "clarifications": [item.model_dump(mode="json") for item in draft.clarifications],
        "status": status,
    }


def _no_job_match(state: TalentRequestState) -> dict[str, Any]:
    clarification = {
        "expression": state["request_text"],
        "reason": "未找到可确认的岗位 JD",
    }
    return {
        "query_plan": {},
        "talent_request": {},
        "clarifications": [clarification],
        "status": "clarification_required",
    }


def build_talent_request_graph(
    *,
    request_interpreter: RequestInterpreter,
    job_lookup: JobLookup,
    checkpointer: Any | None = None,
):
    builder = StateGraph(
        TalentRequestState,
        context_schema=DecisionContext,
        input_schema=TalentRequestInput,
        output_schema=TalentRequestOutput,
    )
    builder.add_node("receive_request", _receive_request)
    builder.add_node("interpret_input", _interpret_input(request_interpreter))
    builder.add_node("lookup_jobs", _lookup_jobs(job_lookup))
    builder.add_node("select_exact_job", _select_exact_job)
    builder.add_node("confirm_job", _confirm_job)
    builder.add_node("compile_selected_job", _compile_selected_job(request_interpreter))
    builder.add_node("build_plan", _build_plan)
    builder.add_node("no_job_match", _no_job_match)
    builder.add_edge(START, "receive_request")
    builder.add_edge("receive_request", "interpret_input")
    builder.add_conditional_edges("interpret_input", _route_input)
    builder.add_conditional_edges("lookup_jobs", _route_job_matches)
    builder.add_edge("select_exact_job", "compile_selected_job")
    builder.add_edge("confirm_job", "compile_selected_job")
    builder.add_edge("compile_selected_job", "build_plan")
    builder.add_edge("build_plan", END)
    builder.add_edge("no_job_match", END)
    return builder.compile(checkpointer=checkpointer) if checkpointer is not None else builder.compile()


MODEL_SYSTEM_PROMPT = """你是人才评估任务编译器。输出 TalentRequestDraft。
如果输入只有岗位名称或岗位简称，input_mode 使用 job_name，只填写 job_query，不根据岗位名称猜测条件。
如果输入包含地区、年限、职级、技能、项目经验或偏好，input_mode 使用 detailed_requirement。
结构化硬条件只能写入 filters。技能、经历和成果写入 semantic_requirements。优先项写入 evaluation_preferences。
含糊、缺失阈值或互相矛盾的条件写入 clarifications。租户和权限不得从用户文本提取。
当输入中包含 confirmed_job 时，按照已确认岗位 JD 编译条件，并保留用户补充要求。"""


def model_request_interpreter(request_text: str, selected_job: JobMatch | None = None) -> TalentRequestDraft:
    model = get_chat_model(temperature=0)
    if model is None:
        raise RuntimeError("任务理解模型未配置，请设置 DASHSCOPE_API_KEY")
    user_content = request_text
    if selected_job is not None:
        user_content = (
            f"original_request={request_text}\n"
            f"confirmed_job={selected_job['name']}\n"
            f"job_description={selected_job['content']}"
        )
    return model.with_structured_output(TalentRequestDraft).invoke(
        [("system", MODEL_SYSTEM_PROMPT), ("user", user_content)]
    )


def database_job_lookup(query: str, context: DecisionContext) -> list[JobMatch]:
    service = TalentToolService(session_factory=SessionLocal)
    tool_context = TalentToolContext(
        tenant_id=context.tenant_id,
        permission_scopes=context.permission_scopes,
        actor_id="langsmith-studio",
        run_id="lesson-14",
    )
    return service.lookup_job_descriptions(query, context=tool_context)


# Agent Server injects its managed checkpointer when this graph is loaded from langgraph.json.
graph = build_talent_request_graph(
    request_interpreter=model_request_interpreter,
    job_lookup=database_job_lookup,
)
