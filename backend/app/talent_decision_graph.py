from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Literal

from langchain.messages import AnyMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from langgraph.types import Command
from typing_extensions import TypedDict


DecisionStatus = Literal[
    "received",
    "request_ready",
    "candidates_ready",
    "drafting",
    "completed",
    "no_candidates",
    "failed",
]


class TalentRequest(TypedDict):
    original_text: str
    task_type: Literal["evaluate_and_recommend"]


class EvaluationResult(TypedDict):
    candidate_id: str
    status: Literal["placeholder"]


class TalentDecisionInput(TypedDict):
    messages: list[AnyMessage]
    request_text: str


class TalentDecisionOutput(TypedDict):
    status: DecisionStatus
    candidate_ids: list[str]
    evaluations: list[EvaluationResult]
    report: str
    errors: list[str]


class TalentDecisionState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    request_text: str
    request: TalentRequest
    candidate_ids: list[str]
    evidence_refs: Annotated[list[str], list.__add__]
    evaluations: Annotated[list[EvaluationResult], list.__add__]
    report: str
    status: DecisionStatus
    errors: Annotated[list[str], list.__add__]
    retry_count: int


@dataclass(frozen=True)
class DecisionContext:
    tenant_id: str
    permission_scopes: tuple[str, ...]


CandidateProvider = Callable[[TalentRequest, DecisionContext], list[str]]


def _receive_request(
    state: TalentDecisionState,
    runtime: Runtime[DecisionContext],
) -> Command[Literal["prepare_request"]]:
    if not runtime.context.tenant_id.strip():
        raise ValueError("Runtime Context 中的 tenant_id 不能为空")
    request_text = state["request_text"].strip()
    if not request_text:
        raise ValueError("request_text 不能为空")
    return Command(
        update={"request_text": request_text, "status": "received", "errors": []},
        goto="prepare_request",
    )


def _prepare_request(state: TalentDecisionState) -> dict:
    return {
        "request": {
            "original_text": state["request_text"],
            "task_type": "evaluate_and_recommend",
        },
        "status": "request_ready",
    }


def _candidate_node(candidate_provider: CandidateProvider):
    def retrieve_candidates(
        state: TalentDecisionState,
        runtime: Runtime[DecisionContext],
    ) -> dict:
        candidate_ids = candidate_provider(state["request"], runtime.context)
        return {"candidate_ids": candidate_ids, "status": "candidates_ready"}

    return retrieve_candidates


def _route_candidates(state: TalentDecisionState) -> Literal["evaluate", "no_candidates"]:
    return "evaluate" if state.get("candidate_ids") else "no_candidates"


def _evaluate(state: TalentDecisionState) -> dict:
    return {
        "evaluations": [
            {"candidate_id": candidate_id, "status": "placeholder"}
            for candidate_id in state["candidate_ids"]
        ]
    }


def _compose_report(state: TalentDecisionState) -> dict:
    candidate_text = "、".join(item["candidate_id"] for item in state["evaluations"])
    return {"report": f"候选人评估占位结果：{candidate_text}", "status": "drafting"}


def _validate_report(state: TalentDecisionState) -> dict:
    if state.get("report"):
        return {"status": "completed"}
    retries = state.get("retry_count", 0) + 1
    if retries > 1:
        return {"retry_count": retries, "status": "failed", "errors": ["报告为空"]}
    return {"retry_count": retries}


def _route_report(state: TalentDecisionState) -> Literal["retry", "done"]:
    if state.get("status") in {"completed", "failed"}:
        return "done"
    return "retry"


def _no_candidates(_: TalentDecisionState) -> dict:
    return {
        "evaluations": [],
        "report": "当前条件下未检索到候选人。",
        "status": "no_candidates",
    }


def build_talent_decision_graph(candidate_provider: CandidateProvider):
    builder = StateGraph(
        TalentDecisionState,
        context_schema=DecisionContext,
        input_schema=TalentDecisionInput,
        output_schema=TalentDecisionOutput,
    )
    builder.add_node("receive_request", _receive_request)
    builder.add_node("prepare_request", _prepare_request)
    builder.add_node("retrieve_candidates", _candidate_node(candidate_provider))
    builder.add_node("evaluate", _evaluate)
    builder.add_node("compose_report", _compose_report)
    builder.add_node("validate_report", _validate_report)
    builder.add_node("no_candidates", _no_candidates)
    builder.add_edge(START, "receive_request")
    builder.add_edge("prepare_request", "retrieve_candidates")
    builder.add_conditional_edges("retrieve_candidates", _route_candidates)
    builder.add_edge("evaluate", "compose_report")
    builder.add_edge("compose_report", "validate_report")
    builder.add_conditional_edges(
        "validate_report",
        _route_report,
        {"retry": "compose_report", "done": END},
    )
    builder.add_edge("no_candidates", END)
    return builder.compile()


class MissingReducerState(TypedDict):
    items: list[str]


def build_missing_reducer_demo():
    builder = StateGraph(MissingReducerState)
    builder.add_node("branch_a", lambda _: {"items": ["A"]})
    builder.add_node("branch_b", lambda _: {"items": ["B"]})
    builder.add_edge(START, "branch_a")
    builder.add_edge(START, "branch_b")
    builder.add_edge("branch_a", END)
    builder.add_edge("branch_b", END)
    return builder.compile()


class ReducerState(TypedDict):
    items: Annotated[list[str], list.__add__]


def build_reducer_demo():
    builder = StateGraph(ReducerState)
    builder.add_node("branch_a", lambda _: {"items": ["A"]})
    builder.add_node("branch_b", lambda _: {"items": ["B"]})
    builder.add_edge(START, "branch_a")
    builder.add_edge(START, "branch_b")
    builder.add_edge("branch_a", END)
    builder.add_edge("branch_b", END)
    return builder.compile()
