from dataclasses import replace

import pytest
from langchain_core.messages import HumanMessage
from langgraph.errors import InvalidUpdateError
from langgraph.graph import END, START, StateGraph

from app.talent_decision_graph import (
    DecisionContext,
    build_missing_reducer_demo,
    build_reducer_demo,
    build_talent_decision_graph,
)
from scripts.verify_talent_decision_graph import _fixture_candidate_provider


def _context() -> DecisionContext:
    return DecisionContext(tenant_id="course-demo", permission_scopes=("hr_private",))


def test_fixture_provider_returns_candidates_only_for_ai_request():
    context = _context()

    assert _fixture_candidate_provider({"original_text": "筛选 AI 工程师"}, context) == ["C001", "C004"]
    assert _fixture_candidate_provider({"original_text": "筛选 Java 工程师"}, context) == []


def test_normal_request_reaches_completed_report():
    graph = build_talent_decision_graph(candidate_provider=lambda request, context: ["C001", "C004"])

    result = graph.invoke(
        {
            "messages": [HumanMessage(content="筛选有 AI 项目经验的技术负责人")],
            "request_text": "筛选有 AI 项目经验的技术负责人",
        },
        context=_context(),
    )

    assert result["status"] == "completed"
    assert result["candidate_ids"] == ["C001", "C004"]
    assert [item["candidate_id"] for item in result["evaluations"]] == ["C001", "C004"]
    assert "C001" in result["report"]
    assert result["errors"] == []


def test_empty_candidate_set_uses_explicit_terminal_branch():
    graph = build_talent_decision_graph(candidate_provider=lambda request, context: [])

    result = graph.invoke(
        {"messages": [], "request_text": "筛选 Java 工程师"},
        context=_context(),
    )

    assert result["status"] == "no_candidates"
    assert result["candidate_ids"] == []
    assert result["evaluations"] == []
    assert result["report"] == "当前条件下未检索到候选人。"


def test_runtime_context_is_required_for_trusted_tenant_data():
    graph = build_talent_decision_graph(candidate_provider=lambda request, context: ["C001"])

    with pytest.raises(ValueError, match="tenant_id"):
        graph.invoke(
            {"messages": [], "request_text": "筛选 AI 工程师"},
            context=replace(_context(), tenant_id=""),
        )


def test_parallel_updates_without_reducer_raise_invalid_update():
    graph = build_missing_reducer_demo()

    with pytest.raises(InvalidUpdateError):
        graph.invoke({"items": []})


def test_parallel_updates_with_reducer_are_aggregated():
    graph = build_reducer_demo()
    print(graph.get_graph().draw_mermaid())

    result = graph.invoke({"items": []})

    assert sorted(result["items"]) == ["A", "B"]
