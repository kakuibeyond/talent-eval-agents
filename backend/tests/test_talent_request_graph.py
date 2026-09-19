from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.query_plan import FilterCondition, SemanticRequirement
from app.talent_decision_graph import DecisionContext
from app.talent_request_graph import TalentRequestDraft, build_talent_request_graph


def _context() -> DecisionContext:
    return DecisionContext(tenant_id="course-demo", permission_scopes=("hr_private",))


def _job(
    code: str,
    name: str,
    *,
    match_type: str,
    score: float = 1.0,
    content: str = "负责企业知识库与 Agent 应用建设",
) -> dict:
    return {
        "job_code": code,
        "name": name,
        "match_score": score,
        "match_type": match_type,
        "version": 1,
        "content": content,
    }


class FixtureInterpreter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def __call__(self, request_text: str, selected_job: dict | None = None) -> TalentRequestDraft:
        selected_code = selected_job["job_code"] if selected_job else None
        self.calls.append((request_text, selected_code))
        if selected_job is not None:
            return TalentRequestDraft(
                input_mode="detailed_requirement",
                job_query=selected_job["name"],
                filters=[],
                semantic_requirements=[
                    SemanticRequirement(requirement_id="S1", query=selected_job["content"], required=True)
                ],
                evaluation_preferences=[],
                clarifications=[],
            )
        if "上海" in request_text:
            return TalentRequestDraft(
                input_mode="detailed_requirement",
                filters=[FilterCondition(field="region", operator="eq", value="上海")],
                semantic_requirements=[
                    SemanticRequirement(requirement_id="S1", query="企业知识库经验", required=True)
                ],
                evaluation_preferences=["有 Agent 评测经验优先"],
                clarifications=[],
            )
        return TalentRequestDraft(
            input_mode="job_name",
            job_query=request_text,
            filters=[],
            semantic_requirements=[],
            evaluation_preferences=[],
            clarifications=[],
        )


def test_detailed_requirement_builds_plan_without_job_lookup():
    interpreter = FixtureInterpreter()
    lookup_calls: list[str] = []

    graph = build_talent_request_graph(
        request_interpreter=interpreter,
        job_lookup=lambda query, context: lookup_calls.append(query) or [],
        checkpointer=InMemorySaver(),
    )
    result = graph.invoke(
        {"request_text": "筛选上海且有企业知识库经验的人才，有 Agent 评测经验优先"},
        {"configurable": {"thread_id": "lesson14-detailed"}},
        context=_context(),
    )

    assert result["status"] == "plan_ready"
    assert result["input_mode"] == "detailed_requirement"
    assert result["job_matches"] == []
    assert result["query_plan"]["filters"] == [
        {"field": "region", "operator": "eq", "value": "上海"}
    ]
    assert result["talent_request"]["evaluation_preferences"] == ["有 Agent 评测经验优先"]
    assert lookup_calls == []


def test_unique_exact_job_match_continues_without_interrupt():
    interpreter = FixtureInterpreter()
    graph = build_talent_request_graph(
        request_interpreter=interpreter,
        job_lookup=lambda query, context: [
            _job("JD-AI-001", "高级 AI 应用工程师", match_type="exact"),
            _job("JD-AI-002", "AI 平台工程师", match_type="fuzzy", score=0.61),
        ],
        checkpointer=InMemorySaver(),
    )

    result = graph.invoke(
        {"request_text": "高级 AI 应用工程师"},
        {"configurable": {"thread_id": "lesson14-exact"}},
        context=_context(),
    )

    assert result["status"] == "plan_ready"
    assert result["selected_job"]["job_code"] == "JD-AI-001"
    assert "__interrupt__" not in result
    assert interpreter.calls == [
        ("高级 AI 应用工程师", None),
        ("高级 AI 应用工程师", "JD-AI-001"),
    ]


def test_ambiguous_job_match_interrupts_and_resumes_same_thread():
    interpreter = FixtureInterpreter()
    lookup_calls: list[str] = []
    matches = [
        _job("JD-AI-001", "高级 AI 应用工程师", match_type="contains"),
        _job("JD-AI-002", "AI 应用工程师（平台方向）", match_type="contains"),
    ]

    def lookup(query: str, context: DecisionContext) -> list[dict]:
        lookup_calls.append(query)
        return matches

    graph = build_talent_request_graph(
        request_interpreter=interpreter,
        job_lookup=lookup,
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "lesson14-ambiguous"}}

    interrupted = graph.invoke({"request_text": "AI 应用工程师"}, config, context=_context())

    payload = interrupted["__interrupt__"][0].value
    assert payload["type"] == "job_selection"
    assert payload["request_text"] == "AI 应用工程师"
    assert [item["job_code"] for item in payload["options"]] == ["JD-AI-001", "JD-AI-002"]
    assert lookup_calls == ["AI 应用工程师"]

    resumed = graph.invoke(
        Command(resume={"action": "select", "job_code": "JD-AI-002"}),
        config,
        context=_context(),
    )

    assert resumed["status"] == "plan_ready"
    assert resumed["selected_job"]["job_code"] == "JD-AI-002"
    assert resumed["talent_request"]["target_job"]["job_code"] == "JD-AI-002"
    assert resumed["query_plan"]["semantic_requirements"][0]["query"] == "负责企业知识库与 Agent 应用建设"
    assert lookup_calls == ["AI 应用工程师"]


def test_no_job_match_returns_clarification_result_without_building_plan():
    graph = build_talent_request_graph(
        request_interpreter=FixtureInterpreter(),
        job_lookup=lambda query, context: [],
        checkpointer=InMemorySaver(),
    )

    result = graph.invoke(
        {"request_text": "量子招聘架构师"},
        {"configurable": {"thread_id": "lesson14-no-match"}},
        context=_context(),
    )

    assert result["status"] == "clarification_required"
    assert result["query_plan"] == {}
    assert result["clarifications"] == [
        {"expression": "量子招聘架构师", "reason": "未找到可确认的岗位 JD"}
    ]
