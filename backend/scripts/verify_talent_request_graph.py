from __future__ import annotations

import json

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.query_plan import FilterCondition, SemanticRequirement
from app.talent_decision_graph import DecisionContext
from app.talent_request_graph import TalentRequestDraft, build_talent_request_graph


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _job(code: str, name: str, match_type: str, content: str) -> dict:
    return {
        "job_code": code,
        "name": name,
        "match_score": 1.0,
        "match_type": match_type,
        "version": 1,
        "content": content,
    }


class DemoInterpreter:
    def __call__(self, request_text: str, selected_job: dict | None = None) -> TalentRequestDraft:
        if selected_job is not None:
            return TalentRequestDraft(
                input_mode="detailed_requirement",
                job_query=selected_job["name"],
                semantic_requirements=[
                    SemanticRequirement(
                        requirement_id="S1",
                        query=selected_job["content"],
                        required=True,
                    )
                ],
            )
        if "上海" in request_text:
            return TalentRequestDraft(
                input_mode="detailed_requirement",
                filters=[FilterCondition(field="region", operator="eq", value="上海")],
                semantic_requirements=[
                    SemanticRequirement(
                        requirement_id="S1",
                        query="企业知识库经验",
                        required=True,
                    )
                ],
                evaluation_preferences=["有 Agent 评测经验优先"],
            )
        return TalentRequestDraft(input_mode="job_name", job_query=request_text)


def main() -> None:
    lookup_calls: list[str] = []
    jobs = [
        _job(
            "JD-AI-001",
            "高级 AI 应用工程师",
            "contains",
            "负责企业知识库、Agent 应用与评测体系建设",
        ),
        _job(
            "JD-AI-002",
            "AI 应用工程师（平台方向）",
            "contains",
            "负责模型服务、Agent 平台与可观测性建设",
        ),
    ]

    def lookup(query: str, context: DecisionContext) -> list[dict]:
        lookup_calls.append(query)
        if query == "高级 AI 应用工程师":
            return [{**jobs[0], "match_type": "exact"}]
        if query == "AI 应用工程师":
            return jobs
        return []

    graph = build_talent_request_graph(
        request_interpreter=DemoInterpreter(),
        job_lookup=lookup,
        checkpointer=InMemorySaver(),
    )
    context = DecisionContext(tenant_id="course-demo", permission_scopes=("hr_private",))

    detailed = graph.invoke(
        {"request_text": "筛选上海且有企业知识库经验的人才，有 Agent 评测经验优先"},
        {"configurable": {"thread_id": "lesson14-detailed"}},
        context=context,
    )
    print("\n[detailed]\n", _json({"status": detailed["status"], "plan": detailed["query_plan"]}))

    exact = graph.invoke(
        {"request_text": "高级 AI 应用工程师"},
        {"configurable": {"thread_id": "lesson14-exact"}},
        context=context,
    )
    print(
        "\n[exact]\n",
        _json(
            {
                "status": exact["status"],
                "selected_job": exact["selected_job"]["job_code"],
            }
        ),
    )

    config = {"configurable": {"thread_id": "lesson14-ambiguous"}}
    interrupted = graph.invoke({"request_text": "AI 应用工程师"}, config, context=context)
    interrupt_payload = interrupted["__interrupt__"][0].value
    print(
        "\n[interrupt]\n",
        _json(
            {
                "thread_id": config["configurable"]["thread_id"],
                "next": graph.get_state(config).next,
                "payload": interrupt_payload,
            }
        ),
    )

    resumed = graph.invoke(
        Command(resume={"action": "select", "job_code": "JD-AI-002"}),
        config,
        context=context,
    )
    print(
        "\n[resume]\n",
        _json(
            {
                "status": resumed["status"],
                "selected_job": resumed["selected_job"]["job_code"],
                "target_job": resumed["talent_request"]["target_job"],
                "next": graph.get_state(config).next,
                "lookup_calls": lookup_calls.count("AI 应用工程师"),
            }
        ),
    )


if __name__ == "__main__":
    main()
