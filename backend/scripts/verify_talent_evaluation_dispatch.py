from __future__ import annotations

import argparse
import json

from app.database import SessionLocal
from app.evidence_citations import load_pack_sources
from app.evidence_index_service import get_evidence_store
from app.evidence_pack import build_evidence_packs, model_extractor
from app.hybrid_search_service import hybrid_search_evidence
from app.milvus_store import EvidenceFilter
from app.model_provider import get_chat_model, get_embedding_model
from app.reranker import get_reranker
from app.talent_decision_graph import DecisionContext
from app.talent_evaluation_dispatch import (
    build_agent_branch_worker,
    build_evaluation_branch_agent,
    build_query_plan_candidate_provider,
    build_structured_dimension_generator,
    build_talent_evaluation_dispatch_graph,
)
from app.talent_tools import TalentToolService, ToolExecutor, build_talent_tools


def build_real_evidence_provider(model):
    embedder = get_embedding_model()
    reranker = get_reranker()
    if embedder is None:
        raise RuntimeError("Embedding 服务未配置")
    if reranker is None:
        raise RuntimeError("Rerank 服务未配置")
    store = get_evidence_store()
    extract = model_extractor(model)

    def evidence_provider(
        *,
        query: str,
        candidate_ids: list[str],
        tenant_id: str,
        permission_scopes: list[str],
        include_evidence_pack: bool,
    ) -> dict:
        ranked = hybrid_search_evidence(
            query=query,
            filters=EvidenceFilter(
                tenant_id=tenant_id,
                permission_scopes=permission_scopes,
                candidate_ids=candidate_ids,
            ),
            store=store,
            embedder=embedder,
            reranker=reranker,
            limit=8,
            rerank_top_n=20,
        )
        chunks = [
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
        result = {
            "candidate_ids": candidate_ids,
            "chunks": chunks,
            "evidence_packs": [],
        }
        if not include_evidence_pack:
            return result
        with SessionLocal() as db:
            sources = load_pack_sources(
                db,
                chunks,
                tenant_id=tenant_id,
                permission_scopes=permission_scopes,
            )
        result["evidence_packs"] = build_evidence_packs(
            candidate_ids=candidate_ids,
            requirements=[
                {
                    "requirement_id": "branch_requirement",
                    "query": query,
                }
            ],
            sources=sources,
            extract=extract,
        )
        return result

    return evidence_provider


def build_graph():
    model = get_chat_model(temperature=0)
    if model is None:
        raise RuntimeError("模型未配置，请先设置项目 .env 中的 DASHSCOPE_API_KEY")
    service = TalentToolService(
        session_factory=SessionLocal,
        evidence_provider=build_real_evidence_provider(model),
    )
    tools = build_talent_tools(service, ToolExecutor())
    branch_agent = build_evaluation_branch_agent(model, tools)
    return build_talent_evaluation_dispatch_graph(
        dimension_generator=build_structured_dimension_generator(lambda: model),
        candidate_provider=build_query_plan_candidate_provider(service),
        branch_worker=build_agent_branch_worker(branch_agent),
        max_work_items=24,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="验证第 15 课动态维度与 Subagent 分发链路")
    parser.add_argument("--tenant-id", default="course-demo")
    parser.add_argument("--region", default="上海")
    parser.add_argument("--max-concurrency", type=int, default=3)
    args = parser.parse_args()

    graph = build_graph()
    result = graph.invoke(
        {
            "talent_request": {
                "original_text": f"筛选{args.region}且具备 RAG 落地与 Agent 评测经验的人才，协作交付经验优先",
                "source": "detailed_requirement",
                "hard_conditions": [
                    {"field": "region", "operator": "eq", "value": args.region}
                ],
                "semantic_conditions": [
                    {"requirement_id": "S1", "query": "RAG 项目落地经验", "required": True},
                    {"requirement_id": "S2", "query": "Agent 评测经验", "required": True},
                ],
                "evaluation_preferences": ["协作与交付经验优先"],
            },
            "query_plan": {
                "task_type": "find_talent",
                "filters": [
                    {"field": "region", "operator": "eq", "value": args.region}
                ],
            },
        },
        config={"max_concurrency": args.max_concurrency},
        context=DecisionContext(
            tenant_id=args.tenant_id,
            permission_scopes=("hr_private",),
        ),
    )

    dimensions = [
        {
            "dimension_id": item["dimension_id"],
            "weight_percent": item["weight_percent"],
            "retrieval_hints": item["retrieval_hints"],
        }
        for item in result["dimensions"]
    ]
    status_counts: dict[str, int] = {}
    for item in result["branch_results"]:
        status_counts[item["status"]] = status_counts.get(item["status"], 0) + 1
    failed = [
        {
            "task_id": item["task_id"],
            "error_code": item["error_code"],
        }
        for item in result["branch_results"]
        if item["status"] == "failed"
    ]

    print("[dimensions] " + json.dumps(dimensions, ensure_ascii=False))
    print(
        "[dispatch] "
        + json.dumps(
            {
                "candidates": result["candidate_ids"],
                "work_items": len(result["work_items"]),
                "max_concurrency": args.max_concurrency,
            },
            ensure_ascii=False,
        )
    )
    print(
        "[result] "
        + json.dumps(
            {
                "status": result["status"],
                "counts": status_counts,
                "failed": failed,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
