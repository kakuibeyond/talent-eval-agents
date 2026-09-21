from __future__ import annotations

import os
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.live_model


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_MODEL_TESTS") != "1",
    reason="set RUN_LIVE_MODEL_TESTS=1 to call the configured chat model",
)
def test_real_model_extraction_is_preserved_in_branch_evidence_pack():
    from app.evidence_pack import model_extractor
    from app.model_provider import get_chat_model
    from app.talent_decision_graph import DecisionContext
    from app.talent_evaluation_dispatch import (
        AssessmentWorkItem,
        build_branch_evidence_provider,
        build_evidence_branch_worker,
    )
    from app.talent_tools import TalentToolService, ToolExecutor, ToolPolicy

    model = get_chat_model(temperature=0)
    assert model is not None, "DASHSCOPE_API_KEY is required for the live model test"

    def search(**kwargs):
        del kwargs
        return [
            SimpleNamespace(
                chunk_id="chunk-rag-001",
                candidate_id="A0001",
                content="参与 RAG 知识库项目，负责文档切分与向量入库。",
                rerank_score=0.95,
                metadata={},
            )
        ]

    def load_sources(chunks, *, tenant_id, permission_scopes):
        del chunks, tenant_id, permission_scopes
        return [
            {
                "chunk_id": "chunk-rag-001",
                "citation_id": "chunk-rag-001",
                "candidate_id": "A0001",
                "content": "参与 RAG 知识库项目，负责文档切分与向量入库。",
                "document_title": "候选人项目材料",
                "requirement_ids": ["branch_requirement"],
            }
        ]

    service = TalentToolService(
        session_factory=lambda: None,
        evidence_provider=build_branch_evidence_provider(
            search=search,
            load_sources=load_sources,
            extract=model_extractor(model),
        ),
    )
    worker = build_evidence_branch_worker(
        service,
        ToolExecutor(
            tool_policies={
                "search_candidate_evidence": ToolPolicy(
                    max_attempts=1,
                    timeout_seconds=30,
                    backoff_seconds=0,
                )
            }
        ),
    )
    result = worker(
        AssessmentWorkItem.model_validate(
            {
                "task_id": "A0001:d1",
                "candidate_id": "A0001",
                "dimension": {
                    "dimension_id": "d1",
                    "name": "RAG 项目落地经验",
                    "definition": "评估候选人的 RAG 项目落地经验",
                    "weight_percent": 100,
                    "evidence_requirements": ["候选人的 RAG 项目职责"],
                    "score_anchors": [
                        {"score": 0, "description": "无相关事实"},
                        {"score": 3, "description": "有参与事实"},
                        {"score": 5, "description": "有主导与结果事实"},
                    ],
                    "retrieval_hints": ["RAG", "知识库", "项目职责"],
                    "source_requirement_ids": ["S1"],
                },
            }
        ),
        DecisionContext(
            tenant_id="course-demo",
            permission_scopes=("hr_private",),
        ),
    )

    print(result.model_dump_json(indent=2))
    assert result.execution_status == "succeeded", result.model_dump(mode="json")
    assert result.requirements[0].extraction_status == "succeeded"
    assert result.requirements[0].facts
    assert result.requirements[0].facts[0].sources[0].chunk_id == "chunk-rag-001"
    assert (
        result.requirements[0].facts[0].sources[0].quote
        == "参与 RAG 知识库项目，负责文档切分与向量入库。"
    )
    assert len(result.tool_call_ids) == 1
