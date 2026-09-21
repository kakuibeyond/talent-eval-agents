from __future__ import annotations

import logging

from app.database import SessionLocal
from app.evidence_citations import load_pack_sources
from app.evidence_pack import model_extractor
from app.evidence_index_service import get_evidence_store
from app.hybrid_search_service import hybrid_search_evidence
from app.milvus_store import EvidenceFilter
from app.model_provider import get_chat_model, get_embedding_model
from app.reranker import get_reranker
from app.talent_evaluation_dispatch import (
    build_branch_evidence_provider,
    build_evidence_branch_worker,
    build_query_plan_candidate_provider,
    build_structured_dimension_generator,
    build_talent_evaluation_dispatch_graph,
)
from app.talent_evaluation_report import (
    build_structured_consistency_reviewer,
    build_structured_dimension_scorer,
    build_structured_report_writer,
    build_talent_evaluation_report_graph,
)
from app.talent_tools import TalentToolService, ToolExecutor, ToolPolicy

logger = logging.getLogger(__name__)


def build_runtime_graph():
    logger.info(
        "event=runtime_graph_build_started function=build_runtime_graph"
    )
    model = get_chat_model(temperature=0)
    embedder = get_embedding_model()
    reranker = get_reranker()
    if model is None:
        raise RuntimeError("模型未配置，请先设置项目 .env 中的 DASHSCOPE_API_KEY")
    if embedder is None:
        raise RuntimeError("Embedding 服务未配置")
    if reranker is None:
        raise RuntimeError("Rerank 服务未配置")
    store = get_evidence_store()

    def search(*, query, candidate_ids, tenant_id, permission_scopes):
        logger.info(
            "event=hybrid_search_started function=search candidate_ids=%s "
            "query_chars=%s permission_scope_count=%s limit=8 rerank_top_n=20",
            candidate_ids, len(query), len(permission_scopes),
        )
        results = hybrid_search_evidence(
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
        logger.info(
            "event=hybrid_search_completed function=search candidate_ids=%s "
            "hit_count=%s chunk_ids=%s",
            candidate_ids, len(results), [item.chunk_id for item in results],
        )
        return results

    def load_sources(chunks, *, tenant_id, permission_scopes):
        logger.info(
            "event=source_loading_started function=load_sources "
            "requested_chunk_count=%s chunk_ids=%s permission_scope_count=%s",
            len(chunks), [item["chunk_id"] for item in chunks],
            len(permission_scopes),
        )
        with SessionLocal() as db:
            sources = load_pack_sources(
                db,
                chunks,
                tenant_id=tenant_id,
                permission_scopes=permission_scopes,
            )
        logger.info(
            "event=source_loading_completed function=load_sources "
            "requested_chunk_count=%s loaded_source_count=%s loaded_chunk_ids=%s",
            len(chunks), len(sources), [item["chunk_id"] for item in sources],
        )
        return sources

    service = TalentToolService(
        session_factory=SessionLocal,
        evidence_provider=build_branch_evidence_provider(
            search=search,
            load_sources=load_sources,
            extract=model_extractor(model),
        ),
    )
    runtime_graph = build_talent_evaluation_dispatch_graph(
        dimension_generator=build_structured_dimension_generator(lambda: model),
        candidate_provider=build_query_plan_candidate_provider(service),
        branch_worker=build_evidence_branch_worker(
            service,
            ToolExecutor(
                tool_policies={
                    "search_candidate_evidence": ToolPolicy(
                        max_attempts=1,
                        timeout_seconds=90,
                    )
                }
            ),
        ),
        max_work_items=24,
    )
    logger.info(
        "event=runtime_graph_build_completed function=build_runtime_graph "
        "max_work_items=24 evidence_timeout_seconds=90"
    )
    return runtime_graph


def build_report_runtime_graph():
    logger.info(
        "event=report_runtime_graph_build_started "
        "function=build_report_runtime_graph"
    )
    model = get_chat_model(temperature=0)
    if model is None:
        raise RuntimeError("模型未配置，请先设置项目 .env 中的 DASHSCOPE_API_KEY")
    runtime_graph = build_talent_evaluation_report_graph(
        dimension_scorer=build_structured_dimension_scorer(lambda: model),
        consistency_reviewer=build_structured_consistency_reviewer(lambda: model),
        report_writer=build_structured_report_writer(lambda: model),
        max_scoring_concurrency=12,
        max_consistency_concurrency=12,
    )
    logger.info(
        "event=report_runtime_graph_build_completed "
        "function=build_report_runtime_graph"
    )
    return runtime_graph


# Agent Server loads this compiled graph through langgraph.json.
graph = build_runtime_graph()
report_graph = build_report_runtime_graph()
