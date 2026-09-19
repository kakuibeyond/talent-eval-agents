from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any

from mcp.server import MCPServer
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver.exceptions import ResourceNotFoundError
from pydantic import Field

from app.query_plan import FilterCondition
from app.talent_tools import TalentToolContext, TalentToolService, ToolExecutor


ContextProvider = Callable[[], TalentToolContext]


def talent_context_from_token(token: AccessToken) -> TalentToolContext:
    claims = token.claims or {}
    tenant_id = str(claims.get("tenant_id", "")).strip()
    raw_permission_scopes = claims.get("permission_scopes", [])
    if not isinstance(raw_permission_scopes, (list, tuple)):
        raise PermissionError("Access Token 的 permission_scopes 必须是列表")
    permission_scopes = tuple(
        str(item).strip() for item in raw_permission_scopes if str(item).strip()
    )
    if not tenant_id:
        raise PermissionError("Access Token 缺少 tenant_id")
    if not permission_scopes:
        raise PermissionError("Access Token 缺少 permission_scopes")
    return TalentToolContext(
        tenant_id=tenant_id,
        permission_scopes=permission_scopes,
        actor_id=token.subject or token.client_id,
        run_id=str(claims.get("run_id", "unknown")),
    )


def authenticated_talent_context() -> TalentToolContext:
    token = get_access_token()
    if token is None:
        raise PermissionError("缺少经过验证的 MCP Access Token")
    return talent_context_from_token(token)


def build_talent_mcp_server(
    service: TalentToolService,
    executor: ToolExecutor,
    *,
    context_provider: ContextProvider,
    token_verifier: TokenVerifier | None = None,
    auth: AuthSettings | None = None,
) -> MCPServer:
    server = MCPServer(
        "Talent Capability Service",
        version="13.1.0",
        instructions="提供受租户与权限约束的人才检索和证据读取能力。",
        token_verifier=token_verifier,
        auth=auth,
    )

    @server.tool(structured_output=True)
    def lookup_job_descriptions(
        query: Annotated[str, Field(min_length=1, max_length=200)],
        limit: Annotated[int, Field(ge=1, le=10)] = 5,
    ) -> dict[str, Any]:
        """按岗位名称查询企业已经维护的岗位 JD。"""
        context = context_provider()
        return executor.execute(
            "lookup_job_descriptions",
            lambda: service.lookup_job_descriptions(query, limit=limit, context=context),
            context=context,
            arguments={"query": query, "limit": limit},
        )

    @server.tool(structured_output=True)
    def filter_candidates(
        filters: Annotated[list[FilterCondition], Field(max_length=20)],
    ) -> dict[str, Any]:
        """按结构化条件筛选授权租户内的候选人。"""
        context = context_provider()
        return executor.execute(
            "filter_candidates",
            lambda: service.filter_candidates(filters, context=context),
            context=context,
            arguments={"filters": filters},
        )

    @server.tool(structured_output=True)
    def search_candidate_evidence(
        query: Annotated[str, Field(min_length=1, max_length=500)],
        candidate_ids: Annotated[list[str], Field(min_length=1, max_length=100)],
    ) -> dict[str, Any]:
        """在候选人范围内检索材料证据并返回 Candidate Evidence Pack 2.0。"""
        context = context_provider()
        return executor.execute(
            "search_candidate_evidence",
            lambda: service.search_candidate_evidence(query, candidate_ids, context=context),
            context=context,
            arguments={"query": query, "candidate_ids": candidate_ids},
        )

    @server.tool(structured_output=True)
    def get_candidate_profiles(
        candidate_ids: Annotated[list[str], Field(min_length=1, max_length=100)],
    ) -> dict[str, Any]:
        """批量读取授权租户内候选人的结构化基础信息。"""
        context = context_provider()
        return executor.execute(
            "get_candidate_profiles",
            lambda: service.get_candidate_profiles(candidate_ids, context=context),
            context=context,
            arguments={"candidate_ids": candidate_ids},
        )

    @server.resource(
        "talent://jobs/{job_code}",
        title="岗位 JD",
        mime_type="application/json",
    )
    def job_description(job_code: str) -> dict[str, Any]:
        item = service.get_job_description(job_code, context=context_provider())
        if item is None:
            raise ResourceNotFoundError("岗位 JD 不存在或当前身份无权访问")
        return item

    @server.resource(
        "talent://candidates/{candidate_id}/profile",
        title="候选人基础档案",
        mime_type="application/json",
    )
    def candidate_profile(candidate_id: str) -> dict[str, Any]:
        items = service.get_candidate_profiles([candidate_id], context=context_provider())
        if not items:
            raise ResourceNotFoundError("候选人不存在或当前身份无权访问")
        return items[0]

    @server.resource(
        "talent://policies/evaluation/current",
        title="人才评估规则",
        mime_type="text/markdown",
    )
    def evaluation_policy() -> str:
        return "# 人才评估规则\n\n所有判断必须引用授权范围内的候选人材料。"

    @server.prompt(name="talent_assessment", title="人才评估与推荐")
    def talent_assessment(job_code: str, request_text: str) -> str:
        return f"按照岗位 {job_code} 处理人才评估与推荐请求：{request_text}"

    return server
