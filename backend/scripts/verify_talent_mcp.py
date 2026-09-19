from __future__ import annotations

import json
from datetime import date
from pathlib import Path
import socket
import sys
from tempfile import TemporaryDirectory
from threading import Thread
import time

import anyio
import httpx2
from langchain_core.messages import AIMessage
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
import uvicorn

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.models import Base, EmployeeProfile, JobDescription  # noqa: E402
from app.talent_mcp_client import build_mcp_tool_graph  # noqa: E402
from app.talent_mcp_server import (  # noqa: E402
    authenticated_talent_context,
    build_talent_mcp_server,
)
from app.talent_tools import TalentToolContext, TalentToolService, ToolExecutor  # noqa: E402


class DemoTokenVerifier:
    def __init__(self, tokens: dict[str, AccessToken]) -> None:
        self.tokens = tokens

    async def verify_token(self, token: str) -> AccessToken | None:
        return self.tokens.get(token)


def build_demo_service(database_path: Path) -> TalentToolService:
    engine = create_engine(f"sqlite:///{database_path}")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        db.add_all(
            [
                JobDescription(
                    job_code="JD-AI-001",
                    tenant_id="tenant-a",
                    name="高级 AI 应用工程师",
                    content="负责企业知识库与 Agent 应用建设",
                    version=2,
                ),
                EmployeeProfile(
                    employee_no="C001",
                    tenant_id="tenant-a",
                    name="林晓岚",
                    birth_date=date(1993, 5, 1),
                    region="上海",
                    current_position="AI 应用工程师",
                    job_level="L3",
                    years_of_experience=7,
                    department="技术中心",
                ),
                EmployeeProfile(
                    employee_no="C002",
                    tenant_id="tenant-b",
                    name="其他租户候选人",
                    region="上海",
                    job_level="L4",
                    years_of_experience=9,
                ),
            ]
        )
        db.commit()

    def session_factory() -> Session:
        return Session(engine)

    return TalentToolService(session_factory=session_factory)


def demo_context() -> TalentToolContext:
    return TalentToolContext(
        tenant_id="tenant-a",
        permission_scopes=("hr_private",),
        actor_id="lesson-13",
        run_id="verify-13",
    )


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def verify_protocol(server) -> dict[str, object]:
    # async with Client(server) as client:
    #     print("[auto]", client.protocol_version)

    # async with Client(server, mode="legacy") as client:
    #     print("[legacy]", client.protocol_version)

    async with Client(server) as client:
        tools = await client.list_tools()
        templates = await client.list_resource_templates()
        resource = await client.read_resource("talent://jobs/JD-AI-001")
        prompt = await client.get_prompt(
            "talent_assessment",
            {"job_code": "JD-AI-001", "request_text": "筛选上海候选人"},
        )
        return {
            "protocol": client.protocol_version,
            "tools": [item.name for item in tools.tools],
            "resource_templates": [item.uri_template for item in templates.resource_templates],
            "resource_job_code": json.loads(resource.contents[0].text)["job_code"],
            "prompt_role": prompt.messages[0].role,
        }


async def verify_auth(service: TalentToolService) -> dict[str, int]:
    resource = "http://127.0.0.1:18081/mcp"
    common_claims = {"tenant_id": "tenant-a", "permission_scopes": ["hr_private"]}
    verifier = DemoTokenVerifier(
        {
            "missing-scope": AccessToken(
                token="missing-scope",
                client_id="talent-chat",
                scopes=[],
                resource=resource,
                claims=common_claims,
            ),
            "wrong-audience": AccessToken(
                token="wrong-audience",
                client_id="talent-chat",
                scopes=["talent:read"],
                resource="https://other.example.com/mcp",
                claims=common_claims,
            ),
            "true-audience": AccessToken(
                token="true-audience",
                client_id="talent-chat",
                scopes=["talent:read"],
                resource=resource,
                claims=common_claims,
            ),
        }
    )
    server = build_talent_mcp_server(
        service,
        ToolExecutor(),
        context_provider=authenticated_talent_context,
        token_verifier=verifier,
        auth=AuthSettings(
            issuer_url="https://auth.example.com", # 谁签发 token
            resource_server_url=resource, # token 发给谁，即 audience
            validate_token_resource=True,
            required_scopes=["talent:read"],
        ),
    )
    transport = httpx2.ASGITransport(app=server.streamable_http_app())
    true_audience_status: int | None = None

    async def capture_true_audience_response(response: httpx2.Response) -> None:
        nonlocal true_audience_status
        if response.request.headers.get("Mcp-Method") == "tools/list":
            true_audience_status = int(response.status_code)

    async with server.session_manager.run():
        async with httpx2.AsyncClient(transport=transport, base_url=resource) as client:
            missing = await client.post(".", json={})
            missing_scope = await client.post(
                ".",
                json={},
                headers={"Authorization": "Bearer missing-scope"},
            )
            wrong_audience = await client.post(
                ".",
                json={},
                headers={"Authorization": "Bearer wrong-audience"},
            )
        async with httpx2.AsyncClient(
            transport=transport,
            base_url=resource,
            headers={"Authorization": "Bearer true-audience"},
            event_hooks={"response": [capture_true_audience_response]},
        ) as authenticated_client:
            async with Client(
                streamable_http_client(resource, http_client=authenticated_client)
            ) as mcp_client:
                await mcp_client.list_tools()

    if true_audience_status is None:
        raise RuntimeError("正确 audience 的 MCP 能力发现没有产生 HTTP 响应")

    return {
        "missing_token": int(missing.status_code),
        "missing_scope": int(missing_scope.status_code),
        "wrong_audience": int(wrong_audience.status_code),
        "true_audience": true_audience_status,
    }


async def verify_langgraph(server) -> dict[str, object]:
    from langchain.mcp import MCPAdapter

    port = free_port()
    url = f"http://127.0.0.1:{port}/mcp"
    uvicorn_server = uvicorn.Server(
        uvicorn.Config(server.streamable_http_app(), host="127.0.0.1", port=port, log_level="error")
    )
    thread = Thread(target=uvicorn_server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not uvicorn_server.started and time.monotonic() < deadline:
        await anyio.sleep(0.05)
    if not uvicorn_server.started:
        raise RuntimeError("MCP HTTP 服务未在 10 秒内启动")

    try:
        async with MCPAdapter(url) as adapter:
            tools = await adapter.list_tools()
            graph = build_mcp_tool_graph(tools)
            mermaid = graph.get_graph().draw_mermaid()
            print(f"[graph mermaid]\n{mermaid}")
            result = await graph.ainvoke(
                {
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "filter_candidates",
                                    "args": {
                                        "filters": [
                                            {"field": "region", "operator": "eq", "value": "上海"}
                                        ]
                                    },
                                    "id": "verify-13",
                                    "type": "tool_call",
                                }
                            ],
                        )
                    ]
                }
            )
            tool_message = result["messages"][-1]
            payload = tool_message.artifact["structured_content"]
            return {
                "discovered_tools": [tool.name for tool in tools],
                "candidate_ids": payload["data"],
                "tool_status": tool_message.status,
            }
    finally:
        uvicorn_server.should_exit = True
        thread.join(timeout=10)


async def main() -> None:
    with TemporaryDirectory(prefix="talent-mcp-") as directory:
        service = build_demo_service(Path(directory) / "talent.db")
        server = build_talent_mcp_server(
            service,
            ToolExecutor(),
            context_provider=demo_context,
        )
        protocol = await verify_protocol(server)
        auth = await verify_auth(service)
        langgraph = await verify_langgraph(server)

    print("[protocol]", json.dumps(protocol, ensure_ascii=False))
    print("\n[auth]", json.dumps(auth, ensure_ascii=False))
    print("\n[langgraph]", json.dumps(langgraph, ensure_ascii=False))


if __name__ == "__main__":
    anyio.run(main)
