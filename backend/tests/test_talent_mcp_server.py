from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import subprocess
import sys

import anyio
import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.shared.exceptions import MCPError
from mcp.types import TextResourceContents
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.models import Base, EmployeeProfile, JobDescription
from app.talent_mcp_server import (
    authenticated_talent_context,
    build_talent_mcp_server,
    talent_context_from_token,
)
from app.talent_tools import TalentToolContext, TalentToolService, ToolExecutor
from scripts.verify_talent_mcp import build_demo_service, verify_auth


def _session_factory(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'talent-mcp.db'}")
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
                JobDescription(
                    job_code="JD-AI-OTHER",
                    tenant_id="tenant-b",
                    name="高级 AI 应用工程师",
                    content="其他租户数据",
                    version=1,
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
                    name="越权数据",
                    region="上海",
                    job_level="L4",
                    years_of_experience=9,
                ),
            ]
        )
        db.commit()

    def factory():
        return Session(engine)

    return factory


def _context() -> TalentToolContext:
    return TalentToolContext(
        tenant_id="tenant-a",
        permission_scopes=("hr_private",),
        actor_id="mcp-course-client",
        run_id="run-13",
    )


def _server(tmp_path):
    def evidence_provider(**kwargs):
        return {
            "candidate_ids": kwargs["candidate_ids"],
            "evidence_packs": [
                {"schema_version": "2.0", "candidate_id": candidate_id, "requirements": []}
                for candidate_id in kwargs["candidate_ids"]
            ],
        }

    service = TalentToolService(
        session_factory=_session_factory(tmp_path),
        evidence_provider=evidence_provider,
    )
    return build_talent_mcp_server(service, ToolExecutor(), context_provider=_context)


def test_server_advertises_talent_tools_resources_and_prompt(tmp_path):
    async def scenario():
        async with Client(_server(tmp_path)) as client:
            tools = await client.list_tools()
            resources = await client.list_resources()
            templates = await client.list_resource_templates()
            prompts = await client.list_prompts()
            return tools, resources, templates, prompts

    tools, resources, templates, prompts = anyio.run(scenario)

    assert {item.name for item in tools.tools} == {
        "lookup_job_descriptions",
        "filter_candidates",
        "search_candidate_evidence",
        "get_candidate_profiles",
    }
    assert {str(item.uri) for item in resources.resources} == {"talent://policies/evaluation/current"}
    assert {item.uri_template for item in templates.resource_templates} == {
        "talent://jobs/{job_code}",
        "talent://candidates/{candidate_id}/profile",
    }
    assert [item.name for item in prompts.prompts] == ["talent_assessment"]


def test_mcp_tools_reuse_existing_service_and_result_protocol(tmp_path):
    async def scenario():
        async with Client(_server(tmp_path)) as client:
            job = await client.call_tool("lookup_job_descriptions", {"query": "AI 应用工程师"})
            candidates = await client.call_tool(
                "filter_candidates",
                {"filters": [{"field": "region", "operator": "eq", "value": "上海"}]},
            )
            profiles = await client.call_tool("get_candidate_profiles", {"candidate_ids": ["C001", "C002"]})
            return job, candidates, profiles

    job, candidates, profiles = anyio.run(scenario)

    assert job.is_error is False
    assert job.structured_content["ok"] is True
    assert job.structured_content["data"][0]["job_code"] == "JD-AI-001"
    assert candidates.structured_content["data"] == ["C001"]
    assert [item["candidate_id"] for item in profiles.structured_content["data"]] == ["C001"]


def test_evidence_tool_preserves_candidate_evidence_pack_v2(tmp_path):
    async def scenario():
        async with Client(_server(tmp_path)) as client:
            return await client.call_tool(
                "search_candidate_evidence",
                {"query": "企业知识库经验", "candidate_ids": ["C001"]},
            )

    result = anyio.run(scenario)

    assert result.is_error is False
    assert result.structured_content["data"]["evidence_packs"][0]["schema_version"] == "2.0"


def test_resource_templates_apply_the_same_tenant_boundary(tmp_path):
    async def scenario():
        async with Client(_server(tmp_path)) as client:
            job = await client.read_resource("talent://jobs/JD-AI-001")
            profile = await client.read_resource("talent://candidates/C001/profile")
            with pytest.raises(MCPError):
                await client.read_resource("talent://jobs/JD-AI-OTHER")
            with pytest.raises(MCPError):
                await client.read_resource("talent://candidates/C002/profile")
            return job, profile

    job, profile = anyio.run(scenario)
    job_content = job.contents[0]
    profile_content = profile.contents[0]

    assert isinstance(job_content, TextResourceContents)
    assert json.loads(job_content.text)["job_code"] == "JD-AI-001"
    assert isinstance(profile_content, TextResourceContents)
    assert json.loads(profile_content.text)["candidate_id"] == "C001"


def test_prompt_renders_user_selected_talent_request(tmp_path):
    async def scenario():
        async with Client(_server(tmp_path)) as client:
            return await client.get_prompt(
                "talent_assessment",
                {"job_code": "JD-AI-001", "request_text": "筛选上海候选人"},
            )

    result = anyio.run(scenario)

    assert result.messages[0].role == "user"
    assert "JD-AI-001" in result.messages[0].content.text
    assert "筛选上海候选人" in result.messages[0].content.text


def test_access_token_claims_become_trusted_talent_context():
    token = AccessToken(
        token="course-token",
        client_id="talent-chat",
        subject="teacher-demo",
        scopes=["talent:read"],
        resource="http://127.0.0.1:18081/mcp",
        claims={
            "tenant_id": "tenant-a",
            "permission_scopes": ["hr_private"],
            "run_id": "run-13",
        },
    )

    context = talent_context_from_token(token)

    assert context == TalentToolContext(
        tenant_id="tenant-a",
        permission_scopes=("hr_private",),
        actor_id="teacher-demo",
        run_id="run-13",
    )


@pytest.mark.parametrize(
    ("claims", "message"),
    [
        ({"permission_scopes": ["hr_private"]}, "tenant_id"),
        ({"tenant_id": "tenant-a"}, "permission_scopes"),
        ({"tenant_id": "tenant-a", "permission_scopes": "hr_private"}, "permission_scopes"),
    ],
)
def test_access_token_requires_tenant_and_data_permissions(claims, message):
    token = AccessToken(
        token="course-token",
        client_id="talent-chat",
        scopes=["talent:read"],
        claims=claims,
    )

    with pytest.raises(PermissionError, match=message):
        talent_context_from_token(token)


class _TokenVerifier:
    def __init__(self, tokens):
        self.tokens = tokens

    async def verify_token(self, token: str):
        return self.tokens.get(token)


def _protected_server(tmp_path):
    resource = "http://127.0.0.1:18081/mcp"
    tokens = {
        "valid": AccessToken(
            token="valid",
            client_id="talent-chat",
            subject="teacher-demo",
            scopes=["talent:read"],
            resource=resource,
            claims={"tenant_id": "tenant-a", "permission_scopes": ["hr_private"], "run_id": "run-13"},
        ),
        "missing-scope": AccessToken(
            token="missing-scope",
            client_id="talent-chat",
            scopes=[],
            resource=resource,
            claims={"tenant_id": "tenant-a", "permission_scopes": ["hr_private"]},
        ),
        "wrong-audience": AccessToken(
            token="wrong-audience",
            client_id="talent-chat",
            scopes=["talent:read"],
            resource="https://other.example.com/mcp",
            claims={"tenant_id": "tenant-a", "permission_scopes": ["hr_private"]},
        ),
    }
    service = TalentToolService(session_factory=_session_factory(tmp_path))
    server = build_talent_mcp_server(
        service,
        ToolExecutor(),
        context_provider=authenticated_talent_context,
        token_verifier=_TokenVerifier(tokens),
        auth=AuthSettings(
            issuer_url="https://auth.example.com",
            resource_server_url=resource,
            validate_token_resource=True,
            required_scopes=["talent:read"],
        ),
    )
    return server, resource


def test_streamable_http_enforces_token_scope_and_audience(tmp_path):
    async def scenario():
        server, resource = _protected_server(tmp_path)
        app = server.streamable_http_app()
        transport = httpx2.ASGITransport(app=app)
        async with server.session_manager.run():
            async with httpx2.AsyncClient(transport=transport, base_url=resource) as http_client:
                missing = await http_client.post(".", json={})
                insufficient = await http_client.post(
                    ".", json={}, headers={"Authorization": "Bearer missing-scope"}
                )
                wrong_audience = await http_client.post(
                    ".", json={}, headers={"Authorization": "Bearer wrong-audience"}
                )
        return missing, insufficient, wrong_audience

    missing, insufficient, wrong_audience = anyio.run(scenario)

    assert missing.status_code == 401
    assert insufficient.status_code == 403
    assert wrong_audience.status_code == 401


def test_streamable_http_token_claims_reach_business_service(tmp_path):
    async def scenario():
        server, resource = _protected_server(tmp_path)
        app = server.streamable_http_app()
        transport = httpx2.ASGITransport(app=app)
        async with server.session_manager.run():
            async with httpx2.AsyncClient(
                transport=transport,
                base_url=resource,
                headers={"Authorization": "Bearer valid"},
            ) as http_client:
                async with Client(streamable_http_client(resource, http_client=http_client)) as client:
                    return await client.call_tool(
                        "get_candidate_profiles",
                        {"candidate_ids": ["C001", "C002"]},
                    )

    result = anyio.run(scenario)

    assert result.is_error is False
    assert [item["candidate_id"] for item in result.structured_content["data"]] == ["C001"]


def test_verify_auth_uses_valid_capability_discovery_for_true_audience(tmp_path):
    service = build_demo_service(tmp_path / "verify-auth.db")

    result = anyio.run(verify_auth, service)

    assert result == {
        "missing_token": 401,
        "missing_scope": 403,
        "wrong_audience": 401,
        "true_audience": 200,
    }


def test_demo_http_server_exposes_tools_over_streamable_http(tmp_path):
    from scripts.run_talent_mcp_http import build_demo_http_server

    async def scenario():
        server = build_demo_http_server(tmp_path / "talent-http.db")
        transport = httpx2.ASGITransport(app=server.streamable_http_app(stateless_http=True))
        async with server.session_manager.run():
            async with httpx2.AsyncClient(
                transport=transport,
                base_url="http://127.0.0.1:18081/mcp",
            ) as http_client:
                async with Client(
                    streamable_http_client(
                        "http://127.0.0.1:18081/mcp",
                        http_client=http_client,
                    )
                ) as client:
                    return await client.list_tools()

    result = anyio.run(scenario)

    assert {tool.name for tool in result.tools} == {
        "lookup_job_descriptions",
        "filter_candidates",
        "search_candidate_evidence",
        "get_candidate_profiles",
    }


def test_demo_http_server_script_can_run_directly():
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_talent_mcp_http.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--host" in result.stdout
    assert "--port" in result.stdout


def test_demo_http_server_stops_cleanly_on_keyboard_interrupt(monkeypatch, capsys):
    from scripts import run_talent_mcp_http

    class StoppedServer:
        def run(self, *args, **kwargs):
            raise KeyboardInterrupt

    monkeypatch.setattr(
        run_talent_mcp_http,
        "build_demo_http_server",
        lambda database_path: StoppedServer(),
    )

    run_talent_mcp_http.main([])

    assert "Talent MCP 已停止" in capsys.readouterr().out
