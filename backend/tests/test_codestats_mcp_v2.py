from __future__ import annotations

import anyio
from mcp import Client

from samples.lesson13.codestats_mcp_v2 import build_server


def test_codestats_tool_counts_python_lines_and_skips_venv(tmp_path):
    project = tmp_path / "demo"
    project.mkdir()
    (project / "app.py").write_text("print('one')\nprint('two')\n", encoding="utf-8")
    (project / "notes.txt").write_text("not python\n", encoding="utf-8")
    venv = project / ".venv"
    venv.mkdir()
    (venv / "ignored.py").write_text("one\ntwo\nthree\n", encoding="utf-8")

    async def scenario():
        async with Client(build_server(tmp_path)) as client:
            return await client.call_tool("count_python_lines", {"project_name": "demo"})

    result = anyio.run(scenario)

    assert result.is_error is False
    assert result.structured_content == {"result": 2}


def test_codestats_resource_describes_project(tmp_path):
    project = tmp_path / "demo"
    project.mkdir()
    (project / "app.py").write_text("one\ntwo\n", encoding="utf-8")

    async def scenario():
        async with Client(build_server(tmp_path)) as client:
            return await client.read_resource("info://demo")

    result = anyio.run(scenario)

    assert result.contents[0].text == "项目 demo 包含 2 行 Python 代码"


def test_codestats_tool_rejects_project_outside_configured_root(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.py").write_text("secret\n", encoding="utf-8")

    async def scenario():
        async with Client(build_server(workspace)) as client:
            return await client.call_tool("count_python_lines", {"project_name": "../outside"})

    result = anyio.run(scenario)

    assert result.is_error is True
