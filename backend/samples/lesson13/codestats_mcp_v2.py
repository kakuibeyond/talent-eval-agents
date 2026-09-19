"""
一个最小化 mcp 示例
用于统计项目中的 代码行数
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from mcp.server import MCPServer


logger = logging.getLogger(__name__)


def _resolve_project(root: Path, project_name: str) -> Path:
    project_dir = (root / project_name).resolve()
    try:
        project_dir.relative_to(root)
    except ValueError as exc:
        logger.warning(f"项目必须位于 CODE_STATS_ROOT 目录内: {project_name}")  
        raise ValueError("项目必须位于 CODE_STATS_ROOT 目录内") from exc
    if not project_dir.is_dir():
        logger.warning(f"项目目录不存在: {project_name}")
        raise ValueError(f"项目目录不存在: {project_name}")
    return project_dir


def _count_python_lines(project_dir: Path) -> int:
    total_lines = 0
    for file_path in project_dir.rglob("*.py"):
        if ".venv" in file_path.relative_to(project_dir).parts:
            continue
        try:
            with file_path.open("r", encoding="utf-8") as source:
                total_lines += sum(1 for _ in source)
        except OSError as exc:
            logger.warning("无法读取文件 %s: %s", file_path, exc)
    return total_lines


def build_server(base_dir: str | Path) -> MCPServer:
    root = Path(base_dir).expanduser().resolve()
    logger.info("CODE_STATS_ROOT: %s", root)
    server = MCPServer("CodeStats")

    @server.tool()
    def count_python_lines(project_name: str) -> int:
        """统计指定项目中 Python 代码的总行数。"""
        return _count_python_lines(_resolve_project(root, project_name))

    @server.resource("info://{project_name}")
    def get_project_info(project_name: str) -> str:
        """读取项目的 Python 代码行数摘要。"""
        project_dir = _resolve_project(root, project_name)
        lines = _count_python_lines(project_dir)
        return f"项目 {project_name} 包含 {lines} 行 Python 代码"

    return server


mcp = build_server(os.environ.get("CODE_STATS_ROOT", str(Path.cwd())))


if __name__ == "__main__":
    mcp.run("stdio")
