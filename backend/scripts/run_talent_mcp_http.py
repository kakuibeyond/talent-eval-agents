from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from mcp.server import MCPServer

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.talent_mcp_server import build_talent_mcp_server  # noqa: E402
from app.talent_tools import ToolExecutor  # noqa: E402
from scripts.verify_talent_mcp import build_demo_service, demo_context  # noqa: E402


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 18081


def build_demo_http_server(database_path: str | Path) -> MCPServer:
    service = build_demo_service(Path(database_path))
    return build_talent_mcp_server(
        service,
        ToolExecutor(),
        context_provider=demo_context,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="启动第 13 课 Talent MCP Streamable HTTP 演示服务",
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    with TemporaryDirectory(prefix="talent-mcp-http-") as directory:
        print(f"Talent MCP 临时目录: {directory}", flush=True)  
        server = build_demo_http_server(Path(directory) / "talent.db")
        endpoint = f"http://{args.host}:{args.port}/mcp"
        print(f"Talent MCP Streamable HTTP: {endpoint}", flush=True)
        print("Demo 模式：隔离 SQLite、固定 tenant-a，无鉴权，仅用于本机调试", flush=True)
        try:
            server.run(
                "streamable-http",
                host=args.host,
                port=args.port,
                streamable_http_path="/mcp",
                json_response=True,
                stateless_http=True,
            )
        except KeyboardInterrupt:
            print("Talent MCP 已停止", flush=True)


if __name__ == "__main__":
    main()
