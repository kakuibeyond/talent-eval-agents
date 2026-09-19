from __future__ import annotations

import anyio
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from app.talent_mcp_client import build_mcp_tool_graph


def test_mcp_tools_can_run_inside_langgraph_tool_node():
    @tool
    async def filter_candidates(region: str) -> list[str]:
        """Filter candidates by region."""
        return ["C001"] if region == "上海" else []

    graph = build_mcp_tool_graph([filter_candidates])

    async def scenario():
        return await graph.ainvoke(
            {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "filter_candidates",
                                "args": {"region": "上海"},
                                "id": "call-13",
                                "type": "tool_call",
                            }
                        ],
                    )
                ]
            }
        )

    result = anyio.run(scenario)

    message = result["messages"][-1]
    assert isinstance(message, ToolMessage)
    assert message.name == "filter_candidates"
    assert message.content == '["C001"]'
