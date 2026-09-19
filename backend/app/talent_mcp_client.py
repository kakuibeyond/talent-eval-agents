from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.tools import BaseTool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode


def build_mcp_tool_graph(tools: Sequence[BaseTool]) -> Any:
    """Build the smallest LangGraph that executes discovered MCP tools."""
    builder = StateGraph(MessagesState)
    builder.add_node("mcp_tools", ToolNode(list(tools)))
    builder.add_edge(START, "mcp_tools")
    builder.add_edge("mcp_tools", END)
    return builder.compile()
