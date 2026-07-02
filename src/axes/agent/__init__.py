"""axes.agent: the container-side authoring framework for Chat Plot agents.

Author agents as ``Agent`` subclasses with ``Tool`` members and other
``Agent`` instances as subagents; the framework's console entrypoint
(``axes-agent``) reads one step off stdin, runs ``plan_step`` or ``run_tool``,
and writes the result to stdout. Chat Plot drives the multi-turn loop and owns
the LLM call, persistence, and streaming.
"""

from __future__ import annotations

from axes.agent.agent import Agent
from axes.agent.context import RunContext
from axes.agent.protocol import (
    ChatMessage,
    Complete,
    Finish,
    Message,
    PlanResult,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from axes.agent.tool import Tool

__all__ = [
    "Agent",
    "Tool",
    "RunContext",
    "ChatMessage",
    "ToolCall",
    "ToolSpec",
    "ToolResult",
    "Complete",
    "Message",
    "Finish",
    "PlanResult",
]
