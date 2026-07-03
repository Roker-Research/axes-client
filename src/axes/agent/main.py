"""The container entrypoint.

Chat Plot runs the image with this console script (``axes-agent``), writing one
JSON ``Request`` to stdin and reading one JSON result from stdout. The image
ships a single top-level (root) agent; subagents are reached by walking the
root's tools. This module locates the root, flattens the tree into a name map,
dispatches one step, and exits.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import sys
from importlib import import_module
from typing import Any

from pydantic import BaseModel

from axes.agent.agent import Agent
from axes.agent.context import RunContext
from axes.agent.protocol import (
    ChatMessage,
    ErrorEnvelope,
    PlanStepRequest,
    SubagentStart,
    request_adapter,
)

#: The root agent's location as ``module:attr``. One convention: a top-level
#: ``agent`` module exposing ``root``. Override with ``AXES_AGENT``.
DEFAULT_ROOT = "agent:root"


def load_root() -> Agent:
    """Resolve the root agent from ``AXES_AGENT`` (default ``agent:root``).

    The working directory is placed on ``sys.path`` so a top-level ``agent``
    module (``WORKDIR/agent.py``) is importable when the console script runs.
    ``attr`` may name an ``Agent`` instance or an ``Agent`` subclass (which is
    instantiated).
    """
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    name, _, attr = os.environ.get("AXES_AGENT", DEFAULT_ROOT).partition(":")
    obj = getattr(import_module(name), attr or "root")
    root: Agent = obj() if inspect.isclass(obj) else obj
    return root


def flatten(root: Agent) -> dict[str, Agent]:
    """Walk the tool tree into a ``name -> Agent`` map.

    Names must be unique across the image (Chat Plot's ``Chat.agent`` is a flat
    string); a collision fails fast rather than silently shadowing.
    """
    registry: dict[str, Agent] = {}

    def visit(agent: Agent) -> None:
        if agent.name in registry:
            if registry[agent.name] is agent:
                return
            raise RuntimeError(
                f"duplicate agent name {agent.name!r} in image; subagent "
                "names must be unique"
            )
        registry[agent.name] = agent
        for tool in agent.tools.values():
            if isinstance(tool, Agent):
                visit(tool)

    visit(root)
    return registry


async def plan(
    agent: Agent,
    messages: list[ChatMessage],
    arguments: dict[str, Any],
) -> BaseModel:
    """Bind arguments and run one ``plan_step``."""
    agent.bind_arguments(arguments)
    planned: Any = agent.plan_step(messages)
    if inspect.isawaitable(planned):
        planned = await planned
    assert isinstance(planned, BaseModel)
    return planned


async def _dispatch(raw: str, root: Agent) -> str:
    """Validate one request, select the agent, run the step, return JSON.

    Pure (no I/O): the entrypoint's testable core. A ``run_tool`` whose
    target is a subagent is not executed; it returns a ``SubagentStart``
    carrying the subagent's first ``plan_step`` on empty history, computed
    with the call's arguments bound.
    """
    registry = flatten(root)
    req = request_adapter.validate_json(raw)
    agent: Agent = root
    if req.agent is not None:
        selected = registry.get(req.agent)
        if selected is None:
            return ErrorEnvelope(
                error=f"unknown agent: {req.agent}"
            ).model_dump_json()
        agent = selected

    result: BaseModel
    if isinstance(req, PlanStepRequest):
        result = await plan(agent, req.messages, req.arguments)
    else:
        target = agent.tools.get(req.tool_call.name)
        if isinstance(target, Agent):
            first_plan = await plan(target, [], req.tool_call.arguments)
            result = SubagentStart(name=target.name, plan=first_plan)
        else:
            ctx = RunContext(chat_id=req.chat_id, arguments=req.arguments)
            result = await agent.run_tool(req.tool_call, ctx)
    return result.model_dump_json()


def main() -> None:
    raw = sys.stdin.read()
    try:
        out = asyncio.run(_dispatch(raw, load_root()))
    except Exception as error:
        sys.stdout.write(
            ErrorEnvelope(
                error=f"{type(error).__name__}: {error}"
            ).model_dump_json()
        )
        sys.stdout.flush()
        raise SystemExit(1) from error
    sys.stdout.write(out)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
