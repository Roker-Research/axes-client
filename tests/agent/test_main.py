import json

import pytest
from pydantic import BaseModel

from axes.agent import Agent, Finish, Message, RunContext, Tool
from axes.agent.main import _dispatch, flatten


class Args(BaseModel):
    x: int


class Out(BaseModel):
    y: int


class Double(Tool[Args, Out]):
    description = "double x"
    arguments_schema = Args
    content_schema = Out

    async def run(self, arguments: Args, ctx: RunContext) -> Out:
        return Out(y=arguments.x * 2)


class Leaf(Agent):
    name = "leaf"
    prompts = []

    def plan_step(self, messages: list) -> Message:
        return Message(content="leaf message")


class TypedLeaf(Agent):
    name = "typed_leaf"
    prompts = []
    arguments_schema = Args

    def system_prompt(self) -> str:
        return "You are a typed leaf."


class Root(Agent):
    name = "root"
    prompts = []
    tools = {
        "double": Double(),
        "leaf": Leaf(),
        "typed_leaf": TypedLeaf(),
    }


def test_flatten_includes_subagents() -> None:
    registry = flatten(Root())
    assert set(registry) == {"root", "leaf", "typed_leaf"}


def test_flatten_detects_name_collision() -> None:
    class Dup(Agent):
        name = "root"  # collides with Root
        prompts = []

    class Bad(Agent):
        name = "root"
        prompts = []
        tools = {"dup": Dup()}

    with pytest.raises(RuntimeError, match="duplicate agent name"):
        flatten(Bad())


async def test_dispatch_run_tool() -> None:
    raw = json.dumps(
        {
            "verb": "run_tool",
            "tool_call": {"name": "double", "arguments": {"x": 21}},
        }
    )
    out = json.loads(await _dispatch(raw, Root()))
    assert out == {"kind": "tool", "content": {"y": 42}, "error": None}


async def test_dispatch_selects_subagent_by_name() -> None:
    raw = json.dumps({"verb": "plan_step", "agent": "leaf", "messages": []})
    out = json.loads(await _dispatch(raw, Root()))
    assert out["action"] == "message"
    assert out["content"] == "leaf message"


async def test_dispatch_unknown_agent() -> None:
    raw = json.dumps({"verb": "plan_step", "agent": "ghost", "messages": []})
    out = json.loads(await _dispatch(raw, Root()))
    assert "unknown agent" in out["error"]


async def test_dispatch_run_tool_on_subagent_returns_start() -> None:
    raw = json.dumps(
        {
            "verb": "run_tool",
            "agent": "root",
            "tool_call": {"name": "leaf", "arguments": {"prompt": "go"}},
        }
    )
    out = json.loads(await _dispatch(raw, Root()))
    assert out["kind"] == "subagent"
    assert out["name"] == "leaf"
    assert out["plan"]["action"] == "message"
    assert out["plan"]["content"] == "leaf message"


async def test_subagent_start_plan_binds_call_arguments() -> None:
    raw = json.dumps(
        {
            "verb": "run_tool",
            "agent": "root",
            "tool_call": {"name": "typed_leaf", "arguments": {"x": 5}},
        }
    )
    out = json.loads(await _dispatch(raw, Root()))
    assert out["kind"] == "subagent"
    assert out["name"] == "typed_leaf"
    assert out["plan"]["action"] == "complete"
    assert json.loads(out["plan"]["user_message"]) == {"x": 5}
    assert out["plan"]["messages"][-1]["role"] == "user"
    assert json.loads(out["plan"]["messages"][-1]["content"]) == {"x": 5}


async def test_dispatch_returns_explicit_finish_content() -> None:
    class Explicit(Agent):
        name = "explicit"
        prompts = []
        content_schema = Out

        def plan_step(self, messages: list) -> Finish:
            return Finish(content={"y": 7})

    raw = json.dumps({"verb": "plan_step", "agent": "explicit"})
    out = json.loads(await _dispatch(raw, Explicit()))
    assert out["action"] == "finish"
    assert out["content"] == {"y": 7}
