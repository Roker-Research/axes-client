import json

import pytest
from pydantic import BaseModel, ValidationError

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


async def test_describe_reports_top_level_contract() -> None:
    class Contented(Agent):
        name = "contented"
        prompts = []
        description = "does a thing"
        arguments_schema = Args
        content_schema = Out

    out = json.loads(await _dispatch('{"verb": "describe"}', Contented()))
    assert out["name"] == "contented"
    assert out["description"] == "does a thing"
    assert out["arguments_schema"]["properties"]["x"]["type"] == "integer"
    assert out["content_schema"]["properties"]["y"]["type"] == "integer"


async def test_describe_falls_back_to_prompt_and_text() -> None:
    out = json.loads(await _dispatch('{"verb": "describe"}', Leaf()))
    assert out["name"] == "leaf"
    # No declared arguments_schema -> the PromptArgs fallback (a `prompt`).
    assert "prompt" in out["arguments_schema"]["properties"]
    # No declared content_schema -> the TextContent fallback (a `text`).
    assert "text" in out["content_schema"]["properties"]


async def test_finish_content_defaults_and_normalizes_to_text() -> None:
    class Plain(Agent):
        name = "plain"
        prompts = []

        def plan_step(self, messages: list) -> Finish:
            return Finish(content={"text": "hi"})

    out = json.loads(await _dispatch('{"verb": "plan_step", "agent": "plain"}', Plain()))
    assert out["action"] == "finish"
    assert out["content"] == {"text": "hi"}
    assert out["error"] is None


async def test_finish_without_content_defaults_to_empty_text() -> None:
    class Empty(Agent):
        name = "empty"
        prompts = []

        def plan_step(self, messages: list) -> Finish:
            return Finish()

    out = json.loads(await _dispatch('{"verb": "plan_step", "agent": "empty"}', Empty()))
    assert out["content"] == {"text": ""}


async def test_finish_error_bypasses_content_validation() -> None:
    class Failing(Agent):
        name = "failing"
        prompts = []
        content_schema = Out  # requires y

        def plan_step(self, messages: list) -> Finish:
            return Finish(error="get_forecast failed")

    out = json.loads(await _dispatch('{"verb": "plan_step", "agent": "failing"}', Failing()))
    assert out["action"] == "finish"
    assert out["error"] == "get_forecast failed"
    assert out["content"] is None


async def test_finish_content_violating_schema_raises() -> None:
    class Bad(Agent):
        name = "bad"
        prompts = []
        content_schema = Out  # requires y: int

        def plan_step(self, messages: list) -> Finish:
            return Finish(content={"wrong": 1})

    with pytest.raises(ValidationError):
        await _dispatch('{"verb": "plan_step", "agent": "bad"}', Bad())


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
