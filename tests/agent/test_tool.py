"""How tools and subagents present to Chat Plot.

A leaf ``Tool`` and a subagent (an ``Agent`` placed in ``tools``) are told
apart by type; these tests pin the resulting ``ToolSpec``s and the refusal to
run a subagent in-container.
"""

from pydantic import BaseModel

from axes.agent import Agent, RunContext, Tool
from axes.agent.agent import PromptArgs


class NoArgs(BaseModel):
    pass


class Ping(Tool[NoArgs, NoArgs]):
    description = "ping"
    arguments_schema = NoArgs

    async def run(self, arguments: NoArgs, ctx: RunContext) -> None:
        return None


class ChildArgs(BaseModel):
    topic: str


class Child(Agent):
    name = "child"
    prompts = []


class TypedChild(Agent):
    name = "typed_child"
    description = "A child with its own input contract."
    arguments_schema = ChildArgs
    prompts = []


class Parent(Agent):
    name = "parent"
    prompts = []
    tools = {
        "ping": Ping(),
        "child": Child(),
        "typed_child": TypedChild(),
    }


def _specs() -> dict:
    return {s.name: s for s in Parent().tool_specs()}


def test_leaf_and_subagent_kinds() -> None:
    specs = _specs()
    assert specs["ping"].kind == "leaf"
    assert specs["child"].kind == "subagent"
    assert specs["child"].subagent_name == "child"


def test_subagent_params_come_from_child_contract() -> None:
    spec = _specs()["typed_child"]
    assert "topic" in spec.parameters["properties"]


def test_subagent_params_fall_back_to_prompt_schema() -> None:
    spec = _specs()["child"]
    assert set(spec.parameters["properties"]) == set(
        PromptArgs.model_json_schema()["properties"]
    )


def test_subagent_inherits_child_description() -> None:
    assert _specs()["typed_child"].description == TypedChild.description


def test_subagent_default_description_mentions_name() -> None:
    assert "child" in _specs()["child"].description


async def test_run_tool_refuses_subagent() -> None:
    result = await Parent().run_tool(_call("child"), RunContext())
    assert result.content is None
    assert "subagent" in (result.error or "")


def _call(name: str):
    from axes.agent import ToolCall

    return ToolCall(name=name)
