import pytest
from pydantic import BaseModel

from axes.agent import (
    Agent,
    ChatMessage,
    Complete,
    Finish,
    Message,
    RunContext,
    Tool,
    ToolCall,
)


class EchoArgs(BaseModel):
    text: str


class EchoOut(BaseModel):
    echoed: str


class Echo(Tool[EchoArgs, EchoOut]):
    description = "Echo the text back."
    arguments_schema = EchoArgs
    content_schema = EchoOut

    async def run(self, arguments: EchoArgs, ctx: RunContext) -> EchoOut:
        return EchoOut(echoed=arguments.text)


class Bare(Agent):
    name = "bare"
    prompts = []
    tools = {"echo": Echo()}

    def system_prompt(self) -> str:
        return "You are a test agent."


def _msgs(*roles_and_tools: tuple[str, bool]) -> list[ChatMessage]:
    out: list[ChatMessage] = []
    for role, has_tool in roles_and_tools:
        calls = [ToolCall(name="echo")] if has_tool else None
        out.append(ChatMessage(role=role, content="x", tool_calls=calls))
    return out


def test_default_plan_step_builds_complete_from_user() -> None:
    result = Bare().plan_step(_msgs(("user", False)))
    assert isinstance(result, Complete)
    assert result.messages[0].role == "system"
    assert result.messages[0].content == "You are a test agent."
    assert [t.name for t in result.tools] == ["echo"]
    assert result.tools[0].kind == "leaf"


def test_default_plan_step_continues_after_tool_result() -> None:
    result = Bare().plan_step(_msgs(("assistant", True), ("tool", False)))
    assert isinstance(result, Complete)


def test_default_done_on_assistant_without_tools() -> None:
    result = Bare().plan_step(_msgs(("user", False), ("assistant", False)))
    assert isinstance(result, Finish)


def test_nudge_is_injected() -> None:
    class Nudger(Bare):
        name = "nudger"

        def nudge(self, messages: list[ChatMessage]) -> str | None:
            return "keep going"

    result = Nudger().plan_step(_msgs(("user", False)))
    assert isinstance(result, Complete)
    assert result.messages[-1].role == "user"
    assert result.messages[-1].content == "keep going"


def test_reasoning_effort_passthrough() -> None:
    class Effortful(Bare):
        name = "effortful"
        reasoning_effort = "high"

    result = Effortful().plan_step(_msgs(("user", False)))
    assert isinstance(result, Complete)
    assert result.reasoning_effort == "high"


async def test_run_tool_validates_and_runs() -> None:
    result = await Bare().run_tool(
        ToolCall(name="echo", arguments={"text": "hi"}),
        RunContext(),
    )
    assert result.error is None
    assert result.content == {"echoed": "hi"}


async def test_run_tool_unknown() -> None:
    result = await Bare().run_tool(ToolCall(name="nope"), RunContext())
    assert result.content is None
    assert "unknown tool" in (result.error or "")


async def test_run_tool_bad_arguments_becomes_error() -> None:
    result = await Bare().run_tool(
        ToolCall(name="echo", arguments={}), RunContext()
    )
    assert result.error is not None


def test_procedural_agent_returns_message() -> None:
    class Proc(Agent):
        name = "proc"
        prompts = []

        def plan_step(self, messages: list[ChatMessage]) -> Message:
            return Message(content="computed forecast")

    result = Proc().plan_step([])
    assert isinstance(result, Message)
    assert result.content == "computed forecast"


class Args(BaseModel):
    ticker: str
    horizon: int = 30


class Contracted(Agent):
    name = "contracted"
    description = "has a contract"
    arguments_schema = Args
    prompts = []

    def system_prompt(self) -> str:
        return f"forecast {self.arguments.ticker}"


def test_bind_arguments_validates_into_schema() -> None:
    agent = Contracted()
    agent.bind_arguments({"ticker": "ZS"})
    assert isinstance(agent.arguments, Args)
    assert agent.arguments.ticker == "ZS"
    assert agent.arguments.horizon == 30


def test_bind_arguments_empty_is_none() -> None:
    agent = Contracted()
    agent.bind_arguments({})
    assert agent.arguments is None


def test_bind_arguments_raises_on_invalid() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Contracted().bind_arguments({"horizon": 5})  # missing ticker


def test_arguments_defaults_to_none_before_bind() -> None:
    assert Contracted().arguments is None


def test_bound_arguments_flow_into_prompt() -> None:
    agent = Contracted()
    agent.bind_arguments({"ticker": "ZC"})
    result = agent.plan_step(_msgs(("user", False)))
    assert isinstance(result, Complete)
    assert result.messages[0].content == "forecast ZC"


def test_agent_without_schema_binds_raw_dict() -> None:
    agent = Bare()
    agent.bind_arguments({"anything": 1})
    assert agent.arguments == {"anything": 1}


def test_name_defaults_to_class_name() -> None:
    class Unnamed(Agent):
        prompts = []

    assert Unnamed.name == "Unnamed"


@pytest.mark.parametrize("effort", ["high", None])
def test_model_defaults_to_none(effort: str | None) -> None:
    class M(Bare):
        name = "m"
        reasoning_effort = effort

    result = M().plan_step(_msgs(("user", False)))
    assert isinstance(result, Complete)
    assert result.model is None
