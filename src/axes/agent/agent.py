"""The single ``Agent`` class.

An agent is a stateless function of history. Its one required behavior is
``plan_step`` — decide how the next assistant message is produced. The default
``plan_step`` is LLM-driven: it prepends the system prompt, applies an optional
nudge, and returns a ``Complete`` for Chat Plot to run the completion on, or a
``Finish`` when ``done``. Override the fine-grained hooks (``system_prompt``,
``done``, ``nudge``, ``model``) to tweak behavior, or override ``plan_step``
wholesale to build a procedural agent that returns a ``Message`` with no LLM
call. ``run_tool`` is provided and dispatches from ``tools``.

The multi-turn loop is not here — Chat Plot drives it, one step per fresh
process.
"""

from __future__ import annotations

import inspect
from typing import Any, ClassVar, Literal

from jinja2 import Template
from pydantic import BaseModel

from axes.agent.context import RunContext
from axes.agent.prompts import load_prompts
from axes.agent.protocol import (
    ChatMessage,
    Complete,
    Finish,
    PlanResult,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from axes.agent.tool import Tool


class PromptArgs(BaseModel):
    """Default invocation schema for an agent that declares no
    ``arguments_schema`` — a single free-form ``prompt`` string."""

    prompt: str


class Agent:
    #: Stable identifier; the name Chat Plot selects with (``Chat.agent``) and
    #: by which a subagent is addressed. Defaults to the class name.
    name: ClassVar[str]

    #: Tool-facing contract. An agent is invoked as a tool — the top-level
    #: agent when its chat is created, a subagent by its parent's LLM — so it
    #: declares how it is called (``arguments_schema``), what it returns
    #: (``content_schema``), and how it reads to a caller (``description``).
    description: ClassVar[str | None] = None
    arguments_schema: ClassVar[type[BaseModel] | None] = None
    content_schema: ClassVar[type[BaseModel] | None] = None

    #: Prompt template keys to load from ``<key>.md`` beside the source file.
    prompts: ClassVar[list[str]] = ["prompt"]

    #: Tools available to this agent, keyed by the name exposed to the LLM.
    #: A ``Tool`` entry is a leaf, run in-container; an ``Agent`` entry is a
    #: subagent, dispatched by Chat Plot to a child chat. The type is the sole
    #: distinction — no wrapper.
    tools: ClassVar[dict[str, Tool[Any, Any] | Agent]] = {}

    #: Model / reasoning effort for the default ``plan_step``. ``None`` lets
    #: Chat Plot apply its configured default.
    model: ClassVar[str | None] = None
    reasoning_effort: ClassVar[str | None] = None

    #: Populated per subclass from its prompt files.
    prompt_templates: ClassVar[dict[str, Template]] = {}

    #: This step's bound invocation arguments (see ``bind_arguments``).
    _arguments: Any = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "name" not in cls.__dict__:
            cls.name = cls.__name__
        try:
            source_file = inspect.getfile(cls)
        except (TypeError, OSError):
            return
        cls.prompt_templates = load_prompts(source_file, cls.prompts)

    # -- prompt hooks -------------------------------------------------------

    def prompt_context(self) -> dict[str, Any]:
        """Variables exposed to the Jinja prompt templates."""
        return {}

    def render(self, key: str, **extra: Any) -> str:
        return self.prompt_templates[key].render(self.prompt_context() | extra)

    def system_prompt(self) -> str:
        return self.render("prompt")

    # -- invocation arguments ----------------------------------------------

    def bind_arguments(self, raw: dict[str, Any] | None) -> None:
        """Bind this step's invocation arguments before ``plan_step``.

        Validated into ``arguments_schema`` when one is declared. Mutating the
        instance is safe: the process handles exactly one step.
        """
        if not raw:
            self._arguments = None
        elif self.arguments_schema is not None:
            self._arguments = self.arguments_schema.model_validate(raw)
        else:
            self._arguments = raw

    @property
    def arguments(self) -> Any:
        """The bound invocation arguments (parsed model, or ``None``)."""
        return self._arguments

    # -- loop hooks (default plan_step) -------------------------------------

    def done(self, messages: list[ChatMessage]) -> bool:
        """Whether the run is complete, derived from history.

        Default: stop once the model has produced an assistant message with no
        tool calls (a natural stop). Continue while the last entry is a user
        message or tool result.
        """
        if not messages:
            return False
        last = messages[-1]
        return last.role == "assistant" and not last.tool_calls

    def nudge(self, messages: list[ChatMessage]) -> str | None:
        """An optional user message to inject before the next completion."""
        return None

    def tool_specs(self) -> list[ToolSpec]:
        specs: list[ToolSpec] = []
        for name, tool in self.tools.items():
            kind: Literal["leaf", "subagent"]
            schema: type[BaseModel] | None
            description: str
            if isinstance(tool, Agent):
                # A subagent's parameters are its own input contract; it reads
                # to the model by its own description.
                kind, subagent_name = "subagent", tool.name
                schema = tool.arguments_schema or PromptArgs
                description = (
                    tool.description
                    or f"Delegate a task to the {tool.name} subagent."
                )
            else:
                kind, subagent_name = "leaf", None
                schema = tool.arguments_schema
                description = tool.description
            parameters = (
                schema.model_json_schema(by_alias=True) if schema else {}
            )
            specs.append(
                ToolSpec(
                    name=name,
                    description=description,
                    parameters=parameters,
                    kind=kind,
                    subagent_name=subagent_name,
                )
            )
        return specs

    # -- the step verbs -----------------------------------------------------

    def plan_step(self, messages: list[ChatMessage]) -> PlanResult:
        """Decide the next message (override for procedural agents)."""
        if self.done(messages):
            return Finish()
        convo: list[ChatMessage] = [
            ChatMessage(role="system", content=self.system_prompt()),
            *messages,
        ]
        nudge = self.nudge(messages)
        if nudge:
            convo.append(ChatMessage(role="user", content=nudge))
        return Complete(
            messages=convo,
            tools=self.tool_specs(),
            model=self.model,
            reasoning_effort=self.reasoning_effort,
        )

    async def run_tool(self, call: ToolCall, ctx: RunContext) -> ToolResult:
        tool = self.tools.get(call.name)
        if tool is None:
            return ToolResult(error=f"unknown tool: {call.name}")
        if isinstance(tool, Agent):
            return ToolResult(
                error=(
                    f"tool {call.name!r} is a subagent and must be dispatched "
                    "by the control plane"
                )
            )
        try:
            arguments: Any = call.arguments
            if tool.arguments_schema is not None:
                arguments = tool.arguments_schema.model_validate(
                    call.arguments
                )
            content = await tool.run(arguments, ctx)
        except Exception as error:  # surfaced to Chat Plot as a tool error
            return ToolResult(error=str(error))
        if content is None:
            return ToolResult(content=None)
        if isinstance(content, BaseModel):
            return ToolResult(content=content.model_dump(by_alias=True))
        return ToolResult(content=content)
