"""The stdin/stdout contract between Chat Plot and a containerized agent.

Chat Plot invokes the container once per step with a JSON ``Request`` on
stdin and reads one JSON result on stdout. There are two verbs:

- ``plan_step`` receives the chat history (already mapped by Chat Plot into a
  provider-neutral ``messages`` array) and returns a ``PlanResult`` — one of
  ``Complete`` (run the LLM), ``Message`` (a procedurally produced assistant
  message), or ``Finish`` (the run is done).
- ``run_tool`` receives one leaf tool call and returns a ``ToolResult``.

Everything here maps onto Chat Plot's ``Message`` / ``ToolCall`` data model.
The framework never calls the LLM, persists anything, or streams — those are
Chat Plot's.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

Role = Literal["system", "user", "assistant", "tool"]


class ToolCall(BaseModel):
    """A tool call requested on an assistant message.

    ``arguments`` is the parsed object (not a JSON string); Chat Plot applies
    any provider-specific serialization when it re-formats the array for the
    completion.
    """

    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    """One entry in the provider-neutral message array.

    Chat Plot builds these from the persisted history; ``plan_step`` may add,
    delete, or rewrite them before returning a ``Complete``.
    """

    model_config = ConfigDict(extra="forbid")

    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    reasoning_details: list[dict[str, Any]] | None = None
    name: str | None = None


class ToolSpec(BaseModel):
    """A tool definition returned to Chat Plot in a ``Complete``.

    ``parameters`` is the JSON schema Chat Plot hands to the LLM as the
    function definition. ``kind`` tells Chat Plot how to dispatch a call to
    this tool: ``leaf`` becomes a ``run_tool`` invocation; ``subagent`` becomes
    a child chat named ``subagent_name``.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    kind: Literal["leaf", "subagent"] = "leaf"
    subagent_name: str | None = None


# --- plan_step results -----------------------------------------------------


# A ``plan_step`` result may carry the agent's **output** — the value that
# becomes the parent tool call's ``content`` (validated by Chat Plot against
# the agent's ``content_schema``). Every step writes the same slot; the name
# marks how settled it is. While the run is in progress it is provisional,
# ``state`` (on ``Complete`` / ``Message``); at ``Finish`` it is authoritative
# and named ``content``, matching where it lands. ``run_tool`` cannot set it —
# a leaf tool never sees history, so it can't form the agent's belief.


class Complete(BaseModel):
    """Ask Chat Plot to run the LLM on this (mutated) messages array."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["complete"] = "complete"
    messages: list[ChatMessage]
    tools: list[ToolSpec] = Field(default_factory=list)
    model: str | None = None
    reasoning_effort: str | None = None
    #: Provisional agent output so far (the parent tool call content).
    state: dict[str, Any] | None = None


class Message(BaseModel):
    """A ready-made assistant message from a procedural agent (no LLM call)."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["message"] = "message"
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    reasoning_details: list[dict[str, Any]] | None = None
    #: Provisional agent output so far (the parent tool call content).
    state: dict[str, Any] | None = None


class Finish(BaseModel):
    """The agent declares the run complete."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["finish"] = "finish"
    reason: str | None = None
    #: The agent's terminal output (the parent tool call content). When
    #: ``None``, Chat Plot derives it from the last assistant message.
    content: dict[str, Any] | None = None


type PlanResult = Annotated[
    Complete | Message | Finish, Field(discriminator="action")
]

plan_result_adapter: TypeAdapter[PlanResult] = TypeAdapter(PlanResult)


# --- run_tool result -------------------------------------------------------


class ToolResult(BaseModel):
    """The outcome of a ``run_tool`` invocation."""

    model_config = ConfigDict(extra="forbid")

    content: dict[str, Any] | None = None
    error: str | None = None


# --- requests (stdin) ------------------------------------------------------


class PlanStepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verb: Literal["plan_step"] = "plan_step"
    agent: str | None = None
    chat_id: str | None = None
    messages: list[ChatMessage] = Field(default_factory=list)
    #: The invocation arguments this agent's chat was created with (frozen for
    #: the chat's life), matching the agent's ``arguments_schema``.
    arguments: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)


class RunToolRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verb: Literal["run_tool"] = "run_tool"
    agent: str | None = None
    chat_id: str | None = None
    tool_call: ToolCall
    arguments: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)


type Request = Annotated[
    PlanStepRequest | RunToolRequest, Field(discriminator="verb")
]

request_adapter: TypeAdapter[Request] = TypeAdapter(Request)


class ErrorEnvelope(BaseModel):
    """Emitted on stdout when a step fails before producing a result."""

    model_config = ConfigDict(extra="forbid")

    error: str
