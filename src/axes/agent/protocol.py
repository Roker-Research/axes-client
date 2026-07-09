"""The stdin/stdout contract between Chat Plot and a containerized agent.

Chat Plot invokes the container once per step with a JSON ``Request`` on
stdin and reads one JSON result on stdout. There are two verbs:

- ``plan_step`` receives the chat history (already mapped by Chat Plot into a
  provider-neutral ``messages`` array) and returns a ``PlanResult`` — one of
  ``Complete`` (run the LLM), ``Message`` (a procedurally produced assistant
  message), or ``Finish`` (the run is done).
- ``run_tool`` receives one tool call. A leaf tool is executed and returns a
  ``ToolResult``; a call that targets a subagent returns a ``SubagentStart``
  instead — the subagent's name plus its first ``plan_step`` — and Chat Plot
  launches the child chat from it.

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
    function definition. Chat Plot dispatches every resulting call back to
    the container as a ``run_tool``; whether the target is a leaf tool or a
    subagent is the container's own knowledge, reported at execution time
    (see ``SubagentStart``).
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)


# --- plan_step results -----------------------------------------------------


# A ``plan_step`` result may carry the agent's **output** — the value that
# becomes the parent tool call's ``content``. Every step writes the same
# slot; the name marks how settled it is. While the run is in progress it is
# provisional, ``state`` (on ``Complete`` / ``Message``); at ``Finish`` it is
# authoritative and named ``content``, matching where it lands. ``run_tool``
# cannot set it — a leaf tool never sees history, so it can't form the
# agent's belief.


class Complete(BaseModel):
    """Ask Chat Plot to run the LLM on this (mutated) messages array.

    ``messages`` is the ephemeral LLM view — the agent rebuilds it every
    step (system prompt, history, transient nudges) and Chat Plot persists
    none of it. To persist a user turn, set ``user_message``: Chat Plot
    stores it as the chat's next user message (so it enters history and the
    UI) and then runs this completion. This is how an agent authors the
    opening prompt of its own run from its invocation arguments — carried on
    the same step that plans the first completion, so no extra container boot
    is spent on it. The same text should also appear in ``messages`` (the
    LLM must see it this turn); on later steps it returns through history.
    """

    model_config = ConfigDict(extra="forbid")

    action: Literal["complete"] = "complete"
    messages: list[ChatMessage]
    tools: list[ToolSpec] = Field(default_factory=list)
    model: str | None = None
    reasoning_effort: str | None = None
    #: A user message to persist before running this completion (e.g. the
    #: opening prompt authored from invocation arguments). Unlike the entries
    #: in ``messages``, this is stored to history.
    user_message: str | None = None
    #: Provisional agent output so far (the parent tool call content).
    state: dict[str, Any] | None = None


class Message(BaseModel):
    """A ready-made assistant message from the agent (no LLM call).

    A procedurally produced assistant turn, optionally carrying tool calls.
    To author a *user* turn, set ``user_message`` on ``Complete`` instead —
    that persists the turn and runs the next completion in one step.
    """

    model_config = ConfigDict(extra="forbid")

    action: Literal["message"] = "message"
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    reasoning_details: list[dict[str, Any]] | None = None
    #: Provisional agent output so far (the parent tool call content).
    state: dict[str, Any] | None = None


class Finish(BaseModel):
    """The agent declares the run complete.

    A successful finish carries ``content`` — the terminal output, which the
    framework validates against the agent's ``content_schema`` (defaulting to
    ``{"text": ...}``) before returning it. A failed finish carries ``error``
    instead and bypasses that validation; the two are mutually exclusive, and
    ``error`` is how an agent reports it could not produce a valid output.
    """

    model_config = ConfigDict(extra="forbid")

    action: Literal["finish"] = "finish"
    reason: str | None = None
    #: The agent's terminal output (the parent tool call content), validated
    #: against ``content_schema``. Left null on a failed finish.
    content: dict[str, Any] | None = None
    #: Set instead of ``content`` when the run failed; bypasses schema
    #: validation and settles the parent tool call as an error.
    error: str | None = None


type PlanResult = Annotated[
    Complete | Message | Finish, Field(discriminator="action")
]

plan_result_adapter: TypeAdapter[PlanResult] = TypeAdapter(PlanResult)


# --- run_tool results --------------------------------------------------------


class ToolResult(BaseModel):
    """The outcome of a ``run_tool`` invocation on a leaf tool."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["tool"] = "tool"
    content: dict[str, Any] | None = None
    error: str | None = None


class SubagentStart(BaseModel):
    """``run_tool`` addressed a subagent, not a leaf tool.

    The entrypoint does not execute the subagent. It reports the subagent's
    ``name`` (Chat Plot creates a child chat under it) together with the
    subagent's first ``plan_step`` — computed on empty history with the
    call's arguments bound — so the launch costs no extra container boot.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["subagent"] = "subagent"
    name: str
    plan: PlanResult


type RunToolResult = Annotated[
    ToolResult | SubagentStart, Field(discriminator="kind")
]

run_tool_result_adapter: TypeAdapter[RunToolResult] = TypeAdapter(
    RunToolResult
)


# --- describe result -------------------------------------------------------


class AgentContract(BaseModel):
    """The image's top-level agent contract, returned to a ``describe``.

    Chat Plot stores this on the ``ExternalAgentVersion`` at build time and
    builds the parent LLM's tool definition from it, so nothing about the
    contract is re-declared on the Chat Plot side. Both schemas are always
    present: ``arguments_schema`` falls back to a single ``prompt`` and
    ``content_schema`` to a single ``text`` when the agent declares neither,
    matching the fallbacks the framework enforces at runtime.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None = None
    arguments_schema: dict[str, Any]
    content_schema: dict[str, Any]


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


class RunToolRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verb: Literal["run_tool"] = "run_tool"
    agent: str | None = None
    chat_id: str | None = None
    tool_call: ToolCall
    arguments: dict[str, Any] = Field(default_factory=dict)


class DescribeRequest(BaseModel):
    """Ask the image to report its top-level agent contract.

    Runs no step and touches no history; Chat Plot sends it once per build to
    extract the schemas it stores on the version.
    """

    model_config = ConfigDict(extra="forbid")

    verb: Literal["describe"] = "describe"


type Request = Annotated[
    PlanStepRequest | RunToolRequest | DescribeRequest,
    Field(discriminator="verb"),
]

request_adapter: TypeAdapter[Request] = TypeAdapter(Request)


class ErrorEnvelope(BaseModel):
    """Emitted on stdout when a step fails before producing a result."""

    model_config = ConfigDict(extra="forbid")

    error: str
