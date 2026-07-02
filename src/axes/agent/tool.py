"""The typed leaf-tool abstraction.

A ``Tool`` is a one-shot unit of work executed by a ``run_tool`` invocation: it
declares argument/content schemas and an async ``run``. Subagents are not a
kind of tool — an ``Agent`` placed directly in another agent's ``tools`` is the
subagent, and the framework tells the two apart by type (see ``Agent``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from pydantic import BaseModel

from axes.agent.context import RunContext


class Tool[TArgs: BaseModel, TContent: BaseModel](ABC):
    """Base class for leaf tools.

    Subclasses set ``description`` and ``arguments_schema`` (and optionally
    ``content_schema``) as class attributes and implement ``run``. Instantiate
    once as a class-level singleton in an agent's ``tools`` dict.
    """

    description: str
    arguments_schema: ClassVar[type[BaseModel] | None] = None
    content_schema: ClassVar[type[BaseModel] | None] = None

    @abstractmethod
    async def run(
        self, arguments: TArgs, ctx: RunContext
    ) -> TContent | None: ...
