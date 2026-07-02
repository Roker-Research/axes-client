"""The runtime context handed to a tool's ``run`` method.

A tool is a one-shot ``run_tool`` process, so ``RunContext`` carries only what
that process needs: the chat identity, the pinned dataset version, and the
free-form ``config`` Chat Plot passes in (connection details, a scoped
capability token, etc.). It deliberately does *not* carry a database session,
a Redis handle, or a socket — those belong to Chat Plot.

Apps construct their own clients (e.g. an ``axes.Client`` handle) from
``config``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RunContext:
    chat_id: str | None = None
    dataset_version_id: str | None = None
    #: The chat's invocation arguments (raw; a tool may validate as it likes).
    arguments: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        chat_id: str | None = None,
        arguments: dict[str, Any] | None = None,
    ) -> RunContext:
        return cls(
            chat_id=chat_id,
            dataset_version_id=config.get("dataset_version_id"),
            arguments=arguments or {},
            config=config,
        )
