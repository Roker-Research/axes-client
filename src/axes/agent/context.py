"""The runtime context handed to a tool's ``run`` method.

A tool is a one-shot ``run_tool`` process, so ``RunContext`` carries only what
that process needs: the chat identity and the invocation arguments. It
deliberately does *not* carry a database session, a Redis handle, or a socket
— those belong to Chat Plot.

Connection details (endpoint, a scoped capability token) reach the tool as
environment variables injected into the container, so apps construct their own
clients (e.g. an ``axes.Client`` handle) without any of that passing through
``RunContext``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RunContext:
    chat_id: str | None = None
    #: The chat's invocation arguments (raw; a tool may validate as it likes).
    arguments: dict[str, Any] = field(default_factory=dict)
