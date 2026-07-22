"""
read_payload() / write_output() — the ingestion task I/O contract.

An ingestion task's input is a single JSON payload and its output is a
single JSON document; the runner injects the file locations as
``AXES_INPUT_FILE`` and ``AXES_OUTPUT_FILE``. stdout and stderr are pure
log channels — print freely; only the output file is parsed.

A script's output shape (all fields except ``status`` optional)::

    {
      "status": "ok" | "not_ready" | "rate_limited" | "error",
      "new_tasks": [{"entrypoint": "fetch.py", "payload": {...}}, ...],
      "cursor": {...},              # proposed high-water mark
      "retry_after_seconds": 900,   # hint for not_ready / rate_limited
      "error": "...",               # required when status == "error"
      "result": {"rows_written": 1234}
    }
"""

from __future__ import annotations

import json
import os
from typing import Any

from axes.exceptions import ConfigError


def read_payload() -> dict[str, Any]:
    """Read the task's payload from ``AXES_INPUT_FILE``."""
    input_file = os.environ.get("AXES_INPUT_FILE")
    if not input_file:
        raise ConfigError(
            "AXES_INPUT_FILE is not set. It is injected by the ingestion "
            "runner; when testing locally, point it at a JSON payload file."
        )
    with open(input_file) as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ConfigError("The task payload must be a JSON object")
    return payload


def write_output(
    status: str = "ok",
    *,
    new_tasks: list[dict[str, Any]] | None = None,
    cursor: dict[str, Any] | None = None,
    retry_after_seconds: int | None = None,
    error: str | None = None,
    result: dict[str, Any] | None = None,
) -> None:
    """Write the task's output document to ``AXES_OUTPUT_FILE``.

    Call exactly once, as the script's final act. Later calls overwrite
    earlier ones — the runner reads the file only after the script exits.
    """
    output_file = os.environ.get("AXES_OUTPUT_FILE")
    if not output_file:
        raise ConfigError(
            "AXES_OUTPUT_FILE is not set. It is injected by the ingestion "
            "runner; when testing locally, point it at a writable path."
        )
    output: dict[str, Any] = {"status": status}
    if new_tasks is not None:
        output["new_tasks"] = new_tasks
    if cursor is not None:
        output["cursor"] = cursor
    if retry_after_seconds is not None:
        output["retry_after_seconds"] = retry_after_seconds
    if error is not None:
        output["error"] = error
    if result is not None:
        output["result"] = result
    with open(output_file, "w") as handle:
        json.dump(output, handle)
