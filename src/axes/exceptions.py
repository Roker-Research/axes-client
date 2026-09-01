"""Typed exceptions raised by the axes client."""

from __future__ import annotations

from typing import Any


class AxesError(Exception):
    """Base class for all axes errors."""


class QueryError(AxesError):
    """The server rejected the SQL query (HTTP 400).

    Attributes:
        message: Human-readable error message from the server.
        query:   The SQL that caused the error, if available.
    """

    def __init__(self, message: str, query: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.query = query


class ResultTooLarge(AxesError):
    """The result exceeded the server's row or byte cap (HTTP 413).

    Attributes:
        message: Human-readable description from the server.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class AuthError(AxesError):
    """Authentication or authorization failure (HTTP 401 / 403).

    Attributes:
        status_code: The HTTP status code (401 or 403).
        message:     Human-readable message from the server.
    """

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class ConfigError(AxesError):
    """The client is not configured correctly (missing env vars, etc.)."""


class SchemaError(AxesError):
    """A table's registered schema could not be resolved.

    Either no table of that name is in the token's scope, or it carries a
    column type this client cannot turn back into an arrow type.
    """


class WriteError(AxesError):
    """The server rejected a table-data write.

    A 409 means the uploaded parquet's schema does not match the table's
    registered column schema (or the dataset has no applied version) —
    for ingestion scripts this is the structural-failure signal.

    Attributes:
        status_code: The HTTP status code.
        message:     Human-readable message from the server.
    """

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def raise_for_query_status(response: Any, *, query: str | None = None) -> None:
    """Translate an error status on a ``/api/query`` response to a typed error.

    For any status >= 400 the (possibly streaming) response body is read so
    its text is available in the message, then the matching exception is
    raised. A status below 400 returns without touching the body — leaving a
    streaming 200 response unread for the caller to consume.

    Shared by the Arrow path (``axes.sql``) and the ndjson path
    (``axes.cli``) so the status ladder lives in exactly one place.
    """
    status = response.status_code
    if status < 400:
        return

    response.read()
    text = response.text

    if status in (401, 403):
        raise AuthError(status, text or f"HTTP {status}")
    if status == 413:
        raise ResultTooLarge(text or "Result exceeded server caps")
    if status == 400:
        raise QueryError(text or "Bad query", query=query)
    raise QueryError(f"HTTP {status}: {text}", query=query)
