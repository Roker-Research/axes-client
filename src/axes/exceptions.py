"""Typed exceptions raised by the axes client."""


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
