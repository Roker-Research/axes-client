"""
Client — connection config and the lazily-initialised module-level default.

Environment variables:
    AXES_ENDPOINT    HTTPS base URL, e.g. https://app.axes.com
    AXES_TOKEN       Bearer token — a short-lived JWT injected by the sandbox
                     runtime, or a personal access token when running externally
"""

from __future__ import annotations

import contextlib
import os
import threading

import httpx

from axes.exceptions import ConfigError


def _build_http_client(
    *,
    endpoint: str,
    token: str,
    timeout: float,
) -> httpx.Client:
    # Every request sets its own Accept (Arrow, ndjson, or JSON) per call,
    # so no default Accept header is needed here.
    return httpx.Client(
        base_url=endpoint.rstrip("/"),
        headers={"Authorization": f"Bearer {token}"},
        timeout=timeout,
    )


class Client:
    """Holds connection config for one Axes endpoint.

    Most callers never instantiate this directly — the module-level
    ``sql``, ``scan``, and ``sql_to_dataframe`` functions use the
    lazily-initialised default client built from environment variables.

    Args:
        endpoint: HTTPS base URL (e.g. ``https://app.axes.com``).  Required.
        token:    Bearer token (JWT or PAT).  Required.
        versions: Default version-pin dict applied to every ``sql``
                  call unless overridden per-call.
        timeout:  HTTP timeout in seconds (default 120).
    """

    def __init__(
        self,
        *,
        endpoint: str,
        token: str,
        versions: dict[str, str] | None = None,
        timeout: float = 120.0,
    ) -> None:
        if not token:
            raise ConfigError("token must not be empty")
        if not endpoint:
            raise ConfigError("endpoint must not be empty")

        self.endpoint = endpoint
        self.token = token
        self.versions: dict[str, str] = versions or {}
        self.timeout = timeout

        self._http: httpx.Client | None = None
        self._lock = threading.Lock()

    @property
    def http(self) -> httpx.Client:
        """Return the shared ``httpx.Client``, creating it on first access."""
        if self._http is None:
            with self._lock:
                if self._http is None:
                    self._http = _build_http_client(
                        endpoint=self.endpoint,
                        token=self.token,
                        timeout=self.timeout,
                    )
        return self._http

    def query(
        self, query: str, *, accept: str
    ) -> contextlib.AbstractContextManager[httpx.Response]:
        """Return a streaming context manager for ``POST /api/query``.

        Args:
            query:  SQL string to execute.
            accept: MIME type for the response format (Arrow IPC or ndjson).

        Usage::

            mime = "application/vnd.apache.arrow.stream"
            with client.query("SELECT 1", accept=mime) as response:
                ...
        """
        return self.http.stream(
            "POST",
            "/api/query",
            json={"query": query},
            headers={"Accept": accept},
        )

    def append_table_data(self, table: str, data: bytes) -> httpx.Response:
        """``POST /api/table-data-files`` — append parquet bytes to a table.

        The target dataset is the one the token is write-scoped to (tokens
        are minted per ingestion task; personal access tokens are
        read-only). Returns the raw response; most callers use
        :func:`axes.append_table_data`.
        """
        return self.http.post(
            "/api/table-data-files",
            params={"table": table},
            content=data,
            headers={
                "Content-Type": "application/octet-stream",
                "Accept": "application/json",
            },
        )

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        if self._http is not None:
            self._http.close()
            self._http = None

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"Client(endpoint={self.endpoint!r})"


# ---------------------------------------------------------------------------
# Module-level default client
# ---------------------------------------------------------------------------

_default_client: Client | None = None


def get_default_client() -> Client:
    """Return the lazily-initialised default client from environment variables.

    Raises ``ConfigError`` if required env vars are missing.
    """
    global _default_client
    if _default_client is None:
        token = os.environ.get("AXES_TOKEN", "")
        if not token:
            raise ConfigError(
                "AXES_TOKEN is not set. "
                "Inside the sandbox it is injected automatically. "
                "Outside, set it to your personal access token."
            )

        endpoint = os.environ.get("AXES_ENDPOINT", "https://app.axes.com")

        _default_client = Client(
            endpoint=endpoint,
            token=token,
        )
    return _default_client
