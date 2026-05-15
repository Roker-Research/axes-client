"""
Client — connection config and the lazily-initialised module-level default.

Environment variables (sandbox):
    AXES_SOCKET      Unix socket path  (preferred over AXES_ENDPOINT)
    AXES_TOKEN       Short-lived JWT injected by the sandbox runtime

Environment variables (external / user):
    AXES_ENDPOINT    HTTPS base URL, e.g. https://app.axes.com
    AXES_TOKEN       Personal access token
"""

from __future__ import annotations

import contextlib
import os
import threading

import httpx

from axes.exceptions import ConfigError


_UNIX_SOCKET_BASE_URL = "http://axes"


def _build_http_client(
    *,
    endpoint: str | None,
    socket_path: str | None,
    token: str,
    timeout: float,
) -> httpx.Client:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.apache.parquet",
    }

    if socket_path:
        return httpx.Client(
            base_url=_UNIX_SOCKET_BASE_URL,
            headers=headers,
            transport=httpx.HTTPTransport(uds=socket_path),
            timeout=timeout,
        )

    return httpx.Client(
        base_url=endpoint.rstrip("/"),  # type: ignore[union-attr]
        headers=headers,
        timeout=timeout,
    )


class Client:
    """Holds connection config for one Axes endpoint.

    Most callers never instantiate this directly — the module-level
    ``sql``, ``scan``, and ``sql_to_dataframe`` functions use the
    lazily-initialised default client built from environment variables.

    Args:
        endpoint:    HTTPS base URL (e.g. ``https://app.axes.com``).
                     Ignored when *socket_path* is provided.
        socket_path: Path to the Unix domain socket exposed by the
                     API inside sandbox containers.
        token:       Bearer token (JWT or PAT).  Required.
        versions:    Default version-pin dict applied to every ``sql``
                     call unless overridden per-call.
        timeout:     HTTP timeout in seconds (default 120).
    """

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        socket_path: str | None = None,
        token: str,
        versions: dict[str, str] | None = None,
        timeout: float = 120.0,
    ) -> None:
        if not token:
            raise ConfigError("token must not be empty")
        if not endpoint and not socket_path:
            raise ConfigError(
                "Provide either endpoint= (HTTPS) or socket_path= (Unix socket)."
            )

        self.endpoint = endpoint
        self.socket_path = socket_path
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
                        socket_path=self.socket_path,
                        token=self.token,
                        timeout=self.timeout,
                    )
        return self._http

    def query(self, query: str, *, accept: str) -> contextlib.AbstractContextManager[httpx.Response]:
        """Return a streaming context manager for ``POST /api/query``.

        Args:
            query:  SQL string to execute.
            accept: MIME type for the response format (Arrow IPC or ndjson).

        Usage::

            with client.query("SELECT 1", accept="application/vnd.apache.arrow.stream") as response:
                ...
        """
        return self.http.stream(
            "POST",
            "/api/query",
            json={"query": query},
            headers={"Accept": accept},
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
        transport = self.socket_path or self.endpoint
        return f"Client(transport={transport!r})"


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

        socket_path = os.environ.get("AXES_SOCKET")
        endpoint = os.environ.get("AXES_ENDPOINT", "https://app.axes.com")

        if not socket_path and not endpoint:
            raise ConfigError(
                "Set AXES_SOCKET (Unix socket path) or AXES_ENDPOINT (HTTPS URL)."
            )

        _default_client = Client(
            endpoint=endpoint,
            socket_path=socket_path,
            token=token,
        )
    return _default_client
