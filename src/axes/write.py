"""
append_table_data() — the write side of the client.

Appends a parquet file to an existing table's data through
``POST /api/table-data-files``. This is the ingestion data-refresh path:
the file becomes visible to the next query with no dataset-version bump.
It requires a write-scoped token bound to the target dataset, which the
ingestion runner injects as ``AXES_TOKEN``; personal access tokens are
read-only and will get ``AuthError``.

The uploaded parquet's column schema must exactly match the table's
registered schema — a mismatch raises ``WriteError`` with status 409 so
ingestion scripts fail structurally instead of corrupting the table.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from axes.client import Client, get_default_client
from axes.exceptions import AuthError, WriteError


@dataclass(frozen=True)
class AppendResult:
    """The created table-data file, as reported by the server."""

    table_data_file_id: str
    parquet_hash: str
    row_count: int
    byte_count: int


def append_table_data(
    table: str,
    parquet: str | Path | bytes | pl.DataFrame,
    *,
    client: Client | None = None,
) -> AppendResult:
    """Append one parquet file to *table* of the token's dataset.

    Args:
        table:   Table name within the dataset's writable version.
        parquet: A parquet file path, raw parquet bytes, or a polars
                 DataFrame (serialized to parquet in memory).
        client:  Optional explicit client; defaults to the env-configured
                 module client.

    Returns:
        AppendResult with the created file's id, hash, and counts.

    Raises:
        AuthError:  Missing/expired token, or the token is not
                    write-scoped (401/403).
        WriteError: Invalid parquet (400), unknown table (404), or schema
                    mismatch (409).
    """
    client = client or get_default_client()
    data = _parquet_bytes(parquet)

    response = client.append_table_data(table, data)
    if response.status_code in (401, 403):
        raise AuthError(response.status_code, response.text)
    if response.status_code >= 400:
        raise WriteError(response.status_code, response.text)

    payload = response.json()["data"]
    return AppendResult(
        table_data_file_id=payload["id"],
        parquet_hash=payload["parquetHash"],
        row_count=payload["rowCount"],
        byte_count=payload["byteCount"],
    )


def _parquet_bytes(parquet: str | Path | bytes | pl.DataFrame) -> bytes:
    if isinstance(parquet, bytes):
        return parquet
    if isinstance(parquet, pl.DataFrame):
        buffer = io.BytesIO()
        parquet.write_parquet(buffer)
        return buffer.getvalue()
    return Path(parquet).read_bytes()
