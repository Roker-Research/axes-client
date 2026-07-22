"""Shared test fixtures and helpers."""

from __future__ import annotations

import io
import json

import polars as pl
import pyarrow.ipc as pa_ipc
import pytest

from axes.client import Client

# ---------------------------------------------------------------------------
# Fixture factories
# ---------------------------------------------------------------------------


def make_arrow(data: dict) -> bytes:
    """Return Arrow IPC stream bytes for a dict of {col: [values]}."""
    table = pl.DataFrame(data).to_arrow()
    sink = io.BytesIO()
    writer = pa_ipc.new_stream(sink, table.schema)
    for batch in table.to_batches():
        writer.write_batch(batch)
    writer.close()
    return sink.getvalue()


def make_parquet(data: dict) -> bytes:
    """Return parquet bytes for a dict of {col: [values]}."""
    buf = io.BytesIO()
    pl.DataFrame(data).write_parquet(buf)
    return buf.getvalue()


def make_ndjson(data: dict) -> bytes:
    """Return ndjson bytes for a dict of {col: [values]}."""
    keys = list(data.keys())
    rows = zip(*[data[k] for k in keys])
    return b"".join(
        (json.dumps(dict(zip(keys, row))) + "\n").encode() for row in rows
    )


# Default small result used across multiple tests.
SIMPLE_DATA = {"state": ["CA", "NY"], "income": [60000.0, 55000.0]}
SIMPLE_ARROW = make_arrow(SIMPLE_DATA)
SIMPLE_PARQUET = make_parquet(SIMPLE_DATA)
SIMPLE_NDJSON = make_ndjson(SIMPLE_DATA)


# ---------------------------------------------------------------------------
# Client fixture (HTTPS, no real network)
# ---------------------------------------------------------------------------


@pytest.fixture()
def client():
    """A Client pointed at a fake HTTPS endpoint."""
    return Client(
        endpoint="http://axes-test",
        token="test-token",
    )
