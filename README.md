# Axes Client Library

Python client for the [Axes](https://axes.com) data API. Send SQL, get parquet back.

```sh
pip install axes-client
```

## Quickstart

```python
from axes import sql

result = sql("SELECT state, AVG(income) FROM american_community_survey.demographics GROUP BY state")

result.rows        # int — number of rows
result.bytes       # int — parquet response size in bytes
result.columns     # list[str] — ordered column names
result.elapsed_ms  # int — wall-clock time for the request

# Check size before loading into memory
if result.bytes < 100 * 1024 * 1024:
    df = result.collect()   # polars DataFrame — full result in memory
else:
    lf = result.scan()      # polars LazyFrame — safe for large results

# Save to parquet
result.save("/work/result.parquet")

# Stream directly to a file — never buffers the full response in memory
result = sql(
    "SELECT * FROM american_community_survey.demographics",
    out="/work/result.parquet",
)
lf = result.scan()  # polars LazyFrame over the file
```

Results under 10 MB are kept in memory. Larger results spill to a temp file
in ``/tmp`` and are deleted when the ``SqlResult`` is garbage collected.
``.scan()`` is always safe — it returns a ``scan_parquet`` LazyFrame when
spilled rather than loading into memory.

## Configuration

```sh
export AXES_TOKEN=your-personal-access-token
```

`AXES_ENDPOINT` defaults to `https://app.axes.com`. Override it if you are
running a self-hosted instance:

```sh
export AXES_ENDPOINT=https://your-axes-instance.com
export AXES_TOKEN=your-personal-access-token
```

## Explicit client

```python
from axes import Client, sql

client = Client(
    endpoint="https://your-axes-instance.com",
    token="your-token",
)

result = sql("SELECT * FROM american_community_survey.demographics", client=client)
```

## CLI

```sh
# Stream rows to stdout as newline-delimited JSON (ndjson), one object per line
axes sql "SELECT state, AVG(income) FROM american_community_survey.demographics GROUP BY state"

# Pipe to jq — each line is a standalone JSON object
axes sql "SELECT state FROM american_community_survey.demographics" | jq '.state'

# Write parquet and print a JSON summary
axes sql "SELECT * FROM american_community_survey.demographics" --out /work/result.parquet
```

When `--out` is provided, a JSON summary is printed to stdout:

```json
{
  "path": "/work/result.parquet",
  "rows": 51,
  "bytes": 4096,
  "columns": ["state", "income"],
  "elapsed_ms": 340
}
```

## Errors

```python
from axes.exceptions import QueryError, ResultTooLarge, AuthError

try:
    result = sql("SELECT * FROM american_community_survey.demographics")
except QueryError as e:
    print(e.message)      # SQL rejected by the server (400)
except ResultTooLarge as e:
    print(e.message)      # Exceeded row/byte cap (413)
except AuthError as e:
    print(e.status_code)  # 401 or 403
```

## Appending data

`append_table_data` uploads a parquet file to an existing table. It needs a
write-scoped token bound to the target dataset — the ingestion runner injects
one as `AXES_TOKEN`; personal access tokens are read-only and raise `AuthError`.

```python
from axes import append_table_data

# Accepts a polars DataFrame, a parquet path, or raw parquet bytes
result = append_table_data("demographics", df)

result.row_count           # rows written
result.byte_count          # bytes written
result.table_data_file_id  # id of the created file

# Atomically retire this firing's earlier files instead of adding to them
append_table_data("demographics", df, replace=True)
```

The uploaded parquet's columns must match the table's registered schema; a
mismatch raises `WriteError` (409).

## Agent framework

`axes.agent` is the container-side framework for authoring [Chat
Plot](../chat-plot) agents. Agents already read the data catalog through this
client, so the framework ships here rather than as a separate package.

```python
from axes.agent import Agent, Tool, RunContext
```

Author an agent as an `Agent` subclass with `Tool` members (and other `Agent`
instances as subagents); the `axes-agent` console script reads one step off
stdin, runs `plan_step` or `run_tool`, and writes the result to stdout. Chat
Plot drives the multi-turn loop and owns the LLM call, persistence, and
streaming. See [`weather-agent`](../weather-agent) for a worked example.

## Development

```sh
uv sync

uv run task test    # pytest
uv run task lint    # ruff check
uv run task format  # ruff format
uv run task check   # mypy (strict)
```

`task fix` runs `ruff check --fix`. Linting (ruff), formatting, and strict type
checking (mypy) all run in CI on every push and pull request.
