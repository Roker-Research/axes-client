# AGENTS.md

## What this repo is

`axes-client` (PyPI: `axes-client`, import: `axes`) — Python client library for
the Axes HTTP API. Sends SQL to `POST /api/query`, streams parquet back.

Spec: `../chat-plot/specs/002-integrated-data-catalog.md` — read this before
touching the public API surface.

## Package layout

Single distribution (PyPI: `axes-client`, import: `axes`) built from a
root-level `src/axes/` tree. The `axes.agent` framework ships in the same
distribution as a subpackage.

```
src/axes/
  __init__.py      public re-exports: Client, sql, SqlResult
  client.py        Client class + get_default_client() from env vars; _build_http_client()
  sql.py           sql(), SqlResult, sql_to_dataframe()
  exceptions.py    QueryError, ResultTooLarge, AuthError, ConfigError
  cli.py           axes sql / axes scan CLI commands (entry point: axes)
  agent/           Agent, Tool, RunContext + the wire protocol (entry point: axes-agent)
tests/
  conftest.py        make_parquet(), SIMPLE_PARQUET fixture, client fixture
  agent/             agent-framework tests
```

## Install / dev setup

```sh
uv sync                           # create .venv, install deps + dev deps
uv run pytest tests/ -v           # run tests
```

Dev dependencies (`pytest`, `pytest-httpx`) are declared under
`[dependency-groups] dev` in `pyproject.toml` and managed by uv.

`hatchling` must be installed for editable installs to work:
```sh
pip install hatchling
```

## Running tests

```sh
uv run pytest tests/ -v                 # all tests (~0.2s, no services needed)
uv run pytest tests/test_sql.py -v      # single file
```

All tests are pure unit tests using `pytest-httpx` to mock the API — no
running server, no GCS, no real network.

## Environment variables

| Var | Purpose |
|---|---|
| `AXES_SOCKET` | Unix socket path (preferred over AXES_ENDPOINT; set in sandbox) |
| `AXES_ENDPOINT` | HTTPS base URL for external/user use |
| `AXES_TOKEN` | JWT (sandbox, injected automatically) or PAT (external users) |
| `AXES_CACHE_DIR` | Directory for parquet tempfiles; defaults to `~/.cache/axes/` |

Library prefers `AXES_SOCKET` over `AXES_ENDPOINT` when both are set.
The module-level default `Client` is lazily initialised from these vars on
first call to `sql()`/`scan()`; tests reset it via `axes.client._default_client = None`.

## Key design constraints from the spec

- `sql()` **never buffers the full response in memory** — streams parquet
  bytes to disk, then reads schema from the footer via `pl.scan_parquet`.
- Error responses (non-200) must have `response.read()` called before
  accessing `.text` on a streaming httpx response — already handled in
  `sql.py`, don't regress this.
- `scan()` is v1 pragmatic: issues `SELECT *`, returns `pl.scan_parquet` over
  the tempfile. Polars filters apply locally after collection, not as pushed
  SQL predicates.
- `scan_cmd` in the CLI calls `sql()` directly — do **not** also call
  `scan()` from the CLI or you'll make two HTTP requests.
- Use `pl.scan_parquet(...).collect_schema()` not `.schema` (deprecated,
  causes `PerformanceWarning`).
- Version pins: per-call `versions=` dict overrides client-level defaults;
  both are merged before sending as the `versions` field in the request body.
  If the merged dict is empty, omit the key entirely.

## CLI

```sh
axes sql "SELECT state, AVG(income) FROM acs.demographics GROUP BY state" \
    --out /work/result.parquet
axes scan acs.demographics --out /work/acs.parquet
```

Both print a JSON summary to stdout. `--versions` accepts a JSON object string.
