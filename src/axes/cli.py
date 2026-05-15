"""
CLI entry point — thin wrapper around the axes library.

    axes sql "<query>" [--out path]

Without --out, rows are streamed as newline-delimited JSON (ndjson) to
stdout as they arrive from the server — pipe to jq or process line by line.
With --out, the full result is fetched as parquet and written to that path,
and a JSON summary is printed instead.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

from axes.client import get_default_client
from axes.exceptions import AxesError, AuthError, QueryError, ResultTooLarge
from axes.sql import sql

_NDJSON_MIME = "application/x-ndjson"


@click.group()
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable debug logging.",
)
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """Axes data client."""
    if verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(relativeCreated)6dms %(name)s %(message)s",
            stream=sys.stderr,
        )


@main.command("sql")
@click.argument("query")
@click.option(
    "--out",
    default=None,
    metavar="PATH",
    help="Write result to this path as parquet and print a JSON summary. "
         "Without this flag, rows are streamed as ndjson to stdout.",
)
@click.option(
    "--pretty",
    is_flag=True,
    default=False,
    help="Pretty-print JSON output with indentation. Only applies without --out.",
)
def sql_cmd(query: str, out: str | None, pretty: bool) -> None:
    """Run a SQL query and print the results.

    Without --out, rows stream to stdout as newline-delimited JSON as they
    arrive from the server — pipe to jq or process line by line.
    With --out, writes parquet to PATH and prints a JSON summary instead.

    \b
    Examples:
        axes sql "SELECT state, AVG(income) FROM acs.demographics GROUP BY state"
        axes sql "SELECT state FROM acs.demographics" | jq '.state'
        axes sql "SELECT * FROM acs.demographics" --out /work/result.parquet
        axes sql "SELECT state FROM acs.demographics" --pretty
    """
    if out is None:
        try:
            _stream_ndjson(query, pretty=pretty)
        except AxesError as exc:
            _die(exc)
    else:
        try:
            result = sql(query, out=out)
        except AxesError as exc:
            _die(exc)
        dest = Path(out)
        click.echo(
            json.dumps(
                {
                    "path": str(dest),
                    "rows": result.rows,
                    "bytes": result.bytes,
                    "columns": result.columns,
                    "elapsed_ms": result.elapsed_ms,
                }
            )
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stream_ndjson(query: str, pretty: bool) -> None:
    """Stream query results as ndjson from the server to stdout."""
    client = get_default_client()
    indent = 2 if pretty else None

    with client.query(query, accept=_NDJSON_MIME) as response:
        if response.status_code == 401 or response.status_code == 403:
            response.read()
            raise AuthError(response.status_code, response.text)
        if response.status_code == 400:
            response.read()
            raise QueryError(response.text or "Bad query", query=query)
        if response.status_code == 413:
            response.read()
            raise ResultTooLarge(response.text or "Result exceeded server caps")
        if response.status_code >= 400:
            response.read()
            _die(QueryError(f"HTTP {response.status_code}: {response.text}", query=query))

        for line in response.iter_lines():
            if not line:
                continue
            if indent is not None:
                line = json.dumps(json.loads(line), indent=indent)
            click.echo(line)


def _die(exc: Exception) -> None:
    click.echo(f"Error: {exc}", err=True)
    sys.exit(1)
