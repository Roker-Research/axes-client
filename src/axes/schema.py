"""
table_schema() — the arrow schema a table is registered with.

An append must match the target table's registered column schema exactly,
so build the parquet against this rather than the types a reader happens
to infer::

    from axes import append_table_data, table_schema

    schema = table_schema("b01002_1yr")
    append_table_data("b01002_1yr", frame.to_arrow().cast(schema))
"""

from __future__ import annotations

import json
import re

import pyarrow as pa

from axes.client import Client
from axes.exceptions import SchemaError
from axes.sql import sql

# The catalog stores each column's type as ``str(field.type)``.
# ``pa.type_for_alias`` reverses that for everything except these two.
_TIMESTAMP = re.compile(r"^timestamp\[(\w+), tz=(.+)\]$")
_DECIMAL = re.compile(r"^decimal(128|256)\((\d+), (\d+)\)$")


def _arrow_type(data_type: str) -> pa.DataType:
    timestamp = _TIMESTAMP.match(data_type)
    if timestamp is not None:
        return pa.timestamp(timestamp.group(1), tz=timestamp.group(2))

    decimal = _DECIMAL.match(data_type)
    if decimal is not None:
        bits, precision, scale = (int(part) for part in decimal.groups())
        factory = pa.decimal128 if bits == 128 else pa.decimal256
        return factory(precision, scale)

    try:
        return pa.type_for_alias(data_type)
    except ValueError as error:
        raise SchemaError(f"Unsupported column type '{data_type}'") from error


def table_schema(table: str, *, client: Client | None = None) -> pa.Schema:
    """Return the arrow schema *table* is registered with.

    Args:
        table:  Table name within the dataset the token is scoped to.
        client: Optional explicit client; defaults to the env-configured
                module client.

    Raises:
        SchemaError: No such table in scope, or it carries a column type
                     this client cannot reconstruct.
    """
    escaped = table.replace("'", "''")
    result = sql(
        "SELECT column_schema FROM metadata.tables "
        f"WHERE table_name = '{escaped}'",
        client=client,
    ).collect()
    if result.height == 0:
        raise SchemaError(f"Table not found: {table}")

    raw = result["column_schema"][0]
    columns = (json.loads(raw) if isinstance(raw, str) else raw)["columns"]
    return pa.schema(
        [
            pa.field(column["name"], _arrow_type(column["data_type"]))
            for column in columns
        ]
    )
