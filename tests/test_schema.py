"""Tests for table_schema()."""

from __future__ import annotations

import json

import pyarrow as pa
import pytest
from conftest import make_arrow
from pytest_httpx import HTTPXMock

from axes.exceptions import SchemaError
from axes.schema import table_schema

# Every arrow type the catalog can hold is stored as ``str(field.type)``;
# these are the ones a registered ACS-shaped table actually uses.
_REGISTERED = [
    {"name": "GEO_ID", "data_type": "string"},
    {"name": "sumlevel", "data_type": "string"},
    {"name": "knowledge_date", "data_type": "date32[day]"},
    {"name": "year", "data_type": "int32"},
    {"name": "B01002_E001", "data_type": "double"},
]


def _catalog_response(columns: list[dict[str, str]]) -> bytes:
    payload: bytes = make_arrow(
        {"column_schema": [json.dumps({"columns": columns})]}
    )
    return payload


class TestTableSchema:
    def test_returns_the_registered_arrow_schema(
        self, client, httpx_mock: HTTPXMock
    ):
        httpx_mock.add_response(content=_catalog_response(_REGISTERED))

        schema = table_schema("b01002_1yr", client=client)

        assert schema == pa.schema(
            [
                pa.field("GEO_ID", pa.string()),
                pa.field("sumlevel", pa.string()),
                pa.field("knowledge_date", pa.date32()),
                pa.field("year", pa.int32()),
                pa.field("B01002_E001", pa.float64()),
            ]
        )

    def test_casting_to_it_produces_the_registered_types(
        self, client, httpx_mock: HTTPXMock
    ):
        """The point of the helper: polars infers large_string and int64,
        and an append comparing str(field.type) rejects both."""
        httpx_mock.add_response(content=_catalog_response(_REGISTERED))
        import datetime

        import polars as pl

        frame = pl.DataFrame(
            {
                "GEO_ID": ["0400000US10"],
                "sumlevel": ["040"],
                "knowledge_date": [datetime.date(2023, 12, 1)],
                "year": [2023],
                "B01002_E001": [41.6],
            }
        )

        schema = table_schema("b01002_1yr", client=client)
        cast = frame.to_arrow().cast(schema)

        assert [str(field.type) for field in cast.schema] == [
            column["data_type"] for column in _REGISTERED
        ]

    def test_queries_the_named_table(self, client, httpx_mock: HTTPXMock):
        httpx_mock.add_response(content=_catalog_response(_REGISTERED))

        table_schema("b01002_1yr", client=client)

        request = httpx_mock.get_request()
        assert "b01002_1yr" in json.loads(request.content)["query"]

    def test_escapes_a_quote_in_the_table_name(
        self, client, httpx_mock: HTTPXMock
    ):
        httpx_mock.add_response(content=_catalog_response(_REGISTERED))

        table_schema("b01002'; DROP", client=client)

        request = httpx_mock.get_request()
        assert "b01002''; DROP" in json.loads(request.content)["query"]

    def test_unknown_table_raises(self, client, httpx_mock: HTTPXMock):
        httpx_mock.add_response(content=make_arrow({"column_schema": []}))

        with pytest.raises(SchemaError, match="Table not found"):
            table_schema("nope", client=client)

    def test_unsupported_column_type_raises(
        self, client, httpx_mock: HTTPXMock
    ):
        httpx_mock.add_response(
            content=_catalog_response(
                [{"name": "x", "data_type": "map<string, int32>"}]
            )
        )

        with pytest.raises(SchemaError, match="Unsupported column type"):
            table_schema("weird", client=client)


class TestRoundTrippedTypes:
    """Every type must survive str(field.type) -> arrow type."""

    @pytest.mark.parametrize(
        "arrow_type",
        [
            pa.string(),
            pa.large_string(),
            pa.int8(),
            pa.int32(),
            pa.int64(),
            pa.uint16(),
            pa.float32(),
            pa.float64(),
            pa.bool_(),
            pa.date32(),
            pa.date64(),
            pa.binary(),
            pa.timestamp("us"),
            pa.timestamp("ns", tz="UTC"),
            pa.decimal128(10, 2),
            pa.decimal256(40, 8),
        ],
    )
    def test_round_trip(self, client, httpx_mock: HTTPXMock, arrow_type):
        httpx_mock.add_response(
            content=_catalog_response(
                [{"name": "x", "data_type": str(arrow_type)}]
            )
        )

        schema = table_schema("t", client=client)

        assert schema.field("x").type == arrow_type
