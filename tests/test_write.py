"""Tests for append_table_data()."""

from __future__ import annotations

import json

import polars as pl
import pytest
from conftest import SIMPLE_DATA, SIMPLE_PARQUET

from axes.exceptions import AuthError, WriteError
from axes.write import append_table_data

_URL = "http://axes-test/api/table-data-files?table=income"

_SUCCESS_BODY = {
    "data": {
        "id": "8b6ee1a2-3c50-4f0d-9a72-0f2f4a9be111",
        "tableDataId": "0f2f4a9b-e111-4f0d-9a72-8b6ee1a23c50",
        "parquetHash": "abc123",
        "rowCount": 2,
        "byteCount": 1234,
        "createdAt": "2026-07-16T00:00:00Z",
        "createdBy": None,
        "retiredAt": None,
    }
}


class TestAppendTableData:
    def test_appends_bytes(self, client, httpx_mock):
        httpx_mock.add_response(method="POST", url=_URL, json=_SUCCESS_BODY)

        result = append_table_data("income", SIMPLE_PARQUET, client=client)

        assert result.table_data_file_id == _SUCCESS_BODY["data"]["id"]
        assert result.parquet_hash == "abc123"
        assert result.row_count == 2
        assert result.byte_count == 1234

        request = httpx_mock.get_request()
        assert request.headers["Content-Type"] == "application/octet-stream"
        assert bytes(request.content) == SIMPLE_PARQUET

    def test_appends_dataframe(self, client, httpx_mock):
        httpx_mock.add_response(method="POST", url=_URL, json=_SUCCESS_BODY)

        append_table_data("income", pl.DataFrame(SIMPLE_DATA), client=client)

        request = httpx_mock.get_request()
        frame = pl.read_parquet(bytes(request.content))
        assert frame.to_dict(as_series=False) == SIMPLE_DATA

    def test_appends_arrow_table(self, client, httpx_mock):
        """The shape a script has after casting to table_schema()."""
        httpx_mock.add_response(method="POST", url=_URL, json=_SUCCESS_BODY)

        append_table_data(
            "income", pl.DataFrame(SIMPLE_DATA).to_arrow(), client=client
        )

        request = httpx_mock.get_request()
        frame = pl.read_parquet(bytes(request.content))
        assert frame.to_dict(as_series=False) == SIMPLE_DATA

    def test_appends_path(self, client, httpx_mock, tmp_path):
        parquet_path = tmp_path / "chunk.parquet"
        parquet_path.write_bytes(SIMPLE_PARQUET)
        httpx_mock.add_response(method="POST", url=_URL, json=_SUCCESS_BODY)

        append_table_data("income", parquet_path, client=client)

        request = httpx_mock.get_request()
        assert bytes(request.content) == SIMPLE_PARQUET

    def test_forbidden_raises_auth_error(self, client, httpx_mock):
        httpx_mock.add_response(
            method="POST",
            url=_URL,
            status_code=403,
            text=json.dumps(
                {"detail": "A write-scoped access token is required"}
            ),
        )

        with pytest.raises(AuthError) as error:
            append_table_data("income", SIMPLE_PARQUET, client=client)
        assert error.value.status_code == 403

    def test_schema_mismatch_raises_write_error(self, client, httpx_mock):
        httpx_mock.add_response(
            method="POST",
            url=_URL,
            status_code=409,
            text=json.dumps({"detail": "Parquet schema does not match"}),
        )

        with pytest.raises(WriteError) as error:
            append_table_data("income", SIMPLE_PARQUET, client=client)
        assert error.value.status_code == 409
