"""Tests for sql() and SqlResult."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from pytest_httpx import HTTPXMock

from axes.exceptions import AuthError, QueryError, ResultTooLarge
from axes.sql import SqlResult, sql

from conftest import SIMPLE_ARROW, SIMPLE_DATA, make_arrow


class TestSqlSuccess:
    def test_returns_sql_result(self, client, httpx_mock: HTTPXMock):
        httpx_mock.add_response(content=SIMPLE_ARROW, status_code=200)
        result = sql("SELECT * FROM acs.demographics", client=client)
        assert isinstance(result, SqlResult)

    def test_result_columns(self, client, httpx_mock: HTTPXMock):
        httpx_mock.add_response(content=SIMPLE_ARROW, status_code=200)
        result = sql("SELECT * FROM acs.demographics", client=client)
        assert set(result.columns) == {"state", "income"}

    def test_result_rows(self, client, httpx_mock: HTTPXMock):
        httpx_mock.add_response(content=SIMPLE_ARROW, status_code=200)
        result = sql("SELECT * FROM acs.demographics", client=client)
        assert result.rows == 2

    def test_result_bytes_positive(self, client, httpx_mock: HTTPXMock):
        httpx_mock.add_response(content=SIMPLE_ARROW, status_code=200)
        result = sql("SELECT * FROM acs.demographics", client=client)
        assert result.bytes > 0

    def test_elapsed_ms_non_negative(self, client, httpx_mock: HTTPXMock):
        httpx_mock.add_response(content=SIMPLE_ARROW, status_code=200)
        result = sql("SELECT * FROM acs.demographics", client=client)
        assert result.elapsed_ms >= 0

    def test_save_writes_parquet(self, client, httpx_mock: HTTPXMock, tmp_path):
        httpx_mock.add_response(content=SIMPLE_ARROW, status_code=200)
        result = sql("SELECT * FROM acs.demographics", client=client)
        out = tmp_path / "output.parquet"
        dest = result.save(out)
        assert dest == out
        assert out.exists()
        assert pl.read_parquet(out).shape == (2, 2)


class TestSqlOutArg:
    def test_out_writes_file(self, client, httpx_mock: HTTPXMock, tmp_path):
        httpx_mock.add_response(content=SIMPLE_ARROW, status_code=200)
        out = tmp_path / "result.parquet"
        sql("SELECT * FROM acs.demographics", out=out, client=client)
        assert out.exists()
        assert pl.read_parquet(out).shape == (2, 2)

    def test_out_returns_sql_result(self, client, httpx_mock: HTTPXMock, tmp_path):
        httpx_mock.add_response(content=SIMPLE_ARROW, status_code=200)
        result = sql("SELECT * FROM acs.demographics", out=tmp_path / "r.parquet", client=client)
        assert isinstance(result, SqlResult)

    def test_out_result_metadata(self, client, httpx_mock: HTTPXMock, tmp_path):
        httpx_mock.add_response(content=SIMPLE_ARROW, status_code=200)
        result = sql("SELECT * FROM acs.demographics", out=tmp_path / "r.parquet", client=client)
        assert result.rows == 2
        assert set(result.columns) == {"state", "income"}
        assert result.bytes > 0

    def test_out_result_collect(self, client, httpx_mock: HTTPXMock, tmp_path):
        httpx_mock.add_response(content=SIMPLE_ARROW, status_code=200)
        result = sql("SELECT * FROM acs.demographics", out=tmp_path / "r.parquet", client=client)
        df = result.collect()
        assert isinstance(df, pl.DataFrame)
        assert df.shape == (2, 2)

    def test_out_result_scan(self, client, httpx_mock: HTTPXMock, tmp_path):
        httpx_mock.add_response(content=SIMPLE_ARROW, status_code=200)
        result = sql("SELECT * FROM acs.demographics", out=tmp_path / "r.parquet", client=client)
        lf = result.scan()
        assert isinstance(lf, pl.LazyFrame)
        assert len(lf.collect()) == 2

    def test_out_file_not_deleted_on_gc(self, client, httpx_mock: HTTPXMock, tmp_path):
        httpx_mock.add_response(content=SIMPLE_ARROW, status_code=200)
        out = tmp_path / "r.parquet"
        result = sql("SELECT * FROM acs.demographics", out=out, client=client)
        del result
        assert out.exists()

    def test_no_versions_key_when_empty(self, client, httpx_mock: HTTPXMock):
        httpx_mock.add_response(content=SIMPLE_ARROW, status_code=200)
        sql("SELECT * FROM acs.demographics", client=client)
        request = httpx_mock.get_request()
        body = json.loads(request.content)
        assert "versions" not in body

    def test_auth_header_sent(self, client, httpx_mock: HTTPXMock):
        httpx_mock.add_response(content=SIMPLE_ARROW, status_code=200)
        sql("SELECT 1", client=client)
        request = httpx_mock.get_request()
        assert request.headers["authorization"] == "Bearer test-token"

    def test_accept_header_is_arrow(self, client, httpx_mock: HTTPXMock):
        httpx_mock.add_response(content=SIMPLE_ARROW, status_code=200)
        sql("SELECT 1", client=client)
        request = httpx_mock.get_request()
        assert request.headers["accept"] == "application/vnd.apache.arrow.stream"


class TestSqlResultAccessors:
    def test_scan_returns_lazy_frame(self, client, httpx_mock: HTTPXMock):
        httpx_mock.add_response(content=SIMPLE_ARROW, status_code=200)
        result = sql("SELECT * FROM acs.demographics", client=client)
        lf = result.scan()
        assert isinstance(lf, pl.LazyFrame)

    def test_collect_returns_dataframe(self, client, httpx_mock: HTTPXMock):
        httpx_mock.add_response(content=SIMPLE_ARROW, status_code=200)
        result = sql("SELECT * FROM acs.demographics", client=client)
        df = result.collect()
        assert isinstance(df, pl.DataFrame)
        assert df.shape == (2, 2)

    def test_collect_data_matches_fixture(self, client, httpx_mock: HTTPXMock):
        httpx_mock.add_response(content=SIMPLE_ARROW, status_code=200)
        result = sql("SELECT * FROM acs.demographics", client=client)
        df = result.collect().sort("state")
        assert df["state"].to_list() == ["CA", "NY"]

    def test_spilled_result_scan_returns_lazy_frame(self, client, httpx_mock: HTTPXMock):
        """A SqlResult backed by a spill file also returns a LazyFrame from scan()."""
        large_data = {"state": ["CA"] * 1000, "income": [60000.0] * 1000}
        large_arrow = make_arrow(large_data)
        httpx_mock.add_response(content=large_arrow, status_code=200)
        result = sql("SELECT * FROM acs.demographics", client=client)
        lf = result.scan()
        assert isinstance(lf, pl.LazyFrame)
        assert len(lf.collect()) == 1000

    def test_spilled_result_collect_returns_dataframe(self, client, httpx_mock: HTTPXMock):
        large_data = {"state": ["CA"] * 1000, "income": [60000.0] * 1000}
        large_arrow = make_arrow(large_data)
        httpx_mock.add_response(content=large_arrow, status_code=200)
        result = sql("SELECT * FROM acs.demographics", client=client)
        df = result.collect()
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 1000


class TestSqlErrors:
    def test_400_raises_query_error(self, client, httpx_mock: HTTPXMock):
        httpx_mock.add_response(status_code=400, text="syntax error near 'FORM'")
        with pytest.raises(QueryError, match="syntax error"):
            sql("SELECT * FORM acs.demographics", client=client)

    def test_413_raises_result_too_large(self, client, httpx_mock: HTTPXMock):
        httpx_mock.add_response(status_code=413, text="result exceeded 10M rows")
        with pytest.raises(ResultTooLarge, match="10M rows"):
            sql("SELECT * FROM acs.demographics", client=client)

    def test_401_raises_auth_error(self, client, httpx_mock: HTTPXMock):
        httpx_mock.add_response(status_code=401, text="unauthorized")
        with pytest.raises(AuthError) as exc_info:
            sql("SELECT 1", client=client)
        assert exc_info.value.status_code == 401

    def test_403_raises_auth_error(self, client, httpx_mock: HTTPXMock):
        httpx_mock.add_response(status_code=403, text="forbidden")
        with pytest.raises(AuthError) as exc_info:
            sql("SELECT 1", client=client)
        assert exc_info.value.status_code == 403
