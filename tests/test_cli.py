"""Tests for the CLI (axes sql)."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner
from conftest import SIMPLE_ARROW, SIMPLE_NDJSON
from pytest_httpx import HTTPXMock

from axes.cli import main


@pytest.fixture()
def runner():
    return CliRunner()


@pytest.fixture(autouse=True)
def _set_env(monkeypatch):
    """Point the default client at the mock server for every CLI test."""
    monkeypatch.setenv("AXES_TOKEN", "cli-test-token")
    monkeypatch.setenv("AXES_ENDPOINT", "http://axes-test")
    import axes.client as _c

    _c._default_client = None


class TestSqlCmd:
    def test_exits_zero(self, runner, httpx_mock: HTTPXMock):
        httpx_mock.add_response(content=SIMPLE_NDJSON, status_code=200)
        result = runner.invoke(main, ["sql", "SELECT * FROM acs.demographics"])
        assert result.exit_code == 0, result.output

    def test_default_output_is_ndjson(self, runner, httpx_mock: HTTPXMock):
        """Without --out, stdout is ndjson — one JSON object per line."""
        httpx_mock.add_response(content=SIMPLE_NDJSON, status_code=200)
        result = runner.invoke(main, ["sql", "SELECT * FROM acs.demographics"])
        lines = [ln for ln in result.output.splitlines() if ln]
        assert len(lines) == 2
        row0 = json.loads(lines[0])
        row1 = json.loads(lines[1])
        assert set(row0.keys()) == {"state", "income"}
        assert row0["state"] == "CA"
        assert row1["state"] == "NY"

    def test_explicit_out_flag_writes_parquet_and_prints_summary(
        self, runner, httpx_mock: HTTPXMock, tmp_path
    ):
        """With --out, parquet is written and a JSON summary is printed."""
        httpx_mock.add_response(content=SIMPLE_ARROW, status_code=200)
        out = tmp_path / "result.parquet"
        result = runner.invoke(
            main,
            ["sql", "SELECT * FROM acs.demographics", "--out", str(out)],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["path"] == str(out)
        assert "rows" in data
        assert "bytes" in data
        assert "columns" in data
        assert "elapsed_ms" in data
        assert out.exists()

    def test_pretty_flag_indents_ndjson(self, runner, httpx_mock: HTTPXMock):
        """--pretty indents each ndjson row across multiple lines."""
        httpx_mock.add_response(content=SIMPLE_NDJSON, status_code=200)
        result = runner.invoke(
            main, ["sql", "SELECT * FROM acs.demographics", "--pretty"]
        )
        assert result.exit_code == 0
        # Pretty output spans multiple lines per object — more lines than rows.
        assert result.output.count("\n") > 2

    def test_pretty_with_out_indents_summary(
        self, runner, httpx_mock: HTTPXMock, tmp_path
    ):
        httpx_mock.add_response(content=SIMPLE_ARROW, status_code=200)
        out = tmp_path / "result.parquet"
        result = runner.invoke(
            main,
            [
                "sql",
                "SELECT * FROM acs.demographics",
                "--out",
                str(out),
                "--pretty",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "rows" in data
        assert "\n" in result.output

    def test_query_error_exits_nonzero(self, runner, httpx_mock: HTTPXMock):
        httpx_mock.add_response(status_code=400, text="bad SQL")
        result = runner.invoke(main, ["sql", "INVALID"])
        assert result.exit_code != 0

    def test_auth_error_exits_nonzero(self, runner, httpx_mock: HTTPXMock):
        httpx_mock.add_response(status_code=401, text="unauthorized")
        result = runner.invoke(main, ["sql", "SELECT 1"])
        assert result.exit_code != 0
