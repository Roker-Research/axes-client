"""Tests for the task I/O helpers (read_payload / write_output)."""

from __future__ import annotations

import json

import pytest

from axes.exceptions import ConfigError
from axes.task import read_payload, write_output


class TestReadPayload:
    def test_reads_payload(self, tmp_path, monkeypatch):
        input_file = tmp_path / "input.json"
        input_file.write_text('{"series": ["A", "B"]}')
        monkeypatch.setenv("AXES_INPUT_FILE", str(input_file))

        assert read_payload() == {"series": ["A", "B"]}

    def test_missing_env_raises(self, monkeypatch):
        monkeypatch.delenv("AXES_INPUT_FILE", raising=False)
        with pytest.raises(ConfigError, match="AXES_INPUT_FILE"):
            read_payload()

    def test_non_object_payload_raises(self, tmp_path, monkeypatch):
        input_file = tmp_path / "input.json"
        input_file.write_text("[1, 2]")
        monkeypatch.setenv("AXES_INPUT_FILE", str(input_file))
        with pytest.raises(ConfigError, match="JSON object"):
            read_payload()


class TestWriteOutput:
    def test_writes_minimal_output(self, tmp_path, monkeypatch):
        output_file = tmp_path / "output.json"
        monkeypatch.setenv("AXES_OUTPUT_FILE", str(output_file))

        write_output()

        assert json.loads(output_file.read_text()) == {"status": "ok"}

    def test_writes_full_output(self, tmp_path, monkeypatch):
        output_file = tmp_path / "output.json"
        monkeypatch.setenv("AXES_OUTPUT_FILE", str(output_file))

        write_output(
            "ok",
            new_tasks=[{"entrypoint": "fetch.py", "payload": {"page": 2}}],
            cursor={"through": "2026-06"},
            result={"rows_written": 10},
        )

        assert json.loads(output_file.read_text()) == {
            "status": "ok",
            "new_tasks": [{"entrypoint": "fetch.py", "payload": {"page": 2}}],
            "cursor": {"through": "2026-06"},
            "result": {"rows_written": 10},
        }

    def test_missing_env_raises(self, monkeypatch):
        monkeypatch.delenv("AXES_OUTPUT_FILE", raising=False)
        with pytest.raises(ConfigError, match="AXES_OUTPUT_FILE"):
            write_output("ok")
