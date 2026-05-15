"""Tests for the Client class and default-client env-var resolution."""

from __future__ import annotations

import pytest

import axes.client as client_module
from axes.client import Client, get_default_client
from axes.exceptions import ConfigError


class TestClientInit:
    def test_requires_token(self):
        with pytest.raises(ConfigError, match="token"):
            Client(endpoint="http://axes", token="")

    def test_requires_transport(self):
        with pytest.raises(ConfigError, match="endpoint"):
            Client(token="tok")

    def test_endpoint_ok(self):
        client = Client(endpoint="http://axes", token="tok")
        assert client.endpoint == "http://axes"
        assert client.socket_path is None

    def test_socket_ok(self):
        client = Client(socket_path="/run/axes/api.sock", token="tok")
        assert client.socket_path == "/run/axes/api.sock"
        assert client.endpoint is None

    def test_default_versions_empty(self):
        client = Client(endpoint="http://axes", token="tok")
        assert client.versions == {}

    def test_custom_versions(self):
        pins = {"acs": "abc-123"}
        client = Client(endpoint="http://axes", token="tok", versions=pins)
        assert client.versions == pins

    def test_context_manager_closes(self):
        client = Client(endpoint="http://axes", token="tok")
        with client:
            _ = client.http
        assert client._http is None


class TestGetDefaultClient:
    def setup_method(self):
        client_module._default_client = None

    def test_missing_token(self, monkeypatch):
        monkeypatch.delenv("AXES_TOKEN", raising=False)
        with pytest.raises(ConfigError, match="AXES_TOKEN"):
            get_default_client()

    def test_default_endpoint(self, monkeypatch):
        monkeypatch.setenv("AXES_TOKEN", "tok")
        monkeypatch.delenv("AXES_SOCKET", raising=False)
        monkeypatch.delenv("AXES_ENDPOINT", raising=False)
        client = get_default_client()
        assert client.endpoint == "https://app.axes.com"

    def test_prefers_socket_over_endpoint(self, monkeypatch):
        monkeypatch.setenv("AXES_TOKEN", "tok")
        monkeypatch.setenv("AXES_SOCKET", "/run/axes/api.sock")
        monkeypatch.setenv("AXES_ENDPOINT", "http://axes")
        client = get_default_client()
        assert client.socket_path == "/run/axes/api.sock"

    def test_endpoint_only(self, monkeypatch):
        monkeypatch.setenv("AXES_TOKEN", "tok")
        monkeypatch.delenv("AXES_SOCKET", raising=False)
        monkeypatch.setenv("AXES_ENDPOINT", "http://axes")
        client = get_default_client()
        assert client.endpoint == "http://axes"
        assert client.socket_path is None

    def test_returns_same_instance(self, monkeypatch):
        monkeypatch.setenv("AXES_TOKEN", "tok")
        monkeypatch.setenv("AXES_ENDPOINT", "http://axes")
        assert get_default_client() is get_default_client()
