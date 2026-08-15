"""Tests for user-supplied path validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from garland.paths import (
    PathTraversalError,
    ensure_directory,
    read_text_file,
    resolve_under_base,
    resolve_user_path,
    write_json_file,
    write_text_file,
)


class TestResolveUserPath:
    def test_relative_path_under_cwd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(tmp_path)
        config = tmp_path / "sim.yaml"
        config.write_text("n_agents: 1\n", encoding="utf-8")
        resolved = resolve_user_path("sim.yaml")
        assert resolved == config.resolve()

    def test_absolute_path(self, tmp_path: Path):
        config = tmp_path / "sim.yaml"
        config.write_text("n_agents: 1\n", encoding="utf-8")
        resolved = resolve_user_path(config)
        assert resolved == config.resolve()

    def test_rejects_traversal(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(PathTraversalError):
            resolve_user_path("../outside.yaml")


class TestResolveUnderBase:
    def test_child_path(self, tmp_path: Path):
        resolved = resolve_under_base(tmp_path, "metrics.csv")
        assert resolved == (tmp_path / "metrics.csv").resolve()

    def test_rejects_escape(self, tmp_path: Path):
        with pytest.raises(PathTraversalError):
            resolve_under_base(tmp_path, "../escape.csv")


class TestValidatedIO:
    def test_read_write_text_roundtrip(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(tmp_path)
        path = write_text_file("out/notes.txt", "hello")
        assert path == (tmp_path / "out" / "notes.txt").resolve()
        assert read_text_file("out/notes.txt") == "hello"

    def test_write_json_and_ensure_directory(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(tmp_path)
        out = ensure_directory("artifacts")
        assert out.is_dir()
        path = write_json_file("artifacts/summary.json", {"ok": True})
        assert json.loads(path.read_text(encoding="utf-8")) == {"ok": True}

    def test_relative_io_rejects_traversal(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(PathTraversalError):
            read_text_file("../secret.txt")
        with pytest.raises(PathTraversalError):
            write_text_file("../escape.txt", "nope")
        with pytest.raises(PathTraversalError):
            write_json_file("../escape.json", {})
        with pytest.raises(PathTraversalError):
            ensure_directory("../escape_dir")
