"""Tests for user-supplied path validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from garland.paths import PathTraversalError, resolve_under_base, resolve_user_path


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
