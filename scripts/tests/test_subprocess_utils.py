"""Tests for subprocess_utils.py."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING, Any

import pytest

import subprocess_utils
from subprocess_utils import (
    ExecutableNotFoundError,
    _build_run_kwargs,
    get_safe_executable,
    run_git_command,
    run_git_command_with_input,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestGetSafeExecutable:
    def test_returns_full_executable_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(subprocess_utils.shutil, "which", lambda command: f"/usr/bin/{command}")

        assert get_safe_executable("git") == "/usr/bin/git"

    def test_raises_when_executable_is_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(subprocess_utils.shutil, "which", lambda _command: None)

        with pytest.raises(ExecutableNotFoundError, match="Required executable 'git' not found in PATH"):
            get_safe_executable("git")


class TestBuildRunKwargs:
    def test_applies_secure_defaults(self) -> None:
        kwargs = _build_run_kwargs("test_function")

        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["check"] is True
        assert kwargs["encoding"] == "utf-8"

    def test_allows_safe_overrides_and_extra_kwargs(self) -> None:
        kwargs = _build_run_kwargs("test_function", capture_output=False, check=False, timeout=30)

        assert kwargs["capture_output"] is False
        assert kwargs["check"] is False
        assert kwargs["timeout"] == 30
        assert kwargs["text"] is True

    def test_ignores_text_override_to_keep_string_output(self) -> None:
        kwargs = _build_run_kwargs("test_function", text=False)

        assert kwargs["text"] is True

    def test_rejects_shell_true(self) -> None:
        kwargs: dict[str, Any] = {"shell": True}
        with pytest.raises(ValueError, match="shell=True is not allowed in test_function"):
            _build_run_kwargs("test_function", **kwargs)

    def test_rejects_executable_override(self) -> None:
        kwargs: dict[str, Any] = {"executable": "/malicious/fake-git"}
        with pytest.raises(ValueError, match="Overriding 'executable' is not allowed in test_function"):
            _build_run_kwargs("test_function", **kwargs)


class TestRunGitCommand:
    def test_runs_git_with_full_path_and_defaults(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        calls: dict[str, Any] = {}
        monkeypatch.setattr(subprocess_utils, "get_safe_executable", lambda command: f"/usr/bin/{command}")

        def fake_run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            calls["args"] = args
            calls["kwargs"] = kwargs
            return subprocess.CompletedProcess(args, 0, stdout="ok\n", stderr="")

        monkeypatch.setattr(subprocess_utils.subprocess, "run", fake_run)

        result = run_git_command(["status", "--short"], cwd=tmp_path)

        assert result.stdout == "ok\n"
        assert calls["args"] == ["/usr/bin/git", "status", "--short"]
        assert calls["kwargs"]["cwd"] == tmp_path
        assert calls["kwargs"]["capture_output"] is True
        assert calls["kwargs"]["text"] is True
        assert calls["kwargs"]["check"] is True
        assert calls["kwargs"]["encoding"] == "utf-8"

    def test_passes_safe_overrides_to_subprocess(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: dict[str, Any] = {}
        monkeypatch.setattr(subprocess_utils, "get_safe_executable", lambda _command: "/usr/bin/git")

        def fake_run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            calls["args"] = args
            calls["kwargs"] = kwargs
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="bad")

        monkeypatch.setattr(subprocess_utils.subprocess, "run", fake_run)

        result = run_git_command(["bad-subcommand"], check=False, timeout=10)

        assert result.returncode == 1
        assert calls["kwargs"]["check"] is False
        assert calls["kwargs"]["timeout"] == 10

    def test_rejects_insecure_kwargs_before_running(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called = False
        monkeypatch.setattr(subprocess_utils, "get_safe_executable", lambda _command: "/usr/bin/git")

        def fake_run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
            nonlocal called
            called = True
            return subprocess.CompletedProcess([], 0)

        monkeypatch.setattr(subprocess_utils.subprocess, "run", fake_run)

        kwargs: dict[str, Any] = {"shell": True}
        with pytest.raises(ValueError, match="shell=True is not allowed"):
            run_git_command(["status"], **kwargs)

        assert called is False


class TestRunGitCommandWithInput:
    def test_passes_stdin_to_git(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        calls: dict[str, Any] = {}
        monkeypatch.setattr(subprocess_utils, "get_safe_executable", lambda _command: "/usr/bin/git")

        def fake_run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            calls["args"] = args
            calls["kwargs"] = kwargs
            return subprocess.CompletedProcess(args, 0, stdout="hash\n", stderr="")

        monkeypatch.setattr(subprocess_utils.subprocess, "run", fake_run)

        result = run_git_command_with_input(["hash-object", "--stdin"], "content", cwd=tmp_path)

        assert result.stdout == "hash\n"
        assert calls["args"] == ["/usr/bin/git", "hash-object", "--stdin"]
        assert calls["kwargs"]["cwd"] == tmp_path
        assert calls["kwargs"]["input"] == "content"
        assert calls["kwargs"]["text"] is True

    def test_rejects_executable_override(self) -> None:
        kwargs: dict[str, Any] = {"executable": "/malicious/fake-git"}
        with pytest.raises(ValueError, match="Overriding 'executable' is not allowed"):
            run_git_command_with_input(["hash-object", "--stdin"], "content", **kwargs)
