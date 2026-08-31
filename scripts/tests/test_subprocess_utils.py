"""Tests for subprocess_utils.py."""

import io
import shutil
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
    run_safe_command,
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
    @pytest.mark.parametrize(
        "contents",
        ["", "line\n", "line\r\n", "line\r", "line \u00e9\nsecond\r\nlast\r\0", b"line\r\n", b"byte\xe9\r\n\0"],
        ids=["empty", "lf", "crlf", "cr", "mixed-unicode", "bytes-crlf", "raw-bytes"],
    )
    def test_preserves_stdin_bytes_with_windows_text_pipes(self, monkeypatch: pytest.MonkeyPatch, contents: str | bytes) -> None:
        monkeypatch.setattr(subprocess_utils, "get_safe_executable", lambda _command: "/tools/git")

        def windows_run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
            payload = kwargs["input"]
            text_mode = kwargs.get("text") or kwargs.get("encoding") or kwargs.get("universal_newlines")
            if text_mode:
                # Windows text-mode stdin translates every LF, including the LF in CRLF.
                with io.BytesIO() as buffer, io.TextIOWrapper(buffer, encoding="utf-8", newline="\r\n", write_through=True) as pipe:
                    pipe.write(payload)
                    payload = buffer.getvalue()
            output = payload.hex() + "\n"
            return subprocess.CompletedProcess(args, 0, output if text_mode else output.encode(), "" if text_mode else b"")

        monkeypatch.setattr(subprocess_utils.subprocess, "run", windows_run)

        result = run_git_command_with_input(["--no-pager", "hash-object", "--stdin"], contents)

        assert result.returncode == 0
        expected = contents.encode("utf-8") if isinstance(contents, str) else contents
        assert result.stdout == expected.hex() + "\n"
        assert result.stderr == ""

    @pytest.mark.parametrize(
        "contents",
        ["", "line\n", "line\r\n", "line\r", "line \u00e9\nsecond\r\nlast\r\0", b"line\r\n", b"byte\xe9\r\n\0"],
        ids=["empty", "lf", "crlf", "cr", "mixed-unicode", "bytes-crlf", "raw-bytes"],
    )
    def test_hashes_the_same_bytes_as_a_literal_file(self, tmp_path: Path, contents: str | bytes) -> None:
        if shutil.which("git") is None:
            pytest.skip("git is required to compare stdin with file hashing")
        source = tmp_path / "payload.txt"
        source.write_bytes(contents.encode("utf-8") if isinstance(contents, str) else contents)
        expected = run_git_command(["--no-pager", "hash-object", "--no-filters", str(source)], cwd=tmp_path, timeout=30)

        actual = run_git_command_with_input(["--no-pager", "hash-object", "--no-filters", "--stdin"], contents, cwd=tmp_path, timeout=30)

        assert actual.returncode == 0
        assert actual.stdout == expected.stdout
        assert actual.stderr == ""

    def test_passes_stdin_to_git(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        calls: dict[str, Any] = {}
        monkeypatch.setattr(subprocess_utils, "get_safe_executable", lambda _command: "/usr/bin/git")

        def fake_run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
            calls["args"] = args
            calls["kwargs"] = kwargs
            return subprocess.CompletedProcess(args, 0, stdout=b"hash\r\n", stderr=b"notice\rnext\r\n")

        monkeypatch.setattr(subprocess_utils.subprocess, "run", fake_run)

        result = run_git_command_with_input(["hash-object", "--stdin"], "content", cwd=tmp_path)

        assert result.stdout == "hash\n"
        assert result.stderr == "notice\nnext\n"
        assert calls["args"] == ["/usr/bin/git", "hash-object", "--stdin"]
        assert calls["kwargs"]["cwd"] == tmp_path
        assert calls["kwargs"]["input"] == b"content"
        assert calls["kwargs"]["text"] is False
        assert "encoding" not in calls["kwargs"]

    @pytest.mark.parametrize("check", [True, False])
    def test_failed_command_preserves_text_diagnostics(self, monkeypatch: pytest.MonkeyPatch, check: bool) -> None:
        monkeypatch.setattr(subprocess_utils, "get_safe_executable", lambda _command: "/tools/git")
        monkeypatch.setattr(
            subprocess_utils.subprocess,
            "run",
            lambda args, **_kwargs: subprocess.CompletedProcess(args, 7, b"partial\r\n", b"failed\r\n"),
        )

        if check:
            with pytest.raises(subprocess.CalledProcessError) as raised:
                run_git_command_with_input(["--no-pager", "hash-object", "--stdin"], "content", check=True)
            outcome = raised.value
            assert outcome.cmd == ["/tools/git", "--no-pager", "hash-object", "--stdin"]
        else:
            outcome = run_git_command_with_input(["--no-pager", "hash-object", "--stdin"], "content", check=False)

        assert outcome.returncode == 7
        assert outcome.stdout == "partial\n"
        assert outcome.stderr == "failed\n"

    @pytest.mark.parametrize(("encoding", "errors", "encoded"), [("latin-1", "strict", b"line \xe9\r\n"), ("ascii", "replace", b"line ?\r\n")])
    def test_honors_encoding_and_error_policy_without_newline_translation(
        self, monkeypatch: pytest.MonkeyPatch, encoding: str, errors: str, encoded: bytes
    ) -> None:
        monkeypatch.setattr(subprocess_utils, "get_safe_executable", lambda _command: "/tools/git")

        def echo(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
            assert kwargs["input"] == encoded
            assert kwargs["text"] is False
            assert "encoding" not in kwargs
            assert "errors" not in kwargs
            assert "universal_newlines" not in kwargs
            return subprocess.CompletedProcess(args, 0, kwargs["input"], b"")

        monkeypatch.setattr(subprocess_utils.subprocess, "run", echo)

        result = run_git_command_with_input([], "line \u00e9\r\n", encoding=encoding, errors=errors, universal_newlines=True)

        assert result.stdout == encoded.decode(encoding).replace("\r\n", "\n")

    def test_allows_uncaptured_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(subprocess_utils, "get_safe_executable", lambda _command: "/tools/git")

        def run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
            assert kwargs["capture_output"] is False
            return subprocess.CompletedProcess(args, 0)

        monkeypatch.setattr(subprocess_utils.subprocess, "run", run)

        result = run_git_command_with_input([], "content", capture_output=False)

        assert result.returncode == 0
        assert result.stdout is None
        assert result.stderr is None

    def test_preserves_timeout_and_partial_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(subprocess_utils, "get_safe_executable", lambda _command: "/tools/git")
        timeout = subprocess.TimeoutExpired(["/tools/git"], 3, output=b"partial\r\n", stderr=b"diagnostic\r\n")

        def run(_args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
            assert kwargs["timeout"] == 3
            raise timeout

        monkeypatch.setattr(subprocess_utils.subprocess, "run", run)

        with pytest.raises(subprocess.TimeoutExpired) as raised:
            run_git_command_with_input([], "content", timeout=3)

        assert raised.value is timeout
        assert raised.value.output == b"partial\r\n"
        assert raised.value.stderr == b"diagnostic\r\n"

    def test_rejects_executable_override(self) -> None:
        kwargs: dict[str, Any] = {"executable": "/malicious/fake-git"}
        with pytest.raises(ValueError, match="Overriding 'executable' is not allowed"):
            run_git_command_with_input(["hash-object", "--stdin"], "content", **kwargs)


class TestRunSafeCommand:
    def test_resolves_arbitrary_command_and_preserves_hardening(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: dict[str, Any] = {}
        monkeypatch.setattr(subprocess_utils, "get_safe_executable", lambda command: f"/tools/{command}")

        def fake_run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            calls["args"] = args
            calls["kwargs"] = kwargs
            return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")

        monkeypatch.setattr(subprocess_utils.subprocess, "run", fake_run)

        result = run_safe_command("ruff", ["check", "-"], input="value = 1\n", timeout=30)

        assert result.stdout == "ok"
        assert calls["args"] == ["/tools/ruff", "check", "-"]
        assert calls["kwargs"]["input"] == "value = 1\n"
        assert calls["kwargs"]["timeout"] == 30
        assert calls["kwargs"]["check"] is True
