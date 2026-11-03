"""Bound the temporary latest-scanner adapter without replacing shared checks."""

import json
import runpy
import subprocess
from pathlib import Path
from typing import Any

import pytest
from research_repo_tools import semgrep_tool

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def adapter(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    namespace = runpy.run_path(str(ROOT / "tooling/semgrep_latest.py"))
    versions = {"research-repo-tools": "0.2.5", "semgrep": "1.183.0"}
    monkeypatch.setitem(namespace["main"].__globals__, "version", versions.__getitem__)
    return namespace | {"versions": versions}


@pytest.mark.parametrize("status", [0, 7])
def test_exact_latest_scanner_forwards_success_and_failure_then_restores_shared_state(
    adapter: dict[str, Any], monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    original = semgrep_tool.VERSION

    def shared(arguments: list[str]) -> int:
        assert arguments == ["semgrep", "check-fixtures"]
        assert semgrep_tool.VERSION == "1.183.0"
        return status

    monkeypatch.setitem(adapter["main"].__globals__, "tooling_main", shared)
    assert adapter["main"](["semgrep", "check-fixtures"]) == status
    assert original == semgrep_tool.VERSION


@pytest.mark.parametrize("drift", ["package", "installed-scanner", "shared-expectation"])
def test_adapter_rejects_version_drift_before_running_the_gate(adapter: dict[str, Any], monkeypatch: pytest.MonkeyPatch, drift: str) -> None:
    if drift == "package":
        adapter["versions"]["research-repo-tools"] = "0.2.6"
    elif drift == "installed-scanner":
        adapter["versions"]["semgrep"] = "1.182.1"
    else:
        monkeypatch.setattr(semgrep_tool, "VERSION", "1.183.0")
    original = semgrep_tool.VERSION
    monkeypatch.setitem(adapter["main"].__globals__, "tooling_main", lambda _args: pytest.fail("drift must not run a scanner"))
    with pytest.raises(RuntimeError):
        adapter["main"](["semgrep", "check-fixtures"])
    assert original == semgrep_tool.VERSION


@pytest.mark.parametrize("arguments", [["semgrep", "check-fixtures"], ["setup"]])
def test_adapter_preserves_shared_installed_version_check(adapter: dict[str, Any], monkeypatch: pytest.MonkeyPatch, arguments: list[str]) -> None:
    original = semgrep_tool.VERSION

    def observed(_executable: object, _arguments: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        payload = {"path": "/unused/semgrep", "version": "1.182.1", "python": "3.15"}
        return subprocess.CompletedProcess(["python"], 0, json.dumps(payload).encode("utf-8"), b"")

    monkeypatch.setattr(semgrep_tool, "run_command_bytes", observed)
    monkeypatch.setitem(adapter["main"].__globals__, "tooling_main", lambda _args: semgrep_tool.setup(ROOT))
    with pytest.raises(RuntimeError, match=r"locked environment must provide Semgrep 1\.183\.0"):
        adapter["main"](arguments)
    assert original == semgrep_tool.VERSION


def test_setup_can_install_the_scanner_before_shared_verification(adapter: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    original = semgrep_tool.VERSION

    def bootstrap_version(name: str) -> str:
        assert name == "research-repo-tools", "setup must not require a scanner before shared synchronization"
        return "0.2.5"

    def shared(arguments: list[str]) -> int:
        assert arguments == ["setup"]
        assert semgrep_tool.VERSION == "1.183.0"
        return 0

    monkeypatch.setitem(adapter["main"].__globals__, "version", bootstrap_version)
    monkeypatch.setitem(adapter["main"].__globals__, "tooling_main", shared)
    assert adapter["main"](["setup"]) == 0
    assert original == semgrep_tool.VERSION


@pytest.mark.parametrize("arguments", [[], ["release", "publish", "v0.5.3"], ["semgrep", "scan"]])
def test_adapter_does_not_expand_into_a_generic_shared_cli_override(adapter: dict[str, Any], arguments: list[str]) -> None:
    with pytest.raises(ValueError, match="supports only shared setup and fixture checks"):
        adapter["main"](arguments)
