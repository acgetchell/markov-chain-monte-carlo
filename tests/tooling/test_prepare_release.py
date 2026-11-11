"""Draft orchestration uses inert providers; no release or workflow is executed."""

import json
import runpy
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from research_repo_tools import dependency_provenance, release_assets
from research_repo_tools.config import load

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = "acgetchell/markov-chain-monte-carlo"
TAG = "v0.5.3"
COMMIT = "a" * 40
NOTES = "Reviewed release notes.\n"


def _published_dependencies(root: Path, *, packages: tuple[str, ...] | None = None) -> tuple[dependency_provenance.DependencyIdentity, ...]:
    """Separate the already-audited dependency graph from inert provider fixtures."""
    assert root == ROOT
    assert packages == ()
    return (dependency_provenance.DependencyIdentity("research-repo-tools", "0.2.8", "https://pypi.org/simple"),)


@pytest.fixture
def preparation(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    namespace = runpy.run_path(str(ROOT / "tooling/prepare_release.py"))
    implementation = namespace["prepare"].__globals__
    state: dict[str, Any] = {"release": None, "writes": [], "selections": [], "checks": 0}
    publishing = load(root=ROOT).publishing
    assert publishing is not None

    def release() -> dict[str, Any]:
        return {
            "id": 73,
            "url": f"https://api.github.com/repos/{REPOSITORY}/releases/73",
            "tag_name": TAG,
            "name": TAG,
            "draft": True,
            "immutable": False,
            "prerelease": False,
            "assets": [],
            "body": NOTES,
        }

    state["new_release"] = release

    def read(endpoint: str, **_kwargs: object) -> Any:
        if endpoint.startswith("releases?"):
            return [] if state["release"] is None else [state["release"]]
        assert endpoint == "releases/73"
        return state["release"]

    def command(executable: str, arguments: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        assert executable == "gh"
        if arguments[:2] == ["release", "create"]:
            assert state["release"] is None
            assert arguments[2:12] == [TAG, "--repo", REPOSITORY, "--verify-tag", "--target", COMMIT, "--draft", "--title", TAG, "--notes-file"]
            assert Path(arguments[12]).read_bytes() == NOTES.encode("utf-8")
            assert len(state["selections"]) == 2
            assert state["checks"] == 2
            state["writes"].append(arguments)
            state["release"] = release()
            return subprocess.CompletedProcess([executable, *arguments], 0, "", "")
        assert arguments[:3] == ["api", "--paginate", "--slurp"]
        assert arguments[-1].endswith("/check-runs?filter=latest&per_page=100")
        state["checks"] += 1
        checks = [
            {"id": index, "name": check.name, "head_sha": COMMIT, "status": "completed", "conclusion": "success", "app": {"id": check.app_id}}
            for index, check in enumerate(publishing.required_checks, 1)
        ]
        if state.get("bad_checks"):
            checks[0]["conclusion"] = "failure"
        return subprocess.CompletedProcess([executable, *arguments], 0, json.dumps([{"total_count": len(checks), "check_runs": checks}]), "")

    def evidence(_policy: object, version: str, commit: str, _read: object, *, expected: bytes | None = None) -> bytes:
        assert (version, commit) == ("0.5.3", COMMIT)
        assert expected == (b"frozen selection" if state["selections"] else None)
        state["selections"].append(expected)
        if state.get("changed_selection") and expected is not None:
            message = "approved artifact identity changed"
            raise ValueError(message)
        return b"frozen selection"

    def transport(executable: str | Path, arguments: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        assert str(executable) == "gh"
        endpoint = arguments[-1]
        if endpoint.endswith("releases?per_page=100"):
            payload: Any = [[state["release"]]]
        elif endpoint.endswith("releases/73"):
            payload = state["release"]
        elif endpoint.endswith("/assets?per_page=100"):
            payload = [state["release"]["assets"]]
        else:
            assert endpoint.endswith(f"git/ref/tags/{TAG}")
            payload = {"ref": f"refs/tags/{TAG}", "object": {"type": "commit", "sha": COMMIT}}
        return subprocess.CompletedProcess([str(executable), *arguments], 0, json.dumps(payload).encode("utf-8"), b"")

    for name, provider in {
        "require_tag_source": lambda _root, _tag: COMMIT,
        "check_metadata": lambda _settings, _tag: None,
        "check_release_dependencies": _published_dependencies,
        "api": read,
        "run_command": command,
        "verify_release_evidence": evidence,
    }.items():
        monkeypatch.setitem(implementation, name, provider)
    monkeypatch.setattr(implementation["changelog"], "notes", lambda _settings, _tag: (NOTES, ROOT / "CHANGELOG.md", "heading"))
    monkeypatch.setattr(release_assets, "run_command_bytes", transport)
    monkeypatch.setattr(release_assets, "require_release_checkout", lambda _root, _commit: None)

    monkeypatch.setattr(dependency_provenance, "check_release_dependencies", _published_dependencies)
    return state | {"state": state, "prepare": namespace["prepare"], "implementation": implementation}


def test_creates_only_a_draft_after_frozen_evidence_and_required_checks(preparation: dict[str, Any]) -> None:
    assert preparation["prepare"](ROOT, TAG) == 73
    assert len(preparation["state"]["writes"]) == 1


def test_identical_draft_retry_does_not_write_or_replace_assets(preparation: dict[str, Any]) -> None:
    state = preparation["state"]
    state["release"] = state["new_release"]()
    asset = {"id": 91, "name": "retained-baseline.tar.gz", "size": 100, "state": "uploaded", "digest": "sha256:" + "b" * 64}
    state["release"]["assets"] = [asset]
    assert preparation["prepare"](ROOT, TAG) == 73
    assert not state["writes"]
    assert state["release"]["assets"] == [asset]


@pytest.mark.parametrize(("field", "value"), [("draft", False), ("immutable", True), ("prerelease", True), ("name", "Wrong"), ("body", "Changed notes")])
def test_conflicting_or_published_release_is_never_edited(preparation: dict[str, Any], field: str, value: object) -> None:
    state = preparation["state"]
    state["release"] = {**state["new_release"](), field: value}
    with pytest.raises(ValueError, match=r"draft|title|notes"):
        preparation["prepare"](ROOT, TAG)
    assert not state["writes"]


@pytest.mark.parametrize(("failure", "message"), [("bad_checks", "Unmet release checks"), ("changed_selection", "approved artifact")])
def test_failed_checks_or_changed_frozen_evidence_prevent_creation(preparation: dict[str, Any], failure: str, message: str) -> None:
    state = preparation["state"]
    state[failure] = True
    with pytest.raises(ValueError, match=message):
        preparation["prepare"](ROOT, TAG)
    assert not state["writes"]


@pytest.mark.parametrize(("field", "value"), [("id", 74), ("body", "Concurrent notes change")])
def test_existing_draft_identity_and_notes_stay_frozen(preparation: dict[str, Any], monkeypatch: pytest.MonkeyPatch, field: str, value: object) -> None:
    state = preparation["state"]
    state["release"] = state["new_release"]()
    implementation = preparation["implementation"]
    original = implementation["verify_release_evidence"]

    def changed(*args: object, **kwargs: object) -> bytes:
        result = original(*args, **kwargs)
        assert isinstance(result, bytes)
        if kwargs.get("expected") is not None:
            state["release"][field] = value
        return result

    monkeypatch.setitem(implementation, "verify_release_evidence", changed)
    with pytest.raises(ValueError, match=r"identity|notes"):
        preparation["prepare"](ROOT, TAG)
    assert not state["writes"]


@pytest.mark.parametrize("failure", ["none", "tag", "event", "sha", "branch", "ancestry", "remote", "dirty"])
def test_source_gate_binds_tag_event_and_protected_main(monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    namespace = runpy.run_path(str(ROOT / "tooling/prepare_release.py"))
    gate = namespace["require_tag_source"]
    implementation = gate.__globals__
    for key, value in {"GITHUB_REPOSITORY": REPOSITORY, "GITHUB_EVENT_NAME": "push", "GITHUB_REF": f"refs/tags/{TAG}", "GITHUB_SHA": COMMIT}.items():
        monkeypatch.setenv(key, value)
    if failure == "event":
        monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    if failure == "sha":
        monkeypatch.setenv("GITHUB_SHA", "b" * 40)
    monkeypatch.setitem(implementation, "resolve_revision", lambda _root, _tag: COMMIT)

    def checkout(_root: Path, _commit: str) -> None:
        if failure == "dirty":
            message = "dirty checkout"
            raise ValueError(message)

    def read(endpoint: str, **_kwargs: object) -> dict[str, Any]:
        if endpoint == "":
            return {"default_branch": "main"}
        if endpoint == "branches/main":
            return {"protected": failure != "branch"}
        assert endpoint == f"compare/{COMMIT}...main"
        return {"status": "diverged" if failure == "ancestry" else "ahead"}

    remote = "b" * 40 if failure == "remote" else COMMIT
    monkeypatch.setitem(implementation, "require_release_checkout", checkout)
    monkeypatch.setitem(implementation, "api", read)
    monkeypatch.setitem(implementation, "run_command", lambda *_args, **_kwargs: SimpleNamespace(stdout=f"{remote}\trefs/tags/{TAG}\n"))
    if failure == "none":
        assert gate(ROOT, TAG) == COMMIT
    else:
        with pytest.raises(ValueError, match=r"stable|tag-push|protected|remote tag|dirty"):
            gate(ROOT, "v0.5.3-rc.1" if failure == "tag" else TAG)


def test_workflow_guards_source_before_isolated_python_without_oidc() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/prepare-release.yml").read_bytes())
    assert workflow.get("on", workflow.get(True)) == {"push": {"tags": ["v*"]}}
    assert workflow["concurrency"]["cancel-in-progress"] is False
    job = workflow["jobs"]["prepare"]
    assert job["permissions"] == {"contents": "write", "actions": "read", "checks": "read"}
    steps = job["steps"]
    guard = next(index for index, step in enumerate(steps) if "merge-base --is-ancestor" in step.get("run", ""))
    runner = next(index for index, step in enumerate(steps) if "python -I tooling/prepare_release.py" in step.get("run", ""))
    assert guard < runner
    assert "tooling/release-evidence.toml" in steps[guard]["run"]
    assert "--no-config --isolated --managed-python" in steps[runner]["run"]
    assert "research-repo-tools==0.2.8" in steps[runner]["run"]
    assert all("semgrep_latest" not in step.get("run", "") for step in steps)
