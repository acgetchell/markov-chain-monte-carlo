"""Coupled MCMC workflow/composite pins use the published offline checker."""

import re
import shutil
from pathlib import Path

import pytest
from research_repo_tools import action_checks, action_updates
from research_repo_tools.just_inspect import dry_run, inspect_justfile

ROOT = Path(__file__).resolve().parents[2]
INPUTS = [".github/workflows", ".github/actions/setup-toolchain/action.yml"]


@pytest.fixture
def actions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Copy the repository's Actions configuration and forbid upstream tag resolution during offline checks."""
    shutil.copytree(ROOT / ".github", tmp_path / ".github")
    monkeypatch.setattr(action_updates, "resolve", lambda *_args: pytest.fail("offline consistency must never resolve upstream tags"))
    return tmp_path


def test_declared_pins_pass_without_resolution_or_file_changes(actions: Path, capsys: pytest.CaptureFixture[str]) -> None:
    before = {path: path.read_bytes() for path in actions.rglob("*") if path.is_file()}
    assert action_checks.check(actions, actions / ".github/action-updates.toml", INPUTS) == 0
    assert "offline, no files changed" in capsys.readouterr().out
    assert all(path.read_bytes() == data for path, data in before.items())


@pytest.mark.parametrize("failure", ["composite", "codeql", "missing-comment", "floating"])
def test_partial_bump_is_rejected_with_native_path_and_without_writes(actions: Path, failure: str) -> None:
    relative = ".github/actions/setup-toolchain/action.yml" if failure in {"composite", "floating"} else ".github/workflows/codeql.yml"
    path = actions / relative
    original = path.read_bytes()
    pattern, replacement = {
        "composite": (rb"(uses: astral-sh/setup-uv@)[0-9a-f]{40}", rb"\g<1>" + b"a" * 40),
        "codeql": (rb"(uses: github/codeql-action/init@[0-9a-f]{40}) # v[\d.]+", rb"\g<1> # v0.0.0"),
        "missing-comment": (rb"(uses: github/codeql-action/init@[0-9a-f]{40}) # v[\d.]+", rb"\g<1>"),
        "floating": (rb"(uses: astral-sh/setup-uv@)[0-9a-f]{40}", rb"\g<1>v10"),
    }[failure]
    text = re.sub(pattern, replacement, original, count=1)
    assert text != original
    path.write_bytes(text)
    with pytest.raises(ValueError, match=r".+") as error:
        action_checks.check(actions, actions / ".github/action-updates.toml", INPUTS)
    assert str(path) in str(error.value)
    assert path.read_bytes() == text


def test_checker_and_updater_share_scope_and_preview_arguments() -> None:
    check = dry_run(ROOT, "actions-check").stderr
    update = dry_run(ROOT, "update-actions", ["--dry-run"]).stderr
    for path in INPUTS:
        assert path in check
        assert path in update
    assert "--no-sync --no-python-downloads --offline" in check
    assert '"$@"' in update
    recipes = inspect_justfile(ROOT).recipes
    for gate in ("check-repository-tooling", "ci", "ci-repository-tooling"):
        assert "actions-check" in {item["recipe"] for item in recipes[gate]["dependencies"]}
