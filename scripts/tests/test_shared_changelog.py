"""Consumer boundaries for the published changelog toolkit; parser tests live upstream."""

import shutil
import subprocess
import tomllib
from importlib.metadata import version
from pathlib import Path

import pytest
from research_repo_tools.cli import main

from .test_release_check import _write_project

ROOT = Path(__file__).resolve().parents[2]


def test_published_tooling_pin_is_locked_and_included_in_dev() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    groups = tomllib.loads(text)["dependency-groups"]
    assert {"include-group": "tooling"} in groups["dev"]
    assert groups["tooling"] == ["research-repo-tools==0.1.2"]
    assert version("research-repo-tools") == "0.1.2"
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    package = next(package for package in lock["package"] if package["name"] == "research-repo-tools")
    assert package["version"] == "0.1.2"
    assert package["source"] == {"registry": "https://pypi.org/simple"}


@pytest.mark.parametrize("archived", [False, True])
def test_shared_notes_preserve_rust_and_reference_links(tmp_path: Path, archived: bool, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "CHANGELOG.md"
    root.write_text("# Changelog\n", encoding="utf-8", newline="\n")
    target = tmp_path / "docs/archives/changelog/0.1.md" if archived else root
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "# Changelog\n\n## [0.1.0] - 2026-03-24\n\n### Added\n\n- Preserve `Chain<S>` and [details][api].\n\n[api]: https://example.com/api\n",
        encoding="utf-8",
        newline="\n",
    )
    assert main(["--root", str(tmp_path), "changelog", "notes", "v0.1.0"]) == 0
    notes = capsys.readouterr().out
    assert "`Chain<S>`" in notes
    assert "[details][api]" in notes
    assert "[api]: https://example.com/api" in notes
    assert main(["--root", str(tmp_path), "changelog", "notes", "v9.9.9"]) == 1
    assert "not found" in capsys.readouterr().err


@pytest.mark.parametrize("archived", [False, True])
@pytest.mark.parametrize("oversized", [False, True])
def test_shared_tag_preserves_notes_and_archive_fallback_without_git_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, archived: bool, oversized: bool
) -> None:
    _write_project(tmp_path)
    manifest = tmp_path / "pyproject.toml"
    manifest.write_text(
        manifest.read_text(encoding="utf-8")
        + '\n[tool.research-repo-tools.release]\ndate-policy = "declared"\n'
        + '[tool.research-repo-tools.changelog]\nowner = "acgetchell"\nrepository = "markov-chain-monte-carlo"\n',
        encoding="utf-8",
        newline="\n",
    )
    target = tmp_path / "CHANGELOG.md"
    if archived:
        target.write_text("# Changelog\n\n## [Unreleased]\n", encoding="utf-8", newline="\n")
        target = tmp_path / "docs/archives/changelog/1.2.md"
        target.parent.mkdir(parents=True)
    body = "- Preserve `Chain<S>` and café.\n" + ("- Detail.\n" * 14000 if oversized else "")
    target.write_text("# Changelog\n\n## [1.2.3] - 2026-08-04\n\n" + body, encoding="utf-8", newline="\n")
    writes: list[tuple[list[str], bytes]] = []

    def stub(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
        if "show-ref" in args:
            return subprocess.CompletedProcess(args, 0, "")
        assert args[:4] == ["git", "tag", "-f", "-a"]
        payload = kwargs["input"]
        assert isinstance(payload, bytes)
        assert kwargs.get("text") is False
        writes.append((args, payload))
        return subprocess.CompletedProcess(args, 0, b"")

    monkeypatch.setattr(shutil, "which", lambda command: command)
    monkeypatch.setattr(subprocess, "run", stub)
    assert main(["--root", str(tmp_path), "changelog", "tag", "v1.2.3", "--force"]) == 0
    assert len(writes) == 1
    payload = writes[0][1].decode("utf-8")
    if oversized:
        assert f"/blob/v1.2.3/{target.relative_to(tmp_path).as_posix()}#" in payload
    else:
        assert body.strip() in payload


def test_python_pin_update_retains_included_tooling_constraints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = tmp_path / "pyproject.toml"
    manifest.write_text(
        '[project]\nname = "consumer"\nversion = "1.0.0"\nrequires-python = ">=3.14"\n'
        '[dependency-groups]\ndev = [{include-group = "tooling"}, "ruff==0.16.1", "pytest>=9.1.1"]\n'
        'tooling = ["research-repo-tools==0.1.2"]\n',
        encoding="utf-8",
        newline="\n",
    )
    original = manifest.read_bytes()
    inputs: list[list[str]] = []

    def resolver(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert args[:3] == ["uv", "export", "--script"]
        source = Path(args[3]).read_text(encoding="utf-8")
        metadata = tomllib.loads(
            "\n".join(line.removeprefix("# ") for line in source.splitlines() if line.startswith("# ") and line not in {"# /// script", "# ///"})
        )
        assert metadata["requires-python"] == ">=3.14"
        inputs.append(metadata["dependencies"])
        return subprocess.CompletedProcess(args, 0, "ruff==0.16.1\n")

    monkeypatch.setattr(shutil, "which", lambda command: command)
    monkeypatch.setattr(subprocess, "run", resolver)
    assert main(["--root", str(tmp_path), "deps", "update-python"]) == 0
    assert len(inputs) == 1
    assert "research-repo-tools==0.1.2" in inputs[0]
    assert "pytest>=9.1.1" in inputs[0]
    assert manifest.read_bytes() == original
