"""Prepared citation dates keep MCMC's prospective changelog commands repeatable."""

import subprocess
from pathlib import Path

import pytest
from research_repo_tools import changelog
from research_repo_tools.cli import main
from research_repo_tools.config import load
from research_repo_tools.just_inspect import dry_run, inspect_justfile

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Exercise the installed CLI with MCMC's configuration and inert Git history."""
    for name in ("pyproject.toml", "rumdl.toml"):
        (tmp_path / name).write_bytes((ROOT / name).read_bytes())

    def cliff(command: str, args: list[str], cwd: Path | None = None, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        assert command == "git-cliff"
        assert cwd == tmp_path
        return subprocess.CompletedProcess(
            [command, *args], 0, stdout="# Changelog\n\n## [0.5.2] - 2099-01-01\n\n### Fixed\n\n- Retain prepared dates.\n", stderr=""
        )

    monkeypatch.setattr(changelog, "run_safe_command", cliff)
    return tmp_path


@pytest.mark.parametrize("quotation", ["", "'", '"'])
def test_omitted_date_reuses_prepared_leap_date_for_generation_and_preview(project: Path, capsys: pytest.CaptureFixture[str], quotation: str) -> None:
    citation = project / "CITATION.cff"
    contents = f"version: 0.5.2\ndate-released: {quotation}2000-02-29{quotation}\n".encode()
    citation.write_bytes(contents)
    command = ["--root", str(project), "changelog", "generate", "--tag", "v0.5.2"]

    assert main(command) == 0
    generated = (project / "CHANGELOG.md").read_bytes()
    assert b"## [0.5.2] - 2000-02-29" in generated
    assert main(command) == 0
    assert (project / "CHANGELOG.md").read_bytes() == generated
    capsys.readouterr()
    assert main([*command, "--dry-run"]) == 0
    assert "## [0.5.2] - 2000-02-29" in capsys.readouterr().out
    assert (project / "CHANGELOG.md").read_bytes() == generated
    assert citation.read_bytes() == contents


@pytest.mark.parametrize("arguments", [["--tag", "v0.5.2", "--date", "2000-02-29"], []])
def test_explicit_override_and_untagged_preview_do_not_require_citation(project: Path, capsys: pytest.CaptureFixture[str], arguments: list[str]) -> None:
    assert main(["--root", str(project), "changelog", "generate", *arguments, "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "## [0.5.2] - " + ("2000-02-29" if arguments else "2099-01-01") in output
    assert not (project / "CITATION.cff").exists()
    assert not (project / "CHANGELOG.md").exists()
    assert not (project / "docs").exists()


@pytest.mark.parametrize(
    "invalid",
    [
        (None, "CITATION.cff"),
        (b"[", "invalid citation YAML"),
        (b"- item\n", "top-level YAML mapping"),
        (b"version: 0.5.2\n", "top-level date-released"),
        (b"date-released: 2000-02-29\n", "top-level version"),
        (b"version: 0.5.1\ndate-released: 2000-02-29\n", "must match v0.5.2"),
        (b"version: 0.5.2\ndate-released: 2001-02-29\n", "YYYY-MM-DD calendar date"),
        (b"version: 0.5.2\ndate-released: 20000229\n", "top-level date-released"),
        (b"version: 0.5.2\ndate-released: \n", "top-level date-released"),
        (b"version: 0.5.2\ndate-released: [2000-02-29]\n", "top-level date-released"),
        (b"version: 0.5.2\ndate-released: 2000-02-29\ndate-released: 2000-03-01\n", "exactly one"),
        (b"version: 0.5.2\nversion: 0.5.3\ndate-released: 2000-02-29\n", "exactly one"),
    ],
)
def test_invalid_default_metadata_preserves_release_artifacts(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], invalid: tuple[bytes | None, str]
) -> None:
    contents, diagnostic = invalid
    citation = project / "CITATION.cff"
    if contents is not None:
        citation.write_bytes(contents)
    retained = project / "CHANGELOG.md"
    retained.write_bytes(b"retained changelog\r\n")
    monkeypatch.setattr(changelog, "run_safe_command", lambda *_args, **_kwargs: pytest.fail("invalid metadata must not generate anything"))
    assert main(["--root", str(project), "changelog", "generate", "--tag", "v0.5.2"]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert diagnostic in output.err
    assert "citation" in output.err
    assert retained.read_bytes() == b"retained changelog\r\n"
    assert not (project / "docs").exists()
    if contents is not None:
        assert citation.read_bytes() == contents


def test_normal_alias_and_preview_support_an_optional_explicit_date() -> None:
    assert load(root=ROOT).changelog.date_source == "citation"
    assert inspect_justfile(ROOT).aliases["changelog-unreleased"] == "changelog-release"
    implicit = dry_run(ROOT, "changelog-unreleased", ["v0.5.2"]).stderr
    empty = dry_run(ROOT, "changelog-release", ["v0.5.2", ""]).stderr
    explicit = dry_run(ROOT, "changelog-release", ["v0.5.2", "2000-02-29"]).stderr
    assert implicit == empty
    assert "args=(--tag 'v0.5.2')" in implicit
    assert "date=''" in implicit
    assert "date='2000-02-29'" in explicit
    assert 'args+=(--date "$date")' in explicit
    assert 'research-repo-tools changelog generate "${args[@]}"' in implicit
    assert 'research-repo-tools changelog generate --dry-run "$@"' in dry_run(ROOT, "changelog-preview", ["--tag", "v0.5.2"]).stderr
