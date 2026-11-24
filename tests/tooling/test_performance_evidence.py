"""MCMC's first-comparison boundary and shared report validation."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from research_repo_tools.cli import main

ROOT = Path(__file__).resolve().parents[2]
REPORT = "docs/performance/current.md"
CONFIGURATION = "tooling/performance-report.toml"


def _project(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the actual recipe/configuration and the already locked test environment."""
    for name in (
        "justfile",
        "pyproject.toml",
        "uv.lock",
        ".python-version",
        CONFIGURATION,
        "tooling/measurement-release-signal.toml",
    ):
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", sys.prefix)


def _check(root: Path) -> subprocess.CompletedProcess[str]:
    """Run just performance-check in the fixture checkout and retain the exit status and both output streams."""
    executable = shutil.which("just")
    assert executable is not None
    return subprocess.run(  # noqa: S603 - resolved executable and fixed consumer recipe.
        [executable, "performance-check"],
        cwd=root,
        check=False,
        capture_output=True,
        encoding="utf-8",
    )


def test_empty_active_series_passes_without_writing_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _project(tmp_path, monkeypatch)
    result = _check(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "Report inventory: uninitialized." in result.stdout
    assert not (tmp_path / REPORT).exists()


@pytest.mark.parametrize(
    "artifact",
    [
        "docs/performance/runs/partial.comparison.json",
        "docs/performance/runs/unexpected.txt",
        "docs/performance/runs/empty-run",
        "target/bench-reports/preparation.json",
        "target/bench-reports/performance.run.json",
        "target/bench-reports/performance.evidence.json",
        "docs/assets/release-performance.csv",
        "docs/assets/release-performance.svg",
    ],
)
def test_new_namespace_rejects_every_orphan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], artifact: str) -> None:
    _project(tmp_path, monkeypatch)
    destination = tmp_path / artifact
    destination.parent.mkdir(parents=True)
    if destination.name == "empty-run":
        destination.mkdir()
    else:
        destination.write_bytes(b"incomplete output\r\n")
    assert main(["--root", str(tmp_path), "performance", "promote", "tooling/performance-report.toml", "--check"]) == 1
    diagnostic = capsys.readouterr().err
    assert "uninitialized report has orphaned inventory" in diagnostic
    assert artifact in diagnostic
    assert not (tmp_path / REPORT).exists()


def test_active_archive_must_be_a_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _project(tmp_path, monkeypatch)
    archive = tmp_path / "docs/performance/runs"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"invalid inventory")
    result = _check(tmp_path)
    assert result.returncode != 0
    assert "inventory directory must be a regular directory" in result.stderr


def test_active_archive_cannot_escape_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "checkout"
    _project(root, monkeypatch)
    config = root / CONFIGURATION
    config.write_bytes(config.read_bytes().replace(b'archive = "docs/performance/runs"', b'archive = "../outside"'))
    result = _check(root)
    assert result.returncode != 0
    assert "normalized repository-relative path: '../outside'" in result.stderr
    assert not (tmp_path / "outside").exists()
