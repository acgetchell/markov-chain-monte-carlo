"""Tests for release metadata and version synchronization checks."""

import tomllib
from pathlib import Path

from research_repo_tools.cli import main

ROOT = Path(__file__).resolve().parents[2]
_DOI = "10.5281/zenodo.20033111"
_VERSION = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))["package"]["version"]


def _write_project(root: Path) -> None:
    """Exercise the actual release policy and metadata without synthetic mirrors."""
    manifest = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    for name in ("Cargo.toml", *manifest["tool"]["research-repo-tools"]["release"]["required-files"]):
        (root / name).write_bytes((ROOT / name).read_bytes())
    assert main(["--root", str(root), "release", "check", "--final-release"]) == 0


def test_configured_checker_requires_current_source_links(tmp_path: Path) -> None:
    _write_project(tmp_path)
    path = tmp_path / "README.md"
    path.write_text(path.read_text(encoding="utf-8").replace(f"/blob/v{_VERSION}/", "/blob/v0.0.0/"), encoding="utf-8", newline="\n")
    assert main(["--root", str(tmp_path), "release", "check", "--final-release"]) == 1


def test_consistent_but_wrong_concept_doi_is_rejected(tmp_path: Path) -> None:
    _write_project(tmp_path)
    for name in ("CITATION.cff", "README.md", "REFERENCES.md"):
        path = tmp_path / name
        path.write_text(path.read_text(encoding="utf-8").replace(_DOI, "10.5281/zenodo.12345"), encoding="utf-8", newline="\n")
    assert main(["--root", str(tmp_path), "release", "check", "--final-release"]) == 1


def test_historical_evidence_and_changelog_archives_are_preserved(tmp_path: Path) -> None:
    _write_project(tmp_path)
    for folder in ("docs/archive/performance", "docs/performance/v1", "docs/archives/changelog", "tests/fixtures"):
        directory = tmp_path / folder
        directory.mkdir(parents=True)
        (directory / "old.md").write_text(
            'markov-chain-monte-carlo = "0.1.0"\njust performance-release v0.1.0 v0.0.9\n',
            encoding="utf-8",
            newline="\n",
        )
    readme = tmp_path / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8") + "[evidence](https://github.com/acgetchell/markov-chain-monte-carlo/blob/v1.0.0/docs/PERFORMANCE.md)\n",
        encoding="utf-8",
        newline="\n",
    )
    assert main(["--root", str(tmp_path), "release", "check", "--final-release"]) == 0
