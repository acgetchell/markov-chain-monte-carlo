"""Tests for release metadata and version synchronization checks."""

from typing import TYPE_CHECKING

import pytest

import release_check

if TYPE_CHECKING:
    from pathlib import Path

_VERSION = "1.2.3"
_DOI = "10.5281/zenodo.20033111"
_RELEASE_DATE = "2026-08-04"
_CARGO_TOML = f"""[package]
name = "markov-chain-monte-carlo"
version = "{_VERSION}"
repository = "https://github.com/acgetchell/markov-chain-monte-carlo"
"""


def _write_project(
    root: Path,
    *,
    metadata_version: str = _VERSION,
    readme: str | None = None,
    citation_doi: str = _DOI,
    citation_date: str = _RELEASE_DATE,
) -> None:
    """Write a minimal repository for release-check tests."""
    readme_text = (
        readme
        if readme is not None
        else (
            f"[![DOI](https://badgen.net/badge/DOI/10.5281%2Fzenodo.20033111/blue)](https://doi.org/{_DOI})\n"
            f'markov-chain-monte-carlo = "{_VERSION}"\n'
            f"cargo add markov-chain-monte-carlo@{_VERSION}\n"
            f"[tagged](https://github.com/acgetchell/markov-chain-monte-carlo/blob/v{_VERSION}/README.md)\n"
        )
    )
    files = {
        "Cargo.toml": _CARGO_TOML,
        "Cargo.lock": f'version = 4\n\n[[package]]\nname = "markov-chain-monte-carlo"\nversion = "{metadata_version}"\n',
        "pyproject.toml": f'[project]\nname = "markov-chain-monte-carlo-tooling"\nversion = "{metadata_version}"\n',
        "uv.lock": (f'version = 1\n\n[[package]]\nname = "markov-chain-monte-carlo-tooling"\nversion = "{metadata_version}"\nsource = {{ editable = "." }}\n'),
        "CITATION.cff": (f"cff-version: 1.2.0\nversion: {metadata_version}\ndoi: {citation_doi}\ndate-released: {citation_date}\n"),
        "CHANGELOG.md": (
            f"# Changelog\n\n## [{_VERSION}] - {_RELEASE_DATE}\n\n- Release\n\n"
            f"[{_VERSION}]: https://github.com/acgetchell/markov-chain-monte-carlo/compare/v1.2.2...v{_VERSION}\n"
        ),
        "README.md": readme_text,
        "REFERENCES.md": f"- DOI: <https://doi.org/{_DOI}>\n",
    }
    for filename, content in files.items():
        (root / filename).write_text(content, encoding="utf-8")


@pytest.mark.parametrize(
    ("name", "old", "new"),
    [
        ("Cargo.lock", 'version = "1.2.3"', 'version = "1.2.2"'),
        ("pyproject.toml", 'version = "1.2.3"', 'version = "1.2.2"'),
        ("uv.lock", 'version = "1.2.3"', 'version = "1.2.2"'),
        ("CITATION.cff", "version: 1.2.3", "version: 1.2.2"),
        ("CITATION.cff", "date-released: 2026-08-04", "date-released: 2026-08-03"),
        ("CHANGELOG.md", "## [1.2.3]", "## [1.2.2]"),
        ("README.md", 'markov-chain-monte-carlo = "1.2.3"', 'markov-chain-monte-carlo = "1.2.2"'),
        ("README.md", "monte-carlo@1.2.3", "monte-carlo@1.2.2"),
        ("README.md", "/blob/v1.2.3/", "/blob/v1.2.2/"),
    ],
)
def test_combined_checker_rejects_stale_metadata(tmp_path: Path, name: str, old: str, new: str) -> None:
    _write_project(tmp_path)
    path = tmp_path / name
    path.write_text(path.read_text(encoding="utf-8").replace(old, new), encoding="utf-8", newline="\n")
    assert release_check.main([str(tmp_path)]) == 1


def test_consistent_but_wrong_concept_doi_is_rejected(tmp_path: Path) -> None:
    _write_project(tmp_path)
    for name in ("CITATION.cff", "README.md", "REFERENCES.md"):
        path = tmp_path / name
        path.write_text(path.read_text(encoding="utf-8").replace(_DOI, "10.5281/zenodo.12345"), encoding="utf-8", newline="\n")
    assert release_check.main([str(tmp_path)]) == 1


@pytest.mark.parametrize("name", ["Cargo.lock", "uv.lock", "CITATION.cff", "REFERENCES.md"])
def test_required_consumer_surface_cannot_disappear(tmp_path: Path, name: str) -> None:
    _write_project(tmp_path)
    (tmp_path / name).unlink()
    assert release_check.main([str(tmp_path)]) == 1


@pytest.mark.parametrize("name", ["README.md", "REFERENCES.md"])
def test_required_doi_reference_cannot_disappear(tmp_path: Path, name: str) -> None:
    _write_project(tmp_path)
    (tmp_path / name).write_text("# No DOI reference\n", encoding="utf-8", newline="\n")
    assert release_check.main([str(tmp_path)]) == 1


def test_active_performance_commands_track_current_release(tmp_path: Path) -> None:
    _write_project(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("just performance-release v1.2.2 v1.2.1\n", encoding="utf-8", newline="\n")
    assert release_check.main([str(tmp_path)]) == 1


def test_historical_evidence_and_changelog_archives_are_preserved(tmp_path: Path) -> None:
    _write_project(tmp_path)
    for folder in ("docs/archive/performance", "docs/archives/changelog", "tests/fixtures"):
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
    assert release_check.main([str(tmp_path)]) == 0


def test_main_reports_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_project(tmp_path)
    assert release_check.main([str(tmp_path)]) == 0
    assert "Release metadata is synchronized at 1.2.3" in capsys.readouterr().out
