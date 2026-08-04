"""Tests for release metadata and version synchronization checks."""

from typing import TYPE_CHECKING

import pytest

import release_check

if TYPE_CHECKING:
    from pathlib import Path

_VERSION = "1.2.3"
_CARGO_TOML = f"""[package]
name = "markov-chain-monte-carlo"
version = "{_VERSION}"
repository = "https://github.com/acgetchell/markov-chain-monte-carlo"
"""


def _write_project(root: Path, *, metadata_version: str = _VERSION, readme: str | None = None) -> None:
    """Write a minimal repository for release-check tests."""
    readme_text = (
        readme
        if readme is not None
        else (
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
        "CITATION.cff": f"cff-version: 1.2.0\nversion: {metadata_version}\n",
        "CHANGELOG.md": (
            f"# Changelog\n\n## [{_VERSION}] - 2026-08-04\n\n- Release\n\n"
            f"[{_VERSION}]: https://github.com/acgetchell/markov-chain-monte-carlo/compare/v1.2.2...v{_VERSION}\n"
        ),
        "README.md": readme_text,
    }
    for filename, content in files.items():
        (root / filename).write_text(content, encoding="utf-8")


def test_find_version_mismatches_accepts_synchronized_release(tmp_path: Path) -> None:
    """Matching structured metadata and active docs pass."""
    _write_project(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "benchmarking.md").write_text(
        "just performance-release v1.2.3 v1.2.2\njust performance-github-assets v9.0.0 v8.0.0\n",
        encoding="utf-8",
    )

    assert release_check.find_version_mismatches(tmp_path) == []


def test_find_version_mismatches_reports_all_structured_metadata(tmp_path: Path) -> None:
    """Cargo, Python, uv, and citation metadata must agree."""
    _write_project(tmp_path, metadata_version="1.2.2")

    mismatches = release_check.find_version_mismatches(tmp_path)

    assert [(mismatch.reference.kind, mismatch.reference.path.name, mismatch.reference.version) for mismatch in mismatches] == [
        (release_check.ReferenceKind.CARGO_LOCK, "Cargo.lock", "1.2.2"),
        (release_check.ReferenceKind.PYPROJECT, "pyproject.toml", "1.2.2"),
        (release_check.ReferenceKind.UV_LOCK, "uv.lock", "1.2.2"),
        (release_check.ReferenceKind.CITATION, "CITATION.cff", "1.2.2"),
    ]


def test_find_version_mismatches_reports_stale_active_documentation(tmp_path: Path) -> None:
    """Versioned install instructions and curated benchmark commands track Cargo."""
    _write_project(
        tmp_path,
        readme=(
            'markov-chain-monte-carlo = { version = "1.2.2", features = ["serde"] }\n'
            "cargo add --features serde markov-chain-monte-carlo@1.2.1\n"
            "[tagged](https://raw.githubusercontent.com/acgetchell/markov-chain-monte-carlo/v1.2.0/README.md)\n"
        ),
    )
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "benchmarking.md").write_text("just performance-release v1.1.0 v1.0.0\n", encoding="utf-8")

    mismatches = release_check.find_version_mismatches(tmp_path)

    assert [mismatch.reference.kind for mismatch in mismatches] == [
        release_check.ReferenceKind.DEPENDENCY_SNIPPET,
        release_check.ReferenceKind.CARGO_ADD,
        release_check.ReferenceKind.BENCHMARK_CURRENT_TAG,
        release_check.ReferenceKind.README_TAG_LINK,
    ]


def test_find_version_mismatches_reports_stale_changelog_release(tmp_path: Path) -> None:
    """The latest generated changelog release must match Cargo.toml."""
    _write_project(tmp_path)
    (tmp_path / "CHANGELOG.md").write_text("# Changelog\n\n## [1.2.2] - 2026-08-03\n", encoding="utf-8")

    mismatches = release_check.find_version_mismatches(tmp_path)

    assert len(mismatches) == 1
    assert mismatches[0].reference.kind is release_check.ReferenceKind.CHANGELOG
    assert mismatches[0].reference.version == "1.2.2"


def test_find_version_mismatches_reports_stale_changelog_comparison_target(tmp_path: Path) -> None:
    """The current changelog link must compare through the Cargo version."""
    _write_project(tmp_path)
    (tmp_path / "CHANGELOG.md").write_text(
        (f"# Changelog\n\n## [{_VERSION}] - 2026-08-04\n\n[{_VERSION}]: https://github.com/acgetchell/markov-chain-monte-carlo/compare/v1.2.1...v1.2.2\n"),
        encoding="utf-8",
    )

    mismatches = release_check.find_version_mismatches(tmp_path)

    assert len(mismatches) == 1
    assert mismatches[0].reference.kind is release_check.ReferenceKind.CHANGELOG_COMPARISON
    assert mismatches[0].reference.version == "1.2.2"


def test_find_version_mismatches_ignores_historical_surfaces(tmp_path: Path) -> None:
    """Archived docs and test fixtures may retain old release examples."""
    _write_project(tmp_path)
    archive = tmp_path / "docs" / "archive"
    fixtures = tmp_path / "tests" / "fixtures"
    archive.mkdir(parents=True)
    fixtures.mkdir(parents=True)
    stale = 'markov-chain-monte-carlo = "0.1.0"\njust performance-release v0.1.0 v0.0.9\n'
    (archive / "old.md").write_text(stale, encoding="utf-8")
    (fixtures / "example.md").write_text(stale, encoding="utf-8")

    assert release_check.find_version_mismatches(tmp_path) == []


def test_find_version_mismatches_rejects_missing_editable_uv_package(tmp_path: Path) -> None:
    """uv.lock must include the local support package as an editable entry."""
    _write_project(tmp_path)
    (tmp_path / "uv.lock").write_text(
        'version = 1\n\n[[package]]\nname = "markov-chain-monte-carlo-tooling"\nversion = "1.2.3"\nsource = { registry = "https://pypi.org/simple" }\n',
        encoding="utf-8",
    )

    with pytest.raises(release_check.ReleaseCheckError, match=r"exactly one uv\.lock editable package"):
        release_check.find_version_mismatches(tmp_path)


def test_find_version_mismatches_rejects_malformed_citation_version(tmp_path: Path) -> None:
    """Malformed citation versions fail before release."""
    _write_project(tmp_path)
    (tmp_path / "CITATION.cff").write_text('cff-version: 1.2.0\nversion: "\n', encoding="utf-8")

    with pytest.raises(release_check.ReleaseCheckError, match=r"CITATION\.cff:2: top-level version"):
        release_check.find_version_mismatches(tmp_path)


def test_main_reports_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI reports the synchronized release version."""
    _write_project(tmp_path)

    exit_code = release_check.main([str(tmp_path)])

    assert exit_code == 0
    assert "Release metadata is synchronized at 1.2.3" in capsys.readouterr().out


def test_main_reports_mismatches(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI reports mismatches on stderr and exits nonzero."""
    _write_project(tmp_path, metadata_version="1.2.2")

    exit_code = release_check.main([str(tmp_path)])

    assert exit_code == 1
    assert "Release-version references are out of sync" in capsys.readouterr().err
