"""Tests for release metadata and version synchronization checks."""

import re
import tomllib
from pathlib import Path

import pytest
from research_repo_tools.cli import main

ROOT = Path(__file__).resolve().parents[2]
_DOI = "10.5281/zenodo.20033111"
_VERSION = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))["package"]["version"]
_GUIDES = (
    "REFERENCES.md",
    "docs/ANALYZING_CHAINS.md",
    "docs/BENCHMARKING.md",
    "docs/RELEASING.md",
    "docs/VALIDATING_PROPOSALS.md",
    "docs/benchmark_distributions.md",
    "docs/code_organization.md",
    "docs/dev/DEVELOPING.md",
    "docs/reviewer_guide.md",
    "docs/roadmap.md",
    "docs/scientific_basis.md",
)


def _write_project(root: Path) -> None:
    """Exercise the actual release policy and metadata without synthetic mirrors."""
    manifest = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    for name in ("Cargo.toml", *manifest["tool"]["research-repo-tools"]["release"]["required-files"]):
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    assert main(["--root", str(root), "release", "check", "--final-release"]) == 0


def test_configured_checker_requires_current_source_links(tmp_path: Path) -> None:
    _write_project(tmp_path)
    path = tmp_path / "README.md"
    path.write_text(path.read_text(encoding="utf-8").replace(f"/blob/v{_VERSION}/", "/blob/v0.0.0/"), encoding="utf-8", newline="\n")
    assert main(["--root", str(tmp_path), "release", "check", "--final-release"]) == 1


@pytest.mark.parametrize(
    ("reference", "replacement"),
    [
        (f"docs.rs/markov-chain-monte-carlo/{_VERSION}/", "docs.rs/markov-chain-monte-carlo/0.0.0/"),
        (f"docs.rs/markov-chain-monte-carlo/{_VERSION}/", "docs.rs/markov-chain-monte-carlo/latest/"),
        (f"published API links below target v{_VERSION}.", "published API links below target v0.0.0."),
    ],
)
def test_published_api_references_require_the_declared_version(tmp_path: Path, reference: str, replacement: str) -> None:
    _write_project(tmp_path)
    readme = tmp_path / "README.md"
    before = readme.read_text(encoding="utf-8")
    stale = before.replace(reference, replacement, 1)
    assert stale != before
    readme.write_text(stale, encoding="utf-8", newline="\n")
    assert main(["--root", str(tmp_path), "release", "check", "--final-release"]) == 1


def test_release_update_advances_api_guides_and_nested_package_version(tmp_path: Path) -> None:
    _write_project(tmp_path)
    readme = tmp_path / "README.md"
    before = readme.read_text(encoding="utf-8")
    api_links = re.findall(r"https://docs\.rs/markov-chain-monte-carlo/[^/\s)]+/[^\s)]+", before)
    assert len(api_links) == 11
    for guide in _GUIDES:
        assert f"https://github.com/acgetchell/markov-chain-monte-carlo/blob/v{_VERSION}/{guide}" in before
    nested_lock = tmp_path / "benches/diagnostic_backends/Cargo.lock"
    nested_packages = tomllib.loads(nested_lock.read_text(encoding="utf-8"))["package"]
    major, minor, patch = _VERSION.split(".")
    next_version = f"{major}.{minor}.{int(patch) + 1}"

    assert (
        main(
            [
                "--root",
                str(tmp_path),
                "release",
                "update",
                f"v{next_version}",
                "--previous-release",
                f"v{_VERSION}",
                "--date",
                "2099-01-01",
                "--offline",
            ]
        )
        == 0
    )

    after = readme.read_text(encoding="utf-8")
    assert tomllib.loads((tmp_path / "Cargo.toml").read_text(encoding="utf-8"))["package"]["version"] == next_version
    assert re.findall(r"https://docs\.rs/markov-chain-monte-carlo/[^/\s)]+/[^\s)]+", after) == [
        link.replace(f"/{_VERSION}/", f"/{next_version}/") for link in api_links
    ]
    assert f"published API links below target v{next_version}." in after
    for guide in _GUIDES:
        assert f"https://github.com/acgetchell/markov-chain-monte-carlo/blob/v{next_version}/{guide}" in after
    assert tomllib.loads(nested_lock.read_text(encoding="utf-8"))["package"] == [
        {**package, "version": next_version} if package["name"] == "markov-chain-monte-carlo" else package for package in nested_packages
    ]


def test_nested_benchmark_lockfile_requires_current_package_version(tmp_path: Path) -> None:
    _write_project(tmp_path)
    lockfile = tmp_path / "benches/diagnostic_backends/Cargo.lock"
    before = lockfile.read_text(encoding="utf-8")
    stale = before.replace(f'name = "markov-chain-monte-carlo"\nversion = "{_VERSION}"', 'name = "markov-chain-monte-carlo"\nversion = "0.0.0"')
    assert stale != before
    lockfile.write_text(stale, encoding="utf-8", newline="\n")
    assert main(["--root", str(tmp_path), "release", "check", "--final-release"]) == 1


def test_consistent_but_wrong_concept_doi_is_rejected(tmp_path: Path) -> None:
    _write_project(tmp_path)
    for name in ("CITATION.cff", "README.md", "REFERENCES.md"):
        path = tmp_path / name
        path.write_text(path.read_text(encoding="utf-8").replace(_DOI, "10.5281/zenodo.12345"), encoding="utf-8", newline="\n")
    assert main(["--root", str(tmp_path), "release", "check", "--final-release"]) == 1


@pytest.mark.parametrize("guide", _GUIDES)
@pytest.mark.parametrize("reference", ["main", "v0.0.0"])
def test_active_guides_require_the_declared_release(tmp_path: Path, guide: str, reference: str) -> None:
    _write_project(tmp_path)
    readme = tmp_path / "README.md"
    before = readme.read_text(encoding="utf-8")
    stale = before.replace(f"/blob/v{_VERSION}/{guide}", f"/blob/{reference}/{guide}")
    assert stale != before
    readme.write_text(stale, encoding="utf-8", newline="\n")
    assert main(["--root", str(tmp_path), "release", "check", "--final-release"]) == 1


def test_historical_evidence_and_changelog_archives_are_preserved(tmp_path: Path) -> None:
    _write_project(tmp_path)
    for folder in ("docs/performance/v1", "docs/archives/changelog", "tests/fixtures"):
        directory = tmp_path / folder
        directory.mkdir(parents=True)
        (directory / "old.md").write_text(
            'markov-chain-monte-carlo = "0.1.0"\njust performance-release v0.1.0 v0.0.9\n',
            encoding="utf-8",
            newline="\n",
        )
    readme = tmp_path / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8")
        + "[evidence](https://github.com/acgetchell/markov-chain-monte-carlo/blob/v1.0.0/docs/performance/v1/v1.0.0-vs-v0.9.0.md)\n",
        encoding="utf-8",
        newline="\n",
    )
    assert main(["--root", str(tmp_path), "release", "check", "--final-release"]) == 0
