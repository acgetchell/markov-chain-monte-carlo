"""Release preparation preserves metadata, evidence, and prior files on failure."""

from typing import TYPE_CHECKING

import pytest
from research_repo_tools.cli import main

import update_release_version as updater

from .test_release_check import _write_project

if TYPE_CHECKING:
    from pathlib import Path


def _snapshot(root: Path) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("tag", ["1.2.4", "v1.2", "v01.2.4", "v1.2.4-rc.1", "v1.2.4+build", " v1.2.4"])
def test_rejects_non_stable_or_noncanonical_tags_before_changes(tmp_path: Path, tag: str) -> None:
    _write_project(tmp_path)
    original = _snapshot(tmp_path)
    with pytest.raises(ValueError, match="stable"):
        updater.update_release_version(tmp_path, tag, previous_tag="v1.2.3")
    assert _snapshot(tmp_path) == original


def test_prepares_all_metadata_without_upgrading_dependencies_or_rewriting_evidence(tmp_path: Path) -> None:
    _write_project(tmp_path)
    cargo_lock = tmp_path / "Cargo.lock"
    cargo_lock.write_text(cargo_lock.read_text() + '\n[[package]]\nname = "dependency"\nversion = "9.8.7"\n', encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text(
        readme.read_text() + "[main guide](https://github.com/acgetchell/markov-chain-monte-carlo/blob/main/docs/guide.md)\n"
        "[plot](https://raw.githubusercontent.com/acgetchell/markov-chain-monte-carlo/v1.2.3/docs/archive/performance/v1.2.3-vs-v1.2.2.svg)\n",
        encoding="utf-8",
    )
    config = tmp_path / "pyproject.toml"
    config.write_bytes(config.read_bytes().replace(b"count = 1", b"count = 2"))
    (tmp_path / "docs").mkdir(exist_ok=True)
    guide = tmp_path / "docs" / "BENCHMARKING.md"
    guide.write_text("just performance-release v1.2.3 v1.2.2\nHistorical v1.0.0 remains unchanged.\n", encoding="utf-8")
    previous_changelog = (tmp_path / "CHANGELOG.md").read_bytes()
    result = updater.update_release_version(tmp_path, "v1.2.4", previous_tag="v1.2.3", release_date="2026-08-30")
    assert result.context.previous_tag == "v1.2.3"
    assert result.context.release_date == "2026-08-30"
    assert 'version = "9.8.7"' in cargo_lock.read_text()
    assert 'name = "markov-chain-monte-carlo"\nversion = "1.2.4"' in cargo_lock.read_text()
    assert 'version = "1.2.4"' in (tmp_path / "uv.lock").read_text()
    assert "doi: 10.5281/zenodo.20033111" in (tmp_path / "CITATION.cff").read_text()
    assert "date-released: 2026-08-30" in (tmp_path / "CITATION.cff").read_text()
    assert "blob/v1.2.4/docs/guide.md" in readme.read_text()
    assert "v1.2.3/docs/archive/performance/" in readme.read_text()
    assert "just performance-release v1.2.4 v1.2.3" in guide.read_text()
    assert "Historical v1.0.0 remains unchanged." in guide.read_text()
    assert (tmp_path / "CHANGELOG.md").read_bytes() == previous_changelog
    assert main(["--root", str(tmp_path), "release", "check", "--final-release"]) == 1  # Final validation still requires the prospective changelog.
    snapshot = _snapshot(tmp_path)
    assert updater.update_release_version(tmp_path, "v1.2.4", previous_tag="v1.2.3", release_date="2026-08-30").changed_paths == ()
    assert _snapshot(tmp_path) == snapshot


def test_preview_runs_shared_and_consumer_validation_without_writes(tmp_path: Path) -> None:
    _write_project(tmp_path)
    original = _snapshot(tmp_path)
    preview = updater.update_release_version(tmp_path, "v1.2.4", previous_tag="v1.2.3", release_date="2026-09-19", dry_run=True)
    assert _snapshot(tmp_path) == original
    applied = updater.update_release_version(tmp_path, "v1.2.4", previous_tag="v1.2.3", release_date="2026-09-19")
    assert preview == applied


def test_shared_synchronization_cannot_silently_repair_a_wrong_consumer_doi(tmp_path: Path) -> None:
    _write_project(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_bytes(readme.read_bytes().replace(b"10.5281/zenodo.20033111", b"10.5281/zenodo.12345"))
    original = _snapshot(tmp_path)
    with pytest.raises(ValueError, match="failed validation"):
        updater.update_release_version(tmp_path, "v1.2.4", previous_tag="v1.2.3")
    assert _snapshot(tmp_path) == original
