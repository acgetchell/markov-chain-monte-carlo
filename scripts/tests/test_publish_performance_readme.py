"""Retained-data README publication validates before writing and preserves prior output."""

import hashlib
import re
import shutil
from dataclasses import replace
from subprocess import CompletedProcess
from typing import TYPE_CHECKING
from xml.etree import ElementTree as ET

import pytest

import archive_performance
import publish_performance_readme as publisher

from .test_archive_performance import REPO_ROOT, _comparison_artifact

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _unpublished_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    def git(args: list[str], **_kwargs: object) -> CompletedProcess[str]:
        assert args == ["--no-pager", "show-ref", "--verify", "--quiet", "refs/tags/v0.5.0"]
        return CompletedProcess(args, 1, "", "")

    monkeypatch.setattr(publisher, "run_git_command", git)


def _prepare(
    root: Path,
    *,
    same_version: bool = False,
    published: bool = False,
    benchmark: str = "chain/shared",
    repository: str = "https://github.com/acgetchell/markov-chain-monte-carlo",
) -> tuple[Path, Path]:
    artifact = _comparison_artifact()
    comparison_set = artifact.comparison_set
    artifact = replace(
        artifact,
        comparison_set=replace(
            comparison_set,
            comparisons=(replace(comparison_set.comparisons[0], benchmark=benchmark),),
            current_sample=tuple(sorted((benchmark if name == "chain/shared" else name, estimate) for name, estimate in comparison_set.current_sample)),
            baseline_sample=tuple(sorted((benchmark if name == "chain/shared" else name, estimate) for name, estimate in comparison_set.baseline_sample)),
        ),
        settings=replace(artifact.settings, current_label=artifact.pair.current_tag if published else f"{artifact.pair.current_tag} working tree"),
        measurement=replace(artifact.measurement, working_tree_applied=not published),
    )
    if same_version:
        pair = archive_performance.ReportId("v0.5.0", "v0.5.0")
        artifact = replace(
            artifact,
            pair=pair,
            settings=replace(artifact.settings, current_label=f"{pair.current_tag} working tree", baseline_label=pair.baseline_tag),
            measurement=replace(artifact.measurement, working_tree_applied=True, baseline=replace(artifact.measurement.baseline, tag=pair.baseline_tag)),
        )
    (root / "Cargo.toml").write_text(
        f'[package]\nname = "markov-chain-monte-carlo"\nversion = "0.5.0"\nrepository = "{repository}"\n', encoding="utf-8", newline="\n"
    )
    (root / "README.md").write_text(f"# Project\n\n{publisher.BEGIN}\n\nNot published yet.\n\n{publisher.END}\n\nKeep this prose.\n", encoding="utf-8")
    archive = root / "docs" / "archive" / "performance"
    archive_performance.promote_report(artifact=artifact, current_path=root / "docs" / "PERFORMANCE.md", archive_dir=archive)
    evidence = archive / f"{artifact.pair.evidence_stem}.csv"
    return evidence, evidence.with_suffix(".svg")


def test_publication_is_deterministic_uses_retained_data_and_never_measures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    evidence, svg = _prepare(tmp_path)
    originals = (evidence.read_bytes(), archive_performance.provenance_path(evidence).read_bytes())
    monkeypatch.setattr(archive_performance, "run_safe_command", lambda *_args, **_kwargs: pytest.fail("publication ran an external tool"))
    monkeypatch.setattr(archive_performance, "generate_local_artifact", lambda *_args, **_kwargs: pytest.fail("publication measured benchmarks"))
    changed = publisher.publish_readme(tmp_path)
    assert changed == (svg, tmp_path / "README.md")
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "80.00 ns" in readme
    assert "100.00 ns" in readme
    assert "1.25x faster" in readme
    assert "1 comparable, 1 current-only, 1 baseline-only" in readme
    assert "v0.5.0/docs/archive/performance/" in readme
    assert "Keep this prose." in readme
    assert "not significance tests" in readme
    assert ET.fromstring(svg.read_text(encoding="utf-8")).tag.endswith("svg")  # noqa: S314 - locally generated SVG, no external input.
    snapshot = (readme, svg.read_bytes())
    assert publisher.publish_readme(tmp_path) == ()
    assert ((tmp_path / "README.md").read_text(encoding="utf-8"), svg.read_bytes()) == snapshot
    assert (evidence.read_bytes(), archive_performance.provenance_path(evidence).read_bytes()) == originals


@pytest.mark.parametrize("suffix", ["", "/", ".git", ".git/"])
def test_publication_uses_manifest_repository_for_all_links(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str) -> None:
    evidence, svg = _prepare(tmp_path, repository=f"https://github.com/release-owner/mcmc-fork{suffix}")
    monkeypatch.setattr(publisher, "_svg", lambda _artifact: "<svg/>\n")

    publisher.publish_readme(tmp_path)

    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert re.findall(r"https://[^\s)]+", readme) == [
        f"https://raw.githubusercontent.com/release-owner/mcmc-fork/v0.5.0/{svg.relative_to(tmp_path).as_posix()}",
        "https://github.com/release-owner/mcmc-fork/blob/v0.5.0/docs/PERFORMANCE.md",
        f"https://github.com/release-owner/mcmc-fork/blob/v0.5.0/{evidence.relative_to(tmp_path).as_posix()}",
        f"https://github.com/release-owner/mcmc-fork/blob/v0.5.0/{archive_performance.provenance_path(evidence).relative_to(tmp_path).as_posix()}",
    ]


def _existing_tag(monkeypatch: pytest.MonkeyPatch, root: Path, assets: dict[str, bytes]) -> None:
    """Model the immutable tag's Git blob IDs independently of the working files."""
    commit = "a" * 64

    def blob_id(contents: bytes) -> str:
        return hashlib.sha256(f"blob {len(contents)}\0".encode() + contents).hexdigest()

    def git(args: list[str], *, cwd: Path, **_kwargs: object) -> CompletedProcess[str]:
        assert cwd == root
        if args == ["--no-pager", "show-ref", "--verify", "--quiet", "refs/tags/v0.5.0"]:
            return CompletedProcess(args, 0, "", "")
        if args == ["--no-pager", "rev-parse", "--verify", "refs/tags/v0.5.0^{commit}"]:
            return CompletedProcess(args, 0, commit + "\n", "")
        assert args[:3] == ["--no-pager", "rev-parse", "--verify"]
        assert args[3].startswith(commit + ":")
        stored = assets.get(args[3].split(":", 1)[1])
        return CompletedProcess(args, 128 if stored is None else 0, "" if stored is None else blob_id(stored) + "\n", "")

    def git_input(args: list[str], contents: str, *, cwd: Path, **_kwargs: object) -> CompletedProcess[str]:
        assert cwd == root
        assert args[:2] == ["--no-pager", "hash-object"]
        assert args[2].startswith("--path=")
        assert args[3:] == ["--stdin"]
        return CompletedProcess(args, 0, blob_id(contents.encode("utf-8")) + "\n", "")

    monkeypatch.setattr(publisher, "run_git_command", git)
    monkeypatch.setattr(publisher, "run_git_command_with_input", git_input)


def test_tag_verification_honors_git_checkout_line_endings(monkeypatch: pytest.MonkeyPatch) -> None:
    if shutil.which("git") is None:
        pytest.skip("git is required to verify checkout line-ending conversion")
    git_input = publisher.run_git_command_with_input
    expected = git_input(["--no-pager", "hash-object", "--stdin"], "line\n", cwd=REPO_ROOT).stdout

    def git(args: list[str], **_kwargs: object) -> CompletedProcess[str]:
        output = "a" * 40 + "\n" if args[-1].endswith("^{commit}") else expected
        return CompletedProcess(args, 0, output, "")

    def autocrlf_input(args: list[str], contents: str, *, cwd: Path, **_kwargs: object) -> CompletedProcess[str]:
        return git_input(["--no-pager", "-c", "core.autocrlf=true", *args[1:]], contents, cwd=cwd)

    monkeypatch.setattr(publisher, "run_git_command", git)
    monkeypatch.setattr(publisher, "run_git_command_with_input", autocrlf_input)
    publisher._validate_publication_tag(REPO_ROOT, _comparison_artifact(), ((REPO_ROOT / "docs/PERFORMANCE.md", "line\r\n"),))


@pytest.mark.parametrize("same_version", [False, True])
def test_existing_tag_requires_and_reuses_exact_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, same_version: bool) -> None:
    evidence, svg = _prepare(tmp_path, published=True, same_version=same_version)
    monkeypatch.setattr(publisher, "_svg", lambda _artifact: "<svg/>\n")
    assets = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in (tmp_path / "docs/PERFORMANCE.md", evidence, evidence.with_suffix(".provenance.json"))
    }
    assets[svg.relative_to(tmp_path).as_posix()] = b"<svg/>\n"
    _existing_tag(monkeypatch, tmp_path, assets)

    assert publisher.publish_readme(tmp_path) == (svg, tmp_path / "README.md")
    assert publisher.publish_readme(tmp_path) == ()
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "v0.5.0/docs/archive/performance/" in readme
    if same_version:
        assert "v0.5.0 working tree against v0.5.0" in readme


@pytest.mark.parametrize("member", ["report", "csv", "json", "svg"])
@pytest.mark.parametrize("missing", [False, True])
def test_existing_tag_missing_or_changed_assets_prevent_all_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, member: str, missing: bool) -> None:
    evidence, svg = _prepare(tmp_path, published=True)
    monkeypatch.setattr(publisher, "_svg", lambda _artifact: "<svg/>\n")
    paths = {"report": tmp_path / "docs/PERFORMANCE.md", "csv": evidence, "json": archive_performance.provenance_path(evidence), "svg": svg}
    assets = {path.relative_to(tmp_path).as_posix(): b"<svg/>\n" if path == svg else path.read_bytes() for path in paths.values()}
    selected = paths[member].relative_to(tmp_path).as_posix()
    if missing:
        del assets[selected]
    else:
        assets[selected] += b"changed"
    _existing_tag(monkeypatch, tmp_path, assets)
    original = (tmp_path / "README.md").read_bytes()
    svg.write_bytes(b"previous plot")

    with pytest.raises(ValueError, match="does not contain the exact publication artifact") as raised:
        publisher.publish_readme(tmp_path)
    assert selected in str(raised.value)
    assert (tmp_path / "README.md").read_bytes() == original
    assert svg.read_bytes() == b"previous plot"


@pytest.mark.parametrize("same_version", [False, True])
def test_existing_release_evidence_requires_its_local_tag(tmp_path: Path, same_version: bool) -> None:
    _evidence, svg = _prepare(tmp_path, published=True, same_version=same_version)
    original = (tmp_path / "README.md").read_bytes()
    with pytest.raises(ValueError, match=r"requires the local v0\.5\.0 tag"):
        publisher.publish_readme(tmp_path)
    assert (tmp_path / "README.md").read_bytes() == original
    assert not svg.exists()


def test_git_failure_is_reported_without_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _evidence, svg = _prepare(tmp_path)
    original = (tmp_path / "README.md").read_bytes()
    monkeypatch.setattr(publisher, "run_git_command", lambda args, **_kwargs: CompletedProcess(args, 128, "", "not a repository"))
    assert publisher.main(["--repo-root", str(tmp_path)]) == 1
    assert "README publication failed" in capsys.readouterr().err
    assert (tmp_path / "README.md").read_bytes() == original
    assert not svg.exists()


def test_markdown_workload_names_remain_literal(tmp_path: Path) -> None:
    _prepare(tmp_path, benchmark="chain/![probe](https://example.com/pixel.png) [link](https://example.com) *emphasis*|`tick`/value")
    publisher.publish_readme(tmp_path)
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert r"| ``chain/![probe](https://example.com/pixel.png) [link](https://example.com) *emphasis*\|`tick`/value`` |" in readme


@pytest.mark.parametrize("missing", ["csv", "json", "report"])
def test_incomplete_evidence_fails_before_any_publication(tmp_path: Path, missing: str) -> None:
    evidence, svg = _prepare(tmp_path)
    destinations = {"csv": evidence, "json": archive_performance.provenance_path(evidence), "report": tmp_path / "docs" / "PERFORMANCE.md"}
    destinations[missing].unlink()
    original = (tmp_path / "README.md").read_bytes()
    with pytest.raises(FileNotFoundError, match="just performance-release"):
        publisher.publish_readme(tmp_path)
    assert (tmp_path / "README.md").read_bytes() == original
    assert not svg.exists()


def test_tampered_evidence_retains_distinct_digest_error(tmp_path: Path) -> None:
    evidence, svg = _prepare(tmp_path)
    evidence.write_text(evidence.read_text(encoding="utf-8").replace("80", "81"), encoding="utf-8")
    original = (tmp_path / "README.md").read_bytes()
    with pytest.raises(ValueError, match="SHA-256 digest"):
        publisher.publish_readme(tmp_path)
    assert (tmp_path / "README.md").read_bytes() == original
    assert not svg.exists()


def test_failed_readme_replacement_restores_existing_svg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _evidence, svg = _prepare(tmp_path)
    svg.write_text("previous plot", encoding="utf-8")
    readme = tmp_path / "README.md"
    original = readme.read_bytes()
    write = archive_performance._write_text

    def fail_readme(path: Path, text: str) -> None:
        if path == readme:
            msg = "simulated README replacement failure"
            raise OSError(msg)
        write(path, text)

    monkeypatch.setattr(archive_performance, "_write_text", fail_readme)
    with pytest.raises(OSError, match="replacement failure"):
        publisher.publish_readme(tmp_path)
    assert readme.read_bytes() == original
    assert svg.read_text(encoding="utf-8") == "previous plot"


@pytest.mark.parametrize("destination", ["README.md", "svg", "archive"])
def test_tracked_publication_paths_cannot_escape_repository(tmp_path: Path, destination: str) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    evidence, svg = _prepare(root)
    outside = tmp_path / "outside"
    outside.mkdir()
    if destination == "archive":
        archive = evidence.parent
        archive.rename(outside / "evidence")
        link, target = archive, outside / "evidence"
    else:
        link = root / "README.md" if destination == "README.md" else svg
        target = outside / "saved.txt"
        target.write_text("must survive", encoding="utf-8")
        link.unlink(missing_ok=True)
    try:
        link.symlink_to(target, target_is_directory=destination == "archive")
    except OSError:
        pytest.skip("symlink creation is unavailable on this runner")
    with pytest.raises(ValueError, match="inside the repository"):
        publisher.publish_readme(root)
    assert link.is_symlink()
    if target.is_file():
        assert target.read_text(encoding="utf-8") == "must survive"


def test_stale_release_evidence_is_not_published_under_a_new_tag(tmp_path: Path) -> None:
    _evidence, svg = _prepare(tmp_path)
    (tmp_path / "Cargo.toml").write_text('[package]\nversion = "0.6.0"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="current package version"):
        publisher.publish_readme(tmp_path)
    assert not svg.exists()


@pytest.mark.parametrize("markers", ["No markers", publisher.BEGIN + publisher.END + publisher.END, publisher.END + publisher.BEGIN])
def test_invalid_readme_boundaries_leave_publication_untouched(tmp_path: Path, markers: str) -> None:
    _evidence, svg = _prepare(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(markers, encoding="utf-8")
    with pytest.raises(ValueError, match="marker pair"):
        publisher.publish_readme(tmp_path)
    assert readme.read_text(encoding="utf-8") == markers
    assert not svg.exists()


def test_report_cannot_publish_evidence_from_another_release_pair(tmp_path: Path) -> None:
    evidence, svg = _prepare(tmp_path)
    sidecar = archive_performance.provenance_path(evidence)
    sidecar.write_text(sidecar.read_text(encoding="utf-8").replace("v0.4.0", "v0.3.0"), encoding="utf-8")
    original = (tmp_path / "README.md").read_bytes()
    with pytest.raises(ValueError, match="release pair does not match"):
        publisher.publish_readme(tmp_path)
    assert (tmp_path / "README.md").read_bytes() == original
    assert not svg.exists()
