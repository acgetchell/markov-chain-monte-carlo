import io
import json
import subprocess
import tarfile
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

import archive_performance

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


def _release(tag: str, days_ago: int = 0) -> archive_performance.PublishedRelease:
    return archive_performance.PublishedRelease(tag, datetime(2026, 8, 1, tzinfo=UTC) - timedelta(days=days_ago))


def _report(current: str, baseline: str, marker: str = "") -> str:
    return (
        "# Benchmark Performance\n\n"
        f"**markov-chain-monte-carlo** {current} · `abc1234`\n"
        "**Statistic**: median\n\n"
        f"Comparison against baseline **{baseline}**:\n\n"
        f"{marker}\n"
    )


def _write_sample(root: Path, benchmark: str, sample: str) -> None:
    sample_dir = root / benchmark / sample
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "estimates.json").write_text(json.dumps({"median": {"point_estimate": 1.0}}), encoding="utf-8")
    (sample_dir / "sample.json").write_text("{}", encoding="utf-8")


def test_stable_published_releases_filters_and_sorts() -> None:
    document = [
        {"tagName": "v0.3.0", "isDraft": False, "isPrerelease": False, "publishedAt": "2026-05-01T00:00:00Z"},
        {"tagName": "v0.5.0-rc.1", "isDraft": False, "isPrerelease": True, "publishedAt": "2026-07-01T00:00:00Z"},
        {"tagName": "v0.4.0", "isDraft": False, "isPrerelease": False, "publishedAt": "2026-06-01T00:00:00+00:00"},
        {"tagName": "nightly", "isDraft": False, "isPrerelease": False, "publishedAt": "2026-08-01T00:00:00Z"},
    ]

    releases = archive_performance.stable_published_releases(document)

    assert [release.tag for release in releases] == ["v0.4.0", "v0.3.0"]


def test_stable_published_releases_rejects_non_boolean_flags() -> None:
    document = [{"tagName": "v0.4.0", "isDraft": "false", "isPrerelease": False, "publishedAt": "2026-06-01T00:00:00Z"}]

    with pytest.raises(TypeError, match="boolean"):
        archive_performance.stable_published_releases(document)


def test_stable_published_releases_rejects_duplicate_tags() -> None:
    document = [
        {"tagName": "v0.4.0", "isDraft": False, "isPrerelease": False, "publishedAt": "2026-06-01T00:00:00Z"},
        {"tagName": "v0.4.0", "isDraft": False, "isPrerelease": False, "publishedAt": "2026-06-02T00:00:00Z"},
    ]

    with pytest.raises(ValueError, match="duplicate"):
        archive_performance.stable_published_releases(document)


def test_infer_release_uses_latest_for_an_unpublished_package() -> None:
    releases = (_release("v0.4.0"), _release("v0.3.0", 30))

    pair = archive_performance.resolve_release_pair(mode="infer-release", releases=releases, package_tag="v0.5.0")

    assert pair == archive_performance.ReportId("v0.5.0", "v0.4.0")


def test_infer_release_uses_previous_when_package_is_already_published() -> None:
    releases = (_release("v0.4.0"), _release("v0.3.0", 30), _release("v0.2.1", 60))

    pair = archive_performance.resolve_release_pair(mode="infer-release", releases=releases, package_tag="v0.4.0")

    assert pair == archive_performance.ReportId("v0.4.0", "v0.3.0")


@pytest.mark.parametrize("package_tag", ["v0.3.9", "v0.1.0"])
def test_infer_release_rejects_an_unpublished_package_not_newer_than_latest(package_tag: str) -> None:
    releases = (_release("v0.4.0"), _release("v0.3.0", 30))

    with pytest.raises(ValueError, match="must be newer"):
        archive_performance.resolve_release_pair(
            mode="infer-release",
            releases=releases,
            package_tag=package_tag,
        )


def test_published_latest_requires_two_stable_releases() -> None:
    with pytest.raises(RuntimeError, match="at least two"):
        archive_performance.resolve_release_pair(mode="published-latest", releases=(_release("v0.4.0"),), package_tag="v0.5.0")


def test_explicit_pair_requires_both_tags() -> None:
    with pytest.raises(ValueError, match="both required"):
        archive_performance.resolve_release_pair(
            mode="explicit",
            releases=(_release("v0.4.0"),),
            package_tag="v0.5.0",
            current_tag="v0.5.0",
        )


def test_github_assets_rejects_working_tree_resolution() -> None:
    with pytest.raises(ValueError, match="requires --published-latest"):
        archive_performance._validate_mode_combination("current-vs-latest", github_assets=True)


def test_published_latest_rejects_local_measurements() -> None:
    with pytest.raises(ValueError, match="reserved"):
        archive_performance._validate_mode_combination("published-latest", github_assets=False)


def test_current_vs_latest_promotion_is_rejected_before_release_lookup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def unexpected_release_lookup(_repo_root: Path) -> tuple[archive_performance.PublishedRelease, ...]:
        pytest.fail("release lookup must not run for an invalid mode combination")

    monkeypatch.setattr(archive_performance, "_published_releases", unexpected_release_lookup)

    status = archive_performance.main(
        [
            "--current-vs-latest",
            "--promote",
            "--repo-root",
            str(tmp_path),
        ]
    )

    assert status == 2
    assert "working-tree reports cannot be promoted" in capsys.readouterr().err


def test_promote_report_archives_the_previous_pair_and_updates_index(tmp_path: Path) -> None:
    current = tmp_path / "docs" / "PERFORMANCE.md"
    archive_dir = tmp_path / "docs" / "archive" / "performance"
    current.parent.mkdir(parents=True)
    current.write_text(_report("v0.4.0", "v0.3.0", "old"), encoding="utf-8")

    archive_performance.promote_report(
        source_text=_report("v0.5.0", "v0.4.0", "new"),
        current_path=current,
        archive_dir=archive_dir,
        expected=archive_performance.ReportId("v0.5.0", "v0.4.0"),
    )

    assert "new" in current.read_text(encoding="utf-8")
    assert "old" in (archive_dir / "v0.4.0-vs-v0.3.0.md").read_text(encoding="utf-8")
    index = (archive_dir / "README.md").read_text(encoding="utf-8")
    assert "[v0.4.0-vs-v0.3.0](v0.4.0-vs-v0.3.0.md)" in index


def test_promote_report_preserves_an_existing_archive(tmp_path: Path) -> None:
    current = tmp_path / "PERFORMANCE.md"
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    current.write_text(_report("v0.4.0", "v0.3.0", "old current"), encoding="utf-8")
    archive = archive_dir / "v0.4.0-vs-v0.3.0.md"
    archive.write_text("preserved archive\n", encoding="utf-8")

    archive_performance.promote_report(
        source_text=_report("v0.5.0", "v0.4.0"),
        current_path=current,
        archive_dir=archive_dir,
        expected=archive_performance.ReportId("v0.5.0", "v0.4.0"),
    )

    assert archive.read_text(encoding="utf-8") == "preserved archive\n"


def test_promote_report_rejects_a_mismatched_release_pair(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not match"):
        archive_performance.promote_report(
            source_text=_report("v0.5.0", "v0.4.0"),
            current_path=tmp_path / "PERFORMANCE.md",
            archive_dir=tmp_path / "archive",
            expected=archive_performance.ReportId("v0.6.0", "v0.5.0"),
        )


def test_generated_working_tree_report_can_be_promoted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair = archive_performance.ReportId("v0.5.0", "v0.4.0")
    applied: list[tuple[Path, Path]] = []

    @contextmanager
    def fake_worktree(
        _repo_root: Path,
        parent: Path,
        name: str,
        _reference: str,
    ) -> Iterator[Path]:
        checkout = parent / name
        checkout.mkdir()
        yield checkout

    def fake_benchmark(checkout: Path, *, save_baseline: str | None = None) -> tuple[str, ...]:
        sample = save_baseline or "new"
        _write_sample(checkout / "target" / "criterion", "chain/step_by_value", sample)
        return ("cargo", "bench")

    def fake_apply(repo_root: Path, worktree: Path) -> None:
        applied.append((repo_root, worktree))

    monkeypatch.setattr(archive_performance, "_ensure_local_tag", lambda *_args: None)
    monkeypatch.setattr(archive_performance, "temporary_worktree", fake_worktree)
    monkeypatch.setattr(archive_performance, "_run_stepping_benchmark", fake_benchmark)
    monkeypatch.setattr(archive_performance, "apply_current_tree", fake_apply)
    monkeypatch.setattr(archive_performance, "_local_measurement_context", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(
        archive_performance,
        "run_git_command",
        lambda args, **_kwargs: subprocess.CompletedProcess(args, 0, stdout="abc1234\n"),
    )

    report = archive_performance.generate_local_report(
        tmp_path,
        pair,
        current_reference="HEAD",
        apply_working_tree=True,
    )
    current = tmp_path / "docs" / "PERFORMANCE.md"
    archive_performance.promote_report(
        source_text=report,
        current_path=current,
        archive_dir=tmp_path / "docs" / "archive" / "performance",
        expected=pair,
    )

    assert applied
    assert archive_performance.parse_report_id(current.read_text(encoding="utf-8")) == pair
    assert "**markov-chain-monte-carlo** v0.5.0 working tree" in report


def test_apply_current_tree_applies_tracked_changes_and_copies_untracked_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "source"
    worktree = tmp_path / "worktree"
    untracked = repo_root / "nested" / "fixture.txt"
    untracked.parent.mkdir(parents=True)
    untracked.write_text("fixture\n", encoding="utf-8")
    worktree.mkdir()
    applied: list[tuple[list[str], str, Path]] = []

    def fake_git(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        stdout = "tracked patch\n" if "diff" in args else "nested/fixture.txt\n"
        return subprocess.CompletedProcess(args, 0, stdout=stdout)

    def fake_git_with_input(args: list[str], input_text: str, *, cwd: Path, **_kwargs: object) -> object:
        applied.append((args, input_text, cwd))
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(archive_performance, "run_git_command", fake_git)
    monkeypatch.setattr(archive_performance, "run_git_command_with_input", fake_git_with_input)

    archive_performance.apply_current_tree(repo_root, worktree)

    assert applied == [
        (["apply", "--binary", "--whitespace=nowarn"], "tracked patch\n", worktree),
    ]
    assert (worktree / "nested" / "fixture.txt").read_text(encoding="utf-8") == "fixture\n"


def test_temporary_worktree_removes_checkout_after_body_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_git(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="")

    monkeypatch.setattr(archive_performance, "run_git_command", fake_git)

    msg = "benchmark failed"
    with pytest.raises(RuntimeError, match="benchmark failed"), archive_performance.temporary_worktree(tmp_path, tmp_path, "checkout", "HEAD"):
        raise RuntimeError(msg)

    assert calls == [
        ["worktree", "add", "--detach", str(tmp_path / "checkout"), "HEAD"],
        ["worktree", "remove", "--force", str(tmp_path / "checkout")],
    ]


def test_copy_criterion_sample_renames_each_benchmark_sample(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    _write_sample(source, "chain/step_by_value", "v0.4.0")
    _write_sample(source, "sampler/run_by_value_100", "v0.4.0")

    copied = archive_performance.copy_criterion_sample(
        source_criterion=source,
        destination_criterion=destination,
        source_sample="v0.4.0",
        destination_sample="new",
    )

    assert copied == 2
    assert (destination / "chain" / "step_by_value" / "new" / "sample.json").is_file()
    assert (destination / "sampler" / "run_by_value_100" / "new" / "estimates.json").is_file()


def test_copy_criterion_sample_fails_when_asset_has_no_requested_baseline(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no Criterion sample"):
        archive_performance.copy_criterion_sample(
            source_criterion=tmp_path / "missing",
            destination_criterion=tmp_path / "destination",
            source_sample="v0.4.0",
            destination_sample="new",
        )


def test_release_metadata_validates_tag_and_measurement_context(tmp_path: Path) -> None:
    metadata = {
        "schema": 1,
        "tag": "v0.5.0",
        "commit": "a" * 40,
        "operating_system": "Linux",
        "architecture": "x86_64",
        "rustc": "rustc 1.97.1",
        "criterion_version": "0.8.2",
    }
    (tmp_path / ".mcmc-release-metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    parsed = archive_performance._release_metadata(tmp_path, "v0.5.0")

    assert parsed.commit == "a" * 40
    assert parsed.criterion_version == "0.8.2"


def test_release_metadata_rejects_a_mismatched_tag(tmp_path: Path) -> None:
    metadata = {
        "schema": 1,
        "tag": "v0.4.0",
        "commit": "a" * 40,
        "operating_system": "Linux",
        "architecture": "x86_64",
        "rustc": "rustc 1.97.1",
        "criterion_version": "0.8.2",
    }
    (tmp_path / ".mcmc-release-metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match"):
        archive_performance._release_metadata(tmp_path, "v0.5.0")


def test_safe_extract_tar_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar.gz"
    payload = b"unsafe"
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo("../outside.txt")
        member.size = len(payload)
        tar.addfile(member, io.BytesIO(payload))

    with pytest.raises(ValueError, match="outside its root"):
        archive_performance.safe_extract_tar(archive, tmp_path / "extract")

    assert not (tmp_path / "outside.txt").exists()


def test_safe_extract_tar_rejects_links(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe-link.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo("criterion/link")
        member.type = tarfile.SYMTYPE
        member.linkname = "../../outside.txt"
        tar.addfile(member)

    with pytest.raises(ValueError, match="unsupported entry"):
        archive_performance.safe_extract_tar(archive, tmp_path / "extract")


def test_command_failure_preserves_stdout_and_stderr() -> None:
    error = subprocess.CalledProcessError(7, ["cargo", "bench"], output="partial output\n", stderr="compiler failed\n")

    message = archive_performance._format_command_failure(error)

    assert "exit 7: cargo bench" in message
    assert "partial output" in message
    assert "compiler failed" in message
