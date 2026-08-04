#!/usr/bin/env python3
"""Generate, promote, and archive release benchmark reports.

Local comparisons run the current tree and a published release in isolated
temporary worktrees.  Historical comparisons consume durable Criterion
archives attached to GitHub Releases and never run Cargo.
"""

import argparse
import json
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from bench_compare import ComparisonSet, ReportSettings, _write_text, collect_comparisons, render_report
from subprocess_utils import ExecutableNotFoundError, run_git_command, run_git_command_with_input, run_safe_command

if TYPE_CHECKING:
    from collections.abc import Iterator

_REPOSITORY = "acgetchell/markov-chain-monte-carlo"
_ASSET_TEMPLATE = "markov-chain-monte-carlo-{tag}-criterion-baseline.tar.gz"
_TAG_RE = re.compile(r"^v(?P<major>0|[1-9][0-9]*)\.(?P<minor>0|[1-9][0-9]*)\.(?P<patch>0|[1-9][0-9]*)$")
_VERSION_RE = re.compile(r"^\*\*markov-chain-monte-carlo\*\* (?P<version>v[0-9]+\.[0-9]+\.[0-9]+)", re.MULTILINE)
_BASELINE_RE = re.compile(r"^Comparison against baseline \*\*(?P<baseline>v[0-9]+\.[0-9]+\.[0-9]+)\*\*:", re.MULTILINE)
_COMMAND_TIMEOUT_SECONDS = 600
_BENCHMARK_TIMEOUT_SECONDS = 7200

type ResolutionMode = Literal["explicit", "published-latest", "infer-release", "current-vs-latest"]


@dataclass(frozen=True, slots=True)
class ReportId:
    """The release pair represented by one curated report."""

    current_tag: str
    baseline_tag: str

    @property
    def archive_name(self) -> str:
        """Return the canonical archive filename."""
        return f"{self.current_tag}-vs-{self.baseline_tag}.md"


@dataclass(frozen=True, slots=True)
class PublishedRelease:
    """Stable GitHub Release metadata used for pair inference."""

    tag: str
    published_at: datetime


@dataclass(frozen=True, slots=True)
class ReleaseMetadata:
    """Validated measurement metadata stored beside release Criterion data."""

    tag: str
    commit: str
    operating_system: str
    architecture: str
    rustc: str
    criterion_version: str


def normalize_tag(tag: str) -> str:
    """Normalize a stable SemVer release tag to a leading-``v`` form."""
    normalized = tag.strip()
    if normalized and not normalized.startswith("v"):
        normalized = f"v{normalized}"
    if _TAG_RE.fullmatch(normalized) is None:
        msg = f"expected a stable SemVer tag like v0.4.0, got {tag!r}"
        raise ValueError(msg)
    return normalized


def _tag_version(tag: str) -> tuple[int, int, int]:
    """Return the numeric stable-SemVer components of a normalized tag."""
    normalized = normalize_tag(tag)
    match = _TAG_RE.fullmatch(normalized)
    if match is None:  # normalize_tag already rejects this state.
        msg = f"expected a normalized stable SemVer tag, got {normalized!r}"
        raise ValueError(msg)
    return (
        int(match.group("major")),
        int(match.group("minor")),
        int(match.group("patch")),
    )


def parse_report_id(text: str) -> ReportId:
    """Read a release pair from a generated benchmark report."""
    version = _VERSION_RE.search(text)
    if version is None:
        msg = "could not find the markov-chain-monte-carlo version line in the benchmark report"
        raise ValueError(msg)
    baseline = _BASELINE_RE.search(text)
    if baseline is None:
        msg = "could not find the comparison baseline in the benchmark report"
        raise ValueError(msg)
    return ReportId(normalize_tag(version.group("version")), normalize_tag(baseline.group("baseline")))


def _parse_publication_time(value: object, index: int) -> datetime:
    if not isinstance(value, str) or not value:
        msg = f"GitHub release entry {index} has invalid publishedAt metadata"
        raise TypeError(msg)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        msg = f"GitHub release entry {index} has invalid publishedAt metadata"
        raise ValueError(msg) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        msg = f"GitHub release entry {index} publishedAt must include a UTC offset"
        raise ValueError(msg)
    return parsed.astimezone(UTC)


def stable_published_releases(document: object) -> tuple[PublishedRelease, ...]:
    """Validate, filter, and newest-first sort GitHub Release metadata."""
    if not isinstance(document, list):
        msg = "GitHub release metadata must be a JSON array"
        raise TypeError(msg)
    releases: list[PublishedRelease] = []
    seen_tags: set[str] = set()
    for index, raw_release in enumerate(document):
        if not isinstance(raw_release, dict):
            msg = f"GitHub release entry {index} must be an object"
            raise TypeError(msg)
        draft = raw_release.get("isDraft")
        prerelease = raw_release.get("isPrerelease")
        if not isinstance(draft, bool) or not isinstance(prerelease, bool):
            msg = f"GitHub release entry {index} must contain boolean draft and prerelease flags"
            raise TypeError(msg)
        if draft or prerelease:
            continue
        raw_tag = raw_release.get("tagName")
        if not isinstance(raw_tag, str) or _TAG_RE.fullmatch(raw_tag) is None:
            continue
        normalized_tag = normalize_tag(raw_tag)
        if normalized_tag in seen_tags:
            msg = f"GitHub release metadata contains duplicate tag {normalized_tag}"
            raise ValueError(msg)
        seen_tags.add(normalized_tag)
        releases.append(
            PublishedRelease(
                tag=normalized_tag,
                published_at=_parse_publication_time(raw_release.get("publishedAt"), index),
            )
        )
    releases.sort(key=lambda release: release.published_at, reverse=True)
    return tuple(releases)


def _published_releases(repo_root: Path) -> tuple[PublishedRelease, ...]:
    result = run_safe_command(
        "gh",
        [
            "release",
            "list",
            "--repo",
            _REPOSITORY,
            "--limit",
            "100",
            "--json",
            "tagName,isDraft,isPrerelease,publishedAt",
        ],
        cwd=repo_root,
        timeout=_COMMAND_TIMEOUT_SECONDS,
    )
    try:
        document = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        msg = f"GitHub release metadata is not valid JSON: {error}"
        raise ValueError(msg) from error
    releases = stable_published_releases(document)
    if not releases:
        msg = "no published stable SemVer GitHub Releases were found"
        raise RuntimeError(msg)
    return releases


def current_package_tag(repo_root: Path) -> str:
    """Read the current release tag from Cargo package metadata."""
    cargo_toml = repo_root / "Cargo.toml"
    document = tomllib.loads(cargo_toml.read_text(encoding="utf-8"))
    package = document.get("package")
    if not isinstance(package, dict) or not isinstance(package.get("version"), str):
        msg = f"could not find package.version in {cargo_toml}"
        raise TypeError(msg)
    return normalize_tag(package["version"])


def _resolve_inferred_pair(
    *,
    mode: ResolutionMode,
    releases: tuple[PublishedRelease, ...],
    package_tag: str,
) -> ReportId:
    """Resolve one non-explicit comparison mode."""
    if mode == "published-latest":
        if len(releases) < 2:
            msg = "at least two published stable releases are required for a historical asset comparison"
            raise RuntimeError(msg)
        return ReportId(releases[0].tag, releases[1].tag)
    if mode == "current-vs-latest":
        if not releases:
            msg = "a published stable release is required for a local comparison"
            raise RuntimeError(msg)
        return ReportId(package_tag, releases[0].tag)
    published_tags = [release.tag for release in releases]
    if package_tag in published_tags:
        index = published_tags.index(package_tag)
        if index + 1 >= len(releases):
            msg = f"published release {package_tag} has no previous stable release"
            raise RuntimeError(msg)
        return ReportId(package_tag, releases[index + 1].tag)
    if not releases:
        msg = "a published stable release is required for release preparation"
        raise RuntimeError(msg)
    if _tag_version(package_tag) <= _tag_version(releases[0].tag):
        msg = f"unpublished package {package_tag} must be newer than the latest published stable release {releases[0].tag}"
        raise ValueError(msg)
    return ReportId(package_tag, releases[0].tag)


def resolve_release_pair(
    *,
    mode: ResolutionMode,
    releases: tuple[PublishedRelease, ...],
    package_tag: str,
    current_tag: str | None = None,
    baseline_tag: str | None = None,
) -> ReportId:
    """Resolve an explicit, local, release-prep, or historical comparison."""
    package_tag = normalize_tag(package_tag)
    if mode == "explicit":
        if current_tag is None or baseline_tag is None:
            msg = "current_tag and baseline_tag are both required for an explicit comparison"
            raise ValueError(msg)
        pair = ReportId(normalize_tag(current_tag), normalize_tag(baseline_tag))
    else:
        if current_tag is not None or baseline_tag is not None:
            msg = f"do not pass explicit tags with {mode!r} resolution"
            raise ValueError(msg)
        pair = _resolve_inferred_pair(mode=mode, releases=releases, package_tag=package_tag)
    if pair.current_tag == pair.baseline_tag and mode != "current-vs-latest":
        msg = f"current and baseline releases must differ: {pair.current_tag}"
        raise ValueError(msg)
    return pair


def _archive_index(archive_dir: Path) -> str:
    reports = sorted(path.name for path in archive_dir.glob("*.md") if path.name != "README.md")
    lines = [
        "# Archived Performance Reports",
        "",
        "Older curated release-to-release benchmark comparisons are archived here. The latest curated report is written to `docs/PERFORMANCE.md`.",
        "",
    ]
    if reports:
        lines.extend(f"- [{name.removesuffix('.md')}]({name})" for name in reports)
    else:
        lines.append("- No archived performance reports yet.")
    return "\n".join(lines) + "\n"


def promote_report(
    *,
    source_text: str,
    current_path: Path,
    archive_dir: Path,
    expected: ReportId,
) -> None:
    """Archive the previous curated report and atomically promote a new one."""
    source_id = parse_report_id(source_text)
    if source_id != expected:
        msg = (
            "benchmark report does not match the requested release pair: "
            f"found {source_id.current_tag} vs {source_id.baseline_tag}, expected {expected.current_tag} vs {expected.baseline_tag}"
        )
        raise ValueError(msg)
    if current_path.exists():
        previous_text = current_path.read_text(encoding="utf-8")
        previous_id = parse_report_id(previous_text)
        if previous_id != source_id:
            archive_path = archive_dir / previous_id.archive_name
            if not archive_path.exists():
                _write_text(archive_path, previous_text)
    _write_text(current_path, source_text)
    _write_text(archive_dir / "README.md", _archive_index(archive_dir))


def _ensure_local_tag(repo_root: Path, tag: str) -> None:
    try:
        run_git_command(["--no-pager", "rev-parse", "--verify", f"refs/tags/{tag}"], cwd=repo_root, timeout=30)
    except subprocess.CalledProcessError:
        run_git_command(["fetch", "origin", f"refs/tags/{tag}:refs/tags/{tag}", "--no-tags"], cwd=repo_root, timeout=_COMMAND_TIMEOUT_SECONDS)


@contextmanager
def temporary_worktree(repo_root: Path, parent: Path, name: str, reference: str) -> Iterator[Path]:
    """Create and reliably remove one detached temporary worktree."""
    worktree = parent / name
    added = False
    try:
        run_git_command(["worktree", "add", "--detach", str(worktree), reference], cwd=repo_root, timeout=_COMMAND_TIMEOUT_SECONDS)
        added = True
        yield worktree
    finally:
        if added:
            active_error = sys.exception()
            try:
                run_git_command(["worktree", "remove", "--force", str(worktree)], cwd=repo_root, timeout=_COMMAND_TIMEOUT_SECONDS)
            except ExecutableNotFoundError, OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired:
                if active_error is None:
                    raise


def apply_current_tree(repo_root: Path, worktree: Path) -> None:
    """Apply tracked and untracked current-tree content to an isolated worktree."""
    patch = run_git_command(["--no-pager", "diff", "--binary", "HEAD"], cwd=repo_root, timeout=_COMMAND_TIMEOUT_SECONDS).stdout
    if patch:
        run_git_command_with_input(
            ["apply", "--binary", "--whitespace=nowarn"],
            patch,
            cwd=worktree,
            timeout=_COMMAND_TIMEOUT_SECONDS,
        )
    untracked = run_git_command(
        ["ls-files", "--others", "--exclude-standard"],
        cwd=repo_root,
        timeout=_COMMAND_TIMEOUT_SECONDS,
    ).stdout.splitlines()
    for relative_name in untracked:
        relative = Path(relative_name)
        source = repo_root / relative
        destination = worktree / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _run_stepping_benchmark(checkout: Path, *, save_baseline: str | None = None) -> None:
    command = ["bench", "--locked", "--bench", "stepping"]
    if save_baseline is not None:
        command.extend(["--", "--save-baseline", save_baseline])
    run_safe_command("cargo", command, cwd=checkout, timeout=_BENCHMARK_TIMEOUT_SECONDS)


def _criterion_dependency_version(checkout: Path) -> str:
    """Read the exact Criterion version from one checkout's Cargo.lock."""
    document = tomllib.loads((checkout / "Cargo.lock").read_text(encoding="utf-8"))
    packages = document.get("package")
    if not isinstance(packages, list):
        msg = f"Cargo.lock in {checkout} does not contain a package array"
        raise TypeError(msg)
    versions = [package.get("version") for package in packages if isinstance(package, dict) and package.get("name") == "criterion"]
    if len(versions) != 1 or not isinstance(versions[0], str):
        msg = f"Cargo.lock in {checkout} must contain exactly one Criterion package"
        raise ValueError(msg)
    return versions[0]


def _release_metadata(criterion_dir: Path, expected_tag: str) -> ReleaseMetadata:
    """Read and validate durable release measurement metadata."""
    path = criterion_dir / ".mcmc-release-metadata.json"
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        msg = f"release benchmark metadata is not valid JSON in {path}: {error}"
        raise ValueError(msg) from error
    if not isinstance(document, dict):
        msg = f"release benchmark metadata in {path} must be an object"
        raise TypeError(msg)
    if document.get("schema") != 1 or isinstance(document.get("schema"), bool):
        msg = f"release benchmark metadata in {path} has an unsupported schema"
        raise ValueError(msg)

    def required_string(field: str) -> str:
        value = document.get(field)
        if not isinstance(value, str) or not value.strip():
            msg = f"release benchmark metadata field {field!r} in {path} must be a non-empty string"
            raise TypeError(msg)
        return value

    metadata = ReleaseMetadata(
        tag=normalize_tag(required_string("tag")),
        commit=required_string("commit"),
        operating_system=required_string("operating_system"),
        architecture=required_string("architecture"),
        rustc=required_string("rustc"),
        criterion_version=required_string("criterion_version"),
    )
    if metadata.tag != expected_tag:
        msg = f"release benchmark metadata tag {metadata.tag} does not match requested release {expected_tag}"
        raise ValueError(msg)
    if re.fullmatch(r"[0-9a-f]{40}", metadata.commit) is None:
        msg = f"release benchmark metadata commit for {metadata.tag} must be a full lowercase Git SHA"
        raise ValueError(msg)
    return metadata


def _local_measurement_context(current_checkout: Path, baseline_checkout: Path, *, working_tree: bool) -> tuple[str, ...]:
    """Describe enough local context to identify and interpret both samples."""
    current_commit = run_git_command(["--no-pager", "rev-parse", "HEAD"], cwd=current_checkout, timeout=30).stdout.strip()
    baseline_commit = run_git_command(["--no-pager", "rev-parse", "HEAD"], cwd=baseline_checkout, timeout=30).stdout.strip()
    current_rustc = run_safe_command("rustc", ["--version"], cwd=current_checkout, timeout=30).stdout.strip()
    baseline_rustc = run_safe_command("rustc", ["--version"], cwd=baseline_checkout, timeout=30).stdout.strip()
    source = "current `HEAD` with tracked and untracked working-tree changes applied" if working_tree else "exact current release tag"
    return (
        "Source mode: same-host isolated worktrees; " + source + ".",
        f"Host: `{platform.platform()}` on `{platform.machine()}`.",
        f"Current commit: `{current_commit}`; rustc: `{current_rustc}`; Criterion: `{_criterion_dependency_version(current_checkout)}`.",
        f"Baseline commit: `{baseline_commit}`; rustc: `{baseline_rustc}`; Criterion: `{_criterion_dependency_version(baseline_checkout)}`.",
    )


def _asset_measurement_context(current: ReleaseMetadata, baseline: ReleaseMetadata) -> tuple[str, ...]:
    """Render validated context from two durable release assets."""
    return (
        "Source mode: durable GitHub Release Criterion assets; no local benchmark runs.",
        (
            f"Current `{current.tag}`: commit `{current.commit}`; `{current.operating_system}` / `{current.architecture}`; rustc: `{current.rustc}`; "
            f"Criterion: `{current.criterion_version}`."
        ),
        (
            f"Baseline `{baseline.tag}`: commit `{baseline.commit}`; `{baseline.operating_system}` / `{baseline.architecture}`; rustc: `{baseline.rustc}`; "
            f"Criterion: `{baseline.criterion_version}`."
        ),
    )


def copy_criterion_sample(
    *,
    source_criterion: Path,
    destination_criterion: Path,
    source_sample: str,
    destination_sample: str,
) -> int:
    """Copy one named sample for every benchmark into a combined tree."""
    copied = 0
    for estimates in sorted(source_criterion.rglob("estimates.json")):
        sample_dir = estimates.parent
        if sample_dir.name != source_sample:
            continue
        benchmark_relative = sample_dir.parent.relative_to(source_criterion)
        destination = destination_criterion / benchmark_relative / destination_sample
        if destination.exists():
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(sample_dir, destination)
        copied += 1
    if copied == 0:
        msg = f"no Criterion sample {source_sample!r} found under {source_criterion}"
        raise FileNotFoundError(msg)
    return copied


def _render_comparison(
    criterion_dir: Path,
    pair: ReportId,
    *,
    current_label: str,
    revision: str,
    measurement_context: tuple[str, ...],
) -> str:
    comparison_set: ComparisonSet = collect_comparisons(criterion_dir, pair.baseline_tag)
    if not comparison_set.comparisons:
        msg = f"no comparable Criterion rows found for {pair.current_tag} vs {pair.baseline_tag}"
        raise RuntimeError(msg)
    settings = ReportSettings(
        current_label=current_label,
        baseline_label=pair.baseline_tag,
        statistic="median",
        revision=revision,
        measurement_context=measurement_context,
    )
    return render_report(comparison_set, settings)


def generate_local_report(
    repo_root: Path,
    pair: ReportId,
    *,
    current_reference: str,
    apply_working_tree: bool,
) -> str:
    """Measure two revisions in isolated worktrees and render their overlap."""
    _ensure_local_tag(repo_root, pair.baseline_tag)
    if current_reference != "HEAD":
        _ensure_local_tag(repo_root, current_reference)
    with tempfile.TemporaryDirectory(prefix="mcmc-performance-") as temporary_name:
        parent = Path(temporary_name)
        with temporary_worktree(repo_root, parent, "current", current_reference) as current_checkout:
            if apply_working_tree:
                apply_current_tree(repo_root, current_checkout)
            with temporary_worktree(repo_root, parent, "baseline", pair.baseline_tag) as baseline_checkout:
                _run_stepping_benchmark(baseline_checkout, save_baseline=pair.baseline_tag)
                _run_stepping_benchmark(current_checkout)
                current_criterion = current_checkout / "target" / "criterion"
                copy_criterion_sample(
                    source_criterion=baseline_checkout / "target" / "criterion",
                    destination_criterion=current_criterion,
                    source_sample=pair.baseline_tag,
                    destination_sample=pair.baseline_tag,
                )
                revision = run_git_command(["--no-pager", "rev-parse", "--short", "HEAD"], cwd=current_checkout, timeout=30).stdout.strip()
                label = f"{pair.current_tag} working tree" if apply_working_tree else pair.current_tag
                context = _local_measurement_context(current_checkout, baseline_checkout, working_tree=apply_working_tree)
                return _render_comparison(
                    current_criterion,
                    pair,
                    current_label=label,
                    revision=revision,
                    measurement_context=context,
                )


def safe_extract_tar(archive: Path, destination: Path) -> None:
    """Extract a regular-file Criterion archive without traversal or links."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(root):
                msg = f"release benchmark archive contains a path outside its root: {member.name}"
                raise ValueError(msg)
            if member.issym() or member.islnk() or not (member.isdir() or member.isfile()):
                msg = f"release benchmark archive contains an unsupported entry: {member.name}"
                raise ValueError(msg)
        tar.extractall(destination, filter="data")


def _download_release_asset(repo_root: Path, tag: str, destination: Path) -> Path:
    asset = _ASSET_TEMPLATE.format(tag=tag)
    destination.mkdir(parents=True, exist_ok=True)
    run_safe_command(
        "gh",
        ["release", "download", tag, "--repo", _REPOSITORY, "--pattern", asset, "--dir", str(destination)],
        cwd=repo_root,
        timeout=_COMMAND_TIMEOUT_SECONDS,
    )
    path = destination / asset
    if not path.is_file():
        msg = f"GitHub Release {tag} does not contain {asset}"
        raise FileNotFoundError(msg)
    return path


def generate_github_asset_report(repo_root: Path, pair: ReportId) -> str:
    """Render a historical comparison solely from GitHub Release assets."""
    with tempfile.TemporaryDirectory(prefix="mcmc-performance-assets-") as temporary_name:
        root = Path(temporary_name)
        current_archive = _download_release_asset(repo_root, pair.current_tag, root / "downloads-current")
        baseline_archive = _download_release_asset(repo_root, pair.baseline_tag, root / "downloads-baseline")
        current_extract = root / "current"
        baseline_extract = root / "baseline"
        safe_extract_tar(current_archive, current_extract)
        safe_extract_tar(baseline_archive, baseline_extract)
        current_criterion = current_extract / "criterion"
        baseline_criterion = baseline_extract / "criterion"
        current_metadata = _release_metadata(current_criterion, pair.current_tag)
        baseline_metadata = _release_metadata(baseline_criterion, pair.baseline_tag)
        combined = root / "combined"
        copy_criterion_sample(
            source_criterion=current_criterion,
            destination_criterion=combined,
            source_sample=pair.current_tag,
            destination_sample="new",
        )
        copy_criterion_sample(
            source_criterion=baseline_criterion,
            destination_criterion=combined,
            source_sample=pair.baseline_tag,
            destination_sample=pair.baseline_tag,
        )
        return _render_comparison(
            combined,
            pair,
            current_label=pair.current_tag,
            revision=current_metadata.commit[:7],
            measurement_context=_asset_measurement_context(current_metadata, baseline_metadata),
        )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and curate release performance reports.")
    parser.add_argument("current_tag", nargs="?")
    parser.add_argument("baseline_tag", nargs="?")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--published-latest", action="store_true")
    modes.add_argument("--infer-release", action="store_true")
    modes.add_argument("--current-vs-latest", action="store_true")
    parser.add_argument("--github-assets", action="store_true", help="Use durable GitHub Release assets without local benchmark runs.")
    parser.add_argument("--promote", action="store_true", help="Promote the report to docs/PERFORMANCE.md and archive the previous report.")
    parser.add_argument("--output", default="target/bench-reports/performance.md")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    return parser.parse_args(argv)


def _resolution_mode(args: argparse.Namespace) -> ResolutionMode:
    if args.published_latest:
        return "published-latest"
    if args.infer_release:
        return "infer-release"
    if args.current_vs_latest:
        return "current-vs-latest"
    return "explicit"


def _validate_mode_combination(mode: ResolutionMode, *, github_assets: bool, promote: bool = False) -> None:
    """Reject mode combinations that could mislabel a working-tree measurement."""
    if promote and mode == "current-vs-latest":
        msg = "working-tree reports cannot be promoted; use --infer-release or explicit release tags"
        raise ValueError(msg)
    if github_assets and mode not in {"explicit", "published-latest"}:
        msg = "--github-assets requires --published-latest or an explicit current/baseline pair"
        raise ValueError(msg)
    if not github_assets and mode == "published-latest":
        msg = "--published-latest is reserved for --github-assets comparisons"
        raise ValueError(msg)


def _format_command_failure(error: subprocess.CalledProcessError) -> str:
    """Preserve command output needed to diagnose a failed release workflow."""
    command = " ".join(str(part) for part in error.cmd) if isinstance(error.cmd, list | tuple) else str(error.cmd)
    parts = [f"command failed with exit {error.returncode}: {command}"]
    if error.stdout:
        parts.append(f"stdout:\n{str(error.stdout).strip()}")
    if error.stderr:
        parts.append(f"stderr:\n{str(error.stderr).strip()}")
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    """Generate a local or historical report and optionally promote it."""
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    repo_root = Path(args.repo_root).resolve()
    mode = _resolution_mode(args)
    try:
        _validate_mode_combination(mode, github_assets=bool(args.github_assets), promote=bool(args.promote))
        releases = () if mode == "explicit" else _published_releases(repo_root)
        pair = resolve_release_pair(
            mode=mode,
            releases=releases,
            package_tag=current_package_tag(repo_root),
            current_tag=args.current_tag,
            baseline_tag=args.baseline_tag,
        )
        if args.github_assets:
            report = generate_github_asset_report(repo_root, pair)
        else:
            published_tags = {release.tag for release in releases}
            current_tag_is_published = mode == "explicit" or (mode == "infer-release" and pair.current_tag in published_tags)
            report = generate_local_report(
                repo_root,
                pair,
                current_reference=pair.current_tag if current_tag_is_published else "HEAD",
                apply_working_tree=not current_tag_is_published,
            )
        if args.promote:
            promote_report(
                source_text=report,
                current_path=repo_root / "docs" / "PERFORMANCE.md",
                archive_dir=repo_root / "docs" / "archive" / "performance",
                expected=pair,
            )
            print(f"Promoted {pair.current_tag} vs {pair.baseline_tag} to {repo_root / 'docs' / 'PERFORMANCE.md'}")
        else:
            output = Path(args.output)
            if not output.is_absolute():
                output = repo_root / output
            _write_text(output, report)
            print(f"Wrote {output}")
    except subprocess.CalledProcessError as error:
        print(f"Performance workflow failed: {_format_command_failure(error)}", file=sys.stderr)
        return 2
    except subprocess.TimeoutExpired as error:
        print(f"Performance workflow timed out after {error.timeout} seconds: {error.cmd}", file=sys.stderr)
        return 2
    except (
        ExecutableNotFoundError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(f"Performance workflow failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
