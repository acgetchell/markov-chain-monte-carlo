#!/usr/bin/env python3
"""Generate, promote, and archive release benchmark reports.

Local comparisons run the current tree and a published release in isolated
temporary worktrees.  Historical comparisons consume durable Criterion
archives attached to GitHub Releases and never run Cargo.
"""

import argparse
import csv
import hashlib
import io
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
from typing import TYPE_CHECKING, Literal, cast

from bench_compare import Comparison, ComparisonSet, Estimate, ReportSettings, _write_text, collect_comparisons, render_report
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
_CSV_SCHEMA = "criterion-comparison/v1"
_PROVENANCE_SCHEMA = "mcmc-performance-provenance/v1"
_BENCHMARK_SUITE = "stepping"
_BENCHMARK_SCOPE = "release-signal"
_CSV_COLUMNS = (
    "schema_version",
    "suite",
    "scope",
    "benchmark",
    "coverage",
    "baseline_point_ns",
    "baseline_lower_ns",
    "baseline_upper_ns",
    "current_point_ns",
    "current_lower_ns",
    "current_upper_ns",
)

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


@dataclass(frozen=True, slots=True)
class SampleProvenance:
    """Structured identity and inputs for one measured Criterion sample."""

    tag: str
    commit: str
    operating_system: str
    architecture: str
    rustc: str
    criterion_version: str
    source_digest_sha256: str | None
    cargo_lock_sha256: str | None
    benchmark_harness_sha256: str | None
    command: tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class MeasurementProvenance:
    """Machine-readable provenance for a comparison's two samples."""

    mode: Literal["local-isolated-worktrees", "github-release-assets"]
    working_tree_applied: bool
    current: SampleProvenance
    baseline: SampleProvenance


@dataclass(frozen=True, slots=True)
class ComparisonArtifact:
    """Validated measurements and rendering metadata for one release pair."""

    pair: ReportId
    comparison_set: ComparisonSet
    settings: ReportSettings
    measurement: MeasurementProvenance


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


def _archive_index(archive_dir: Path, *, additional_reports: tuple[str, ...] = ()) -> str:
    reports = {path.name for path in archive_dir.glob("*.md") if path.name != "README.md"}
    reports.update(additional_reports)
    lines = [
        "# Archived Performance Reports",
        "",
        "Older curated release-to-release benchmark comparisons are archived here. The latest curated report is written to `docs/PERFORMANCE.md`.",
        "",
    ]
    if reports:
        lines.extend(f"- [{name.removesuffix('.md')}]({name})" for name in sorted(reports))
    else:
        lines.append("- No archived performance reports yet.")
    return "\n".join(lines) + "\n"


def _publish_texts(outputs: tuple[tuple[Path, str], ...]) -> None:
    """Publish a set of text files, restoring every prior target on failure."""
    paths = tuple(path for path, _text in outputs)
    if len(paths) != len(set(paths)):
        msg = "transactional publication requires unique output paths"
        raise ValueError(msg)

    previous = {path: path.read_text(encoding="utf-8") if path.exists() else None for path in paths}
    completed: list[Path] = []
    try:
        for path, text in outputs:
            _write_text(path, text)
            completed.append(path)
    except (OSError, RuntimeError) as error:
        rollback_failures: list[str] = []
        for path in reversed(completed):
            try:
                previous_text = previous[path]
                if previous_text is None:
                    path.unlink(missing_ok=True)
                else:
                    _write_text(path, previous_text)
            except (OSError, RuntimeError) as rollback_error:
                rollback_failures.append(f"{path}: {rollback_error}")
        if rollback_failures:
            details = "; ".join(rollback_failures)
            msg = f"publication failed and rollback was incomplete: {details}"
            raise RuntimeError(msg) from error
        raise


def _sample_mapping(sample: tuple[tuple[str, Estimate], ...], label: str) -> dict[str, Estimate]:
    """Validate one deterministically ordered sample and return a lookup."""
    names = [name for name, _estimate in sample]
    if names != sorted(names):
        msg = f"{label} sample rows must be sorted by benchmark name"
        raise ValueError(msg)
    if len(names) != len(set(names)):
        msg = f"{label} sample contains duplicate benchmark names"
        raise ValueError(msg)
    for name in names:
        if not name or name != name.strip() or any(character in name for character in "\0\n\r"):
            msg = f"{label} sample contains an invalid benchmark name: {name!r}"
            raise ValueError(msg)
    return dict(sample)


def _comparison_set_from_samples(current: dict[str, Estimate], baseline: dict[str, Estimate]) -> ComparisonSet:
    """Construct the canonical comparison and coverage view of two samples."""
    shared = sorted(current.keys() & baseline.keys())
    return ComparisonSet(
        comparisons=tuple(Comparison(name, baseline[name], current[name]) for name in shared),
        missing_baseline=tuple(sorted(current.keys() - baseline.keys())),
        missing_current=tuple(sorted(baseline.keys() - current.keys())),
        current_sample=tuple(sorted(current.items())),
        baseline_sample=tuple(sorted(baseline.items())),
    )


def _validated_comparison_set(comparison_set: ComparisonSet) -> ComparisonSet:
    """Reject internally inconsistent derived coverage fields."""
    current = _sample_mapping(comparison_set.current_sample, "current")
    baseline = _sample_mapping(comparison_set.baseline_sample, "baseline")
    canonical = _comparison_set_from_samples(current, baseline)
    if comparison_set != canonical:
        msg = "comparison rows or coverage notes do not match the stored current and baseline samples"
        raise ValueError(msg)
    return canonical


def _format_float(value: float | None) -> str:
    return "" if value is None else repr(value)


def serialize_comparison_csv(comparison_set: ComparisonSet) -> str:
    """Serialize both samples as deterministic, analysis-friendly CSV."""
    validated = _validated_comparison_set(comparison_set)
    current = dict(validated.current_sample)
    baseline = dict(validated.baseline_sample)
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(_CSV_COLUMNS)
    for benchmark in sorted(current.keys() | baseline.keys()):
        current_estimate = current.get(benchmark)
        baseline_estimate = baseline.get(benchmark)
        if current_estimate is not None and baseline_estimate is not None:
            coverage = "comparable"
        elif current_estimate is not None:
            coverage = "current_only"
        else:
            coverage = "baseline_only"
        writer.writerow(
            (
                _CSV_SCHEMA,
                _BENCHMARK_SUITE,
                _BENCHMARK_SCOPE,
                benchmark,
                coverage,
                _format_float(None if baseline_estimate is None else baseline_estimate.point_ns),
                _format_float(None if baseline_estimate is None else baseline_estimate.lower_ns),
                _format_float(None if baseline_estimate is None else baseline_estimate.upper_ns),
                _format_float(None if current_estimate is None else current_estimate.point_ns),
                _format_float(None if current_estimate is None else current_estimate.lower_ns),
                _format_float(None if current_estimate is None else current_estimate.upper_ns),
            )
        )
    return output.getvalue()


def _parse_estimate_fields(row: dict[str, str], prefix: str, *, required: bool, row_number: int) -> Estimate | None:
    fields = tuple(row[f"{prefix}_{part}_ns"] for part in ("point", "lower", "upper"))
    if not required:
        if any(fields):
            msg = f"CSV row {row_number} must leave absent {prefix} estimate fields blank"
            raise ValueError(msg)
        return None
    if not fields[0]:
        msg = f"CSV row {row_number} is missing {prefix}_point_ns"
        raise ValueError(msg)
    if bool(fields[1]) != bool(fields[2]):
        msg = f"CSV row {row_number} must provide both {prefix} confidence bounds or neither"
        raise ValueError(msg)
    try:
        point = float(fields[0])
        lower = float(fields[1]) if fields[1] else None
        upper = float(fields[2]) if fields[2] else None
    except ValueError as error:
        msg = f"CSV row {row_number} contains a non-numeric {prefix} estimate"
        raise ValueError(msg) from error
    return Estimate(point, lower, upper)


def parse_comparison_csv(text: str) -> ComparisonSet:
    """Parse and strictly validate a versioned comparison CSV document."""
    try:
        rows = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except csv.Error as error:
        msg = f"comparison CSV is malformed: {error}"
        raise ValueError(msg) from error
    if not rows:
        msg = "comparison CSV is empty"
        raise ValueError(msg)
    if tuple(rows[0]) != _CSV_COLUMNS:
        msg = "comparison CSV has an unsupported or reordered header"
        raise ValueError(msg)
    if len(rows) == 1:
        msg = "comparison CSV contains no benchmark rows"
        raise ValueError(msg)
    current: dict[str, Estimate] = {}
    baseline: dict[str, Estimate] = {}
    previous_name: str | None = None
    for row_number, values in enumerate(rows[1:], start=2):
        if len(values) != len(_CSV_COLUMNS):
            msg = f"CSV row {row_number} has {len(values)} fields; expected {len(_CSV_COLUMNS)}"
            raise ValueError(msg)
        row = dict(zip(_CSV_COLUMNS, values, strict=True))
        if row["schema_version"] != _CSV_SCHEMA or row["suite"] != _BENCHMARK_SUITE or row["scope"] != _BENCHMARK_SCOPE:
            msg = f"CSV row {row_number} has unsupported schema, suite, or scope metadata"
            raise ValueError(msg)
        benchmark = row["benchmark"]
        if not benchmark or benchmark != benchmark.strip() or any(character in benchmark for character in "\0\n\r"):
            msg = f"CSV row {row_number} has an invalid benchmark name"
            raise ValueError(msg)
        if previous_name is not None and benchmark <= previous_name:
            msg = f"CSV benchmark rows must be unique and sorted: {benchmark!r}"
            raise ValueError(msg)
        previous_name = benchmark
        coverage = row["coverage"]
        if coverage not in {"comparable", "current_only", "baseline_only"}:
            msg = f"CSV row {row_number} has unsupported coverage {coverage!r}"
            raise ValueError(msg)
        has_current = coverage != "baseline_only"
        has_baseline = coverage != "current_only"
        current_estimate = _parse_estimate_fields(row, "current", required=has_current, row_number=row_number)
        baseline_estimate = _parse_estimate_fields(row, "baseline", required=has_baseline, row_number=row_number)
        if current_estimate is not None:
            current[benchmark] = current_estimate
        if baseline_estimate is not None:
            baseline[benchmark] = baseline_estimate
    return _comparison_set_from_samples(current, baseline)


def _sample_provenance_document(sample: SampleProvenance) -> dict[str, object]:
    return {
        "architecture": sample.architecture,
        "benchmark_harness_sha256": sample.benchmark_harness_sha256,
        "cargo_lock_sha256": sample.cargo_lock_sha256,
        "command": None if sample.command is None else list(sample.command),
        "commit": sample.commit,
        "criterion_version": sample.criterion_version,
        "operating_system": sample.operating_system,
        "rustc": sample.rustc,
        "source_digest_sha256": sample.source_digest_sha256,
        "tag": sample.tag,
    }


def serialize_provenance(artifact: ComparisonArtifact) -> str:
    """Serialize report settings and measurement provenance deterministically."""
    csv_sha256 = hashlib.sha256(serialize_comparison_csv(artifact.comparison_set).encode("utf-8")).hexdigest()
    document = {
        "csv_sha256": csv_sha256,
        "csv_schema": _CSV_SCHEMA,
        "measurement": {
            "baseline": _sample_provenance_document(artifact.measurement.baseline),
            "current": _sample_provenance_document(artifact.measurement.current),
            "mode": artifact.measurement.mode,
            "working_tree_applied": artifact.measurement.working_tree_applied,
        },
        "release": {
            "baseline_tag": artifact.pair.baseline_tag,
            "current_tag": artifact.pair.current_tag,
        },
        "report": {
            "baseline_label": artifact.settings.baseline_label,
            "current_label": artifact.settings.current_label,
            "measurement_context": list(artifact.settings.measurement_context),
            "revision": artifact.settings.revision,
            "statistic": artifact.settings.statistic,
        },
        "schema": _PROVENANCE_SCHEMA,
    }
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def _object_field(document: dict[str, object], field: str, context: str) -> dict[str, object]:
    value = document.get(field)
    if not isinstance(value, dict):
        msg = f"{context} field {field!r} must be an object"
        raise TypeError(msg)
    return {str(key): item for key, item in value.items()}


def _required_string(document: dict[str, object], field: str, context: str) -> str:
    value = document.get(field)
    if not isinstance(value, str) or not value.strip():
        msg = f"{context} field {field!r} must be a non-empty string"
        raise TypeError(msg)
    return value


def _optional_sha256(document: dict[str, object], field: str, context: str) -> str | None:
    value = document.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        msg = f"{context} field {field!r} must be null or a lowercase SHA-256 digest"
        raise ValueError(msg)
    return value


def _parse_sample_provenance(document: dict[str, object], *, expected_tag: str, context: str) -> SampleProvenance:
    expected_fields = {
        "architecture",
        "benchmark_harness_sha256",
        "cargo_lock_sha256",
        "command",
        "commit",
        "criterion_version",
        "operating_system",
        "rustc",
        "source_digest_sha256",
        "tag",
    }
    if set(document) != expected_fields:
        msg = f"{context} fields do not match the provenance schema"
        raise ValueError(msg)
    tag = normalize_tag(_required_string(document, "tag", context))
    if tag != expected_tag:
        msg = f"{context} tag {tag} does not match release pair tag {expected_tag}"
        raise ValueError(msg)
    commit = _required_string(document, "commit", context)
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        msg = f"{context} commit must be a full lowercase Git SHA"
        raise ValueError(msg)
    raw_command = document.get("command")
    command: tuple[str, ...] | None
    if raw_command is None:
        command = None
    elif isinstance(raw_command, list) and raw_command and all(isinstance(part, str) and part for part in raw_command):
        command = tuple(cast("list[str]", raw_command))
    else:
        msg = f"{context} command must be null or a non-empty string array"
        raise TypeError(msg)
    return SampleProvenance(
        tag=tag,
        commit=commit,
        operating_system=_required_string(document, "operating_system", context),
        architecture=_required_string(document, "architecture", context),
        rustc=_required_string(document, "rustc", context),
        criterion_version=_required_string(document, "criterion_version", context),
        source_digest_sha256=_optional_sha256(document, "source_digest_sha256", context),
        cargo_lock_sha256=_optional_sha256(document, "cargo_lock_sha256", context),
        benchmark_harness_sha256=_optional_sha256(document, "benchmark_harness_sha256", context),
        command=command,
    )


def _parse_release_pair(document: dict[str, object]) -> ReportId:
    release = _object_field(document, "release", "comparison provenance")
    if set(release) != {"baseline_tag", "current_tag"}:
        msg = "release provenance fields do not match the supported schema"
        raise ValueError(msg)
    return ReportId(
        normalize_tag(_required_string(release, "current_tag", "release provenance")),
        normalize_tag(_required_string(release, "baseline_tag", "release provenance")),
    )


def _parse_report_settings(document: dict[str, object], pair: ReportId) -> ReportSettings:
    report = _object_field(document, "report", "comparison provenance")
    if set(report) != {"baseline_label", "current_label", "measurement_context", "revision", "statistic"}:
        msg = "report provenance fields do not match the supported schema"
        raise ValueError(msg)
    statistic = _required_string(report, "statistic", "report provenance")
    if statistic not in {"mean", "median"}:
        msg = f"report provenance has unsupported statistic {statistic!r}"
        raise ValueError(msg)
    raw_context = report.get("measurement_context")
    if not isinstance(raw_context, list) or not all(isinstance(item, str) and item for item in raw_context):
        msg = "report provenance measurement_context must be a string array"
        raise TypeError(msg)
    settings = ReportSettings(
        current_label=_required_string(report, "current_label", "report provenance"),
        baseline_label=_required_string(report, "baseline_label", "report provenance"),
        statistic=statistic,
        revision=_required_string(report, "revision", "report provenance"),
        measurement_context=tuple(cast("list[str]", raw_context)),
    )
    if settings.baseline_label != pair.baseline_tag:
        msg = "report baseline label does not match the release pair"
        raise ValueError(msg)
    return settings


def _parse_measurement_provenance(document: dict[str, object], pair: ReportId) -> MeasurementProvenance:
    measurement = _object_field(document, "measurement", "comparison provenance")
    if set(measurement) != {"baseline", "current", "mode", "working_tree_applied"}:
        msg = "measurement provenance fields do not match the supported schema"
        raise ValueError(msg)
    mode = _required_string(measurement, "mode", "measurement provenance")
    if mode not in {"local-isolated-worktrees", "github-release-assets"}:
        msg = f"measurement provenance has unsupported mode {mode!r}"
        raise ValueError(msg)
    working_tree_applied = measurement.get("working_tree_applied")
    if not isinstance(working_tree_applied, bool):
        msg = "measurement provenance working_tree_applied must be boolean"
        raise TypeError(msg)
    current = _parse_sample_provenance(
        _object_field(measurement, "current", "measurement provenance"),
        expected_tag=pair.current_tag,
        context="current sample provenance",
    )
    baseline = _parse_sample_provenance(
        _object_field(measurement, "baseline", "measurement provenance"),
        expected_tag=pair.baseline_tag,
        context="baseline sample provenance",
    )
    local_values = (
        current.source_digest_sha256,
        current.cargo_lock_sha256,
        current.benchmark_harness_sha256,
        current.command,
        baseline.source_digest_sha256,
        baseline.cargo_lock_sha256,
        baseline.benchmark_harness_sha256,
        baseline.command,
    )
    if mode == "local-isolated-worktrees" and any(value is None for value in local_values):
        msg = "local measurement provenance requires commands and source-input hashes for both samples"
        raise ValueError(msg)
    if mode == "github-release-assets" and (working_tree_applied or any(value is not None for value in local_values)):
        msg = "GitHub asset provenance cannot claim local commands, source hashes, or working-tree changes"
        raise ValueError(msg)
    return MeasurementProvenance(mode, working_tree_applied, current, baseline)


def parse_provenance(text: str) -> tuple[ReportId, ReportSettings, MeasurementProvenance, str]:
    """Parse the strict JSON sidecar paired with a comparison CSV."""
    try:
        raw_document = json.loads(text)
    except json.JSONDecodeError as error:
        msg = f"comparison provenance is not valid JSON: {error}"
        raise ValueError(msg) from error
    if not isinstance(raw_document, dict):
        msg = "comparison provenance must be a JSON object"
        raise TypeError(msg)
    document = {str(key): value for key, value in raw_document.items()}
    if set(document) != {"csv_sha256", "csv_schema", "measurement", "release", "report", "schema"}:
        msg = "comparison provenance fields do not match the supported schema"
        raise ValueError(msg)
    if document.get("schema") != _PROVENANCE_SCHEMA or document.get("csv_schema") != _CSV_SCHEMA:
        msg = "comparison provenance has an unsupported schema"
        raise ValueError(msg)
    pair = _parse_release_pair(document)
    settings = _parse_report_settings(document, pair)
    measurement = _parse_measurement_provenance(document, pair)
    csv_sha256 = _required_string(document, "csv_sha256", "comparison provenance")
    if re.fullmatch(r"[0-9a-f]{64}", csv_sha256) is None:
        msg = "comparison provenance csv_sha256 must be a lowercase SHA-256 digest"
        raise ValueError(msg)
    expected_label = f"{pair.current_tag} working tree" if measurement.working_tree_applied else pair.current_tag
    if settings.current_label != expected_label:
        msg = "report current label does not match the measurement source mode"
        raise ValueError(msg)
    if settings.revision != measurement.current.commit[:7]:
        msg = "report revision does not match the current sample commit"
        raise ValueError(msg)
    return pair, settings, measurement, csv_sha256


def provenance_path(csv_path: Path) -> Path:
    """Return the deterministic JSON sidecar path for one comparison CSV."""
    return csv_path.with_suffix(".provenance.json")


def _artifact_from_text(csv_text: str, provenance_text: str) -> ComparisonArtifact:
    comparison_set = parse_comparison_csv(csv_text)
    pair, settings, measurement, expected_csv_sha256 = parse_provenance(provenance_text)
    actual_csv_sha256 = hashlib.sha256(csv_text.encode("utf-8")).hexdigest()
    if actual_csv_sha256 != expected_csv_sha256:
        msg = "comparison CSV does not match its provenance SHA-256 digest"
        raise ValueError(msg)
    if not comparison_set.comparisons:
        msg = f"comparison artifact has no comparable rows for {pair.current_tag} vs {pair.baseline_tag}"
        raise ValueError(msg)
    return ComparisonArtifact(pair, comparison_set, settings, measurement)


def load_comparison_artifact(csv_path: Path) -> ComparisonArtifact:
    """Load and validate a CSV comparison and its adjacent provenance JSON."""
    sidecar = provenance_path(csv_path)
    return _artifact_from_text(csv_path.read_text(encoding="utf-8"), sidecar.read_text(encoding="utf-8"))


def save_comparison_artifact(artifact: ComparisonArtifact, csv_path: Path) -> ComparisonArtifact:
    """Transactionally save, reload, and validate a comparison artifact pair."""
    csv_text = serialize_comparison_csv(artifact.comparison_set)
    provenance_text = serialize_provenance(artifact)
    parsed = _artifact_from_text(csv_text, provenance_text)
    if parsed != artifact:
        msg = "comparison artifact changed during serialization"
        raise ValueError(msg)
    sidecar = provenance_path(csv_path)
    _publish_texts(((csv_path, csv_text), (sidecar, provenance_text)))
    return load_comparison_artifact(csv_path)


def promote_report(
    *,
    source_text: str,
    current_path: Path,
    archive_dir: Path,
    expected: ReportId,
) -> None:
    """Archive the previous curated report and promote all outputs with rollback."""
    source_id = parse_report_id(source_text)
    if source_id != expected:
        msg = (
            "benchmark report does not match the requested release pair: "
            f"found {source_id.current_tag} vs {source_id.baseline_tag}, expected {expected.current_tag} vs {expected.baseline_tag}"
        )
        raise ValueError(msg)
    outputs: list[tuple[Path, str]] = []
    additional_reports: list[str] = []
    if current_path.exists():
        previous_text = current_path.read_text(encoding="utf-8")
        previous_id = parse_report_id(previous_text)
        if previous_id != source_id:
            archive_path = archive_dir / previous_id.archive_name
            if not archive_path.exists():
                outputs.append((archive_path, previous_text))
                additional_reports.append(archive_path.name)
    outputs.append((current_path, source_text))
    outputs.append(
        (
            archive_dir / "README.md",
            _archive_index(archive_dir, additional_reports=tuple(additional_reports)),
        )
    )
    _publish_texts(tuple(outputs))


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


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_digest(checkout: Path) -> str:
    """Hash the Rust/package inputs that define the measured implementation."""
    rust_sources = sorted((checkout / "src").rglob("*.rs"))
    required_paths = [
        checkout / "Cargo.toml",
        checkout / "Cargo.lock",
        checkout / "rust-toolchain.toml",
        checkout / "benches" / "stepping.rs",
    ]
    paths = [*required_paths, *rust_sources]
    missing = [path for path in paths if not path.is_file()]
    if not rust_sources:
        missing.append(checkout / "src" / "*.rs")
    if missing:
        msg = f"cannot hash benchmark inputs; missing {', '.join(str(path) for path in missing)}"
        raise FileNotFoundError(msg)
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            continue
        relative = path.relative_to(checkout).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _local_sample_provenance(checkout: Path, tag: str, command: tuple[str, ...]) -> SampleProvenance:
    return SampleProvenance(
        tag=tag,
        commit=run_git_command(["--no-pager", "rev-parse", "HEAD"], cwd=checkout, timeout=30).stdout.strip(),
        operating_system=platform.platform(),
        architecture=platform.machine(),
        rustc=run_safe_command("rustc", ["--version"], cwd=checkout, timeout=30).stdout.strip(),
        criterion_version=_criterion_dependency_version(checkout),
        source_digest_sha256=_source_digest(checkout),
        cargo_lock_sha256=_file_sha256(checkout / "Cargo.lock"),
        benchmark_harness_sha256=_file_sha256(checkout / "benches" / "stepping.rs"),
        command=command,
    )


def _local_measurement_provenance(
    current_checkout: Path,
    baseline_checkout: Path,
    pair: ReportId,
    *,
    working_tree: bool,
) -> MeasurementProvenance:
    baseline_command = ("cargo", "bench", "--locked", "--bench", "stepping", "--", "--save-baseline", pair.baseline_tag)
    current_command = ("cargo", "bench", "--locked", "--bench", "stepping")
    return MeasurementProvenance(
        mode="local-isolated-worktrees",
        working_tree_applied=working_tree,
        current=_local_sample_provenance(current_checkout, pair.current_tag, current_command),
        baseline=_local_sample_provenance(baseline_checkout, pair.baseline_tag, baseline_command),
    )


def _local_measurement_context(measurement: MeasurementProvenance) -> tuple[str, ...]:
    """Render concise human context from structured local provenance."""
    source = "current `HEAD` with tracked and untracked working-tree changes applied" if measurement.working_tree_applied else "exact current release tag"
    current_harness = measurement.current.benchmark_harness_sha256 or "unavailable"
    baseline_harness = measurement.baseline.benchmark_harness_sha256 or "unavailable"
    context = [
        "Source mode: same-host isolated worktrees; " + source + ".",
        f"Host: `{measurement.current.operating_system}` on `{measurement.current.architecture}`.",
        (f"Current commit: `{measurement.current.commit}`; rustc: `{measurement.current.rustc}`; Criterion: `{measurement.current.criterion_version}`."),
        (f"Baseline commit: `{measurement.baseline.commit}`; rustc: `{measurement.baseline.rustc}`; Criterion: `{measurement.baseline.criterion_version}`."),
        f"Benchmark harness SHA-256 prefixes: current `{current_harness[:12]}`; baseline `{baseline_harness[:12]}`.",
    ]
    if current_harness != baseline_harness:
        context.append("Benchmark harness hashes differ; verify that every shared name retains the same workload contract.")
    return tuple(context)


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


def _asset_measurement_provenance(current: ReleaseMetadata, baseline: ReleaseMetadata) -> MeasurementProvenance:
    def sample(metadata: ReleaseMetadata) -> SampleProvenance:
        return SampleProvenance(
            tag=metadata.tag,
            commit=metadata.commit,
            operating_system=metadata.operating_system,
            architecture=metadata.architecture,
            rustc=metadata.rustc,
            criterion_version=metadata.criterion_version,
            source_digest_sha256=None,
            cargo_lock_sha256=None,
            benchmark_harness_sha256=None,
            command=None,
        )

    return MeasurementProvenance(
        mode="github-release-assets",
        working_tree_applied=False,
        current=sample(current),
        baseline=sample(baseline),
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


def _comparison_artifact(
    criterion_dir: Path,
    pair: ReportId,
    *,
    settings: ReportSettings,
    measurement: MeasurementProvenance,
) -> ComparisonArtifact:
    comparison_set: ComparisonSet = collect_comparisons(criterion_dir, pair.baseline_tag)
    if not comparison_set.comparisons:
        msg = f"no comparable Criterion rows found for {pair.current_tag} vs {pair.baseline_tag}"
        raise RuntimeError(msg)
    return ComparisonArtifact(pair, comparison_set, settings, measurement)


def generate_local_artifact(
    repo_root: Path,
    pair: ReportId,
    *,
    current_reference: str,
    apply_working_tree: bool,
) -> ComparisonArtifact:
    """Measure two revisions in isolated worktrees and collect an artifact."""
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
                measurement = _local_measurement_provenance(
                    current_checkout,
                    baseline_checkout,
                    pair,
                    working_tree=apply_working_tree,
                )
                label = f"{pair.current_tag} working tree" if apply_working_tree else pair.current_tag
                return _comparison_artifact(
                    current_criterion,
                    pair,
                    settings=ReportSettings(
                        current_label=label,
                        baseline_label=pair.baseline_tag,
                        statistic="median",
                        revision=measurement.current.commit[:7],
                        measurement_context=_local_measurement_context(measurement),
                    ),
                    measurement=measurement,
                )


def generate_local_report(
    repo_root: Path,
    pair: ReportId,
    *,
    current_reference: str,
    apply_working_tree: bool,
) -> str:
    """Compatibility wrapper that renders a newly generated local artifact."""
    artifact = generate_local_artifact(
        repo_root,
        pair,
        current_reference=current_reference,
        apply_working_tree=apply_working_tree,
    )
    return render_report(artifact.comparison_set, artifact.settings)


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


def generate_github_asset_artifact(repo_root: Path, pair: ReportId) -> ComparisonArtifact:
    """Collect a historical comparison solely from GitHub Release assets."""
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
        measurement = _asset_measurement_provenance(current_metadata, baseline_metadata)
        return _comparison_artifact(
            combined,
            pair,
            settings=ReportSettings(
                current_label=pair.current_tag,
                baseline_label=pair.baseline_tag,
                statistic="median",
                revision=current_metadata.commit[:7],
                measurement_context=_asset_measurement_context(current_metadata, baseline_metadata),
            ),
            measurement=measurement,
        )


def generate_github_asset_report(repo_root: Path, pair: ReportId) -> str:
    """Compatibility wrapper that renders a newly generated asset artifact."""
    artifact = generate_github_asset_artifact(repo_root, pair)
    return render_report(artifact.comparison_set, artifact.settings)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and curate release performance reports.")
    parser.add_argument("current_tag", nargs="?")
    parser.add_argument("baseline_tag", nargs="?")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--published-latest", action="store_true")
    modes.add_argument("--infer-release", action="store_true")
    modes.add_argument("--current-vs-latest", action="store_true")
    parser.add_argument("--github-assets", action="store_true", help="Use durable GitHub Release assets without local benchmark runs.")
    parser.add_argument("--measurements-output", help="Write the comparison CSV and adjacent provenance JSON to this path.")
    parser.add_argument("--rerender", metavar="CSV", help="Render an existing comparison CSV and adjacent provenance JSON without measuring.")
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


def _validate_rerender_combination(args: argparse.Namespace) -> None:
    """Keep rerendering independent of GitHub, Git, Cargo, and pair inference."""
    if args.rerender is None:
        return
    if (
        args.current_tag is not None
        or args.baseline_tag is not None
        or args.published_latest
        or args.infer_release
        or args.current_vs_latest
        or args.github_assets
        or args.measurements_output is not None
    ):
        msg = "--rerender cannot be combined with tags, measurement modes, --github-assets, or --measurements-output"
        raise ValueError(msg)


def main(argv: list[str] | None = None) -> int:
    """Generate or rerender a report and optionally promote it."""
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    repo_root = Path(args.repo_root).resolve()
    try:
        _validate_rerender_combination(args)
        if args.rerender is not None:
            measurements_path = Path(args.rerender)
            if not measurements_path.is_absolute():
                measurements_path = repo_root / measurements_path
            artifact = load_comparison_artifact(measurements_path)
            pair = artifact.pair
        else:
            mode = _resolution_mode(args)
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
                generated = generate_github_asset_artifact(repo_root, pair)
            else:
                published_tags = {release.tag for release in releases}
                current_tag_is_published = mode == "explicit" or (mode == "infer-release" and pair.current_tag in published_tags)
                generated = generate_local_artifact(
                    repo_root,
                    pair,
                    current_reference=pair.current_tag if current_tag_is_published else "HEAD",
                    apply_working_tree=not current_tag_is_published,
                )
            raw_measurements_path = args.measurements_output or "target/bench-reports/performance.csv"
            measurements_path = Path(raw_measurements_path)
            if not measurements_path.is_absolute():
                measurements_path = repo_root / measurements_path
            artifact = save_comparison_artifact(generated, measurements_path)
            print(f"Wrote {measurements_path} and {provenance_path(measurements_path)}")
        report = render_report(artifact.comparison_set, artifact.settings)
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
