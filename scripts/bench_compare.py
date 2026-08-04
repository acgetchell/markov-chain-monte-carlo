#!/usr/bin/env python3
"""Render Markdown comparisons from Criterion benchmark samples.

Criterion stores every benchmark below ``target/criterion`` with one directory
per sample.  This utility compares the ordinary ``new`` sample with a named
saved baseline and writes a compact, reviewable report.
"""

import argparse
import json
import math
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from subprocess_utils import ExecutableNotFoundError, run_git_command

type Statistic = Literal["mean", "median"]


@dataclass(frozen=True, slots=True)
class Estimate:
    """One validated Criterion estimate in nanoseconds."""

    point_ns: float
    lower_ns: float | None
    upper_ns: float | None

    def __post_init__(self) -> None:
        """Reject incomplete, non-positive, or non-finite timings."""
        values = (self.point_ns, self.lower_ns, self.upper_ns)
        if any(value is not None and (not math.isfinite(value) or value <= 0) for value in values):
            msg = f"Criterion timings must be finite and positive: {values!r}"
            raise ValueError(msg)
        if (self.lower_ns is None) != (self.upper_ns is None):
            msg = "Criterion confidence intervals require both bounds"
            raise ValueError(msg)
        if self.lower_ns is not None and self.upper_ns is not None and self.lower_ns > self.upper_ns:
            msg = "Criterion confidence interval lower bound exceeds upper bound"
            raise ValueError(msg)
        if self.lower_ns is not None and self.upper_ns is not None and not self.lower_ns <= self.point_ns <= self.upper_ns:
            msg = "Criterion point estimate must lie within its confidence interval"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Comparison:
    """A current estimate paired with its saved baseline."""

    benchmark: str
    baseline: Estimate
    current: Estimate

    @property
    def percent_change(self) -> float:
        """Return signed current-vs-baseline change; negative is faster."""
        return ((self.current.point_ns - self.baseline.point_ns) / self.baseline.point_ns) * 100.0

    @property
    def speedup(self) -> float:
        """Return baseline/current; values above one are faster."""
        return self.baseline.point_ns / self.current.point_ns


@dataclass(frozen=True, slots=True)
class ComparisonSet:
    """Comparable rows plus samples missing from either revision."""

    comparisons: tuple[Comparison, ...]
    missing_baseline: tuple[str, ...]
    missing_current: tuple[str, ...]
    current_sample: tuple[tuple[str, Estimate], ...]


@dataclass(frozen=True, slots=True)
class ReportSettings:
    """Labels, statistic, and provenance rendered into one report."""

    current_label: str
    baseline_label: str
    statistic: Statistic
    revision: str
    measurement_context: tuple[str, ...] = ()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _numeric_field(data: dict[str, object], field: str, path: Path) -> float:
    value = data.get(field)
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"{field!r} in {path} must be numeric"
        raise TypeError(msg)
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        msg = f"{field!r} in {path} must be finite and positive"
        raise ValueError(msg)
    return number


def read_estimate(path: Path, statistic: Statistic = "median") -> Estimate:
    """Read one Criterion estimate file with contextual validation errors."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        msg = f"malformed Criterion JSON in {path}: {error}"
        raise ValueError(msg) from error
    if not isinstance(document, dict):
        msg = f"Criterion estimate in {path} must be a JSON object"
        raise TypeError(msg)
    raw_statistic = document.get(statistic)
    if not isinstance(raw_statistic, dict):
        msg = f"Criterion statistic {statistic!r} is missing from {path}"
        raise KeyError(msg)
    statistic_data = {str(key): value for key, value in raw_statistic.items()}
    point = _numeric_field(statistic_data, "point_estimate", path)
    raw_interval = statistic_data.get("confidence_interval")
    if raw_interval is None:
        return Estimate(point, None, None)
    if not isinstance(raw_interval, dict):
        msg = f"Criterion confidence interval in {path} must be an object"
        raise TypeError(msg)
    interval = {str(key): value for key, value in raw_interval.items()}
    return Estimate(
        point,
        _numeric_field(interval, "lower_bound", path),
        _numeric_field(interval, "upper_bound", path),
    )


def collect_sample(criterion_dir: Path, sample: str, statistic: Statistic = "median") -> dict[str, Estimate]:
    """Collect all estimates whose immediate sample directory matches *sample*."""
    results: dict[str, Estimate] = {}
    if not criterion_dir.is_dir():
        return results
    for path in sorted(criterion_dir.rglob("estimates.json")):
        if path.parent.name != sample:
            continue
        relative_benchmark = path.parent.parent.relative_to(criterion_dir)
        benchmark = "/".join(relative_benchmark.parts)
        if benchmark in results:
            msg = f"duplicate Criterion result for {benchmark!r} in sample {sample!r}"
            raise ValueError(msg)
        results[benchmark] = read_estimate(path, statistic)
    return results


def collect_comparisons(
    criterion_dir: Path,
    baseline_name: str,
    statistic: Statistic = "median",
) -> ComparisonSet:
    """Pair the ``new`` sample with *baseline_name* across all benchmarks."""
    current = collect_sample(criterion_dir, "new", statistic)
    baseline = collect_sample(criterion_dir, baseline_name, statistic)
    shared = sorted(current.keys() & baseline.keys())
    return ComparisonSet(
        comparisons=tuple(Comparison(name, baseline[name], current[name]) for name in shared),
        missing_baseline=tuple(sorted(current.keys() - baseline.keys())),
        missing_current=tuple(sorted(baseline.keys() - current.keys())),
        current_sample=tuple(sorted(current.items())),
    )


def _format_duration(nanoseconds: float) -> str:
    if nanoseconds < 1_000:
        return f"{nanoseconds:.2f} ns"
    if nanoseconds < 1_000_000:
        return f"{nanoseconds / 1_000:.2f} µs"
    if nanoseconds < 1_000_000_000:
        return f"{nanoseconds / 1_000_000:.2f} ms"
    return f"{nanoseconds / 1_000_000_000:.2f} s"


def _format_estimate(estimate: Estimate) -> str:
    point = _format_duration(estimate.point_ns)
    if estimate.lower_ns is None or estimate.upper_ns is None:
        return point
    return f"{point} ({_format_duration(estimate.lower_ns)} - {_format_duration(estimate.upper_ns)})"


def _interval_relation(comparison: Comparison) -> str:
    baseline = comparison.baseline
    current = comparison.current
    if baseline.lower_ns is None or baseline.upper_ns is None or current.lower_ns is None or current.upper_ns is None:
        return "unknown"
    if current.upper_ns < baseline.lower_ns:
        return "current lower"
    if baseline.upper_ns < current.lower_ns:
        return "current higher"
    return "overlap"


def _git_revision(root: Path) -> str:
    try:
        return run_git_command(["--no-pager", "rev-parse", "--short", "HEAD"], cwd=root, timeout=10).stdout.strip()
    except ExecutableNotFoundError, OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired:
        return "unknown"


def render_report(
    comparison_set: ComparisonSet,
    settings: ReportSettings,
) -> str:
    """Render a deterministic Markdown comparison report."""
    lines = [
        "# Benchmark Performance",
        "",
        f"**markov-chain-monte-carlo** {settings.current_label} · `{settings.revision}`",
        f"**Statistic**: {settings.statistic}",
        "",
        f"Comparison against baseline **{settings.baseline_label}**:",
        "",
        (
            "Negative change means the current point estimate is lower. The CI relation describes only overlap between Criterion's marginal intervals; "
            "it is not a paired significance test."
        ),
    ]
    if settings.measurement_context:
        lines.extend(["", "## Measurement Context", ""])
        lines.extend(f"- {item}" for item in settings.measurement_context)
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Benchmark | Baseline | Current | Change | Baseline/current | CI relation |",
            "|:----------|---------:|--------:|-------:|-----------------:|:------------|",
        ]
    )
    for comparison in comparison_set.comparisons:
        lines.append(
            f"| `{comparison.benchmark}` | {_format_estimate(comparison.baseline)} | {_format_estimate(comparison.current)} | "
            f"{comparison.percent_change:+.2f}% | {comparison.speedup:.2f}x | {_interval_relation(comparison)} |"
        )

    if comparison_set.missing_baseline or comparison_set.missing_current:
        lines.extend(["", "## Coverage Notes", ""])
        if comparison_set.missing_baseline:
            lines.append("Current-only rows without a saved baseline:")
            lines.extend(f"- `{name}`" for name in comparison_set.missing_baseline)
        if comparison_set.missing_current:
            if comparison_set.missing_baseline:
                lines.append("")
            lines.append("Baseline-only rows without a current sample:")
            lines.extend(f"- `{name}`" for name in comparison_set.missing_current)

    lines.extend(
        [
            "",
            "## How to Update",
            "",
            "```bash",
            "just performance-local",
            "just performance-github-assets",
            "just performance-release",
            "just performance-release <current-tag> <baseline-tag>",
            "```",
            "",
            (
                "Local reports live under `target/bench-reports/`. The curated release report is `docs/PERFORMANCE.md`; older curated reports are indexed "
                "under `docs/archive/performance/`."
            ),
            "",
            "See `docs/BENCHMARKING.md` for command semantics and reproducibility limits.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        if temporary is None:
            msg = f"failed to create a temporary output for {path}"
            raise RuntimeError(msg)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Criterion's current sample with a saved baseline.")
    parser.add_argument("baseline", nargs="?", default="last", help="Criterion baseline name (default: last).")
    parser.add_argument("--criterion-dir", default="target/criterion")
    parser.add_argument("--output", default="target/bench-reports/performance.md")
    parser.add_argument("--current-label", default="working tree")
    parser.add_argument("--baseline-label")
    parser.add_argument("--revision", help="Revision label to record instead of the current checkout's short commit.")
    parser.add_argument("--stat", choices=("mean", "median"), default="median")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Render a comparison report, returning 2 for missing or invalid samples."""
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    root = _repo_root()
    criterion_dir = Path(args.criterion_dir)
    if not criterion_dir.is_absolute():
        criterion_dir = root / criterion_dir
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    baseline_name = str(args.baseline)
    statistic: Statistic = args.stat

    try:
        comparison_set = collect_comparisons(criterion_dir, baseline_name, statistic)
    except (OSError, KeyError, TypeError, ValueError) as error:
        print(f"Invalid Criterion data: {error}", file=sys.stderr)
        return 2
    if not comparison_set.current_sample:
        print(f"No current Criterion results found under {criterion_dir}. Run `just bench-latest` first.", file=sys.stderr)
        return 2
    if not comparison_set.comparisons:
        print(
            f"No comparable Criterion results found for baseline {baseline_name!r}. Run `just bench-save-baseline {baseline_name}` first.",
            file=sys.stderr,
        )
        return 2

    baseline_label = str(args.baseline_label or baseline_name)
    settings = ReportSettings(
        current_label=str(args.current_label),
        baseline_label=baseline_label,
        statistic=statistic,
        revision=str(args.revision or _git_revision(root)),
    )
    report = render_report(comparison_set, settings)
    try:
        _write_text(output, report)
    except (OSError, RuntimeError) as error:
        print(f"Could not write benchmark report: {error}", file=sys.stderr)
        return 2
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
