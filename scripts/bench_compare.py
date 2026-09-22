#!/usr/bin/env python3
"""Render Markdown comparisons from Criterion benchmark samples.

Criterion stores every benchmark below ``target/criterion`` with one directory
per sample.  This utility compares the ordinary ``new`` sample with a named
saved baseline and writes a compact, reviewable report.
"""

import argparse
import subprocess
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal

from research_repo_tools.criterion import Comparison, Estimate, Sample, collect_sample as shared_collect_sample, compare_samples
from research_repo_tools.files import replace_many
from research_repo_tools.process import ExecutableNotFoundError, format_exception_diagnostics, run_command

run_git_command = partial(run_command, "git")

type Statistic = Literal["mean", "median"]

_MAX_FACTOR_PRECISION = 16


@dataclass(frozen=True, slots=True)
class ComparisonSet:
    """Comparable rows plus samples missing from either revision."""

    comparisons: tuple[Comparison, ...]
    missing_baseline: tuple[str, ...]
    missing_current: tuple[str, ...]
    current_sample: tuple[tuple[str, Estimate], ...]
    baseline_sample: tuple[tuple[str, Estimate], ...]


@dataclass(frozen=True, slots=True)
class ReportSettings:
    """Labels, statistic, and provenance rendered into one report."""

    current_label: str
    baseline_label: str
    statistic: Statistic
    revision: str
    measurement_context: tuple[str, ...] = ()


def collect_sample(criterion_dir: Path, sample: str, statistic: Statistic = "median") -> dict[str, Estimate]:
    """Read the MCMC wall-time harness through the shared nanosecond parser."""
    sample_data = shared_collect_sample(criterion_dir, sample, statistic=statistic, unit="ns")
    # The retained CSV schema records bounds but has no confidence-level column.
    return {name: Estimate(estimate.point, estimate.lower, estimate.upper) for name, estimate in sample_data.estimates}


def comparison_from_samples(current: dict[str, Estimate], baseline: dict[str, Estimate]) -> ComparisonSet:
    """Adapt shared complete inventories to the retained MCMC CSV/report schema."""
    comparison = compare_samples(Sample(tuple(baseline.items())), Sample(tuple(current.items())))
    return ComparisonSet(
        comparison.comparisons, comparison.missing_baseline, comparison.missing_current, comparison.current.estimates, comparison.baseline.estimates
    )


def collect_comparisons(criterion_dir: Path, baseline_name: str, statistic: Statistic = "median") -> ComparisonSet:
    """Pair MCMC's current and saved wall-time samples."""
    return comparison_from_samples(collect_sample(criterion_dir, "new", statistic), collect_sample(criterion_dir, baseline_name, statistic))


def _format_duration(nanoseconds: float) -> str:
    if nanoseconds < 1_000:
        return f"{nanoseconds:.2f} ns"
    if nanoseconds < 1_000_000:
        return f"{nanoseconds / 1_000:.2f} µs"
    if nanoseconds < 1_000_000_000:
        return f"{nanoseconds / 1_000_000:.2f} ms"
    return f"{nanoseconds / 1_000_000_000:.2f} s"


def _format_estimate(estimate: Estimate) -> str:
    point = _format_duration(estimate.point)
    if estimate.lower is None or estimate.upper is None:
        return point
    return f"{point} ({_format_duration(estimate.lower)} - {_format_duration(estimate.upper)})"


def _format_relative_performance(comparison: Comparison) -> str:
    """Describe the current duration as an explicit faster/slower factor."""
    speedup = comparison.speedup
    if speedup == 1.0:
        return "unchanged"
    factor = speedup if speedup > 1.0 else 1.0 / speedup
    precision = 3 if factor < 1.01 else 2
    while precision < _MAX_FACTOR_PRECISION and round(factor, precision) == 1.0:
        precision += 1
    direction = "faster" if speedup > 1.0 else "slower"
    return f"{factor:.{precision}f}x {direction}"


def _markdown_code_span(value: str) -> str:
    """Render arbitrary benchmark text safely inside Markdown and pipe tables."""
    longest_backtick_run = 0
    current_backtick_run = 0
    for character in value:
        if character == "`":
            current_backtick_run += 1
            longest_backtick_run = max(longest_backtick_run, current_backtick_run)
        else:
            current_backtick_run = 0
    delimiter = "`" * (longest_backtick_run + 1)
    escaped = value.replace("|", r"\|")
    padding = " " if value.startswith("`") or value.endswith("`") else ""
    return f"{delimiter}{padding}{escaped}{padding}{delimiter}"


def _git_revision(root: Path) -> str:
    try:
        return run_git_command(["--no-pager", "rev-parse", "--short", "HEAD"], cwd=root, timeout=10).stdout.strip()
    except (
        # Keep tuple syntax for the pinned Semgrep Python parser.
        ExecutableNotFoundError,
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
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
        "Positive time reduction means the current duration is lower (faster); negative means it is higher (slower).",
        "The relative-performance column states how many times the current version is faster or slower.",
        "Shown confidence intervals are Criterion's marginal intervals; they are not a paired significance test.",
    ]
    if settings.measurement_context:
        lines.extend(["", "## Measurement Context", ""])
        lines.extend(f"- {item}" for item in settings.measurement_context)
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Benchmark | Baseline | Current | Time reduction | Current vs baseline |",
            "|:----------|---------:|--------:|---------------:|--------------------:|",
        ]
    )
    for comparison in comparison_set.comparisons:
        lines.append(
            f"| {_markdown_code_span(comparison.benchmark)} | {_format_estimate(comparison.baseline)} | {_format_estimate(comparison.current)} | "
            f"{comparison.percent_reduction:+.2f}% | {_format_relative_performance(comparison)} |"
        )

    if comparison_set.missing_baseline or comparison_set.missing_current:
        lines.extend(["", "## Coverage Notes", ""])
        if comparison_set.missing_baseline:
            lines.extend(["Current-only rows without a saved baseline:", ""])
            lines.extend(f"- {_markdown_code_span(name)}" for name in comparison_set.missing_baseline)
        if comparison_set.missing_current:
            if comparison_set.missing_baseline:
                lines.append("")
            lines.extend(["Baseline-only rows without a current sample:", ""])
            lines.extend(f"- {_markdown_code_span(name)}" for name in comparison_set.missing_current)

    lines.extend(
        [
            "",
            "## How to Update",
            "",
            "```bash",
            "just performance-local",
            "just performance-github-assets",
            "just performance-release",
            "just performance-doc",
            "just performance-readme",
            "just performance-release <current-tag> <baseline-tag>",
            "```",
            "",
            "Generated Markdown, CSV measurements, and JSON provenance live under `target/bench-reports/`.",
            "The curated release report is `docs/PERFORMANCE.md`.",
            "Older curated reports are indexed under `docs/archive/performance/`.",
            "",
            "See `docs/BENCHMARKING.md` for command semantics and reproducibility limits.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_text(path: Path, text: str) -> None:
    """Publish UTF-8 report bytes through the shared transaction."""
    replace_many({path: text.encode("utf-8")})


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Criterion's current sample with a saved baseline.")
    parser.add_argument("baseline", nargs="?", default="last", help="Criterion baseline name (default: last).")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root for relative inputs and outputs (default: current directory).",
    )
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
    root = args.repo_root.resolve()
    criterion_dir = Path(args.criterion_dir)
    if not criterion_dir.is_absolute():
        criterion_dir = root / criterion_dir
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    baseline_name = str(args.baseline)
    statistic: Statistic = args.stat

    try:
        if output.resolve().is_relative_to(criterion_dir.resolve()):
            msg = "benchmark report output must be outside the Criterion input tree"
            raise ValueError(msg)
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
    except (OSError, RuntimeError, ValueError, ExceptionGroup) as error:
        print(f"Could not write benchmark report: {format_exception_diagnostics(error)}", file=sys.stderr)
        return 2
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
