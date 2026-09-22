"""Publish a README comparison and SVG from validated retained release evidence."""

import argparse
import io
import math
import re
import subprocess
import sys
import tomllib
from importlib import import_module
from pathlib import Path

from research_repo_tools.process import ExecutableNotFoundError, format_exception_diagnostics, run_command
from research_repo_tools.publication import GitCheck, MarkerPair, plan_publication, publish_publication

from archive_performance import (
    ComparisonArtifact,
    _artifact_from_text,
    _rerender_measurements_path,
    parse_report_id,
    provenance_path,
)
from bench_compare import _format_estimate, _format_relative_performance, _markdown_code_span

BEGIN = "<!-- PERFORMANCE:BEGIN -->"
END = "<!-- PERFORMANCE:END -->"


def _svg(artifact: ComparisonArtifact) -> str:
    """Render deterministic point-estimate ratios, without measuring any workload."""
    matplotlib = import_module("matplotlib")
    figures = import_module("matplotlib.figure")
    rows = sorted(artifact.comparison_set.comparisons, key=lambda row: row.benchmark)
    ratios = [row.speedup for row in rows]
    if any(not math.isfinite(ratio) or ratio <= 0 for ratio in ratios):
        msg = "comparison ratios must be finite and positive for README publication"
        raise ValueError(msg)
    with matplotlib.rc_context({"svg.hashsalt": "mcmc-performance", "font.family": "DejaVu Sans"}):
        figure = figures.Figure(figsize=(9, max(2.5, 0.4 * len(rows) + 1.5)), layout="constrained")
        axes = figure.subplots()
        axes.barh([row.benchmark for row in rows], ratios, color="#267394")
        axes.axvline(1, color="#555555", linestyle="--", linewidth=1)
        axes.invert_yaxis()
        axes.set_xlabel("Baseline time / current time (point estimates; >1 is faster)")
        axes.set_title(f"{artifact.settings.current_label} against {artifact.settings.baseline_label}")
        output = io.StringIO()
        figure.savefig(output, format="svg", metadata={"Date": None, "Creator": "markov-chain-monte-carlo"})
        return output.getvalue()


def _markdown(artifact: ComparisonArtifact, evidence: Path, svg: Path, repository_slug: str) -> str:
    pair = artifact.pair
    base = f"https://github.com/{repository_slug}/blob/{pair.current_tag}/"
    raw = f"https://raw.githubusercontent.com/{repository_slug}/{pair.current_tag}/"
    rows = sorted(artifact.comparison_set.comparisons, key=lambda row: row.benchmark)
    lines = [
        (
            f"**{artifact.settings.current_label} against {artifact.settings.baseline_label}**; "
            f"{artifact.settings.statistic} elapsed time, with recorded confidence bounds where available."
        ),
        "",
        f"![Release workload time ratios]({raw}{svg.as_posix()})",
        "",
        "| Workload | Baseline | Current | Relative time |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        name = _markdown_code_span(row.benchmark)
        lines.append(f"| {name} | {_format_estimate(row.baseline)} | {_format_estimate(row.current)} | {_format_relative_performance(row)} |")
    lines.extend(
        [
            "",
            (
                f"Coverage: {len(rows)} comparable, {len(artifact.comparison_set.missing_baseline)} current-only, "
                f"{len(artifact.comparison_set.missing_current)} baseline-only workloads."
            ),
            "",
            "These workload timings do not measure mixing, convergence, or effective sample size. Ratios are point estimates, not significance tests.",
            "",
            f"- [Report and measurement context]({base}docs/PERFORMANCE.md)",
            f"- [CSV measurements]({base}{evidence.as_posix()})",
            f"- [JSON provenance]({base}{provenance_path(evidence).as_posix()})",
        ]
    )
    return "\n".join(lines)


def _publication_checks(root: Path, artifact: ComparisonArtifact, assets: tuple[str, ...]) -> tuple[GitCheck, ...]:
    """Allow MCMC's future-release preparation; require exact blobs for existing tags."""
    tag = artifact.pair.current_tag
    exists = run_command("git", ["--no-pager", "show-ref", "--verify", "--quiet", f"refs/tags/{tag}"], cwd=root, check=False, timeout=30)
    if exists.returncode == 1:
        if tag == artifact.pair.baseline_tag or not artifact.measurement.working_tree_applied:
            raise ValueError(f"publication of existing-release evidence requires the local {tag} tag")
        return ()
    exists.check_returncode()
    return (GitCheck(tag, assets),)


def publish_readme(root: Path) -> tuple[Path, ...]:
    """Validate MCMC evidence, then let the shared planner publish exact candidates."""
    root = root.resolve()
    report = root / "docs" / "PERFORMANCE.md"
    if not report.is_file():
        msg = "retained release report is missing; run just performance-release before publication"
        raise FileNotFoundError(msg)
    evidence = _rerender_measurements_path(root, "")
    svg = evidence.with_suffix(".svg")
    paths = (report, evidence, provenance_path(evidence), root / "Cargo.toml")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"retained release evidence is missing: {path}; run just performance-release before publication")
    inputs = {path.relative_to(root).as_posix(): path.read_bytes() for path in paths}
    evidence_name = evidence.relative_to(root).as_posix()
    provenance_name = provenance_path(evidence).relative_to(root).as_posix()
    artifact = _artifact_from_text(inputs[evidence_name].decode("utf-8"), inputs[provenance_name].decode("utf-8"))
    if artifact.pair != parse_report_id(inputs["docs/PERFORMANCE.md"].decode("utf-8")):
        msg = "retained evidence release pair does not match docs/PERFORMANCE.md"
        raise ValueError(msg)
    package = tomllib.loads(inputs["Cargo.toml"].decode("utf-8")).get("package")
    if not isinstance(package, dict) or not isinstance(package.get("version"), str):
        msg = "Cargo.toml requires a package table with a string version"
        raise TypeError(msg)
    repository = package.get("repository")
    match = re.fullmatch(r"https://github\.com/(?P<slug>[^/]+/[^/]+?)(?:\.git)?/?", repository) if isinstance(repository, str) else None
    if match is None:
        raise ValueError(f"Cargo package repository must be a GitHub HTTPS URL: {repository!r}")
    if artifact.pair.current_tag != "v" + package["version"]:
        msg = "retained evidence does not describe the current package version; run just performance-release before publication"
        raise ValueError(msg)
    svg_name = svg.relative_to(root).as_posix()
    section = _markdown(artifact, Path(evidence_name), Path(svg_name), match["slug"])
    plan = plan_publication(
        root,
        "README.md",
        MarkerPair(BEGIN, END),
        section,
        inputs=inputs,
        figures={svg_name: _svg(artifact).encode("utf-8")},
        git_checks=_publication_checks(root, artifact, ("docs/PERFORMANCE.md", evidence_name, provenance_name, svg_name)),
    )
    return publish_publication(plan)


def main(argv: list[str] | None = None) -> int:
    """Publish retained evidence with read-only Git checks, without release discovery or benchmarks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        changed = publish_readme(args.repo_root)
    except ImportError as error:
        print(f"README plotting requires markov-chain-monte-carlo-tooling[notebook]: {error}", file=sys.stderr)
        return 1
    except (ExecutableNotFoundError, OSError, RuntimeError, TypeError, ValueError, subprocess.SubprocessError, ExceptionGroup) as error:
        print(f"README publication failed: {format_exception_diagnostics(error)}", file=sys.stderr)
        return 1
    for path in changed:
        print(f"Updated {path.relative_to(args.repo_root.resolve())}")
    if not changed:
        print("README performance publication is already current.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
