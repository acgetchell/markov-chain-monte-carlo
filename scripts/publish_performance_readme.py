"""Publish a README comparison and SVG from validated retained release evidence."""

import argparse
import io
import math
import subprocess
import sys
from importlib import import_module
from pathlib import Path

from archive_performance import (
    ComparisonArtifact,
    _publish_texts,
    _rerender_measurements_path,
    current_package_tag,
    load_comparison_artifact,
    parse_report_id,
    provenance_path,
)
from bench_compare import _format_estimate, _format_relative_performance, _markdown_code_span
from release_check import _read_cargo_package_info
from subprocess_utils import ExecutableNotFoundError, run_git_command, run_git_command_with_input

BEGIN = "<!-- PERFORMANCE:BEGIN -->"
END = "<!-- PERFORMANCE:END -->"


def _require_contained(root: Path, path: Path) -> None:
    """Reject tracked destinations that redirect outside the repository or alias a link."""
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        msg = f"tracked publication path must remain inside the repository and must not be a symbolic link: {path}"
        raise ValueError(msg)


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


def _validate_publication_tag(root: Path, artifact: ComparisonArtifact, assets: tuple[tuple[Path, str], ...]) -> None:
    """Allow a future release tag or verify every linked artifact at an existing tag."""
    tag = artifact.pair.current_tag
    reference = f"refs/tags/{tag}"
    exists = run_git_command(["--no-pager", "show-ref", "--verify", "--quiet", reference], cwd=root, check=False, timeout=30)
    if exists.returncode == 1:
        if artifact.pair.current_tag == artifact.pair.baseline_tag or not artifact.measurement.working_tree_applied:
            msg = f"publication of existing-release evidence requires the local {tag} tag; make release tags available before publication"
            raise ValueError(msg)
        return
    exists.check_returncode()
    commit = run_git_command(["--no-pager", "rev-parse", "--verify", f"{reference}^{{commit}}"], cwd=root, timeout=30).stdout.strip()
    for path, contents in assets:
        relative = path.relative_to(root).as_posix()
        tagged = run_git_command(["--no-pager", "rev-parse", "--verify", f"{commit}:{relative}"], cwd=root, check=False, timeout=30)
        expected = run_git_command_with_input(["--no-pager", "hash-object", f"--path={relative}", "--stdin"], contents, cwd=root, timeout=30).stdout.strip()
        if tagged.returncode != 0 or tagged.stdout.strip() != expected:
            msg = (
                f"tag {tag} does not contain the exact publication artifact {relative}; "
                "keep repaired or same-version evidence local, or prepare a new release before README publication"
            )
            raise ValueError(msg)


def publish_readme(root: Path) -> tuple[Path, ...]:
    """Validate all retained inputs and outputs before transactionally publishing both files."""
    root = root.resolve()
    readme = root / "README.md"
    report = root / "docs" / "PERFORMANCE.md"
    for path in (readme, report):
        _require_contained(root, path)
    if not report.is_file():
        msg = "retained release report is missing; run just performance-release before publication"
        raise FileNotFoundError(msg)
    evidence = _rerender_measurements_path(root, "")
    svg = evidence.with_suffix(".svg")
    # Tracked destinations are paths, independent of their human-readable link labels.
    tracked_paths = frozenset({readme, report, evidence, provenance_path(evidence), svg})
    for path in tracked_paths:
        _require_contained(root, path)
    artifact = load_comparison_artifact(evidence)
    if artifact.pair != parse_report_id(report.read_text(encoding="utf-8")):
        msg = "retained evidence release pair does not match docs/PERFORMANCE.md"
        raise ValueError(msg)
    if artifact.pair.current_tag != current_package_tag(root):
        msg = "retained evidence does not describe the current package version; run just performance-release before publication"
        raise ValueError(msg)
    original = readme.read_bytes().decode("utf-8")
    if original.count(BEGIN) != 1 or original.count(END) != 1 or original.index(BEGIN) >= original.index(END):
        msg = "README.md must contain exactly one ordered PERFORMANCE marker pair"
        raise ValueError(msg)
    repository_slug = _read_cargo_package_info(root / "Cargo.toml").repository_slug
    contents = _markdown(artifact, evidence.relative_to(root), svg.relative_to(root), repository_slug)
    updated = original[: original.index(BEGIN) + len(BEGIN)] + "\n\n" + contents + "\n\n" + original[original.index(END) :]
    rendered_svg = _svg(artifact)
    assets = (*((path, path.read_bytes().decode("utf-8")) for path in (report, evidence, provenance_path(evidence))), (svg, rendered_svg))
    _validate_publication_tag(root, artifact, assets)
    outputs = ((svg, rendered_svg), (readme, updated))
    changed = tuple((path, text) for path, text in outputs if not path.exists() or path.read_bytes() != text.encode("utf-8"))
    _publish_texts(changed)
    return tuple(path for path, _ in changed)


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
    except (ExecutableNotFoundError, OSError, RuntimeError, TypeError, ValueError, subprocess.SubprocessError) as error:
        print(f"README publication failed: {error}", file=sys.stderr)
        return 1
    for path in changed:
        print(f"Updated {path.relative_to(args.repo_root.resolve())}")
    if not changed:
        print("README performance publication is already current.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
