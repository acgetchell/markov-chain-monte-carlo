"""MCMC's first-comparison boundary and shared report validation."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from research_repo_tools.cli import main
from research_repo_tools.criterion import COMPARISON_SCHEMA, Estimate, Sample, compare_samples, serialize_comparison
from research_repo_tools.evidence import Evidence, Provenance, publish_evidence

ROOT = Path(__file__).resolve().parents[2]
REPORT = "docs/performance/v1/performance.md"
CONFIGURATION = "tooling/performance-report.toml"


def _project(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the actual recipe/configuration and the already locked test environment."""
    for name in ("justfile", "pyproject.toml", "uv.lock", ".python-version", CONFIGURATION, "tooling/performance-interpretation.md"):
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", sys.prefix)


def _check(root: Path) -> subprocess.CompletedProcess[str]:
    executable = shutil.which("just")
    assert executable is not None
    return subprocess.run(  # noqa: S603 - resolved executable and fixed consumer recipe.
        [executable, "performance-check"],
        cwd=root,
        check=False,
        capture_output=True,
        encoding="utf-8",
    )


def test_first_comparison_gate_allows_experiments_without_release_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _project(tmp_path, monkeypatch)
    experiment = tmp_path / "docs/performance/v1/experiments/diagnostic-backends.json"
    experiment.parent.mkdir(parents=True)
    experiment.write_bytes((ROOT / experiment.relative_to(tmp_path)).read_bytes())
    (experiment.parent.parent / "README.md").write_text("# Benchmark evidence\n\nAwaiting the first release comparison.\n", encoding="utf-8", newline="\n")

    result = _check(tmp_path)

    assert result.returncode == 0, result.stderr
    assert "No release comparison yet" in result.stdout
    assert not (tmp_path / REPORT).exists()


@pytest.mark.parametrize(
    "artifact",
    [
        "docs/performance/v1/v0.2.0-vs-v0.1.0.comparison.json",
        "docs/performance/v1/v0.2.0-vs-v0.1.0.evidence.json",
        "docs/performance/v1/v0.2.0-vs-v0.1.0.csv",
        "docs/performance/v1/v0.2.0-vs-v0.1.0.svg",
        "docs/performance/v1/v0.2.0-vs-v0.1.0.md",
        "tooling/performance-readme.toml",
    ],
)
def test_release_artifact_without_current_report_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, artifact: str) -> None:
    _project(tmp_path, monkeypatch)
    path = tmp_path / artifact
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"partial release artifact\n")

    result = _check(tmp_path)

    assert result.returncode != 0
    assert "Release evidence exists without its current report" in result.stderr
    assert "No release comparison yet" not in result.stdout


def test_current_report_delegates_reproduction_and_drift_to_shared_validator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _project(tmp_path, monkeypatch)
    comparison = compare_samples(Sample((("chain/step_by_value", Estimate(12)),)), Sample((("chain/step_by_value", Estimate(10)),)))
    evidence = Evidence(
        serialize_comparison(comparison),
        COMPARISON_SCHEMA,
        (
            ("baseline", Provenance("1" * 40, context=(("release", "v0.1.0"),))),
            ("current", Provenance("2" * 40, context=(("release", "v0.2.0"),))),
        ),
    )
    payload, manifest = "incoming.comparison.json", "incoming.evidence.json"
    publish_evidence(evidence, tmp_path / payload, tmp_path / manifest)
    assert main(["--root", str(tmp_path), "performance", "promote", CONFIGURATION, "--payload", payload, "--manifest", manifest]) == 0

    result = _check(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "No release comparison yet" not in result.stdout

    report = tmp_path / REPORT
    report.write_bytes(report.read_bytes() + b"\nUnreproduced report change.\n")
    result = _check(tmp_path)
    assert result.returncode != 0
    assert "No release comparison yet" not in result.stdout
