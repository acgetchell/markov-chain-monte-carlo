"""Consumer evidence transition and representative shared-command integration."""

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from research_repo_tools.cli import main
from research_repo_tools.criterion import parse_comparison
from research_repo_tools.evidence import load_evidence, serialize_evidence
from research_repo_tools.legacy_evidence import convert_csv
from research_repo_tools.performance_reports import load_report_plan
from research_repo_tools.publication_config import load_publication

if TYPE_CHECKING:
    from research_repo_tools.evidence import Evidence

ROOT = Path(__file__).resolve().parents[2]
STEM = "v0.4.2-vs-v0.4.1"
LEGACY = ROOT / "docs/archive/performance"
SHARED = ROOT / "docs/performance/v1"


def converted() -> Evidence:
    return convert_csv(
        (LEGACY / f"{STEM}.csv").read_bytes(),
        (LEGACY / f"{STEM}.provenance.json").read_bytes(),
        (ROOT / "tooling/legacy-csv.toml").read_bytes(),
    )


def test_conversion_preserves_every_mcmc_value_bound_and_coverage() -> None:
    evidence = converted()
    comparison = parse_comparison(evidence.payload)
    rows = list(csv.DictReader(io.StringIO((LEGACY / f"{STEM}.csv").read_bytes().decode("utf-8"))))
    for side in ("baseline", "current"):
        sample = getattr(comparison, side)
        actual = dict(sample.estimates)
        expected = {row["benchmark"]: row for row in rows if row[f"{side}_point_ns"]}
        assert actual.keys() == expected.keys()
        for name, row in expected.items():
            assert actual[name].point == float(row[f"{side}_point_ns"])
            assert actual[name].lower == float(row[f"{side}_lower_ns"])
            assert actual[name].upper == float(row[f"{side}_upper_ns"])
            assert actual[name].confidence_level is None
    assert set(comparison.missing_baseline) == {row["benchmark"] for row in rows if row["coverage"] == "current_only"}
    assert set(comparison.missing_current) == {row["benchmark"] for row in rows if row["coverage"] == "baseline_only"}
    original = json.loads((LEGACY / f"{STEM}.provenance.json").read_bytes())
    for side, source in evidence.sources:
        assert source.source_sha256 is None
        assert source.harness_sha256 is None
        context = dict(source.context)
        assert json.loads(context["legacy.record"]) == original["measurement"][side]
        assert context["legacy.payload-sha256"] == original["csv_sha256"]
        assert context["legacy.manifest-sha256"] == hashlib.sha256((LEGACY / f"{STEM}.provenance.json").read_bytes()).hexdigest()
    retained = load_evidence(SHARED / f"{STEM}.comparison.json", SHARED / f"{STEM}.evidence.json")
    assert serialize_evidence(evidence) == serialize_evidence(retained)


def test_shared_report_reproduces_from_retained_evidence_offline() -> None:
    plan = load_report_plan(ROOT, "tooling/performance-report.toml")
    assert plan.changed_paths == ()
    report = (SHARED / "current.md").read_text(encoding="utf-8")
    assert "do not establish convergence" in report
    assert "baseline time divided by current time" in report
    assert "legacy.record" in report


def test_historical_evidence_cannot_be_republished_as_a_new_measurement() -> None:
    # The old README and SVG remain immutable. New publication needs freshly
    # measured source fingerprints and independently reviewed provenance pins.
    with pytest.raises(ValueError, match="prepare policy requires prospective working-tree evidence"):
        load_publication(ROOT, "tooling/performance-readme.toml")


def test_mcmc_rows_and_references_publish_through_shared_cli(tmp_path: Path) -> None:
    # Exercise this repository's selected rows, report references, marker pair,
    # and renderer through the installed CLI. Tag enforcement is tested upstream;
    # this fixture uses document-relative links and no Git repository.
    config = (ROOT / "tooling/performance-readme.toml").read_text(encoding="utf-8")
    config = "\n".join(line for line in config.splitlines() if not line.startswith(("repository =", "tag-policy =", "current-sources =", "current-harness =")))
    (tmp_path / "publication.toml").write_text(config, encoding="utf-8", newline="\n")
    for relative in (
        "Cargo.toml",
        "tooling/performance-interpretation.md",
        "docs/performance/v1/current.md",
        f"docs/performance/v1/{STEM}.comparison.json",
        f"docs/performance/v1/{STEM}.evidence.json",
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    document = tmp_path / "README.md"
    document.write_bytes(b"Before\n<!-- PERFORMANCE:BEGIN -->\nold\n<!-- PERFORMANCE:END -->\nAfter\n")
    command = ["--root", str(tmp_path), "performance", "publish", "publication.toml"]
    assert main(command) == 0
    assert main([*command, "--check"]) == 0
    published = document.read_text(encoding="utf-8")
    assert published.startswith("Before\n")
    assert published.endswith("After\n")
    assert "In-place rollback step" in published
    assert (tmp_path / f"docs/performance/v1/{STEM}.svg").is_file()
