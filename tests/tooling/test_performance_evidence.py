"""MCMC historical values and future publication eligibility."""

import csv
import hashlib
import io
import json
from pathlib import Path

import pytest
from research_repo_tools.criterion import parse_comparison
from research_repo_tools.evidence import load_evidence
from research_repo_tools.publication_config import load_publication

ROOT = Path(__file__).resolve().parents[2]
STEM = "v0.4.2-vs-v0.4.1"
LEGACY = ROOT / "docs/archive/performance"
SHARED = ROOT / "docs/performance/v1"


def test_retained_evidence_preserves_every_mcmc_value_bound_and_coverage() -> None:
    evidence = load_evidence(SHARED / f"{STEM}.comparison.json", SHARED / f"{STEM}.evidence.json")
    comparison = parse_comparison(evidence.payload)
    original_csv = (LEGACY / f"{STEM}.csv").read_bytes()
    rows = list(csv.DictReader(io.StringIO(original_csv.decode("utf-8"))))
    for side in ("baseline", "current"):
        sample = getattr(comparison, side)
        assert (sample.statistic, sample.unit) == ("median", "ns")
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
    assert original["csv_sha256"] == hashlib.sha256(original_csv).hexdigest()
    for side, source in evidence.sources:
        assert source.revision == original["measurement"][side]["commit"]
        assert source.source_sha256 is None
        assert source.harness_sha256 is None
        context = dict(source.context)
        assert context["release"] == original["release"][f"{side}_tag"]
        assert json.loads(context["legacy.manifest"]) == original
        assert json.loads(context["legacy.record"]) == original["measurement"][side]
        assert context["legacy.payload-sha256"] == original["csv_sha256"]
        assert context["legacy.manifest-sha256"] == hashlib.sha256((LEGACY / f"{STEM}.provenance.json").read_bytes()).hexdigest()


def test_historical_evidence_cannot_be_republished_as_a_new_measurement() -> None:
    # The old README and SVG remain immutable. New publication needs freshly
    # measured source fingerprints and independently reviewed provenance pins.
    with pytest.raises(ValueError, match="prepare policy requires prospective working-tree evidence"):
        load_publication(ROOT, "tooling/performance-readme.toml")
