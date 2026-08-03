import json
from typing import TYPE_CHECKING

import pytest

import bench_compare

if TYPE_CHECKING:
    from pathlib import Path


def _write_estimate(
    root: Path,
    benchmark: str,
    sample: str,
    point: float,
    interval: tuple[float, float] | None = None,
) -> None:
    path = root / benchmark / sample / "estimates.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    statistic: dict[str, object] = {"point_estimate": point}
    if interval is not None:
        statistic["confidence_interval"] = {"lower_bound": interval[0], "upper_bound": interval[1]}
    path.write_text(json.dumps({"median": statistic, "mean": statistic}), encoding="utf-8")


def test_collect_comparisons_reports_shared_and_missing_rows(tmp_path: Path) -> None:
    criterion = tmp_path / "criterion"
    _write_estimate(criterion, "chain/step_by_value", "v0.4.0", 100.0, (95.0, 105.0))
    _write_estimate(criterion, "chain/step_by_value", "new", 80.0, (75.0, 85.0))
    _write_estimate(criterion, "observing/new_metric", "new", 200.0)
    _write_estimate(criterion, "chain/removed_metric", "v0.4.0", 300.0)

    collected = bench_compare.collect_comparisons(criterion, "v0.4.0")

    assert len(collected.comparisons) == 1
    assert collected.comparisons[0].benchmark == "chain/step_by_value"
    assert collected.comparisons[0].percent_change == pytest.approx(-20.0)
    assert collected.comparisons[0].speedup == pytest.approx(1.25)
    assert collected.missing_baseline == ("observing/new_metric",)
    assert collected.missing_current == ("chain/removed_metric",)


def test_render_report_includes_release_pair_and_coverage_notes(tmp_path: Path) -> None:
    criterion = tmp_path / "criterion"
    _write_estimate(criterion, "chain/step_by_value", "v0.4.0", 100.0, (95.0, 105.0))
    _write_estimate(criterion, "chain/step_by_value", "new", 80.0, (75.0, 85.0))
    _write_estimate(criterion, "chain/new_path", "new", 90.0)
    comparison_set = bench_compare.collect_comparisons(criterion, "v0.4.0")

    settings = bench_compare.ReportSettings(
        current_label="v0.5.0",
        baseline_label="v0.4.0",
        statistic="median",
        revision="abc1234",
        measurement_context=("Source mode: fixture.",),
    )
    report = bench_compare.render_report(comparison_set, settings)

    assert "**markov-chain-monte-carlo** v0.5.0 · `abc1234`" in report
    assert "Comparison against baseline **v0.4.0**:" in report
    assert "## Measurement Context" in report
    assert "Source mode: fixture." in report
    assert "| `chain/step_by_value` |" in report
    assert "-20.00%" in report
    assert "current lower" in report
    assert "Current-only rows without a saved baseline:" in report
    assert "`chain/new_path`" in report


def test_main_supports_an_explicit_saved_baseline(tmp_path: Path) -> None:
    criterion = tmp_path / "criterion"
    output = tmp_path / "report.md"
    _write_estimate(criterion, "sampler/run_by_value_100", "release-a", 10_000.0)
    _write_estimate(criterion, "sampler/run_by_value_100", "new", 9_000.0)

    status = bench_compare.main(
        [
            "release-a",
            "--criterion-dir",
            str(criterion),
            "--output",
            str(output),
            "--current-label",
            "working tree",
            "--revision",
            "fixture-revision",
        ]
    )

    assert status == 0
    output_text = output.read_text(encoding="utf-8")
    assert "Comparison against baseline **release-a**:" in output_text
    assert "`fixture-revision`" in output_text


def test_main_fails_cleanly_when_the_baseline_is_missing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    criterion = tmp_path / "criterion"
    _write_estimate(criterion, "chain/step_by_value", "new", 100.0)

    status = bench_compare.main(["missing", "--criterion-dir", str(criterion), "--output", str(tmp_path / "report.md")])

    assert status == 2
    assert "No comparable Criterion results" in capsys.readouterr().err


def test_read_estimate_rejects_invalid_timing_data(tmp_path: Path) -> None:
    criterion = tmp_path / "criterion"
    _write_estimate(criterion, "chain/step_by_value", "new", -1.0)

    with pytest.raises(ValueError, match="finite and positive"):
        bench_compare.collect_sample(criterion, "new")


@pytest.mark.parametrize(
    ("point", "interval"),
    [
        (90.0, (100.0, 120.0)),
        (130.0, (100.0, 120.0)),
    ],
)
def test_read_estimate_rejects_a_point_outside_its_confidence_interval(
    tmp_path: Path,
    point: float,
    interval: tuple[float, float],
) -> None:
    criterion = tmp_path / "criterion"
    _write_estimate(criterion, "chain/step_by_value", "new", point, interval)

    with pytest.raises(ValueError, match="must lie within"):
        bench_compare.collect_sample(criterion, "new")
