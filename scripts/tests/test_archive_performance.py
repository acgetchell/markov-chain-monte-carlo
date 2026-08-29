import io
import json
import os
import re
import subprocess
import tarfile
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import archive_performance
import bench_compare

if TYPE_CHECKING:
    from collections.abc import Iterator


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_HARNESS = REPO_ROOT / "benches" / "stepping.rs"
STEADY_STATE_BENCHMARKS = {
    "chain/step_by_value": ("scalar_chain", "StdRng::seed_from_u64"),
    "chain/step_mut_accept": ("spin_chain", "StdRng::seed_from_u64"),
    "chain/step_mut_reject_rollback": ("spin_chain", "StdRng::seed_from_u64"),
    "chain/step_delayed_accept_commit": ("scalar_chain", "StdRng::seed_from_u64"),
    "chain/step_delayed_reject_plan": ("scalar_chain", "StdRng::seed_from_u64"),
    "chain/step_delayed_no_plan": ("scalar_chain", "StdRng::seed_from_u64"),
    "sampler/run_by_value_100": ("Sampler::new", "StdRng::seed_from_u64"),
    "sampler/run_mut_100": ("Sampler::new", "StdRng::seed_from_u64"),
    "sampler/run_delayed_100": ("Sampler::new", "StdRng::seed_from_u64"),
    "observing/run_observing_buffer_100": ("Sampler::new", "StdRng::seed_from_u64"),
}
FRESH_BATCH_BENCHMARKS = {
    "observing/manual_online_sum_100",
    "observing/run_observing_into_online_stats_100",
    "observing/run_observing_into_binning_100",
}


def _benchmark_block(source: str, name: str) -> str:
    pattern = re.compile(
        rf'^    c\.bench_function\("{re.escape(name)}", \|b\| \{{(?P<body>.*?)^    \}}\);',
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(source)
    assert match is not None, name
    return match.group("body")


def _release(tag: str, days_ago: int = 0) -> archive_performance.PublishedRelease:
    return archive_performance.PublishedRelease(tag, datetime(2026, 8, 1, tzinfo=UTC) - timedelta(days=days_ago))


def _report(current: str, baseline: str, marker: str = "") -> str:
    return (
        "# Benchmark Performance\n\n"
        f"**markov-chain-monte-carlo** {current} · `abc1234`\n"
        "**Statistic**: median\n\n"
        f"Comparison against baseline **{baseline}**:\n\n"
        f"{marker}\n"
    )


def _write_sample(root: Path, benchmark: str, sample: str, point: float = 1.0) -> None:
    sample_dir = root / benchmark / sample
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "estimates.json").write_text(json.dumps({"median": {"point_estimate": point}}), encoding="utf-8")
    (sample_dir / "sample.json").write_text("{}", encoding="utf-8")


def _write_release_archive(
    root: Path,
    tag: str,
    point: float,
    commit: str,
    *,
    benchmark_harness_sha256: str | None = None,
) -> Path:
    criterion = root / tag / "criterion"
    _write_sample(criterion, "chain/step_by_value", tag, point)
    metadata = {
        "schema": 1 if benchmark_harness_sha256 is None else 2,
        "tag": tag,
        "commit": commit,
        "operating_system": "Linux",
        "architecture": "x86_64",
        "rustc": "rustc 1.97.1",
        "criterion_version": "0.8.2",
    }
    if benchmark_harness_sha256 is not None:
        metadata["benchmark_harness_sha256"] = benchmark_harness_sha256
    (criterion / ".mcmc-release-metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    archive = root / f"{tag}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(criterion, arcname="criterion")
    return archive


def _sample_provenance(
    tag: str,
    commit_character: str,
    *,
    local: bool = True,
    benchmark_harness_sha256: str | None = None,
) -> archive_performance.SampleProvenance:
    return archive_performance.SampleProvenance(
        tag=tag,
        commit=commit_character * 40,
        operating_system="TestOS",
        architecture="test-arch",
        rustc="rustc 1.97.1",
        criterion_version="0.8.2",
        source_digest_sha256="1" * 64 if local else None,
        cargo_lock_sha256="2" * 64 if local else None,
        benchmark_harness_sha256="3" * 64 if local else benchmark_harness_sha256,
        command=("cargo", "bench") if local else None,
    )


def _measurement(
    pair: archive_performance.ReportId,
    *,
    local: bool = True,
    working_tree: bool = False,
) -> archive_performance.MeasurementProvenance:
    return archive_performance.MeasurementProvenance(
        mode="local-isolated-worktrees" if local else "github-release-assets",
        working_tree_applied=working_tree,
        current=_sample_provenance(pair.current_tag, "a", local=local),
        baseline=_sample_provenance(pair.baseline_tag, "b", local=local),
    )


def _comparison_artifact(
    pair: archive_performance.ReportId | None = None,
    *,
    current_point: float = 80.0,
) -> archive_performance.ComparisonArtifact:
    resolved_pair = pair or archive_performance.ReportId("v0.5.0", "v0.4.0")
    baseline_shared = bench_compare.Estimate(100.0, 95.0, 105.0)
    current_shared = bench_compare.Estimate(current_point, 75.0, 85.0)
    current_only = bench_compare.Estimate(200.0, None, None)
    baseline_only = bench_compare.Estimate(300.0, None, None)
    comparison_set = bench_compare.ComparisonSet(
        comparisons=(bench_compare.Comparison("chain/shared", baseline_shared, current_shared),),
        missing_baseline=("chain/current_only",),
        missing_current=("chain/baseline_only",),
        current_sample=(("chain/current_only", current_only), ("chain/shared", current_shared)),
        baseline_sample=(("chain/baseline_only", baseline_only), ("chain/shared", baseline_shared)),
    )
    settings = bench_compare.ReportSettings(
        current_label=resolved_pair.current_tag,
        baseline_label=resolved_pair.baseline_tag,
        statistic="median",
        revision="aaaaaaa",
        measurement_context=("Source mode: fixture.",),
    )
    return archive_performance.ComparisonArtifact(
        pair=resolved_pair,
        comparison_set=comparison_set,
        settings=settings,
        measurement=_measurement(resolved_pair),
    )


def test_release_signal_benchmark_names_are_explicit_contracts() -> None:
    source = BENCHMARK_HARNESS.read_text(encoding="utf-8")
    registered = set(re.findall(r'c\.bench_function\("([^"]+)"', source))

    assert registered == set(STEADY_STATE_BENCHMARKS) | FRESH_BATCH_BENCHMARKS


def test_steady_state_contracts_construct_fixtures_before_timing() -> None:
    source = BENCHMARK_HARNESS.read_text(encoding="utf-8")

    for name, setup_markers in STEADY_STATE_BENCHMARKS.items():
        block = _benchmark_block(source, name)
        setup, separator, timed = block.partition("b.iter(||")
        assert separator, name
        assert "iter_batched" not in timed, name
        for marker in setup_markers:
            assert marker in setup, f"{name}: {marker} must remain outside the timed loop"


def test_fresh_batch_contracts_use_criterion_batch_setup() -> None:
    source = BENCHMARK_HARNESS.read_text(encoding="utf-8")

    for name in FRESH_BATCH_BENCHMARKS:
        block = _benchmark_block(source, name)
        assert "b.iter_batched(" in block, name
        assert "StdRng::seed_from_u64(SEED)" in block, name


def test_local_measurement_context_flags_changed_harnesses() -> None:
    pair = archive_performance.ReportId("v0.5.0", "v0.4.0")
    measurement = archive_performance.MeasurementProvenance(
        mode="local-isolated-worktrees",
        working_tree_applied=True,
        current=_sample_provenance(pair.current_tag, "a"),
        baseline=archive_performance.SampleProvenance(
            tag=pair.baseline_tag,
            commit="b" * 40,
            operating_system="TestOS",
            architecture="test-arch",
            rustc="rustc 1.96.0",
            criterion_version="0.8.2",
            source_digest_sha256="4" * 64,
            cargo_lock_sha256="5" * 64,
            benchmark_harness_sha256="6" * 64,
            command=("cargo", "bench"),
        ),
    )

    context = archive_performance._local_measurement_context(measurement)

    assert "current `333333333333`; baseline `666666666666`" in context[-2]
    assert "workload contract" in context[-1]


def test_stable_published_releases_filters_and_sorts() -> None:
    document = [
        {"tagName": "v0.3.0", "isDraft": False, "isPrerelease": False, "publishedAt": "2026-05-01T00:00:00Z"},
        {"tagName": "v0.5.0-rc.1", "isDraft": False, "isPrerelease": True, "publishedAt": "2026-07-01T00:00:00Z"},
        {"tagName": "v0.4.0", "isDraft": False, "isPrerelease": False, "publishedAt": "2026-06-01T00:00:00+00:00"},
        {"tagName": "nightly", "isDraft": False, "isPrerelease": False, "publishedAt": "2026-08-01T00:00:00Z"},
    ]

    releases = archive_performance.stable_published_releases(document)

    assert [release.tag for release in releases] == ["v0.4.0", "v0.3.0"]


def test_stable_published_releases_rejects_non_boolean_flags() -> None:
    document = [{"tagName": "v0.4.0", "isDraft": "false", "isPrerelease": False, "publishedAt": "2026-06-01T00:00:00Z"}]

    with pytest.raises(TypeError, match="boolean"):
        archive_performance.stable_published_releases(document)


def test_stable_published_releases_rejects_duplicate_tags() -> None:
    document = [
        {"tagName": "v0.4.0", "isDraft": False, "isPrerelease": False, "publishedAt": "2026-06-01T00:00:00Z"},
        {"tagName": "v0.4.0", "isDraft": False, "isPrerelease": False, "publishedAt": "2026-06-02T00:00:00Z"},
    ]

    with pytest.raises(ValueError, match="duplicate"):
        archive_performance.stable_published_releases(document)


def test_infer_release_uses_latest_for_an_unpublished_package() -> None:
    releases = (_release("v0.4.0"), _release("v0.3.0", 30))

    pair = archive_performance.resolve_release_pair(mode="infer-release", releases=releases, package_tag="v0.5.0")

    assert pair == archive_performance.ReportId("v0.5.0", "v0.4.0")


def test_infer_release_uses_previous_when_package_is_already_published() -> None:
    releases = (_release("v0.4.0"), _release("v0.3.0", 30), _release("v0.2.1", 60))

    pair = archive_performance.resolve_release_pair(mode="infer-release", releases=releases, package_tag="v0.4.0")

    assert pair == archive_performance.ReportId("v0.4.0", "v0.3.0")


@pytest.mark.parametrize("package_tag", ["v0.3.9", "v0.1.0"])
def test_infer_release_rejects_an_unpublished_package_not_newer_than_latest(package_tag: str) -> None:
    releases = (_release("v0.4.0"), _release("v0.3.0", 30))

    with pytest.raises(ValueError, match="must be newer"):
        archive_performance.resolve_release_pair(
            mode="infer-release",
            releases=releases,
            package_tag=package_tag,
        )


def test_published_latest_requires_two_stable_releases() -> None:
    with pytest.raises(RuntimeError, match="at least two"):
        archive_performance.resolve_release_pair(mode="published-latest", releases=(_release("v0.4.0"),), package_tag="v0.5.0")


def test_explicit_pair_requires_both_tags() -> None:
    with pytest.raises(ValueError, match="both required"):
        archive_performance.resolve_release_pair(
            mode="explicit",
            releases=(_release("v0.4.0"),),
            package_tag="v0.5.0",
            current_tag="v0.5.0",
        )


def test_github_assets_rejects_working_tree_resolution() -> None:
    with pytest.raises(ValueError, match="requires --published-latest"):
        archive_performance._validate_mode_combination("current-vs-latest", github_assets=True)


def test_published_latest_rejects_local_measurements() -> None:
    with pytest.raises(ValueError, match="reserved"):
        archive_performance._validate_mode_combination("published-latest", github_assets=False)


def test_current_vs_latest_promotion_is_rejected_before_release_lookup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def unexpected_release_lookup(_repo_root: Path) -> tuple[archive_performance.PublishedRelease, ...]:
        pytest.fail("release lookup must not run for an invalid mode combination")

    monkeypatch.setattr(archive_performance, "_published_releases", unexpected_release_lookup)

    status = archive_performance.main(
        [
            "--current-vs-latest",
            "--promote",
            "--repo-root",
            str(tmp_path),
        ]
    )

    assert status == 2
    assert "working-tree reports cannot be promoted" in capsys.readouterr().err


def test_comparison_csv_round_trip_preserves_both_samples_and_coverage() -> None:
    artifact = _comparison_artifact()

    text = archive_performance.serialize_comparison_csv(artifact.comparison_set)
    parsed = archive_performance.parse_comparison_csv(text)

    assert parsed == artifact.comparison_set
    assert text.splitlines()[1].split(",")[4] == "baseline_only"
    assert text.splitlines()[2].split(",")[4] == "current_only"
    assert text.splitlines()[3].split(",")[4] == "comparable"
    assert archive_performance.serialize_comparison_csv(parsed) == text


def test_comparison_csv_rejects_unsorted_or_incomplete_rows() -> None:
    text = archive_performance.serialize_comparison_csv(_comparison_artifact().comparison_set)
    lines = text.splitlines()
    unsorted = "\n".join((lines[0], lines[2], lines[1], lines[3], ""))

    with pytest.raises(ValueError, match="unique and sorted"):
        archive_performance.parse_comparison_csv(unsorted)

    fields = lines[3].split(",")
    fields[8] = ""
    incomplete = "\n".join((*lines[:3], ",".join(fields), ""))
    with pytest.raises(ValueError, match="missing current_point_ns"):
        archive_performance.parse_comparison_csv(incomplete)


def test_provenance_round_trip_rejects_a_mismatched_release_tag() -> None:
    artifact = _comparison_artifact()
    text = archive_performance.serialize_provenance(artifact)

    pair, settings, measurement, _csv_sha256 = archive_performance.parse_provenance(text)

    assert (pair, settings, measurement) == (artifact.pair, artifact.settings, artifact.measurement)
    document = json.loads(text)
    document["measurement"]["current"]["tag"] = "v0.6.0"
    with pytest.raises(ValueError, match="does not match"):
        archive_performance.parse_provenance(json.dumps(document))


def test_asset_provenance_round_trip_preserves_harness_hashes() -> None:
    artifact = _comparison_artifact()
    asset_artifact = archive_performance.ComparisonArtifact(
        pair=artifact.pair,
        comparison_set=artifact.comparison_set,
        settings=artifact.settings,
        measurement=archive_performance.MeasurementProvenance(
            mode="github-release-assets",
            working_tree_applied=False,
            current=_sample_provenance(
                artifact.pair.current_tag,
                "a",
                local=False,
                benchmark_harness_sha256="c" * 64,
            ),
            baseline=_sample_provenance(
                artifact.pair.baseline_tag,
                "b",
                local=False,
                benchmark_harness_sha256="d" * 64,
            ),
        ),
    )

    _pair, _settings, measurement, _csv_sha256 = archive_performance.parse_provenance(archive_performance.serialize_provenance(asset_artifact))

    assert measurement.current.benchmark_harness_sha256 == "c" * 64
    assert measurement.baseline.benchmark_harness_sha256 == "d" * 64


def test_load_comparison_artifact_rejects_csv_tampering(tmp_path: Path) -> None:
    csv_path = tmp_path / "release-performance.csv"
    archive_performance.save_comparison_artifact(_comparison_artifact(), csv_path)
    csv_path.write_text(csv_path.read_text(encoding="utf-8").replace("200.0", "201.0", 1), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match its provenance"):
        archive_performance.load_comparison_artifact(csv_path)


def test_save_comparison_artifact_rolls_back_csv_and_provenance_together(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / "release-performance.csv"
    sidecar = archive_performance.provenance_path(csv_path)
    csv_path.write_text("old csv\n", encoding="utf-8")
    sidecar.write_text("old provenance\n", encoding="utf-8")
    real_write = archive_performance._write_text

    def fail_sidecar(path: Path, text: str) -> None:
        if path == sidecar:
            msg = "injected provenance failure"
            raise OSError(msg)
        real_write(path, text)

    monkeypatch.setattr(archive_performance, "_write_text", fail_sidecar)

    with pytest.raises(OSError, match="injected provenance failure"):
        archive_performance.save_comparison_artifact(_comparison_artifact(), csv_path)

    assert csv_path.read_text(encoding="utf-8") == "old csv\n"
    assert sidecar.read_text(encoding="utf-8") == "old provenance\n"


def test_rerender_promotes_saved_artifact_without_release_lookup_or_measurement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / "target" / "bench-reports" / "release-performance.csv"
    artifact = _comparison_artifact()
    archive_performance.save_comparison_artifact(artifact, csv_path)

    def unexpected(*_args: object, **_kwargs: object) -> object:
        pytest.fail("rerender must not resolve releases or run benchmarks")

    monkeypatch.setattr(archive_performance, "_published_releases", unexpected)
    monkeypatch.setattr(archive_performance, "current_package_tag", unexpected)
    monkeypatch.setattr(archive_performance, "generate_local_artifact", unexpected)
    monkeypatch.setattr(archive_performance, "generate_github_asset_artifact", unexpected)

    status = archive_performance.main(
        [
            "--rerender",
            str(csv_path),
            "--promote",
            "--repo-root",
            str(tmp_path),
        ]
    )

    assert status == 0
    promoted = (tmp_path / "docs" / "PERFORMANCE.md").read_text(encoding="utf-8")
    assert archive_performance.parse_report_id(promoted) == artifact.pair
    assert "| `chain/shared` |" in promoted
    evidence_stem = artifact.pair.evidence_stem
    evidence_dir = tmp_path / "docs" / "archive" / "performance"
    assert (evidence_dir / f"{evidence_stem}.csv").read_text(encoding="utf-8") == archive_performance.serialize_comparison_csv(artifact.comparison_set)
    assert (evidence_dir / f"{evidence_stem}.provenance.json").read_text(encoding="utf-8") == archive_performance.serialize_provenance(artifact)
    assert f"archive/performance/{evidence_stem}.csv" in promoted


def test_rerender_without_path_uses_tracked_curated_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _comparison_artifact()
    evidence_dir = tmp_path / "docs" / "archive" / "performance"
    csv_path = evidence_dir / f"{artifact.pair.evidence_stem}.csv"
    archive_performance.save_comparison_artifact(artifact, csv_path)
    current_report = tmp_path / "docs" / "PERFORMANCE.md"
    current_report.parent.mkdir(parents=True, exist_ok=True)
    current_report.write_text(
        archive_performance.render_report(artifact.comparison_set, artifact.settings),
        encoding="utf-8",
    )

    def unexpected(*_args: object, **_kwargs: object) -> object:
        pytest.fail("rerender must not resolve releases or run benchmarks")

    monkeypatch.setattr(archive_performance, "_published_releases", unexpected)
    monkeypatch.setattr(archive_performance, "current_package_tag", unexpected)
    monkeypatch.setattr(archive_performance, "generate_local_artifact", unexpected)
    monkeypatch.setattr(archive_performance, "generate_github_asset_artifact", unexpected)

    status = archive_performance.main(["--rerender", "--promote", "--repo-root", str(tmp_path)])

    assert status == 0
    promoted = current_report.read_text(encoding="utf-8")
    assert archive_performance.parse_report_id(promoted) == artifact.pair
    assert f"archive/performance/{artifact.pair.evidence_stem}.csv" in promoted


def test_invalid_rerender_combination_is_rejected_before_release_lookup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def unexpected(*_args: object, **_kwargs: object) -> object:
        pytest.fail("invalid rerender arguments must fail before release lookup")

    monkeypatch.setattr(archive_performance, "_published_releases", unexpected)

    status = archive_performance.main(
        [
            "--rerender",
            "release-performance.csv",
            "--infer-release",
            "--repo-root",
            str(tmp_path),
        ]
    )

    assert status == 2
    assert "--rerender cannot be combined" in capsys.readouterr().err


def test_generation_saves_and_reloads_artifacts_before_rendering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair = archive_performance.ReportId("v0.5.0", "v0.4.0")
    artifact = _comparison_artifact(pair)
    csv_path = tmp_path / "measurements.csv"
    report_path = tmp_path / "report.md"
    monkeypatch.setattr(archive_performance, "current_package_tag", lambda _root: pair.current_tag)
    monkeypatch.setattr(archive_performance, "generate_local_artifact", lambda *_args, **_kwargs: artifact)

    status = archive_performance.main(
        [
            pair.current_tag,
            pair.baseline_tag,
            "--measurements-output",
            str(csv_path),
            "--output",
            str(report_path),
            "--repo-root",
            str(tmp_path),
        ]
    )

    assert status == 0
    assert archive_performance.load_comparison_artifact(csv_path) == artifact
    assert "| `chain/shared` |" in report_path.read_text(encoding="utf-8")


def test_source_digest_changes_with_measured_rust_inputs(tmp_path: Path) -> None:
    files = {
        "Cargo.toml": "[package]\nname = 'fixture'\n",
        "Cargo.lock": "version = 4\n",
        "rust-toolchain.toml": "[toolchain]\nchannel = 'stable'\n",
        "benches/stepping.rs": "fn benchmark() {}\n",
        "src/lib.rs": "pub fn sample() {}\n",
    }
    for relative, content in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    original = archive_performance._source_digest(tmp_path)

    (tmp_path / "src" / "lib.rs").write_text("pub fn changed() {}\n", encoding="utf-8")

    assert archive_performance._source_digest(tmp_path) != original


def test_source_digest_rejects_source_removed_after_enumeration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    files = {
        "Cargo.toml": "[package]\nname = 'fixture'\n",
        "Cargo.lock": "version = 4\n",
        "rust-toolchain.toml": "[toolchain]\nchannel = 'stable'\n",
        "benches/stepping.rs": "fn benchmark() {}\n",
        "src/lib.rs": "pub fn sample() {}\n",
    }
    for relative, content in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    original_rglob = Path.rglob

    def remove_sources_after_enumeration(path: Path, pattern: str) -> Iterator[Path]:
        sources = list(original_rglob(path, pattern))
        for source in sources:
            source.unlink()
        return iter(sources)

    monkeypatch.setattr(Path, "rglob", remove_sources_after_enumeration)

    with pytest.raises(FileNotFoundError, match="cannot hash benchmark inputs; missing") as error:
        archive_performance._source_digest(tmp_path)
    assert str(tmp_path / "src" / "lib.rs") in str(error.value)


def test_promote_report_archives_the_previous_pair_and_updates_index(tmp_path: Path) -> None:
    current = tmp_path / "docs" / "PERFORMANCE.md"
    archive_dir = tmp_path / "docs" / "archive" / "performance"
    current.parent.mkdir(parents=True)
    current.write_text(_report("v0.4.0", "v0.3.0", "old"), encoding="utf-8")
    artifact = _comparison_artifact()

    archive_performance.promote_report(
        artifact=artifact,
        current_path=current,
        archive_dir=archive_dir,
    )

    assert "| `chain/shared` |" in current.read_text(encoding="utf-8")
    assert "old" in (archive_dir / "v0.4.0-vs-v0.3.0.md").read_text(encoding="utf-8")
    assert (archive_dir / "v0.5.0-vs-v0.4.0.csv").read_text(encoding="utf-8") == archive_performance.serialize_comparison_csv(artifact.comparison_set)
    assert (archive_dir / "v0.5.0-vs-v0.4.0.provenance.json").read_text(encoding="utf-8") == archive_performance.serialize_provenance(artifact)
    index = (archive_dir / "README.md").read_text(encoding="utf-8")
    assert "[v0.4.0-vs-v0.3.0](v0.4.0-vs-v0.3.0.md)" in index


def test_promote_report_rebases_evidence_links_when_archiving(tmp_path: Path) -> None:
    current = tmp_path / "docs" / "PERFORMANCE.md"
    archive_dir = tmp_path / "docs" / "archive" / "performance"
    first = _comparison_artifact(archive_performance.ReportId("v0.5.0", "v0.4.0"))
    second = _comparison_artifact(archive_performance.ReportId("v0.6.0", "v0.5.0"))

    archive_performance.promote_report(artifact=first, current_path=current, archive_dir=archive_dir)
    archive_performance.promote_report(artifact=second, current_path=current, archive_dir=archive_dir)

    archived = (archive_dir / "v0.5.0-vs-v0.4.0.md").read_text(encoding="utf-8")
    assert "](v0.5.0-vs-v0.4.0.csv)" in archived
    assert "](v0.5.0-vs-v0.4.0.provenance.json)" in archived
    assert "archive/performance/v0.5.0-vs-v0.4.0" not in archived


def test_promote_report_rolls_back_every_output_if_index_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = tmp_path / "docs" / "PERFORMANCE.md"
    archive_dir = tmp_path / "docs" / "archive" / "performance"
    index = archive_dir / "README.md"
    current.parent.mkdir(parents=True)
    archive_dir.mkdir(parents=True)
    old_report = _report("v0.4.0", "v0.3.0", "old")
    old_index = "# Existing index\n"
    current.write_text(old_report, encoding="utf-8")
    index.write_text(old_index, encoding="utf-8")
    archive = archive_dir / "v0.4.0-vs-v0.3.0.md"
    artifact = _comparison_artifact()
    real_write = archive_performance._write_text
    failed = False

    def fail_index_once(path: Path, text: str) -> None:
        nonlocal failed
        if path == index and not failed:
            failed = True
            msg = "injected index write failure"
            raise OSError(msg)
        real_write(path, text)

    monkeypatch.setattr(archive_performance, "_write_text", fail_index_once)

    with pytest.raises(OSError, match="injected index write failure"):
        archive_performance.promote_report(
            artifact=artifact,
            current_path=current,
            archive_dir=archive_dir,
        )

    assert current.read_text(encoding="utf-8") == old_report
    assert index.read_text(encoding="utf-8") == old_index
    assert not archive.exists()
    assert not (archive_dir / "v0.5.0-vs-v0.4.0.csv").exists()
    assert not (archive_dir / "v0.5.0-vs-v0.4.0.provenance.json").exists()

    monkeypatch.setattr(archive_performance, "_write_text", real_write)
    archive_performance.promote_report(
        artifact=artifact,
        current_path=current,
        archive_dir=archive_dir,
    )

    assert "| `chain/shared` |" in current.read_text(encoding="utf-8")
    assert archive.read_text(encoding="utf-8") == old_report
    assert "[v0.4.0-vs-v0.3.0](v0.4.0-vs-v0.3.0.md)" in index.read_text(encoding="utf-8")


def test_promote_report_preserves_an_existing_archive(tmp_path: Path) -> None:
    current = tmp_path / "PERFORMANCE.md"
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    current.write_text(_report("v0.4.0", "v0.3.0", "old current"), encoding="utf-8")
    archive = archive_dir / "v0.4.0-vs-v0.3.0.md"
    archive.write_text("preserved archive\n", encoding="utf-8")

    archive_performance.promote_report(
        artifact=_comparison_artifact(),
        current_path=current,
        archive_dir=archive_dir,
    )

    assert archive.read_text(encoding="utf-8") == "preserved archive\n"


def test_promote_report_replaces_active_pair_evidence_atomically(tmp_path: Path) -> None:
    current = tmp_path / "PERFORMANCE.md"
    archive_dir = tmp_path / "archive"
    first = _comparison_artifact()
    archive_performance.promote_report(artifact=first, current_path=current, archive_dir=archive_dir)
    csv_path = archive_dir / f"{first.pair.evidence_stem}.csv"
    original_csv = csv_path.read_text(encoding="utf-8")
    revised = _comparison_artifact(current_point=81.0)

    archive_performance.promote_report(artifact=revised, current_path=current, archive_dir=archive_dir)

    assert csv_path.read_text(encoding="utf-8") != original_csv
    assert archive_performance.load_comparison_artifact(csv_path) == revised


def test_promote_report_rejects_replacing_historical_pair_evidence(tmp_path: Path) -> None:
    current = tmp_path / "PERFORMANCE.md"
    archive_dir = tmp_path / "archive"
    historical = _comparison_artifact()
    current_pair = _comparison_artifact(archive_performance.ReportId("v0.6.0", "v0.5.0"))
    archive_performance.promote_report(artifact=historical, current_path=current, archive_dir=archive_dir)
    archive_performance.promote_report(artifact=current_pair, current_path=current, archive_dir=archive_dir)
    csv_path = archive_dir / f"{historical.pair.evidence_stem}.csv"
    provenance_path = archive_performance.provenance_path(csv_path)
    original_csv = csv_path.read_text(encoding="utf-8")
    original_provenance = provenance_path.read_text(encoding="utf-8")
    revised = _comparison_artifact(current_point=81.0)

    with pytest.raises(ValueError, match="refusing to replace immutable release benchmark evidence"):
        archive_performance.promote_report(artifact=revised, current_path=current, archive_dir=archive_dir)

    assert csv_path.read_text(encoding="utf-8") == original_csv
    assert provenance_path.read_text(encoding="utf-8") == original_provenance


def test_promote_report_rejects_an_internally_mismatched_artifact(tmp_path: Path) -> None:
    artifact = _comparison_artifact()
    mismatched = archive_performance.ComparisonArtifact(
        pair=archive_performance.ReportId("v0.6.0", "v0.5.0"),
        comparison_set=artifact.comparison_set,
        settings=artifact.settings,
        measurement=artifact.measurement,
    )

    with pytest.raises(ValueError, match="does not match"):
        archive_performance.promote_report(
            artifact=mismatched,
            current_path=tmp_path / "PERFORMANCE.md",
            archive_dir=tmp_path / "archive",
        )


def test_generated_working_tree_report_can_be_promoted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair = archive_performance.ReportId("v0.5.0", "v0.4.0")
    applied: list[tuple[Path, Path]] = []

    @contextmanager
    def fake_worktree(
        _repo_root: Path,
        parent: Path,
        name: str,
        _reference: str,
    ) -> Iterator[Path]:
        checkout = parent / name
        checkout.mkdir()
        yield checkout

    def fake_benchmark(checkout: Path, *, save_baseline: str | None = None) -> None:
        sample = save_baseline or "new"
        _write_sample(checkout / "target" / "criterion", "chain/step_by_value", sample)

    def fake_apply(repo_root: Path, worktree: Path) -> None:
        applied.append((repo_root, worktree))

    monkeypatch.setattr(archive_performance, "_ensure_local_tag", lambda *_args: None)
    monkeypatch.setattr(archive_performance, "temporary_worktree", fake_worktree)
    monkeypatch.setattr(archive_performance, "_run_stepping_benchmark", fake_benchmark)
    monkeypatch.setattr(archive_performance, "apply_current_tree", fake_apply)
    monkeypatch.setattr(
        archive_performance,
        "_local_measurement_provenance",
        lambda *_args, **_kwargs: _measurement(pair, working_tree=True),
    )
    monkeypatch.setattr(archive_performance, "_local_measurement_context", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(
        archive_performance,
        "run_git_command",
        lambda args, **_kwargs: subprocess.CompletedProcess(args, 0, stdout="abc1234\n"),
    )

    artifact = archive_performance.generate_local_artifact(
        tmp_path,
        pair,
        current_reference="HEAD",
        apply_working_tree=True,
    )
    report = bench_compare.render_report(artifact.comparison_set, artifact.settings)
    current = tmp_path / "docs" / "PERFORMANCE.md"
    archive_performance.promote_report(
        artifact=artifact,
        current_path=current,
        archive_dir=tmp_path / "docs" / "archive" / "performance",
    )

    assert applied
    assert archive_performance.parse_report_id(current.read_text(encoding="utf-8")) == pair
    assert "**markov-chain-monte-carlo** v0.5.0 working tree" in report


@pytest.mark.parametrize(
    "relative_name",
    [
        pytest.param("nested/fixture name.txt", id="space"),
        pytest.param(
            "nested/fixture\nname.txt",
            id="newline",
            marks=pytest.mark.skipif(os.name == "nt", reason="Windows filenames cannot contain newlines"),
        ),
    ],
)
def test_apply_current_tree_applies_tracked_changes_and_copies_untracked_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_name: str,
) -> None:
    repo_root = tmp_path / "source"
    worktree = tmp_path / "worktree"
    untracked = repo_root / relative_name
    untracked.parent.mkdir(parents=True)
    untracked.write_text("fixture\n", encoding="utf-8")
    worktree.mkdir()
    applied: list[tuple[list[str], str, Path]] = []
    git_calls: list[list[str]] = []

    def fake_git(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        git_calls.append(args)
        stdout = "tracked patch\n" if "diff" in args else f"{relative_name}\0"
        return subprocess.CompletedProcess(args, 0, stdout=stdout)

    def fake_git_with_input(args: list[str], input_text: str, *, cwd: Path, **_kwargs: object) -> object:
        applied.append((args, input_text, cwd))
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(archive_performance, "run_git_command", fake_git)
    monkeypatch.setattr(archive_performance, "run_git_command_with_input", fake_git_with_input)

    archive_performance.apply_current_tree(repo_root, worktree)

    assert applied == [
        (["apply", "--binary", "--whitespace=nowarn"], "tracked patch\n", worktree),
    ]
    assert git_calls[-1] == ["ls-files", "--others", "--exclude-standard", "-z", "--"]
    assert (worktree / relative_name).read_text(encoding="utf-8") == "fixture\n"


def test_apply_current_tree_rejects_untracked_symlinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "source"
    worktree = tmp_path / "worktree"
    repo_root.mkdir()
    worktree.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    link = repo_root / "link.txt"
    link.symlink_to(outside)

    def fake_git(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        stdout = "" if "diff" in args else "link.txt\0"
        return subprocess.CompletedProcess(args, 0, stdout=stdout)

    monkeypatch.setattr(archive_performance, "run_git_command", fake_git)

    with pytest.raises(ValueError, match="must be a regular file"):
        archive_performance.apply_current_tree(repo_root, worktree)

    assert not (worktree / "link.txt").exists()


def test_temporary_worktree_removes_checkout_after_body_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_git(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="")

    monkeypatch.setattr(archive_performance, "run_git_command", fake_git)

    msg = "benchmark failed"
    with pytest.raises(RuntimeError, match="benchmark failed"), archive_performance.temporary_worktree(tmp_path, tmp_path, "checkout", "HEAD"):
        raise RuntimeError(msg)

    assert calls == [
        ["worktree", "add", "--detach", str(tmp_path / "checkout"), "HEAD"],
        ["worktree", "remove", "--force", str(tmp_path / "checkout")],
    ]


def test_copy_criterion_sample_renames_each_benchmark_sample(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    _write_sample(source, "chain/step_by_value", "v0.4.0")
    _write_sample(source, "sampler/run_by_value_100", "v0.4.0")

    copied = archive_performance.copy_criterion_sample(
        source_criterion=source,
        destination_criterion=destination,
        source_sample="v0.4.0",
        destination_sample="new",
    )

    assert copied == 2
    assert (destination / "chain" / "step_by_value" / "new" / "sample.json").is_file()
    assert (destination / "sampler" / "run_by_value_100" / "new" / "estimates.json").is_file()


def test_copy_criterion_sample_fails_when_asset_has_no_requested_baseline(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no Criterion sample"):
        archive_performance.copy_criterion_sample(
            source_criterion=tmp_path / "missing",
            destination_criterion=tmp_path / "destination",
            source_sample="v0.4.0",
            destination_sample="new",
        )


def test_generate_github_asset_report_renames_current_sample_and_renders_release_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair = archive_performance.ReportId("v0.5.0", "v0.4.0")
    archives = {
        pair.current_tag: _write_release_archive(tmp_path, pair.current_tag, 80.0, "a" * 40),
        pair.baseline_tag: _write_release_archive(tmp_path, pair.baseline_tag, 100.0, "b" * 40),
    }

    def fake_download(_repo_root: Path, tag: str, _destination: Path) -> Path:
        return archives[tag]

    monkeypatch.setattr(archive_performance, "_download_release_asset", fake_download)

    report = archive_performance.generate_github_asset_report(tmp_path, pair)

    assert "**markov-chain-monte-carlo** v0.5.0 · `aaaaaaa`" in report
    assert "Comparison against baseline **v0.4.0**:" in report
    assert "Source mode: durable GitHub Release Criterion assets" in report
    assert "Benchmark harness identity is unavailable" in report
    assert "| `chain/step_by_value` | 100.00 ns | 80.00 ns | +20.00% | 1.25x faster |" in report


def test_release_metadata_validates_tag_and_measurement_context(tmp_path: Path) -> None:
    metadata = {
        "schema": 1,
        "tag": "v0.5.0",
        "commit": "a" * 40,
        "operating_system": "Linux",
        "architecture": "x86_64",
        "rustc": "rustc 1.97.1",
        "criterion_version": "0.8.2",
    }
    (tmp_path / ".mcmc-release-metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    parsed = archive_performance._release_metadata(tmp_path, "v0.5.0")

    assert parsed.commit == "a" * 40
    assert parsed.criterion_version == "0.8.2"
    assert parsed.benchmark_harness_sha256 is None


def test_release_metadata_schema_two_requires_a_valid_harness_hash(tmp_path: Path) -> None:
    metadata = {
        "schema": 2,
        "tag": "v0.5.0",
        "commit": "a" * 40,
        "operating_system": "Linux",
        "architecture": "x86_64",
        "rustc": "rustc 1.98.0",
        "criterion_version": "0.8.2",
        "benchmark_harness_sha256": "b" * 64,
    }
    path = tmp_path / ".mcmc-release-metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")

    parsed = archive_performance._release_metadata(tmp_path, "v0.5.0")

    assert parsed.benchmark_harness_sha256 == "b" * 64
    metadata.pop("benchmark_harness_sha256")
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="do not match schema 2"):
        archive_performance._release_metadata(tmp_path, "v0.5.0")

    metadata["benchmark_harness_sha256"] = "not-a-digest"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        archive_performance._release_metadata(tmp_path, "v0.5.0")


def test_asset_measurement_context_reports_matching_and_changed_harnesses() -> None:
    current = archive_performance.ReleaseMetadata("v0.5.0", "a" * 40, "Linux", "x86_64", "rustc 1.98.0", "0.8.2", "c" * 64)
    matching = archive_performance.ReleaseMetadata("v0.4.0", "b" * 40, "Linux", "x86_64", "rustc 1.98.0", "0.8.2", "c" * 64)
    changed = archive_performance.ReleaseMetadata("v0.4.0", "b" * 40, "Linux", "x86_64", "rustc 1.98.0", "0.8.2", "d" * 64)

    matching_context = archive_performance._asset_measurement_context(current, matching)
    changed_context = archive_performance._asset_measurement_context(current, changed)

    assert "current `cccccccccccc`; baseline `cccccccccccc`" in matching_context[-1]
    assert "workload contract" in changed_context[-1]


def test_release_metadata_rejects_a_mismatched_tag(tmp_path: Path) -> None:
    metadata = {
        "schema": 1,
        "tag": "v0.4.0",
        "commit": "a" * 40,
        "operating_system": "Linux",
        "architecture": "x86_64",
        "rustc": "rustc 1.97.1",
        "criterion_version": "0.8.2",
    }
    (tmp_path / ".mcmc-release-metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match"):
        archive_performance._release_metadata(tmp_path, "v0.5.0")


def test_safe_extract_tar_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar.gz"
    payload = b"unsafe"
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo("../outside.txt")
        member.size = len(payload)
        tar.addfile(member, io.BytesIO(payload))

    with pytest.raises(ValueError, match="outside its root"):
        archive_performance.safe_extract_tar(archive, tmp_path / "extract")

    assert not (tmp_path / "outside.txt").exists()


def test_safe_extract_tar_rejects_links(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe-link.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo("criterion/link")
        member.type = tarfile.SYMTYPE
        member.linkname = "../../outside.txt"
        tar.addfile(member)

    with pytest.raises(ValueError, match="unsupported entry"):
        archive_performance.safe_extract_tar(archive, tmp_path / "extract")


def test_safe_extract_tar_rejects_oversized_archive_before_creating_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "large.tar.gz"
    with tarfile.open(archive, "w:gz"):
        pass
    monkeypatch.setattr(archive_performance, "_MAX_RELEASE_ARCHIVE_SIZE_BYTES", 0)
    destination = tmp_path / "extract"

    with pytest.raises(ValueError, match="archive is too large"):
        archive_performance.safe_extract_tar(archive, destination)

    assert not destination.exists()


def test_safe_extract_tar_rejects_too_many_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "many.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.addfile(tarfile.TarInfo("criterion"))
        tar.addfile(tarfile.TarInfo("criterion/extra"))
    monkeypatch.setattr(archive_performance, "_MAX_RELEASE_ARCHIVE_MEMBER_COUNT", 1)
    destination = tmp_path / "extract"

    with pytest.raises(ValueError, match="too many entries"):
        archive_performance.safe_extract_tar(archive, destination)

    assert not destination.exists()


def test_safe_extract_tar_rejects_excessive_expanded_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "expanded.tar.gz"
    payload = b"123456"
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo("criterion/data.txt")
        member.size = len(payload)
        tar.addfile(member, io.BytesIO(payload))
    monkeypatch.setattr(archive_performance, "_MAX_RELEASE_ARCHIVE_CONTENT_BYTES", 5)
    destination = tmp_path / "extract"

    with pytest.raises(ValueError, match="expands beyond"):
        archive_performance.safe_extract_tar(archive, destination)

    assert not destination.exists()


def test_command_failure_preserves_stdout_and_stderr() -> None:
    error = subprocess.CalledProcessError(7, ["cargo", "bench"], output="partial output\n", stderr="compiler failed\n")

    message = archive_performance._format_command_failure(error)

    assert "exit 7: cargo bench" in message
    assert "partial output" in message
    assert "compiler failed" in message
