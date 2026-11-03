"""MCMC workload lifecycles and declared measurement policy."""

import re
import shutil
import subprocess
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import pytest
from research_repo_tools import performance_release
from research_repo_tools.common_measurement import CompleteMeasurementPlan, DiscoveryPolicy
from research_repo_tools.just_inspect import dry_run
from research_repo_tools.measurement import MeasurementConfig, load_measurement
from research_repo_tools.performance_release import load_performance_release
from research_repo_tools.release_discovery import PublishedRelease
from research_repo_tools.release_metadata import PackageInfo
from research_repo_tools.report_inventory import check_report

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_HARNESS = REPO_ROOT / "benches" / "stepping.rs"
STEADY_STATE_BENCHMARKS = {
    "chain/step_by_value": ("scalar_chain", "StdRng::seed_from_u64"),
    "chain/step_mut_accept": ("spin_chain", "StdRng::seed_from_u64"),
    "chain/step_mut_reject_rollback": ("spin_chain", "StdRng::seed_from_u64"),
    "chain/step_delayed_accept_reflection": ("scalar_chain", "StdRng::seed_from_u64"),
    "chain/step_delayed_reject_reflection": ("scalar_chain", "StdRng::seed_from_u64"),
    "chain/step_delayed_no_plan": ("scalar_chain", "StdRng::seed_from_u64"),
    "sampler/run_by_value_100": ("Sampler::new", "StdRng::seed_from_u64"),
    "sampler/run_mut_100": ("Sampler::new", "StdRng::seed_from_u64"),
    "sampler/run_delayed_reflection_100": ("Sampler::new", "StdRng::seed_from_u64"),
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
    start, end = match.span("body")
    return source[start:end]


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


def test_measurement_configuration_preserves_the_release_signal() -> None:
    config = load_measurement(REPO_ROOT, "tooling/benchmark.toml")
    assert isinstance(config, MeasurementConfig)
    assert config.command == ("cargo", "bench", "--locked", "--bench", "stepping")
    assert config.sample == "new"
    assert config.statistic == "median"
    assert config.unit == "ns"
    assert config.sources == ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml", "src/**/*.rs", "benches/stepping.rs")
    assert config.harness == ("benches/stepping.rs",)
    assert config.compatible == ("context.os", "context.architecture", "context.cpu")
    assert dict(config.context) == {"suite": "stepping", "scope": "release-signal"}


def test_complete_release_declaration_covers_every_native_workload() -> None:
    plan = load_measurement(REPO_ROOT, "tooling/measurement-release-signal.toml")
    assert isinstance(plan, CompleteMeasurementPlan)
    assert (plan.schema, plan.harness_mode, plan.backend) == (4, "common", "export")
    source = BENCHMARK_HARNESS.read_text(encoding="utf-8")
    unthinned = set(STEADY_STATE_BENCHMARKS) | FRESH_BATCH_BENCHMARKS
    groups = set(re.findall(r'c\.benchmark_group\("([^"]+)"', source))
    assert groups == {"sampler/run_by_value_thinned_100", "sampler/run_mut_thinned_100", "sampler/run_delayed_thinned_100"}
    intervals = re.search(r"const THIN_INTERVALS: \[usize; 3\] = \[([^]]+)\];", source)
    assert intervals is not None
    assert tuple(int(value.strip()) for value in intervals[1].split(",")) == (1, 2, 16)
    expected = unthinned | {f"{group}/{interval}" for group in groups for interval in (1, 2, 16)}
    assert len(expected) == 22
    assert ".sample_size(" not in source
    assert "config =" not in source
    assert plan.measurement.harness == ("benches/stepping.rs",)
    assert plan.measurement.workloads == ("benches/stepping.rs",)
    assert plan.saved_sample == "retained"
    assert plan.raw_files == ("tukey.json",)
    for phase in (plan.baseline, plan.current):
        assert isinstance(phase.policy, DiscoveryPolicy)
        assert set(phase.policy.discovery.required) == expected
        assert phase.policy.sample_count == 100
        assert phase.policy.statistics == ("mean", "median")
        assert phase.policy.confidence_level == 0.95
        assert phase.command == ("cargo", "bench", "--locked", "--bench", "stepping", "--", "--save-baseline=retained", "--noplot")
        assert phase.gate == ("cargo", "bench", "--locked", "--bench", "stepping", "--", "--test")
        assert len(phase.policy.discovery.suites) == 1
        assert phase.policy.discovery.suites[0].command == ("cargo", "bench", "--locked", "--bench", "stepping", "--", "--list")
    publication = tomllib.loads((REPO_ROOT / "tooling/performance-publication.toml").read_text(encoding="utf-8"))
    assert {row["key"] for row in publication["rows"]} == unthinned
    assert len(publication["rows"]) == 13
    for series in publication["series"]:
        assert series["rows"] == {name: name for name in unthinned}


def test_preparation_infers_both_releases_from_package_and_published_history(monkeypatch: pytest.MonkeyPatch) -> None:
    workflow = load_performance_release(REPO_ROOT, "tooling/performance-release.toml")
    assert (workflow.mode, workflow.order, workflow.current, workflow.baseline) == ("infer-release", "published", None, None)
    monkeypatch.setattr(performance_release, "read_package_info", lambda _root: PackageInfo("markov-chain-monte-carlo", "0.5.3"))
    # Out-of-order input proves selection comes from publication chronology.
    releases = (PublishedRelease("v0.5.2", datetime(2026, 10, 28, tzinfo=UTC)), PublishedRelease("v0.5.1", datetime(2026, 10, 21, tzinfo=UTC)))
    monkeypatch.setattr(performance_release, "published_releases", lambda _root, **_kwargs: tuple(reversed(releases)))
    pair, working_tree = performance_release.select_measurement_pair(
        REPO_ROOT, workflow.measurement, repository=workflow.repository, mode=workflow.mode, order=workflow.order
    )
    assert (pair.current, pair.baseline, working_tree) == ("v0.5.3", "v0.5.2", True)


def test_new_report_inventory_preserves_legacy_evidence_without_promoting_it(tmp_path: Path) -> None:
    workflow = load_performance_release(REPO_ROOT, "tooling/performance-release.toml")
    for relative in (workflow.measurement, workflow.report):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((REPO_ROOT / relative).read_bytes())
    legacy = tmp_path / "docs/performance/v1/README.md"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"Retained historical series\r\n")
    assert check_report(tmp_path, workflow.report).status == "uninitialized"
    assert legacy.read_bytes() == b"Retained historical series\r\n"
    assert not (tmp_path / "docs/performance/current.md").exists()
    partial = tmp_path / "docs/performance/runs/partial"
    partial.mkdir(parents=True)
    with pytest.raises(ValueError, match="orphaned inventory"):
        check_report(tmp_path, workflow.report)


@pytest.mark.parametrize(
    ("recipe", "operation", "arguments"),
    [("performance-prepare", "prepare", []), ("performance-release", "release", []), ("performance-readme", "readme", ["--preview"])],
)
def test_release_recipes_keep_preparation_measurement_and_publication_separate(recipe: str, operation: str, arguments: list[str]) -> None:
    commands = dry_run(REPO_ROOT, recipe, arguments).stderr.splitlines()
    performance = [line for line in commands if "research-repo-tools performance " in line]
    assert len(performance) == 1
    executable = shutil.which("bash")
    assert executable is not None
    provider = 'uv() { printf "%s\\0" "$@"; }\n'
    result = subprocess.run(  # noqa: S603 - inert uv provider captures real shell argument forwarding without acquisition.
        [executable, "-c", provider + performance[0], recipe, *arguments], check=True, capture_output=True
    )
    command = result.stdout.decode("utf-8").rstrip("\0").split("\0")
    start = command.index("performance")
    expected = ["performance", operation, "tooling/performance-release.toml"]
    if operation == "release":
        expected += ["--progress-cadence", "300"]
    assert command[start:] == [*expected, *arguments]
    assert not {"--allow-git-mutations", "--fetch-tags", "--current", "--baseline"} & set(command)
