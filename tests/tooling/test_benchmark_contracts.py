"""MCMC workload lifecycles and declared measurement policy."""

import re
from pathlib import Path

from research_repo_tools.measurement import load_measurement

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
    return match.group("body")


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
    assert config.command == ("cargo", "bench", "--locked", "--bench", "stepping")
    assert config.sample == "new"
    assert config.statistic == "median"
    assert config.unit == "ns"
    assert config.sources == ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml", "src/**/*.rs", "benches/stepping.rs")
    assert config.harness == ("benches/stepping.rs",)
    assert config.compatible == ("context.os", "context.architecture", "context.cpu")
    assert dict(config.context) == {"suite": "stepping", "scope": "release-signal"}
