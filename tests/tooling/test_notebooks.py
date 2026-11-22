"""Consumer notebook policies exercised through the pinned shared public CLI."""

import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from research_repo_tools.cli import main
from research_repo_tools.notebook_testing import isolated_project

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from research_repo_tools.notebook_testing import NotebookProject

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[NotebookProject]:
    """Keep MCMC input selection explicit while borrowing the locked interpreter."""
    for name in ("MCMC_TRACE_PATH", "MCMC_REPO_ROOT", "MCMC_NOTEBOOK_OUTPUT_DIR", "MCMC_DIAGNOSTICS_PATH"):
        monkeypatch.delenv(name, raising=False)
    with isolated_project(
        REPO_ROOT,
        [Path("notebooks/ising_trace_analysis.ipynb"), Path("notebooks/diagnostic_plots.ipynb")],
        parent=tmp_path,
        environment=Path(sys.prefix),
    ) as fixture:
        yield fixture


def write_ising_trace(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"chain_id,step,accepted,proposed,log_prob,energy,magnetization\n0,0,false,false,-1.0,-2.0,0.5\n")


def test_notebook_lint_rejects_dependency_installation(project: NotebookProject, capsys: pytest.CaptureFixture[str]) -> None:
    """MCMC notebooks consume the locked environment rather than installing dependencies."""
    notebook = project.root / "notebooks/ising_trace_analysis.ipynb"
    payload = json.loads(notebook.read_bytes())
    payload["cells"].append(
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "forbidden-install",
            "metadata": {},
            "outputs": [],
            "source": ["%pip install numpy\n"],
        }
    )
    notebook.write_text(json.dumps(payload), encoding="utf-8", newline="\n")
    before = notebook.read_bytes()
    assert main(["--root", str(project.root), "notebooks", "lint", str(notebook)]) == 1
    output = capsys.readouterr()
    assert "install" in output.out + output.err
    assert notebook.read_bytes() == before


def test_explicit_repository_root_never_falls_back_to_ambient_trace(project: NotebookProject, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_root = project.root
    configured_root = tmp_path / "configured"
    notebook = project.root / "notebooks/ising_trace_analysis.ipynb"
    (configured_root / "examples").mkdir(parents=True)
    (configured_root / "Cargo.toml").write_bytes(b"[package]\nname = 'fixture'\n")
    (configured_root / "examples" / "ising_1d.rs").write_bytes(b"// fixture\n")
    ambient_trace = runtime_root / "target" / "ising_1d_trace.csv"
    write_ising_trace(ambient_trace)
    monkeypatch.setenv("MCMC_REPO_ROOT", str(configured_root))
    monkeypatch.delenv("MCMC_TRACE_PATH", raising=False)
    monkeypatch.delenv("MCMC_NOTEBOOK_OUTPUT_DIR", raising=False)
    before = notebook.read_bytes()

    execution = project.execute(notebook)
    assert execution.returncode == 1

    report = execution.report
    assert report["status"] == "failed"
    assert report["failed_cell"] == {"index": 2, "id": "load-and-validate-trace"}
    assert str(configured_root / "target/ising_1d_trace.csv") in report["error"]["message"]
    assert str(ambient_trace) not in report["error"]["message"]
    assert notebook.read_bytes() == before
    assert not (configured_root / "target/notebooks/ising_energy_trace.png").exists()


def test_explicit_trace_preserves_source_and_writes_only_selected_figure(project: NotebookProject, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = project.root
    monkeypatch.setenv("MCMC_DIAGNOSTICS_PATH", str(tmp_path / "ambient-missing.json"))
    notebook = project.root / "notebooks/ising_trace_analysis.ipynb"
    trace_path = tmp_path / "mounted-input" / "trace.csv"
    figure_root = tmp_path / "figure-output"
    write_ising_trace(trace_path)
    monkeypatch.setenv("MCMC_TRACE_PATH", str(trace_path))
    monkeypatch.setenv("MCMC_REPO_ROOT", str(tmp_path / "invalid-root-is-ignored"))
    monkeypatch.setenv("MCMC_NOTEBOOK_OUTPUT_DIR", str(figure_root))
    monkeypatch.setenv("MPLBACKEND", "TkAgg")
    before = notebook.read_bytes()

    execution = project.execute(notebook, env={"MCMC_DIAGNOSTICS_PATH": None})
    assert execution.returncode == 0

    assert notebook.read_bytes() == before
    assert (figure_root / "ising_energy_trace.png").is_file()
    assert not (trace_path.parent / "notebooks/ising_energy_trace.png").exists()
    assert not (root / "target/notebooks/ising_energy_trace.png").exists()
    assert not (project.artifacts / "ising_energy_trace.png").exists()


def test_autocorrelation_known_values_and_unavailable_inputs(project: NotebookProject, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the actual notebook cells, including chain/spacing boundaries."""
    notebook = project.root / "notebooks/ising_trace_analysis.ipynb"
    trace_path = tmp_path / "target/ising_1d_trace.csv"
    trace_path.parent.mkdir(parents=True)
    lines = ["chain_id,step,accepted,proposed,log_prob,energy,magnetization\n"]
    fixtures = [
        (0, [1, 2, 3, 4], [1, 2, 3, 4], [3, -1, -1, -1]),
        (1, [1, 2, 3, 4], [2, 2, 2, 2], [2, 2, 2, 2]),
        (2, [1], [1], [1]),
        (3, [1, 2, 4, 5], [1, 2, 3, 4], [1, 2, 3, 4]),
        (4, [1, 2, 3, 4], [1, -1, 1, -1], [1, -1, 0, 0]),
    ]
    for chain, steps, energy, magnetization in fixtures:
        for step, e_value, m_value in zip(steps, energy, magnetization, strict=True):
            lines.append(f"{chain},{step},false,false,0,{e_value},{m_value}\n")
    trace_path.write_text("".join(lines), encoding="utf-8", newline="\n")
    checks = """
rows = {(row["chain_id"], row["observable"]): row for row in analysis_rows}
assert math.isclose(rows[0, "energy"]["tau"], 1.5, abs_tol=1e-14)
assert rows[0, "energy"]["window"] == 1
assert math.isclose(rows[0, "magnetization"]["tau"], 5 / 6, abs_tol=1e-14)
assert rows[0, "energy"]["tau_steps"] == rows[0, "energy"]["tau"]
assert "Constant trace" in rows[1, "energy"]["status"]
assert "At least two" in rows[2, "energy"]["status"]
assert "uniform positive step spacing" in rows[3, "energy"]["status"]
assert "No truncation pair" in rows[4, "energy"]["status"]
assert "not positive" in rows[4, "magnetization"]["status"]
assert all(row["tau"] is None for key, row in rows.items() if key[0] != 0)
expected = [1.0, 0.25, -0.3, -0.45]
actual = scalar_acf(pl.Series([1.0, 2.0, 3.0, 4.0]), 3)
assert all(math.isclose(a, b, abs_tol=1e-14) for a, b in zip(actual, expected, strict=True))
tiny = math.ulp(0.0)
actual = scalar_acf(pl.Series([0.0, tiny, 2 * tiny, 3 * tiny]), 3)
assert all(math.isclose(a, b, abs_tol=1e-14) for a, b in zip(actual, expected, strict=True))
huge = float.fromhex("0x1.fffffffffffffp+1023")
actual = scalar_acf(pl.Series([-huge, huge, -huge, huge]), 3)
assert all(math.isclose(a, b, abs_tol=1e-14) for a, b in zip(actual, [1.0, -0.75, 0.5, -0.25], strict=True))
assert initial_monotone_time([1.0, 0.5, 0.3, 0.2, 0.4, 0.3, -0.2, 0.1, 0.9]) == (4.0, 5)
ess = {(row["chain_id"], row["observable"]): row for row in ess_rows}
assert math.isclose(ess[0, "energy"]["ess"], 8 / 3, abs_tol=1e-14)
assert math.isclose(ess[0, "magnetization"]["ess"], 24 / 5, abs_tol=1e-14)
assert all(row["ess_per_second"] is None for row in ess_rows)
assert all(row["rhat"] is None for row in rhat_rows)
left, right = [0.0, 2.0, 0.0, 2.0], [2.0, 4.0, 2.0, 4.0]
assert math.isclose(classical_split_rhat([left, right]), math.sqrt(7 / 6), abs_tol=1e-14)
assert math.isclose(classical_split_rhat([left, left]), math.sqrt(0.5), abs_tol=1e-14)
assert math.isclose(classical_split_rhat([[0, 2, huge, 0, 2], [2, 4, -huge, 2, 4]]), math.sqrt(7 / 6), abs_tol=1e-14)
assert classical_split_rhat([[0, 1, 10, 11], [11, 10, 1, 0]]) > 8
for bad in ([], [left], [left, [1, 2]], [left, left + [1]], [left, [1] * 4], [left, [1, 2, 3, math.nan]]):
    try:
        classical_split_rhat(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"Unexpected R-hat for {bad}")
"""
    payload = json.loads(notebook.read_bytes())
    payload["cells"].append(
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "verify-autocorrelation-contract",
            "metadata": {},
            "outputs": [],
            "source": checks.strip().splitlines(keepends=True),
        }
    )
    notebook.write_text(json.dumps(payload), encoding="utf-8", newline="\n")
    monkeypatch.setenv("MCMC_TRACE_PATH", str(trace_path))
    monkeypatch.setenv("MCMC_NOTEBOOK_OUTPUT_DIR", str(tmp_path / "figures"))
    before = notebook.read_bytes()

    execution = project.execute(notebook)
    assert execution.returncode == 0
    assert notebook.read_bytes() == before


_PRODUCTION_SCOPE = "production_including_observation_excluding_warmup_export_and_analysis"


@pytest.mark.parametrize(
    "timing_case",
    [
        (4, 0.5, 0, None, _PRODUCTION_SCOPE, 16 / 3, "measured production"),
        (5, 0.5, 0, "Timing sample counts do not match the analyzed trace.", _PRODUCTION_SCOPE, None, "unavailable for analyzed samples"),
        (4, -1.0, 0, "Elapsed seconds must be a finite nonnegative number.", _PRODUCTION_SCOPE, None, "unavailable for analyzed samples"),
        (4, 0.0, 0, None, _PRODUCTION_SCOPE, None, "unavailable for analyzed samples"),
        (4, 0.5, 1, None, _PRODUCTION_SCOPE, None, "unavailable for analyzed samples"),
        (4, 0.5, 0, None, None, None, "unavailable for analyzed samples"),
        (4, 0.5, 0, None, "including_warmup", None, "unavailable for analyzed samples"),
        (4, 5e-324, 0, None, _PRODUCTION_SCOPE, None, "unrepresentable rate"),
    ],
)
def test_ess_timing_requires_matching_analyzed_samples(
    project: NotebookProject, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, timing_case: tuple[int, float, int, str | None, str | None, float | None, str]
) -> None:
    """Measured rates are available only for the complete matching workload."""
    samples, elapsed, discard, expected_error, scope, expected_rate, expected_status = timing_case
    notebook = project.root / "notebooks/ising_trace_analysis.ipynb"
    trace_path = tmp_path / "trace.csv"
    trace_path.write_text(
        "chain_id,step,accepted,proposed,log_prob,energy,magnetization\n"
        + "".join(f"{chain},{step},true,true,0,{step + 2 * chain},{step + 2 * chain}\n" for chain in range(2) for step in range(1, 5)),
        encoding="utf-8",
        newline="\n",
    )
    report_path = tmp_path / "diagnostics.json"
    report = {
        "schema_version": 1,
        "chains": [
            {
                "chain_id": chain,
                "samples": samples,
                "recording_interval_steps": 1,
                "elapsed_seconds": elapsed,
                "observables": [{"observable": name, "tau": 1.5} for name in ("energy", "magnetization")],
            }
            for chain in range(2)
        ],
    }
    if scope is not None:
        report["timing_scope"] = scope
    report_path.write_text(json.dumps(report), encoding="utf-8", newline="\n")
    payload = json.loads(notebook.read_bytes())
    for cell in payload["cells"]:
        if cell["id"] == "analyze-autocorrelation-by-chain":
            cell["source"][0] = f"analysis_discard = {discard}\n"
    checks = f"""
if analysis_discard == 0:
    assert all(math.isclose(row["rhat"], math.sqrt(35 / 6), abs_tol=1e-14) for row in rhat_rows)
assert sorted((row["chain_id"], row["observable"]) for row in ess_rows) == [
    (0, "energy"), (0, "magnetization"), (1, "energy"), (1, "magnetization"),
]
expected_rate = {expected_rate!r}
for row in ess_rows:
    assert row["timing_status"] == {expected_status!r}
    if expected_rate is None:
        assert row["ess_per_second"] is None
    else:
        assert math.isclose(row["ess_per_second"], expected_rate, abs_tol=1e-14)
"""
    if expected_error is None:
        payload["cells"].append(
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "verify-measured-ess-rate",
                "metadata": {},
                "outputs": [],
                "source": checks.strip().splitlines(keepends=True),
            }
        )
    notebook.write_text(json.dumps(payload), encoding="utf-8", newline="\n")
    monkeypatch.setenv("MCMC_TRACE_PATH", str(trace_path))
    monkeypatch.setenv("MCMC_DIAGNOSTICS_PATH", str(report_path))
    monkeypatch.setenv("MCMC_NOTEBOOK_OUTPUT_DIR", str(tmp_path / "figures"))
    execution = project.execute(notebook)
    assert execution.returncode == (0 if expected_error is None else 1)
    if expected_error is not None:
        execution_report = execution.report
        assert execution_report["status"] == "failed"
        assert execution_report["failed_cell"]["id"] == "compute-effective-sample-sizes"
        assert execution_report["error"]["type"] == "CellExecutionError"
        assert "ValueError" in execution_report["error"]["message"]
        assert expected_error in execution_report["error"]["message"]


@pytest.mark.parametrize("change", ["boolean_schema", "boolean_chain", "duplicate_observable", "overflow_elapsed", "overflow_tau", "missing_observables"])
def test_invalid_timing_preserves_existing_analysis_outputs(project: NotebookProject, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str) -> None:
    notebook = project.root / "notebooks/ising_trace_analysis.ipynb"
    trace_path = tmp_path / "trace.csv"
    trace_path.write_text(
        "chain_id,step,accepted,proposed,log_prob,energy,magnetization\n"
        + "".join(f"{chain},{step},true,true,0,{step},{step}\n" for chain in range(2) for step in range(1, 5)),
        encoding="utf-8",
        newline="\n",
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "timing_scope": _PRODUCTION_SCOPE,
        "chains": [
            {
                "chain_id": chain,
                "samples": 4,
                "recording_interval_steps": 1,
                "elapsed_seconds": 0.5,
                "observables": [{"observable": name, "tau": 1.5} for name in ("energy", "magnetization")],
            }
            for chain in range(2)
        ],
    }
    item: dict[str, Any] = report["chains"][0]
    if change == "boolean_schema":
        report["schema_version"] = True
    elif change == "boolean_chain":
        item["chain_id"] = False
    elif change == "duplicate_observable":
        item["observables"].append({"observable": "energy", "tau": 1.5})
    elif change == "overflow_elapsed":
        item["elapsed_seconds"] = 10**1000
    elif change == "overflow_tau":
        item["observables"][0]["tau"] = 10**1000
    else:
        del item["observables"]
    report_path = tmp_path / "diagnostics.json"
    report_path.write_text(json.dumps(report), encoding="utf-8", newline="\n")
    output = tmp_path / "figures"
    output.mkdir()
    retained = {name: f"existing {name}\r\n".encode() for name in ("ising_energy_trace.png", "ising_ess_summary.csv", "ising_rhat_summary.csv")}
    for name, contents in retained.items():
        (output / name).write_bytes(contents)
    monkeypatch.setenv("MCMC_TRACE_PATH", str(trace_path))
    monkeypatch.setenv("MCMC_DIAGNOSTICS_PATH", str(report_path))
    monkeypatch.setenv("MCMC_NOTEBOOK_OUTPUT_DIR", str(output))
    execution = project.execute(notebook)
    assert execution.returncode == 1
    assert execution.report["failed_cell"]["id"] == "compute-effective-sample-sizes"
    assert "ValueError" in execution.report["error"]["message"]
    assert {path.name: path.read_bytes() for path in output.iterdir()} == retained


@pytest.mark.parametrize(
    ("chains", "expected_count", "expected_split"),
    [
        (((0, 2, 0, 2), (2, 4, 2, 4, 6)), "", None),
        (((1, 1, 1, 1), (2, 4, 2, 4)), "4", None),
        (((0, 2, 99, 0, 2), (2, 4, -99, 2, 4)), "5", (2, 1)),
    ],
    ids=["unequal-lengths", "constant-half", "valid-odd-length"],
)
def test_rhat_export_counts_describe_analyzed_chains(
    project: NotebookProject,
    monkeypatch: pytest.MonkeyPatch,
    chains: tuple[tuple[int, ...], ...],
    expected_count: str,
    expected_split: tuple[int, int] | None,
) -> None:
    """Export common input counts and successful split counts without inventing metadata."""
    tmp_path = project.root
    notebook = project.root / "notebooks/ising_trace_analysis.ipynb"
    trace_path = tmp_path / "trace.csv"
    trace_path.write_text(
        "chain_id,step,accepted,proposed,log_prob,energy,magnetization\n"
        + "".join(f"{chain_id},{step},true,true,0,{value},{value}\n" for chain_id, values in enumerate(chains) for step, value in enumerate(values, start=1)),
        encoding="utf-8",
        newline="\n",
    )
    figure_root = tmp_path / "figures"
    monkeypatch.setenv("MCMC_TRACE_PATH", str(trace_path))
    monkeypatch.setenv("MCMC_NOTEBOOK_OUTPUT_DIR", str(figure_root))
    monkeypatch.delenv("MCMC_DIAGNOSTICS_PATH", raising=False)
    before = notebook.read_bytes()

    execution = project.execute(notebook)
    assert execution.returncode == 0

    assert notebook.read_bytes() == before
    with (figure_root / "ising_rhat_summary.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert {row["observable"] for row in rows} == {"energy", "magnetization"}
    for row in rows:
        assert row["chain_count"] == str(len(chains))
        assert row["samples_per_chain"] == expected_count
        if expected_split is None:
            assert row["rhat"] == ""
            assert row["samples_per_split_chain"] == ""
            assert row["omitted_middle_draws_per_chain"] == ""
        else:
            assert float(row["rhat"]) == pytest.approx((7 / 6) ** 0.5, rel=1e-14)
            assert row["samples_per_split_chain"] == str(expected_split[0])
            assert row["omitted_middle_draws_per_chain"] == str(expected_split[1])


def plot_report() -> dict[str, Any]:
    """Small rendering fixture; pinned ranks/ESS, stubbed unrelated scalar panels."""
    fixtures = REPO_ROOT / "tests/fixtures"
    original = json.loads((fixtures / "ess.json").read_bytes())["cases"][0]["chains"]
    references = json.loads((fixtures / "diagnostic_plots.json").read_bytes())["cases"][0]["prefixes"]
    missing = {"status": "unavailable", "value": None, "reason": "fixture_unavailable"}
    estimated = {"status": "estimated", "value": 0.25}
    prefixes = []
    for reference in references:
        length = reference["draws_per_chain"]
        retained = 4 * (length // 2)
        complete = length == 31
        metrics = {}
        for name in ("mean", "bulk", "tail"):
            value = reference["estimates"][f"{name}_ess"]
            metrics[name] = {
                "estimate": {"status": "estimated", "value": value},
                "relative": value / retained,
                "rate": {"status": "estimated", "value": value} if complete else missing,
            }
        # Preserve an explicit missing tail in the plot/table instead of a sentinel.
        if length == 7:
            metrics["tail"] = {"estimate": missing, "relative": None, "rate": missing}
        prefixes.append(
            {
                "samples_per_chain": length,
                "chain_count": 2,
                "original_draws": 2 * length,
                "used_split_draws": retained,
                "samples_per_split_chain": length // 2,
                "omitted_middle_per_chain": length % 2,
                "timing": {
                    "status": "measured" if complete else "unavailable",
                    "elapsed_seconds": 1.0 if complete else None,
                    "reason": None if complete else "prefix_not_timed",
                },
                "ranks": [{"chain_id": f"chain-{index}", "ranks": ranks} for index, ranks in enumerate(reference["ranks"])],
                "ess": metrics,
                "mean_mcse": estimated,
                "pooled_sample_sd": estimated,
                "quantiles": [{"probability": probability, "mcse": estimated} for probability in (0.05, 0.5, 0.95)],
                "rhat": dict.fromkeys(("classical", "rank_normalized", "folded", "combined"), estimated),
                "single_chain": [
                    {
                        "chain_id": f"chain-{index}",
                        "acf": {"status": "estimated", "values": [1.0, 0.5]},
                        "blocked_mean_error": {
                            "status": "estimated",
                            "levels": [
                                {
                                    "block_size": 2,
                                    "block_count": length // 2,
                                    "used_draws": 2 * (length // 2),
                                    "standard_error": {"status": "estimated", "value": 0.0},
                                }
                            ],
                        },
                    }
                    for index in range(2)
                ],
            }
        )
    return {
        "schema_version": 2,
        "workflow": "rank_ess_efficiency_v1",
        "crate_version": "consumer-fixture",
        "source_revision": None,
        "source_dirty": None,
        "rank_convention": "one_based_average_ties_all_original_prefix_draws_signed_zero_equal",
        "split_convention": "first_and_last_floor_N_over_2",
        "quantile_convention": "linear_type_7_all_original_prefix_draws",
        "timing_scope": "sequential_production_sampling_and_recording_excluding_allocation_warmup_analysis_export",
        "prefixes": [7, 16, 31],
        "runs": [
            {
                "run_id": "rendering-fixture",
                "scenario": "rendering_fixture",
                "observable": "position",
                "units": "position",
                "chain_identity": "original_unsplit",
                "recording_interval": 1,
                "prefixes": prefixes,
                "chains": [
                    {"chain_id": f"chain-{index}", "draws": draws[:31], "elapsed_seconds": seconds}
                    for index, (draws, seconds) in enumerate(zip(original, (0.25, 0.75), strict=True))
                ],
            }
        ],
    }


def test_rank_notebook_preserves_missing_values_and_matching_evidence(project: NotebookProject, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    notebook = project.root / "notebooks/diagnostic_plots.ipynb"
    report_path = tmp_path / "source.json"
    # CRLF input must survive the evidence copy byte-for-byte on every platform.
    report_bytes = json.dumps(plot_report(), indent=2, allow_nan=False).replace("\n", "\r\n").encode("utf-8")
    report_path.write_bytes(report_bytes)
    output = tmp_path / "figures"
    monkeypatch.setenv("MCMC_DIAGNOSTICS_PATH", str(report_path))
    monkeypatch.setenv("MCMC_NOTEBOOK_OUTPUT_DIR", str(output))
    monkeypatch.setenv("MPLBACKEND", "TkAgg")
    source = notebook.read_bytes()
    execution = project.execute(notebook)
    assert execution.returncode == 0
    assert notebook.read_bytes() == source
    assert (output / "diagnostics.json").read_bytes() == report_bytes
    manifest = json.loads((output / "manifest.json").read_bytes())
    assert manifest["report_sha256"] == hashlib.sha256(report_bytes).hexdigest()
    assert manifest["rank_prefix_per_chain"] == 31
    for name in manifest["figures"]:
        assert (output / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    with (output / "efficiency.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 9
    first_tail = next(row for row in rows if row["estimator"] == "tail" and row["original_prefix_per_chain"] == "7")
    assert first_tail["ess"] == first_tail["relative_ess"] == first_tail["ess_per_second"] == ""
    assert first_tail["status"] == "unavailable"
    assert first_tail["reason"] == "fixture_unavailable"
    assert all(row["ess_per_second"] == "" for row in rows if row["original_prefix_per_chain"] != "31")
    with (output / "rank_bins.csv").open(encoding="utf-8", newline="") as stream:
        bins = list(csv.DictReader(stream))
    assert {row["chain_id"] for row in bins} == {"chain-0", "chain-1", "pooled reference"}
    for chain in ("chain-0", "chain-1", "pooled reference"):
        assert math.fsum(float(row["proportion"]) for row in bins if row["chain_id"] == chain) == pytest.approx(1.0)
    with (output / "blocked_errors.csv").open(encoding="utf-8", newline="") as stream:
        blocks = list(csv.DictReader(stream))
    assert all(float(row["standard_error"]) == 0.0 and row["status"] == "estimated" for row in blocks)


def test_rank_notebook_renders_short_prefix_cadence_and_unavailable_chains(project: NotebookProject, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    notebook = project.root / "notebooks/diagnostic_plots.ipynb"
    report = plot_report()
    run = report["runs"][0]
    run["recording_interval"] = 3
    for chain in run["prefixes"][-1]["single_chain"]:
        chain["acf"] = {"status": "unavailable", "value": None, "reason": "constant_samples"}
        chain["blocked_mean_error"] = {"status": "unavailable", "value": None, "reason": "fixture_blocks_unavailable"}
    report_path = tmp_path / "source.json"
    report_path.write_text(json.dumps(report), encoding="utf-8", newline="\n")
    output = tmp_path / "figures"
    monkeypatch.setenv("MCMC_DIAGNOSTICS_PATH", str(report_path))
    monkeypatch.setenv("MCMC_NOTEBOOK_OUTPUT_DIR", str(output))
    payload = json.loads(notebook.read_bytes())
    rank_source = next(cell["source"] for cell in payload["cells"] if cell["id"] == "render-original-chain-rank-overlays")
    rank_source[rank_source.index('rank_prefix = report["prefixes"][-1]\n')] = "rank_prefix = 7\n"
    # Inspect the actual rendered axes in the notebook kernel, where plotting dependencies live.
    payload["cells"].append(
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "assert-rendered-cadence-and-missing-chains",
            "metadata": {},
            "outputs": [],
            "source": [
                'assert trace_axes[0, 0].get_xlabel() == "Recorded production draw (cadence = 3)"\n',
                "assert trace_axes[0, 1].get_legend_handles_labels()[1] == [\n",
                '    "chain-0: ACF unavailable (constant_samples)", "chain-1: ACF unavailable (constant_samples)"\n',
                "]\n",
            ],
        }
    )
    notebook.write_text(json.dumps(payload), encoding="utf-8", newline="\n")
    execution = project.execute(notebook)
    assert execution.returncode == 0
    manifest = json.loads((output / "manifest.json").read_bytes())
    assert manifest["rank_prefix_per_chain"] == 7
    with (output / "rank_bins.csv").open(encoding="utf-8", newline="") as stream:
        bins = list(csv.DictReader(stream))
    assert {row["original_prefix_per_chain"] for row in bins} == {"7"}
    for chain, expected_draws in (("chain-0", 7), ("chain-1", 7), ("pooled reference", 14)):
        assert sum(int(row["count"]) for row in bins if row["chain_id"] == chain) == expected_draws
    with (output / "blocked_errors.csv").open(encoding="utf-8", newline="") as stream:
        blocks = list(csv.DictReader(stream))
    assert [row["chain_id"] for row in blocks] == ["chain-0", "chain-1"]
    for row in blocks:
        assert row["run_id"] == "rendering-fixture"
        assert row["status"] == "unavailable"
        assert row["reason"] == "fixture_blocks_unavailable"
        assert row["block_size"] == row["block_count"] == row["used_draws"] == row["standard_error"] == ""


@pytest.fixture(scope="module")
def validate_plot_report() -> Callable[[dict[str, Any], Path], None]:
    """Execute the checked-in report validation cells in a fresh namespace."""
    notebook = REPO_ROOT / "notebooks/diagnostic_plots.ipynb"
    source = {cell["id"]: "".join(cell["source"]) for cell in json.loads(notebook.read_bytes())["cells"]}
    instructions = tuple(
        compile(source[cell_id], f"{notebook}:{cell_id}", "exec") for cell_id in ("validate-numeric-availability", "validate-report-and-preserve-evidence")
    )

    def validate(report: dict[str, Any], output: Path) -> None:
        namespace = {"math": math, "sys": sys, "report": report, "report_bytes": json.dumps(report).encode("utf-8"), "output_dir": output}
        for cell in instructions:
            exec(cell, namespace)  # noqa: S102 - run the checked-in notebook validation cells.

    return validate


def prior_plot_outputs(output: Path) -> dict[str, bytes]:
    output.mkdir()
    previous = {"diagnostics.json": b"prior report\r\n", "manifest.json": b"prior manifest\n", "rank_overlays.png": b"prior figure"}
    for name, content in previous.items():
        (output / name).write_bytes(content)
    return previous


@pytest.mark.parametrize(
    "trace_case",
    [
        pytest.param(((0, 1, 2, 3), 1, True), id="regular"),
        pytest.param(((0, 1, 2, 3), 2, False), id="regular-mismatch"),
        pytest.param(((0, 1, 3, 4), 1, False), id="irregular-mismatch"),
        pytest.param(((0,), 1, True), id="singleton"),
    ],
)
def test_ising_trace_accepts_unsigned_steps_and_checks_timing_cadence(
    project: NotebookProject, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trace_case: tuple[tuple[int, ...], int, bool]
) -> None:
    step_offsets, recording_interval, cadence_matches = trace_case
    trace = tmp_path / "trace.csv"
    steps = tuple(2**63 + offset for offset in step_offsets)
    rows = [f"{chain},{step},true,true,0,{index},{index}\n" for chain in range(2) for index, step in enumerate(steps)]
    trace.write_text("chain_id,step,accepted,proposed,log_prob,energy,magnetization\n" + "".join(rows), encoding="utf-8", newline="\n")
    report = tmp_path / "timing.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "timing_scope": _PRODUCTION_SCOPE,
                "chains": [
                    {
                        "chain_id": chain,
                        "samples": len(steps),
                        "recording_interval_steps": recording_interval,
                        "elapsed_seconds": 0.5,
                        "observables": [{"observable": name, "tau": 1.5} for name in ("energy", "magnetization")],
                    }
                    for chain in range(2)
                ],
            }
        ),
        encoding="utf-8",
        newline="\n",
    )
    output = tmp_path / "figures"
    monkeypatch.setenv("MCMC_TRACE_PATH", str(trace))
    monkeypatch.setenv("MCMC_DIAGNOSTICS_PATH", str(report))
    monkeypatch.setenv("MCMC_NOTEBOOK_OUTPUT_DIR", str(output))
    notebook = project.root / "notebooks/ising_trace_analysis.ipynb"
    execution = project.execute(notebook)
    if not cadence_matches:
        assert execution.returncode == 1
        assert execution.report["failed_cell"]["id"] == "compute-effective-sample-sizes"
        assert "Timing recording interval does not match the analyzed trace" in execution.report["error"]["message"]
        assert not output.exists()
        return
    assert execution.returncode == 0, execution.report
    with (output / "ising_ess_summary.csv").open(encoding="utf-8", newline="") as stream:
        estimates = list(csv.DictReader(stream))
    assert len(estimates) == 4
    if len(steps) == 1:
        assert {row["status"] for row in estimates} == {"At least two samples are required."}
        assert {row["timing_status"] for row in estimates} == {"unavailable for analyzed samples"}
        assert all(row["ess_per_second"] == "" for row in estimates)
        return
    assert all(float(row["tau_steps"]) == pytest.approx(1.5) and float(row["ess_per_second"]) == pytest.approx(16 / 3) for row in estimates)
    with (output / "ising_rhat_summary.csv").open(encoding="utf-8", newline="") as stream:
        estimates = list(csv.DictReader(stream))
    assert len(estimates) == 2
    assert all(float(row["rhat"]) == pytest.approx(math.sqrt(19 / 6)) for row in estimates)


def test_rank_notebook_keeps_previous_outputs_when_rendering_fails(project: NotebookProject, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    notebook = project.root / "notebooks/diagnostic_plots.ipynb"
    report = tmp_path / "report.json"
    report.write_text(json.dumps(plot_report()), encoding="utf-8", newline="\n")
    output = tmp_path / "figures"
    previous = prior_plot_outputs(output)
    payload = json.loads(notebook.read_bytes())
    cell = next(cell for cell in payload["cells"] if cell["id"] == "render-traces-and-autocorrelations")
    cell["source"].insert(cell["source"].index("manifest = {\n"), 'raise ValueError("fixture render failure")\n')
    notebook.write_text(json.dumps(payload), encoding="utf-8", newline="\n")
    monkeypatch.setenv("MCMC_DIAGNOSTICS_PATH", str(report))
    monkeypatch.setenv("MCMC_NOTEBOOK_OUTPUT_DIR", str(output))
    execution = project.execute(notebook)
    assert execution.returncode == 1
    assert execution.report["failed_cell"]["id"] == "render-traces-and-autocorrelations"
    assert "fixture render failure" in execution.report["error"]["message"]
    assert {path.name: path.read_bytes() for path in output.iterdir()} == previous


def test_rank_notebook_exports_estimates_after_many_unavailable_rows(tmp_path: Path) -> None:
    import polars as pl

    estimated = plot_report()["runs"][0]
    unavailable = plot_report()["runs"][0]
    for chain in unavailable["prefixes"][-1]["single_chain"]:
        chain["blocked_mean_error"] = {"status": "unavailable", "value": None, "reason": "fixture_unavailable"}
    runs = [{**unavailable, "run_id": f"unavailable-{index}"} for index in range(51)] + [estimated]
    path = REPO_ROOT / "notebooks/diagnostic_plots.ipynb"
    cell = next(cell for cell in json.loads(path.read_bytes())["cells"] if cell["id"] == "export-scalar-and-blocked-summaries")
    namespace = {"runs": runs, "pl": pl, "staging_dir": tmp_path}
    exec(compile("".join(cell["source"]), f"{path}:{cell['id']}", "exec"), namespace)  # noqa: S102 - run the checked-in export cell.
    with (tmp_path / "blocked_errors.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 104
    assert all(row["block_size"] == row["standard_error"] == "" for row in rows[:-2])
    assert [(row["run_id"], int(row["block_size"]), float(row["standard_error"])) for row in rows[-2:]] == [
        ("rendering-fixture", 2, 0.0),
        ("rendering-fixture", 2, 0.0),
    ]


def test_rank_notebook_rejects_invalid_report_before_replacing_outputs(project: NotebookProject, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    notebook = project.root / "notebooks/diagnostic_plots.ipynb"
    report = plot_report()
    del report["runs"][0]["prefixes"][-1]["mean_mcse"]["status"]
    report_path = tmp_path / "incomplete.json"
    report_path.write_bytes(json.dumps(report).encode("utf-8"))
    output = tmp_path / "figures"
    previous = prior_plot_outputs(output)
    monkeypatch.setenv("MCMC_DIAGNOSTICS_PATH", str(report_path))
    monkeypatch.setenv("MCMC_NOTEBOOK_OUTPUT_DIR", str(output))
    execution = project.execute(notebook)
    assert execution.returncode == 1
    result = execution.report
    assert result["failed_cell"]["id"] == "validate-report-and-preserve-evidence"
    assert "Scalar metric requires status and value" in result["error"]["message"]
    assert {path.name: path.read_bytes() for path in output.iterdir()} == previous


def test_rank_notebook_rejects_missing_scalar_value(validate_plot_report: Callable[[dict[str, Any], Path], None], tmp_path: Path) -> None:
    report = plot_report()
    del report["runs"][0]["prefixes"][-1]["mean_mcse"]["value"]
    output = tmp_path / "figures"
    previous = prior_plot_outputs(output)
    with pytest.raises(ValueError, match="Scalar metric requires status and value"):
        validate_plot_report(report, output)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == previous


@pytest.mark.parametrize(
    "metric",
    [
        None,
        [],
        {"status": "estimated", "value": 1.0, "reason": "failed"},
        {"status": "estimated", "value": 10**1000},
        {"status": "unavailable", "value": None, "reason": True},
        {"status": "unavailable", "value": None, "reason": "  "},
    ],
)
def test_rank_notebook_rejects_invalid_scalar_alternatives_without_replacing_evidence(
    validate_plot_report: Callable[[dict[str, Any], Path], None], tmp_path: Path, metric: object
) -> None:
    report = plot_report()
    report["runs"][0]["prefixes"][-1]["mean_mcse"] = metric
    output = tmp_path / "figures"
    previous = prior_plot_outputs(output)
    with pytest.raises(ValueError, match=r"Scalar metric requires status and value|Diagnostic availability/value/reason mismatch"):
        validate_plot_report(report, output)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == previous


def test_rank_notebook_rejects_noninteger_report_schema(validate_plot_report: Callable[[dict[str, Any], Path], None], tmp_path: Path) -> None:
    report = plot_report()
    report["schema_version"] = 2.0
    output = tmp_path / "figures"
    with pytest.raises(ValueError, match="Expected diagnostics schema"):
        validate_plot_report(report, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "blocked",
    [
        {"status": "estimated"},
        {
            "status": "estimated",
            "levels": [{"block_size": 2, "block_count": 3, "used_draws": 1, "standard_error": {"status": "estimated", "value": 0.0}}],
        },
    ],
    ids=["missing-levels", "inconsistent-draw-count"],
)
def test_rank_notebook_rejects_invalid_block_levels(
    validate_plot_report: Callable[[dict[str, Any], Path], None], tmp_path: Path, blocked: dict[str, Any]
) -> None:
    report = plot_report()
    report["runs"][0]["prefixes"][0]["single_chain"][0]["blocked_mean_error"] = blocked
    output = tmp_path / "figures"
    with pytest.raises(ValueError, match="Invalid blocked error availability or draw counts"):
        validate_plot_report(report, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "corruption",
    [
        "chain_identity",
        "acf_identity",
        "split_count",
        "prefix_timing",
        "missing_value",
        "acf_null",
        "acf_nonfinite",
        "acf_range",
        "acf_lag_zero",
        "acf_empty",
        "acf_unavailable_values",
        "cadence",
        "boolean_count",
        "measured_reason",
        "unavailable_reason",
        "acf_reason",
    ],
)
def test_rank_notebook_rejects_inconsistent_report_before_rendering(
    validate_plot_report: Callable[[dict[str, Any], Path], None], tmp_path: Path, corruption: str
) -> None:
    report = plot_report()
    prefix = report["runs"][0]["prefixes"][0]
    if corruption == "chain_identity":
        prefix["ranks"][0]["chain_id"] = "wrong-chain"
    elif corruption == "acf_identity":
        prefix["single_chain"][0]["chain_id"] = "wrong-chain"
    elif corruption == "split_count":
        prefix["used_split_draws"] = prefix["original_draws"]
    elif corruption == "prefix_timing":
        prefix["timing"] = {"status": "measured", "elapsed_seconds": 1.0, "reason": None}
    elif corruption == "missing_value":
        prefix["ess"]["tail"]["estimate"]["value"] = 0.0
    elif corruption == "cadence":
        report["runs"][0]["recording_interval"] = True
    elif corruption == "boolean_count":
        prefix["omitted_middle_per_chain"] = True
    elif corruption == "measured_reason":
        run_full = report["runs"][0]["prefixes"][-1]
        run_full["timing"]["reason"] = "not measured"
    elif corruption == "unavailable_reason":
        prefix["timing"]["reason"] = True
    elif corruption == "acf_reason":
        prefix["single_chain"][0]["acf"]["reason"] = "not estimated"
    elif corruption == "acf_unavailable_values":
        prefix["single_chain"][0]["acf"] = {"status": "unavailable", "values": [1.0], "reason": "fixture_unavailable"}
    else:
        prefix["single_chain"][0]["acf"]["values"] = {
            "acf_null": [1.0, None],
            "acf_nonfinite": [1.0, float("nan")],
            "acf_range": [1.0, 1.01],
            "acf_lag_zero": [0.9, 0.5],
            "acf_empty": [],
        }[corruption]
    output = tmp_path / "figures"
    diagnostics = r"Diagnostic rows|Prefix original/split counts|Measured timing|Diagnostic availability|Recording interval|Invalid (?:ACF|unavailable timing)"
    with pytest.raises(ValueError, match=diagnostics):
        validate_plot_report(report, output)
    assert not output.exists()
