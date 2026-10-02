"""Consumer notebook policies exercised through the pinned shared public CLI."""

import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest
from research_repo_tools.cli import main

REPO_ROOT = Path(__file__).resolve().parents[2]
ISING_NOTEBOOK = REPO_ROOT / "notebooks" / "ising_trace_analysis.ipynb"


def notebook_project(root: Path, monkeypatch: pytest.MonkeyPatch, source: Path = ISING_NOTEBOOK) -> Path:
    """Borrow the locked test interpreter without creating another environment."""
    for name in ("MCMC_TRACE_PATH", "MCMC_REPO_ROOT", "MCMC_NOTEBOOK_OUTPUT_DIR", "MCMC_DIAGNOSTICS_PATH"):
        monkeypatch.delenv(name, raising=False)
    notebook = root / "notebooks" / source.name
    notebook.parent.mkdir(parents=True)
    notebook.write_bytes(source.read_bytes())
    for name in ("pyproject.toml", "uv.lock", ".python-version"):
        (root / name).write_bytes((REPO_ROOT / name).read_bytes())
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", sys.prefix)
    # These tests exercise Python notebook behavior without installing a Rust toolchain.
    manifest = (root / "pyproject.toml").read_text(encoding="utf-8")
    start = manifest.index("[tool.research-repo-tools.toolchain.cargo]")
    end = manifest.index("[tool.research-repo-tools.notebooks]", start)
    (root / "pyproject.toml").write_text(manifest[:start] + manifest[end:], encoding="utf-8", newline="\n")
    return notebook


def write_ising_trace(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"chain_id,step,accepted,proposed,log_prob,energy,magnetization\n0,0,false,false,-1.0,-2.0,0.5\n")


def test_notebook_lint_rejects_dependency_installation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """MCMC notebooks consume the locked environment rather than installing dependencies."""
    notebook = notebook_project(tmp_path, monkeypatch)
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
    assert main(["--root", str(tmp_path), "notebooks", "lint", str(notebook)]) == 1
    output = capsys.readouterr()
    assert "install" in output.out + output.err
    assert notebook.read_bytes() == before


def execute(root: Path, notebook: Path) -> int:
    return main(["--root", str(root), "notebooks", "execute", str(notebook)])


def test_explicit_repository_root_never_falls_back_to_ambient_trace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_root = tmp_path / "runtime"
    configured_root = tmp_path / "configured"
    notebook = notebook_project(runtime_root, monkeypatch)
    (configured_root / "examples").mkdir(parents=True)
    (configured_root / "Cargo.toml").write_bytes(b"[package]\nname = 'fixture'\n")
    (configured_root / "examples" / "ising_1d.rs").write_bytes(b"// fixture\n")
    ambient_trace = runtime_root / "target" / "ising_1d_trace.csv"
    write_ising_trace(ambient_trace)
    monkeypatch.setenv("MCMC_REPO_ROOT", str(configured_root))
    monkeypatch.delenv("MCMC_TRACE_PATH", raising=False)
    monkeypatch.delenv("MCMC_NOTEBOOK_OUTPUT_DIR", raising=False)
    before = notebook.read_bytes()

    assert execute(runtime_root, notebook) == 1

    report = json.loads((runtime_root / "target/notebooks/notebooks/ising_trace_analysis.report.json").read_bytes())
    assert report["status"] == "failed"
    assert report["failed_cell"] == {"index": 2, "id": "load-and-validate-trace"}
    assert str(configured_root / "target/ising_1d_trace.csv") in report["error"]["message"]
    assert str(ambient_trace) not in report["error"]["message"]
    assert notebook.read_bytes() == before
    assert not (configured_root / "target/notebooks/ising_energy_trace.png").exists()


def test_explicit_trace_preserves_source_and_writes_only_selected_figure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "runtime"
    monkeypatch.setenv("MCMC_DIAGNOSTICS_PATH", str(tmp_path / "ambient-missing.json"))
    notebook = notebook_project(root, monkeypatch)
    trace_path = tmp_path / "mounted-input" / "trace.csv"
    figure_root = tmp_path / "figure-output"
    write_ising_trace(trace_path)
    monkeypatch.setenv("MCMC_TRACE_PATH", str(trace_path))
    monkeypatch.setenv("MCMC_REPO_ROOT", str(tmp_path / "invalid-root-is-ignored"))
    monkeypatch.setenv("MCMC_NOTEBOOK_OUTPUT_DIR", str(figure_root))
    monkeypatch.setenv("MPLBACKEND", "TkAgg")
    before = notebook.read_bytes()

    assert execute(root, notebook) == 0

    assert notebook.read_bytes() == before
    assert (figure_root / "ising_energy_trace.png").is_file()
    assert not (trace_path.parent / "notebooks/ising_energy_trace.png").exists()
    assert not (root / "target/notebooks/ising_energy_trace.png").exists()


def test_autocorrelation_known_values_and_unavailable_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the actual notebook cells, including chain/spacing boundaries."""
    notebook = notebook_project(tmp_path, monkeypatch)
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

    assert execute(tmp_path, notebook) == 0
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, timing_case: tuple[int, float, int, str | None, str | None, float | None, str]
) -> None:
    """Measured rates are available only for the complete matching workload."""
    samples, elapsed, discard, expected_error, scope, expected_rate, expected_status = timing_case
    notebook = notebook_project(tmp_path, monkeypatch)
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
    assert execute(tmp_path, notebook) == (0 if expected_error is None else 1)
    if expected_error is not None:
        execution_report = json.loads((tmp_path / "target/notebooks/notebooks/ising_trace_analysis.report.json").read_bytes())
        assert execution_report["status"] == "failed"
        assert execution_report["failed_cell"]["id"] == "compute-effective-sample-sizes"
        assert execution_report["error"]["type"] == "CellExecutionError"
        assert "ValueError" in execution_report["error"]["message"]
        assert expected_error in execution_report["error"]["message"]


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
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    chains: tuple[tuple[int, ...], ...],
    expected_count: str,
    expected_split: tuple[int, int] | None,
) -> None:
    """Export common input counts and successful split counts without inventing metadata."""
    notebook = notebook_project(tmp_path, monkeypatch)
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

    assert execute(tmp_path, notebook) == 0

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


def test_rank_notebook_preserves_missing_values_and_matching_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    notebook = notebook_project(tmp_path, monkeypatch, REPO_ROOT / "notebooks/diagnostic_plots.ipynb")
    report_path = tmp_path / "source.json"
    # CRLF input must survive the evidence copy byte-for-byte on every platform.
    report_bytes = json.dumps(plot_report(), indent=2, allow_nan=False).replace("\n", "\r\n").encode("utf-8")
    report_path.write_bytes(report_bytes)
    output = tmp_path / "figures"
    monkeypatch.setenv("MCMC_DIAGNOSTICS_PATH", str(report_path))
    monkeypatch.setenv("MCMC_NOTEBOOK_OUTPUT_DIR", str(output))
    monkeypatch.setenv("MPLBACKEND", "TkAgg")
    source = notebook.read_bytes()
    assert execute(tmp_path, notebook) == 0
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


def test_rank_notebook_labels_cadence_and_each_unavailable_chain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    notebook = notebook_project(tmp_path, monkeypatch, REPO_ROOT / "notebooks/diagnostic_plots.ipynb")
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
    # Inspect the actual rendered axes in the notebook kernel, where plotting dependencies live.
    payload = json.loads(notebook.read_bytes())
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
    assert execute(tmp_path, notebook) == 0
    with (output / "blocked_errors.csv").open(encoding="utf-8", newline="") as stream:
        blocks = list(csv.DictReader(stream))
    assert [row["chain_id"] for row in blocks] == ["chain-0", "chain-1"]
    for row in blocks:
        assert row["run_id"] == "rendering-fixture"
        assert row["status"] == "unavailable"
        assert row["reason"] == "fixture_blocks_unavailable"
        assert row["block_size"] == row["block_count"] == row["used_draws"] == row["standard_error"] == ""


@pytest.mark.parametrize("missing_field", ["status", "value"])
def test_rank_notebook_rejects_incomplete_scalar_metrics_before_replacing_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing_field: str) -> None:
    notebook = notebook_project(tmp_path, monkeypatch, REPO_ROOT / "notebooks/diagnostic_plots.ipynb")
    report = plot_report()
    del report["runs"][0]["prefixes"][-1]["mean_mcse"][missing_field]
    report_path = tmp_path / "incomplete.json"
    report_path.write_text(json.dumps(report), encoding="utf-8", newline="\n")
    output = tmp_path / "figures"
    output.mkdir()
    previous = {"diagnostics.json": b"prior report\r\n", "manifest.json": b"prior manifest\n", "rank_overlays.png": b"prior figure"}
    for name, content in previous.items():
        (output / name).write_bytes(content)
    monkeypatch.setenv("MCMC_DIAGNOSTICS_PATH", str(report_path))
    monkeypatch.setenv("MCMC_NOTEBOOK_OUTPUT_DIR", str(output))
    assert execute(tmp_path, notebook) == 1
    result = json.loads((tmp_path / "target/notebooks/notebooks/diagnostic_plots.report.json").read_bytes())
    assert result["failed_cell"]["id"] == "validate-report-and-preserve-evidence"
    assert "Scalar metric requires status and value" in result["error"]["message"]
    assert {path.name: path.read_bytes() for path in output.iterdir()} == previous


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
    ],
)
def test_rank_notebook_rejects_inconsistent_report_before_rendering(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str) -> None:
    notebook = notebook_project(tmp_path, monkeypatch, REPO_ROOT / "notebooks/diagnostic_plots.ipynb")
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
    report_path = tmp_path / "invalid.json"
    report_path.write_text(json.dumps(report), encoding="utf-8", newline="\n")
    output = tmp_path / "figures"
    monkeypatch.setenv("MCMC_DIAGNOSTICS_PATH", str(report_path))
    monkeypatch.setenv("MCMC_NOTEBOOK_OUTPUT_DIR", str(output))
    assert execute(tmp_path, notebook) == 1
    result = json.loads((tmp_path / "target/notebooks/notebooks/diagnostic_plots.report.json").read_bytes())
    assert result["failed_cell"]["id"] == "validate-report-and-preserve-evidence"
    assert "ValueError" in result["error"]["message"]
    assert not output.exists()
