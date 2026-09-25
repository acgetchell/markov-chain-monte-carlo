"""Consumer notebook policies exercised through the pinned shared public CLI."""

import csv
import json
import sys
from pathlib import Path

import pytest
from research_repo_tools.cli import main

REPO_ROOT = Path(__file__).resolve().parents[2]
ISING_NOTEBOOK = REPO_ROOT / "notebooks" / "ising_trace_analysis.ipynb"


def notebook_project(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Borrow the locked test interpreter without creating another environment."""
    notebook = root / "notebooks" / ISING_NOTEBOOK.name
    notebook.parent.mkdir(parents=True)
    notebook.write_bytes(ISING_NOTEBOOK.read_bytes())
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


@pytest.mark.parametrize("timing_case", [(4, 0.5, 0, 0), (5, 0.5, 0, 1), (4, -1.0, 0, 1), (4, 0.0, 0, 0), (4, 0.5, 1, 0)])
def test_ess_timing_requires_matching_analyzed_samples(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, timing_case: tuple[int, float, int, int]) -> None:
    """Measured rates are available only for the complete matching workload."""
    samples, elapsed, discard, expected_exit = timing_case
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
    report_path.write_text(json.dumps(report), encoding="utf-8", newline="\n")
    payload = json.loads(notebook.read_bytes())
    for cell in payload["cells"]:
        if cell["id"] == "analyze-autocorrelation-by-chain":
            cell["source"][0] = f"analysis_discard = {discard}\n"
    checks = """
if analysis_discard == 0:
    assert all(math.isclose(row["rhat"], math.sqrt(35 / 6), abs_tol=1e-14) for row in rhat_rows)
if elapsed_by_chain:
    assert all(math.isclose(row["ess_per_second"], 16 / 3, abs_tol=1e-14) for row in ess_rows)
else:
    assert all(row["ess_per_second"] is None for row in ess_rows)
"""
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
    assert execute(tmp_path, notebook) == expected_exit


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
