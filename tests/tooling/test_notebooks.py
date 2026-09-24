"""Consumer notebook policies exercised through the pinned shared public CLI."""

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from research_repo_tools.cli import main

if TYPE_CHECKING:
    import pytest

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
