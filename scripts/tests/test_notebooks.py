"""Consumer notebook policies exercised through the pinned shared public CLI."""

import hashlib
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
    artifact = root / "target/notebooks/notebooks/ising_trace_analysis.ipynb"
    report = json.loads(artifact.with_suffix(".report.json").read_bytes())
    assert report["status"] == "passed"
    assert report["source_sha256"] == hashlib.sha256(before).hexdigest()
    assert report["lock_sha256"] == hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()
    assert report["packages"]["research-repo-tools"] == "0.1.2"
    executed = json.loads(artifact.read_bytes())
    assert [cell["id"] for cell in executed["cells"]] == [cell["id"] for cell in json.loads(before)["cells"]]
    assert all(cell["execution_count"] is not None for cell in executed["cells"] if cell["cell_type"] == "code")


def test_native_notebook_lint_preserves_source() -> None:
    before = ISING_NOTEBOOK.read_bytes()

    assert main(["--root", str(REPO_ROOT), "notebooks", "lint", str(ISING_NOTEBOOK)]) == 0
    assert ISING_NOTEBOOK.read_bytes() == before
