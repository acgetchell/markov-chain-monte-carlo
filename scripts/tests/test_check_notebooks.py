"""Behavioral tests for the notebook checker."""

import configparser
import json
import os
import shutil
import stat
import subprocess
import sys
import zipfile
from email.parser import Parser
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from nbclient.exceptions import CellExecutionError

import check_notebooks
from subprocess_utils import ExecutableNotFoundError

INVALID_EXECUTION_COUNT = True
REPO_ROOT = Path(__file__).resolve().parents[2]
ISING_NOTEBOOK = REPO_ROOT / "notebooks" / "ising_trace_analysis.ipynb"
CONSOLE_SCRIPTS = {
    "archive-performance": "archive_performance",
    "bench-compare": "bench_compare",
    "check-notebooks": "check_notebooks",
    "postprocess-changelog": "postprocess_changelog",
    "release-check": "release_check",
    "tag-release": "tag_release",
    "update-python-dev-pins": "update_python_dev_pins",
    "update-tool-pins": "update_cargo_tool_pins",
}


def copy_ising_notebook(root: Path) -> Path:
    """Copy the real Ising notebook into a temporary repository fixture."""
    notebook = root / "notebooks" / ISING_NOTEBOOK.name
    notebook.parent.mkdir(parents=True, exist_ok=True)
    notebook.write_bytes(ISING_NOTEBOOK.read_bytes())
    return notebook


def write_ising_trace(path: Path) -> None:
    """Write the smallest valid trace accepted by the Ising notebook."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "chain_id,step,accepted,proposed,log_prob,energy,magnetization\n0,0,false,false,-1.0,-2.0,0.5\n",
        encoding="utf-8",
    )


def code_cell(
    source: str | list[str] = "value = 1",
    *,
    cell_id: str = "code-cell",
    metadata: dict[str, object] | None = None,
    outputs: list[object] | None = None,
    execution_count: int | None = None,
) -> dict[str, object]:
    """Return one valid code-cell fixture."""
    return {
        "cell_type": "code",
        "execution_count": execution_count,
        "id": cell_id,
        "metadata": {} if metadata is None else metadata,
        "outputs": [] if outputs is None else outputs,
        "source": source,
    }


def markdown_cell(source: str = "# Title", *, cell_id: str = "markdown-cell") -> dict[str, object]:
    """Return one valid Markdown-cell fixture."""
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": source}


def write_notebook(path: Path, cells: list[dict[str, object]]) -> None:
    """Write a minimal nbformat 4 notebook fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "cells": cells,
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def check_options(
    repo_root: Path,
    *,
    mode: check_notebooks.CheckMode = "lint",
    paths: tuple[Path, ...] = (),
    output_dir: Path | None = None,
    run_tools: bool = False,
) -> check_notebooks.CheckOptions:
    """Return explicit checker options for tests."""
    return check_notebooks.CheckOptions(
        mode=mode,
        paths=paths,
        repo_root=repo_root,
        output_dir=output_dir or repo_root / "target" / "notebooks",
        timeout=120,
        run_ruff=run_tools,
        run_format=run_tools,
        run_ty=run_tools,
    )


class TestDiscoverNotebooks:
    def test_discovers_sorted_notebooks_and_excludes_checkpoints(self, tmp_path: Path) -> None:
        first = tmp_path / "notebooks" / "Z.ipynb"
        second = tmp_path / "notebooks" / "a.ipynb"
        checkpoint = tmp_path / "notebooks" / ".ipynb_checkpoints" / "a.ipynb"
        elsewhere = tmp_path / "elsewhere.ipynb"
        for path in (first, second, checkpoint, elsewhere):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")

        assert check_notebooks.discover_notebooks(tmp_path) == [first, second]

    def test_returns_empty_when_notebook_directory_is_missing(self, tmp_path: Path) -> None:
        assert check_notebooks.discover_notebooks(tmp_path) == []


class TestLoadNotebook:
    def test_accepts_maximum_length_cell_id(self, tmp_path: Path) -> None:
        notebook = tmp_path / "valid.ipynb"
        cell_id = "a" * 64
        write_notebook(notebook, [code_cell(cell_id=cell_id)])

        loaded = check_notebooks.load_notebook(notebook)

        assert loaded.document.cells[0].cell_id == cell_id

    def test_parses_valid_cells_into_immutable_records(self, tmp_path: Path) -> None:
        notebook = tmp_path / "valid.ipynb"
        write_notebook(notebook, [markdown_cell(), code_cell(["value = 1\n", "value + 1"])])

        loaded = check_notebooks.load_notebook(notebook)

        assert loaded.document.nbformat == 4
        assert loaded.document.nbformat_minor == 5
        assert loaded.document.cells[1].source == "value = 1\nvalue + 1"
        assert loaded.document.cells[1].cell_id == "code-cell"

    @pytest.mark.parametrize(
        ("mutation", "message"),
        [
            (lambda payload: payload.pop("cells"), "cells must be a list"),
            (lambda payload: payload.__setitem__("unknown", None), "unsupported notebook field.*unknown"),
            (lambda payload: payload.__setitem__("nbformat", 3), "expected nbformat 4"),
            (lambda payload: payload.__setitem__("nbformat", 4.0), "nbformat must be integer 4"),
            (lambda payload: payload.__setitem__("nbformat_minor", "5"), "nbformat_minor must be an integer"),
            (lambda payload: payload.__setitem__("metadata", []), "metadata must be a JSON object"),
            (lambda payload: payload["cells"][0].__setitem__("metadata", []), "cell 1: metadata must be a JSON object"),
            (lambda payload: payload["cells"][0].__setitem__("source", 42), "source must be a string"),
            (lambda payload: payload["cells"][0].__setitem__("id", ""), "id must be a nonempty string"),
            (
                lambda payload: payload["cells"][0].__setitem__("id", "contains spaces"),
                "id must contain 1-64 ASCII letters",
            ),
            (
                lambda payload: payload["cells"][0].__setitem__("id", "a" * 65),
                "id must contain 1-64 ASCII letters",
            ),
            (lambda payload: payload["cells"][0].__setitem__("outputs", {}), "outputs must be a list"),
            (
                lambda payload: payload["cells"][0].__setitem__("execution_count", INVALID_EXECUTION_COUNT),
                "execution_count must be an integer or null",
            ),
        ],
    )
    def test_rejects_distinct_invalid_json_shapes(
        self,
        tmp_path: Path,
        mutation: Any,
        message: str,
    ) -> None:
        notebook = tmp_path / "invalid.ipynb"
        payload: dict[str, Any] = {
            "cells": [code_cell()],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        mutation(payload)
        notebook.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises((TypeError, ValueError), match=message):
            check_notebooks.load_notebook(notebook)

    @pytest.mark.parametrize("minor", range(5))
    def test_rejects_nbformat_versions_before_stable_cell_ids(self, tmp_path: Path, minor: int) -> None:
        notebook = tmp_path / f"nbformat-4-{minor}.ipynb"
        payload = {"cells": [code_cell()], "metadata": {}, "nbformat": 4, "nbformat_minor": minor}
        notebook.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match=rf"expected nbformat 4\.5 or newer.*got 4\.{minor}"):
            check_notebooks.load_notebook(notebook)

    @pytest.mark.parametrize(
        ("cell", "message"),
        [
            ({key: value for key, value in code_cell().items() if key != "execution_count"}, "missing required code field.*execution_count"),
            ({**code_cell(), "attachments": {}}, "unsupported field.*code cell: attachments"),
            ({**code_cell(), "unknown": None}, "unsupported field.*code cell: unknown"),
            ({**markdown_cell(), "outputs": []}, "unsupported field.*markdown cell: outputs"),
            (
                {"cell_type": "raw", "execution_count": None, "id": "raw-cell", "metadata": {}, "source": "text"},
                "unsupported field.*raw cell: execution_count",
            ),
            ({**markdown_cell(), "attachments": []}, "attachments must be a JSON object"),
            (
                {**markdown_cell(), "attachments": {"image.png": "encoded"}},
                "attachment 'image.png' must contain a JSON object MIME bundle",
            ),
            (
                {**markdown_cell(), "attachments": {"image.png": []}},
                "attachment 'image.png' must contain a JSON object MIME bundle",
            ),
            (
                {**markdown_cell(), "attachments": {"image.png": {"image/png": 42}}},
                "MIME value 'image/png' must be a string or list of strings",
            ),
            (
                {**markdown_cell(), "attachments": {"image.png": {"image/png": ["encoded", 42]}}},
                "MIME value 'image/png' must be a string or list of strings",
            ),
        ],
    )
    def test_rejects_cell_type_schema_violations(self, tmp_path: Path, cell: dict[str, object], message: str) -> None:
        notebook = tmp_path / "invalid.ipynb"
        write_notebook(notebook, [cell])

        with pytest.raises((TypeError, ValueError), match=message):
            check_notebooks.load_notebook(notebook)

    def test_accepts_schema_valid_attachment_mime_values(self, tmp_path: Path) -> None:
        notebook = tmp_path / "attachments.ipynb"
        cell = {
            **markdown_cell(),
            "attachments": {
                "image.png": {
                    "image/png": ["encoded", "data"],
                    "text/plain": "description",
                    "application/json": {"width": 10, "visible": True},
                    "application/vnd.example+json": [1, None, {"nested": "value"}],
                }
            },
        }
        write_notebook(notebook, [cell])

        loaded = check_notebooks.load_notebook(notebook)

        assert loaded.document.cells[0].cell_type == "markdown"

    def test_rejects_duplicate_cell_ids(self, tmp_path: Path) -> None:
        notebook = tmp_path / "duplicate.ipynb"
        write_notebook(notebook, [markdown_cell(cell_id="duplicate"), code_cell(cell_id="duplicate")])

        with pytest.raises(ValueError, match="cell IDs must be unique"):
            check_notebooks.load_notebook(notebook)

    def test_rejects_non_object_json(self, tmp_path: Path) -> None:
        notebook = tmp_path / "list.ipynb"
        notebook.write_text("[]", encoding="utf-8")

        with pytest.raises(TypeError, match="notebook JSON must be an object"):
            check_notebooks.load_notebook(notebook)


class TestNotebookDiagnostics:
    def test_reports_syntax_outputs_and_execution_count_by_cell(self, tmp_path: Path) -> None:
        notebook_path = tmp_path / "invalid.ipynb"
        write_notebook(
            notebook_path,
            [
                code_cell("value = 1", cell_id="clean"),
                code_cell("if True print('bad')", cell_id="dirty", outputs=[{"output_type": "stream"}], execution_count=4),
            ],
        )
        notebook = check_notebooks.load_notebook(notebook_path).document

        diagnostics = check_notebooks.code_cell_diagnostics(notebook)

        assert [(diagnostic.cell, diagnostic.message) for diagnostic in diagnostics] == [
            (2, "syntax error at line 1: invalid syntax"),
            (2, "has 1 output block(s); clear outputs before committing"),
            (2, "execution_count=4; clear execution counts before committing"),
        ]

    @pytest.mark.parametrize(
        ("source", "ipython_syntax"),
        [
            ("%matplotlib inline\nvalue = 1", "%matplotlib inline"),
            ("%%time\nvalue = 1", "%%time"),
            ("!echo hello\nvalue = 1", "!echo hello"),
            ("%%bash\necho hello", "%%bash"),
        ],
    )
    def test_neutralizes_ipython_syntax_before_python_validation(
        self,
        tmp_path: Path,
        source: str,
        ipython_syntax: str,
    ) -> None:
        notebook_path = tmp_path / "magics.ipynb"
        write_notebook(notebook_path, [code_cell(source)])
        notebook = check_notebooks.load_notebook(notebook_path).document

        assert check_notebooks.code_cell_diagnostics(notebook) == []
        snapshot = check_notebooks.extract_code(notebook)
        compile(snapshot.source, "extracted-notebook.py", "exec")
        assert ipython_syntax not in snapshot.source

    def test_extract_code_maps_generated_lines_to_cells(self, tmp_path: Path) -> None:
        notebook_path = tmp_path / "valid.ipynb"
        write_notebook(
            notebook_path,
            [
                code_cell("first = 1\nfirst", cell_id="first"),
                markdown_cell(),
                code_cell("second = 2", cell_id="second"),
            ],
        )
        notebook = check_notebooks.load_notebook(notebook_path).document

        snapshot = check_notebooks.extract_code(notebook)

        assert "# %% notebook cell 1 (first)" in snapshot.source
        assert "# %% notebook cell 3 (second)" in snapshot.source
        second_line = next(line for line, cell in snapshot.line_to_cell.items() if cell == 3)
        assert snapshot.source.splitlines()[second_line - 1].startswith("# %% notebook cell 3")

    def test_ruff_diagnostics_map_extracted_line_to_cell(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        snapshot = check_notebooks.CodeSnapshot("value\n", {1: 7})

        def fake_run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess([], 1, stdout="fixture.py:1:1: F821 Undefined name `value`\n", stderr="")

        monkeypatch.setattr(check_notebooks, "run_safe_command", fake_run)

        diagnostics = check_notebooks.ruff_check_diagnostics(Path("fixture.ipynb"), snapshot, tmp_path)

        assert diagnostics == [check_notebooks.Diagnostic(7, "ruff check: F821 Undefined name `value`")]

    def test_missing_external_tool_is_a_notebook_diagnostic(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def missing(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
            raise ExecutableNotFoundError

        monkeypatch.setattr(check_notebooks, "run_safe_command", missing)

        diagnostics = check_notebooks.ruff_format_diagnostics(Path("fixture.ipynb"), check_notebooks.CodeSnapshot("", {}), tmp_path)

        assert diagnostics == [check_notebooks.Diagnostic(None, "ruff is required; run the checker through `uv run --locked`")]

    def test_ty_exit_code_two_is_an_unexpected_tool_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        result = subprocess.CompletedProcess([], 2, stdout="invalid configuration\n", stderr="")
        monkeypatch.setattr(check_notebooks, "tool_result", lambda *_args, **_kwargs: result)

        diagnostics = check_notebooks.ty_diagnostics(Path("fixture.ipynb"), check_notebooks.CodeSnapshot("", {}), tmp_path)

        assert diagnostics == [check_notebooks.Diagnostic(None, "ty failed with exit code 2: invalid configuration")]

    def test_lint_notebooks_reports_clean_path(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        notebook = tmp_path / "valid.ipynb"
        write_notebook(notebook, [code_cell()])
        options = check_options(tmp_path, paths=(notebook,))

        assert check_notebooks.lint_notebooks((notebook,), options) == 0
        assert f"OK linted {notebook}" in capsys.readouterr().out

    def test_lint_notebooks_reports_load_failure_and_continues(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        invalid = tmp_path / "invalid.ipynb"
        invalid.write_text("not-json", encoding="utf-8")
        valid = tmp_path / "valid.ipynb"
        write_notebook(valid, [code_cell()])
        options = check_options(tmp_path, paths=(invalid, valid))

        assert check_notebooks.lint_notebooks((invalid, valid), options) == 1

        captured = capsys.readouterr()
        assert f"{invalid}: notebook: error:" in captured.err
        assert f"OK linted {valid}" in captured.out


class TestExecuteNotebooks:
    def test_rejects_notebook_outside_repository_notebook_root(self, tmp_path: Path) -> None:
        notebook = tmp_path / "external.ipynb"
        write_notebook(notebook, [code_cell()])
        options = check_options(tmp_path, mode="execute", paths=(notebook,))

        with pytest.raises(ValueError, match="executed notebook must be under"):
            check_notebooks.execute_notebook(notebook, options)

    def test_rejects_output_directory_inside_source_notebooks(self, tmp_path: Path) -> None:
        notebook = tmp_path / "notebooks" / "valid.ipynb"
        write_notebook(notebook, [code_cell()])
        source_before = notebook.read_bytes()
        options = check_options(tmp_path, mode="execute", paths=(notebook,), output_dir=tmp_path / "notebooks")

        with pytest.raises(ValueError, match="must remain outside the source notebook directory"):
            check_notebooks.execute_notebook(notebook, options)

        assert notebook.read_bytes() == source_before

    def test_writes_executed_artifact_without_mutating_source(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        notebook = tmp_path / "notebooks" / "valid.ipynb"
        write_notebook(notebook, [code_cell()])
        source_before = notebook.read_bytes()
        calls: dict[str, Any] = {}

        def fake_read(handle: Any, *, as_version: int) -> dict[str, Any]:
            calls["read_path"] = handle.name
            calls["as_version"] = as_version
            return {"executed": False}

        def fake_write(value: dict[str, Any], handle: Any) -> None:
            json.dump(value, handle)

        class FakeNotebookClient:
            def __init__(self, notebook_value: dict[str, Any], **kwargs: Any) -> None:
                calls["notebook"] = notebook_value
                calls["kwargs"] = kwargs

            def execute(self) -> None:
                calls["environment"] = {
                    "IPYTHONDIR": check_notebooks.os.environ["IPYTHONDIR"],
                    "MPLBACKEND": check_notebooks.os.environ["MPLBACKEND"],
                    "MPLCONFIGDIR": check_notebooks.os.environ["MPLCONFIGDIR"],
                }
                calls["notebook"]["executed"] = True

        fake_nbformat = SimpleNamespace(read=fake_read, write=fake_write)
        fake_nbclient = SimpleNamespace(NotebookClient=FakeNotebookClient)

        def fake_import(name: str) -> SimpleNamespace:
            return fake_nbclient if name == "nbclient" else fake_nbformat

        monkeypatch.setattr(check_notebooks, "import_module", fake_import)
        monkeypatch.setenv("MPLBACKEND", "TkAgg")
        monkeypatch.setenv("IPYTHONDIR", str(tmp_path / "ambient-ipython"))
        monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "ambient-matplotlib"))
        options = check_options(tmp_path, mode="execute", paths=(notebook,))

        output_path = check_notebooks.execute_notebook(notebook, options)

        assert notebook.read_bytes() == source_before
        assert output_path == tmp_path / "target" / "notebooks" / "valid.ipynb"
        assert json.loads(output_path.read_text(encoding="utf-8")) == {"executed": True}
        if os.name != "nt":
            assert stat.S_IMODE(output_path.stat().st_mode) == 0o644
        assert calls["kwargs"] == {
            "timeout": 120,
            "kernel_name": "python3",
            "resources": {"metadata": {"path": str(tmp_path)}},
        }
        assert calls["environment"] == {
            "IPYTHONDIR": str(tmp_path / "target" / "notebooks" / ".ipython"),
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(tmp_path / "target" / "notebooks" / ".matplotlib"),
        }
        assert check_notebooks.os.environ["MPLBACKEND"] == "TkAgg"
        assert Path(check_notebooks.os.environ["IPYTHONDIR"]) == tmp_path / "ambient-ipython"
        assert Path(check_notebooks.os.environ["MPLCONFIGDIR"]) == tmp_path / "ambient-matplotlib"

    def test_explicit_repository_root_never_falls_back_to_an_ambient_trace(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime_root = tmp_path / "runtime"
        configured_root = tmp_path / "configured"
        notebook = copy_ising_notebook(runtime_root)
        (configured_root / "examples").mkdir(parents=True)
        (configured_root / "Cargo.toml").write_text("[package]\nname = 'fixture'\n", encoding="utf-8")
        (configured_root / "examples" / "ising_1d.rs").write_text("// fixture\n", encoding="utf-8")
        ambient_trace = runtime_root / "target" / "ising_1d_trace.csv"
        write_ising_trace(ambient_trace)
        monkeypatch.setenv("MCMC_REPO_ROOT", str(configured_root))
        monkeypatch.setenv("IPYTHONDIR", str(tmp_path / "ambient-ipython"))
        monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "ambient-matplotlib"))
        monkeypatch.delenv("MCMC_TRACE_PATH", raising=False)
        monkeypatch.delenv("MCMC_NOTEBOOK_OUTPUT_DIR", raising=False)
        options = check_options(runtime_root, mode="execute", paths=(notebook,))

        with pytest.raises(CellExecutionError) as error:
            check_notebooks.execute_notebook(notebook, options)

        message = str(error.value)
        assert str(configured_root / "target" / "ising_1d_trace.csv") in message
        assert Path(check_notebooks.os.environ["IPYTHONDIR"]) == tmp_path / "ambient-ipython"
        assert Path(check_notebooks.os.environ["MPLCONFIGDIR"]) == tmp_path / "ambient-matplotlib"
        assert str(ambient_trace) not in message
        assert not (configured_root / "target" / "notebooks" / "ising_energy_trace.png").exists()

    def test_explicit_trace_writes_figure_only_to_configured_output_directory(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime_root = tmp_path / "runtime"
        notebook = copy_ising_notebook(runtime_root)
        trace_path = tmp_path / "mounted-input" / "trace.csv"
        figure_root = tmp_path / "figure-output"
        write_ising_trace(trace_path)
        monkeypatch.setenv("MCMC_TRACE_PATH", str(trace_path))
        monkeypatch.setenv("MCMC_REPO_ROOT", str(tmp_path / "invalid-root-is-ignored"))
        monkeypatch.setenv("MCMC_NOTEBOOK_OUTPUT_DIR", str(figure_root))
        options = check_options(runtime_root, mode="execute", paths=(notebook,))

        executed = check_notebooks.execute_notebook(notebook, options)

        assert executed.is_file()
        assert (figure_root / "ising_energy_trace.png").is_file()
        assert not (trace_path.parent / "notebooks" / "ising_energy_trace.png").exists()
        assert not (runtime_root / "target" / "notebooks" / "ising_energy_trace.png").exists()


class TestClearNotebooks:
    def test_clears_outputs_and_counts(self, tmp_path: Path) -> None:
        notebook = tmp_path / "dirty.ipynb"
        write_notebook(
            notebook,
            [
                code_cell(
                    metadata={"execution": {"iopub.status.busy": "2026-01-01T00:00:00Z"}, "tags": ["keep"]},
                    outputs=[{"output_type": "stream"}],
                    execution_count=3,
                )
            ],
        )
        if os.name != "nt":
            notebook.chmod(0o640)

        assert check_notebooks.clear_notebook(notebook) is True

        loaded = check_notebooks.load_notebook(notebook).document
        assert loaded.cells[0].output_count == 0
        assert loaded.cells[0].execution_count is None
        serialized = notebook.read_text(encoding="utf-8")
        raw = json.loads(serialized)
        assert raw["cells"][0]["metadata"] == {"tags": ["keep"]}
        assert serialized.startswith('{\n  "cells": [')
        assert serialized.endswith("\n")
        if os.name != "nt":
            assert stat.S_IMODE(notebook.stat().st_mode) == 0o640

    def test_clears_execution_metadata_without_other_generated_state(self, tmp_path: Path) -> None:
        notebook = tmp_path / "metadata.ipynb"
        write_notebook(notebook, [code_cell(metadata={"execution": {"shell.execute_reply": "timestamp"}})])

        assert check_notebooks.clear_notebook(notebook) is True

        raw = json.loads(notebook.read_text(encoding="utf-8"))
        assert raw["cells"][0]["metadata"] == {}

    def test_does_not_rewrite_clean_notebook(self, tmp_path: Path) -> None:
        notebook = tmp_path / "clean.ipynb"
        write_notebook(notebook, [code_cell()])
        before = notebook.read_bytes()

        assert check_notebooks.clear_notebook(notebook) is False
        assert notebook.read_bytes() == before


class TestCli:
    def test_parse_args_discovers_notebooks_and_resolves_output(self, tmp_path: Path) -> None:
        notebook = tmp_path / "notebooks" / "valid.ipynb"
        write_notebook(notebook, [code_cell()])

        options = check_notebooks.parse_args(["lint", "--repo-root", str(tmp_path)])

        assert options.paths == (notebook,)
        assert options.output_dir == tmp_path / "target" / "notebooks"

    def test_main_reports_invalid_json_without_traceback(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        notebook = tmp_path / "invalid.ipynb"
        notebook.write_text("not-json", encoding="utf-8")

        assert check_notebooks.main(["lint", str(notebook), "--no-ruff", "--no-format", "--no-ty"]) == 1

        stderr = capsys.readouterr().err
        assert "error:" in stderr
        assert "Traceback" not in stderr

    def test_main_reports_missing_notebook_dependency_with_locked_uv_guidance(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        notebook = tmp_path / "notebooks" / "valid.ipynb"
        write_notebook(notebook, [code_cell()])

        def missing_dependency(name: str) -> None:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)

        monkeypatch.setattr(check_notebooks, "import_module", missing_dependency)

        assert check_notebooks.main(["execute", str(notebook), "--repo-root", str(tmp_path)]) == 1
        assert capsys.readouterr().err == (
            "error: nbclient is required; install `markov-chain-monte-carlo-tooling[notebook]` or run the checker through `uv run --locked`\n"
        )


@pytest.fixture(scope="session")
def tooling_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the published wheel once for outside-checkout consumer tests."""
    uv = shutil.which("uv")
    assert uv is not None
    output_dir = tmp_path_factory.mktemp("tooling-wheel")
    source_dir = output_dir / "source"
    source_dir.mkdir()
    for filename in ("LICENSE", "pyproject.toml"):
        shutil.copy2(REPO_ROOT / filename, source_dir / filename)
    shutil.copytree(REPO_ROOT / "scripts", source_dir / "scripts", ignore=shutil.ignore_patterns("__pycache__", "tests"))
    subprocess.run(  # noqa: S603 - uv is resolved and every argument is a test constant or fixture path.
        [
            uv,
            "build",
            "--wheel",
            "--offline",
            "--no-build-logs",
            "--out-dir",
            str(output_dir),
            str(source_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = tuple(output_dir.glob("*.whl"))
    assert len(wheels) == 1
    return wheels[0]


class TestBuiltToolingPackage:
    def test_wheel_declares_notebook_extra_and_console_scripts(self, tooling_wheel: Path) -> None:
        with zipfile.ZipFile(tooling_wheel) as archive:
            metadata_name = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
            entry_points_name = next(name for name in archive.namelist() if name.endswith(".dist-info/entry_points.txt"))
            metadata = Parser().parsestr(archive.read(metadata_name).decode("utf-8"))
            parser = configparser.ConfigParser()
            parser.read_string(archive.read(entry_points_name).decode("utf-8"))

        requirements = metadata.get_all("Requires-Dist") or []
        assert metadata.get_all("Provides-Extra") == ["notebook"]
        assert "# Tooling Scripts" in metadata.get_payload()
        for dependency in ("ipykernel", "matplotlib", "nbclient", "nbformat", "polars"):
            assert any(requirement.startswith(dependency) and 'extra == "notebook"' in requirement for requirement in requirements)
        assert {name: value.partition(":")[0] for name, value in parser["console_scripts"].items()} == CONSOLE_SCRIPTS

    @pytest.mark.parametrize("module_name", CONSOLE_SCRIPTS.values(), ids=CONSOLE_SCRIPTS)
    def test_wheel_console_script_help_works_outside_checkout(self, tooling_wheel: Path, tmp_path: Path, module_name: str) -> None:
        command = (
            "import importlib, pathlib, sys; "
            "module = importlib.import_module(sys.argv[1]); "
            "assert str(module.__file__).startswith(str(pathlib.Path(sys.argv[2]).resolve())); "
            "raise SystemExit(module.main(['--help']))"
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(tooling_wheel)

        result = subprocess.run(  # noqa: S603 - the current interpreter and fixed module names exercise the built wheel.
            [sys.executable, "-c", command, module_name, str(tooling_wheel)],
            cwd=tmp_path,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout.lower()

    def test_wheel_notebook_extra_supports_execution_outside_checkout(self, tooling_wheel: Path, tmp_path: Path) -> None:
        notebook = tmp_path / "notebooks" / "minimal.ipynb"
        write_notebook(notebook, [code_cell("value = 1")])
        output_dir = tmp_path / "artifacts"
        command = (
            "import check_notebooks, pathlib, sys; "
            "assert str(check_notebooks.__file__).startswith(str(pathlib.Path(sys.argv[1]).resolve())); "
            "raise SystemExit(check_notebooks.main(['execute', sys.argv[2], '--repo-root', sys.argv[3], "
            "'--output-dir', sys.argv[4], '--no-ruff', '--no-format', '--no-ty']))"
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(tooling_wheel)

        result = subprocess.run(  # noqa: S603 - the current interpreter exercises the built wheel with declared extra dependencies.
            [sys.executable, "-c", command, str(tooling_wheel), str(notebook), str(tmp_path), str(output_dir)],
            cwd=tmp_path,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        assert (output_dir / "minimal.ipynb").is_file()

    def test_wheel_bench_compare_defaults_to_invocation_repository(self, tooling_wheel: Path, tmp_path: Path) -> None:
        criterion = tmp_path / "target" / "criterion" / "chain" / "step_by_value"
        for sample, point in (("release-a", 100.0), ("new", 80.0)):
            estimate = criterion / sample / "estimates.json"
            estimate.parent.mkdir(parents=True, exist_ok=True)
            statistic = {"point_estimate": point}
            estimate.write_text(json.dumps({"median": statistic, "mean": statistic}), encoding="utf-8")
        command = (
            "import bench_compare, pathlib, sys; "
            "assert str(bench_compare.__file__).startswith(str(pathlib.Path(sys.argv[1]).resolve())); "
            "raise SystemExit(bench_compare.main(['release-a', '--revision', 'fixture']))"
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(tooling_wheel)

        result = subprocess.run(  # noqa: S603 - the current interpreter exercises the built wheel's operational default paths.
            [sys.executable, "-c", command, str(tooling_wheel)],
            cwd=tmp_path,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        assert (tmp_path / "target" / "bench-reports" / "performance.md").is_file()

    def test_wheel_archive_performance_defaults_to_invocation_repository(self, tooling_wheel: Path, tmp_path: Path) -> None:
        command = (
            "import archive_performance, pathlib, sys; "
            "assert str(archive_performance.__file__).startswith(str(pathlib.Path(sys.argv[1]).resolve())); "
            "raise SystemExit(archive_performance.main(['--rerender', 'missing.csv']))"
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(tooling_wheel)

        result = subprocess.run(  # noqa: S603 - the current interpreter exercises the built wheel's operational default paths.
            [sys.executable, "-c", command, str(tooling_wheel)],
            cwd=tmp_path,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 2
        assert str(tmp_path / "missing.csv") in result.stderr

    def test_returns_success_when_no_notebooks_are_found(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        assert check_notebooks.main(["lint", "--repo-root", str(tmp_path)]) == 0
        assert "No notebooks found." in capsys.readouterr().out

    def test_rejects_non_positive_timeout(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            check_notebooks.parse_args(["execute", "--timeout", "0"])

        assert "expected a positive integer" in capsys.readouterr().err
