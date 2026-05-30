"""Tests for check_notebooks.py."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

import check_notebooks

if TYPE_CHECKING:
    from pathlib import Path


def write_notebook(path: Path, cells: list[dict[str, Any]]) -> None:
    """Write a minimal notebook fixture."""
    path.write_text(json.dumps({"cells": cells}), encoding="utf-8")


class TestDiscoverNotebooks:
    def test_discovers_notebooks_in_notebooks_directory(self, tmp_path: Path) -> None:
        notebooks = tmp_path / "notebooks"
        nested = notebooks / "nested"
        nested.mkdir(parents=True)
        first = notebooks / "a.ipynb"
        second = nested / "b.ipynb"
        ignored = tmp_path / "elsewhere.ipynb"
        first.write_text("{}", encoding="utf-8")
        second.write_text("{}", encoding="utf-8")
        ignored.write_text("{}", encoding="utf-8")

        assert check_notebooks.discover_notebooks(tmp_path) == [first, second]

    def test_returns_empty_list_when_notebook_directory_is_missing(self, tmp_path: Path) -> None:
        assert check_notebooks.discover_notebooks(tmp_path) == []


class TestCellSource:
    def test_joins_list_source(self) -> None:
        assert check_notebooks.cell_source({"source": ["a = 1\n", "a"]}) == "a = 1\na"

    def test_returns_string_source(self) -> None:
        assert check_notebooks.cell_source({"source": "a = 1"}) == "a = 1"

    def test_defaults_missing_source_to_empty_string(self) -> None:
        assert check_notebooks.cell_source({}) == ""

    def test_rejects_invalid_source_type(self) -> None:
        with pytest.raises(TypeError, match="cell source must be a string or list of strings"):
            check_notebooks.cell_source({"source": 42})


class TestLintNotebook:
    def test_lints_valid_code_cells(self, tmp_path: Path) -> None:
        notebook = tmp_path / "valid.ipynb"
        write_notebook(
            notebook,
            [
                {"cell_type": "markdown", "source": "# Title\n"},
                {"cell_type": "code", "source": ["value = 1\n", "value + 1"]},
            ],
        )

        check_notebooks.lint_notebook(notebook)

    def test_reports_code_cell_syntax_errors_with_cell_index(self, tmp_path: Path) -> None:
        notebook = tmp_path / "invalid.ipynb"
        write_notebook(notebook, [{"cell_type": "code", "source": "if True print('bad')"}])

        with pytest.raises(SyntaxError) as exc_info:
            check_notebooks.lint_notebook(notebook)

        assert exc_info.value.filename is not None
        assert f"{notebook}:1" in exc_info.value.filename

    def test_lint_notebooks_prints_checked_paths(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        notebook = tmp_path / "valid.ipynb"
        write_notebook(notebook, [{"cell_type": "code", "source": "value = 1"}])

        check_notebooks.lint_notebooks([notebook])

        assert f"✓ linted {notebook}" in capsys.readouterr().out


class TestExecuteNotebooks:
    def test_executes_notebooks_in_memory_with_repo_root_resource(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        repo_root = tmp_path
        notebook = tmp_path / "notebooks" / "valid.ipynb"
        notebook.parent.mkdir()
        notebook.write_text("{}", encoding="utf-8")
        calls: dict[str, Any] = {}

        fake_nbformat = SimpleNamespace(read=lambda handle, as_version: {"handle": handle.name, "as_version": as_version})

        class FakeNotebookClient:
            def __init__(self, notebook_value: dict[str, Any], **kwargs: Any) -> None:
                calls["notebook"] = notebook_value
                calls["kwargs"] = kwargs

            def execute(self) -> None:
                calls["executed"] = True

        fake_nbclient = SimpleNamespace(NotebookClient=FakeNotebookClient)

        def fake_import(name: str) -> SimpleNamespace:
            if name == "nbformat":
                return fake_nbformat
            if name == "nbclient":
                return fake_nbclient
            msg = f"unexpected import {name}"
            raise AssertionError(msg)

        monkeypatch.delenv("MPLBACKEND", raising=False)
        monkeypatch.setattr(check_notebooks.importlib, "import_module", fake_import)

        check_notebooks.execute_notebooks([notebook], repo_root)

        assert calls["notebook"]["as_version"] == 4
        assert calls["kwargs"]["timeout"] == 120
        assert calls["kwargs"]["kernel_name"] == "python3"
        assert calls["kwargs"]["resources"] == {"metadata": {"path": str(repo_root)}}
        assert calls["executed"] is True
        assert check_notebooks.os.environ["MPLBACKEND"] == "Agg"
        assert f"✓ executed {notebook}" in capsys.readouterr().out


class TestMain:
    def test_lint_mode_uses_explicit_paths(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        notebook = tmp_path / "explicit.ipynb"
        calls: dict[str, Any] = {}
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(check_notebooks, "lint_notebooks", lambda paths: calls.setdefault("paths", paths))

        assert check_notebooks.main(["lint", notebook.name]) == 0

        assert calls["paths"] == [notebook]

    def test_returns_success_when_no_notebooks_are_found(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.chdir(tmp_path)

        assert check_notebooks.main(["lint"]) == 0

        assert "No notebooks found." in capsys.readouterr().out
