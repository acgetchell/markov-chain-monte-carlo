"""Validate and execute repository notebooks."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any


def discover_notebooks(repo_root: Path) -> list[Path]:
    """Return repository notebooks in the conventional notebook directory."""
    return sorted((repo_root / "notebooks").glob("**/*.ipynb"))


def cell_source(cell: dict[str, Any]) -> str:
    """Return a code cell's source as a single string."""
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    if isinstance(source, str):
        return source
    msg = f"cell source must be a string or list of strings, got {type(source).__name__}"
    raise TypeError(msg)


def lint_notebook(path: Path) -> None:
    """Validate notebook JSON and compile code cells."""
    with path.open(encoding="utf-8") as handle:
        notebook = json.load(handle)
    for index, cell in enumerate(notebook.get("cells", []), start=1):
        if cell.get("cell_type") == "code":
            compile(cell_source(cell), f"{path}:{index}", "exec")


def lint_notebooks(paths: list[Path]) -> None:
    """Lint every notebook in `paths`."""
    for path in paths:
        lint_notebook(path)
        print(f"✓ linted {path}")


def execute_notebooks(paths: list[Path], repo_root: Path) -> None:
    """Execute every notebook in memory without writing outputs back to disk."""
    nbformat = importlib.import_module("nbformat")
    nbclient = importlib.import_module("nbclient")
    os.environ.setdefault("MPLBACKEND", "Agg")
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            notebook = nbformat.read(handle, as_version=4)
        client = nbclient.NotebookClient(
            notebook,
            timeout=120,
            kernel_name="python3",
            resources={"metadata": {"path": str(repo_root)}},
        )
        client.execute()
        print(f"✓ executed {path}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["lint", "execute"], help="notebook check mode")
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="notebooks to check; defaults to tracked *.ipynb files",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run notebook validation."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    repo_root = Path.cwd()
    paths = [repo_root / path for path in args.paths] if args.paths else discover_notebooks(repo_root)
    if not paths:
        print("No notebooks found.")
        return 0
    if args.mode == "lint":
        lint_notebooks(paths)
    else:
        execute_notebooks(paths, repo_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
