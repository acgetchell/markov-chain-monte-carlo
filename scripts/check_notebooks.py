"""Validate, execute, and clean repository notebooks."""

import argparse
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Literal, TypeIs, cast

from subprocess_utils import ExecutableNotFoundError, run_safe_command

type CheckMode = Literal["lint", "execute", "clear"]
type CellType = Literal["code", "markdown", "raw"]

VALID_CELL_TYPES: set[CellType] = {"code", "markdown", "raw"}
NOTEBOOK_ROOT_FIELDS = frozenset({"cells", "metadata", "nbformat", "nbformat_minor"})
COMMON_CELL_FIELDS = frozenset({"cell_type", "id", "metadata", "source"})
CELL_ALLOWED_FIELDS: dict[CellType, frozenset[str]] = {
    "code": COMMON_CELL_FIELDS | {"execution_count", "outputs"},
    "markdown": COMMON_CELL_FIELDS | {"attachments"},
    "raw": COMMON_CELL_FIELDS | {"attachments"},
}
CELL_REQUIRED_FIELDS: dict[CellType, frozenset[str]] = {
    "code": COMMON_CELL_FIELDS | {"execution_count", "outputs"},
    "markdown": COMMON_CELL_FIELDS,
    "raw": COMMON_CELL_FIELDS,
}
PYTHON_CELL_MAGICS = frozenset({"capture", "prun", "time", "timeit"})
CELL_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
JSON_MIME_RE = re.compile(r"^application/(.*\+)?json$")
RUFF_LOCATION_RE = re.compile(r"^.+?:(?P<line>\d+):(?P<column>\d+):\s+(?P<message>.+)$")
TY_LOCATION_RE = re.compile(r"^.+?:(?P<line>\d+):(?P<column>\d+):\s+(?P<message>.+)$")
DEFAULT_OUTPUT_MODE = 0o644


def is_json_object(value: object) -> TypeIs[dict[str, object]]:
    """Return whether a raw JSON value is an object with string keys."""
    return isinstance(value, dict) and all(isinstance(key, str) for key in value)


def is_string_list(value: object) -> TypeIs[list[str]]:
    """Return whether a raw JSON value is a list of strings."""
    return isinstance(value, list) and all(isinstance(part, str) for part in value)


@dataclass(frozen=True, slots=True)
class NotebookCell:
    """Validated notebook cell data used by linting."""

    index: int
    cell_id: str
    cell_type: CellType
    source: str
    output_count: int
    execution_count: int | None

    @property
    def is_code(self) -> bool:
        """Return whether this is a Python code cell."""
        return self.cell_type == "code"


@dataclass(frozen=True, slots=True)
class NotebookDocument:
    """Validated notebook structure loaded from JSON."""

    path: Path
    nbformat: int
    nbformat_minor: int
    cells: tuple[NotebookCell, ...]


@dataclass(frozen=True, slots=True)
class LoadedNotebook:
    """Raw notebook JSON paired with its validated representation."""

    raw: dict[str, object]
    document: NotebookDocument


@dataclass(frozen=True, slots=True)
class CodeSnapshot:
    """Notebook code extracted into one Python source string."""

    source: str
    line_to_cell: dict[int, int]


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """Cell-aware notebook validation failure."""

    cell: int | None
    message: str


@dataclass(frozen=True, slots=True)
class CheckOptions:
    """Parsed command-line options trusted by the checker."""

    mode: CheckMode
    paths: tuple[Path, ...]
    repo_root: Path
    output_dir: Path
    timeout: int
    run_ruff: bool
    run_format: bool
    run_ty: bool


def discover_notebooks(repo_root: Path) -> list[Path]:
    """Return repository notebooks while excluding Jupyter checkpoints."""
    notebook_root = repo_root / "notebooks"
    notebooks = (path for path in notebook_root.glob("**/*.ipynb") if ".ipynb_checkpoints" not in path.parts)
    return sorted(notebooks, key=lambda path: path.relative_to(notebook_root).as_posix())


def parse_cell_type(value: object, *, path: Path, index: int) -> CellType:
    """Parse a raw notebook cell type."""
    if not isinstance(value, str):
        msg = f"{path}: cell {index}: cell_type must be a string"
        raise TypeError(msg)
    if value not in VALID_CELL_TYPES:
        expected = ", ".join(sorted(VALID_CELL_TYPES))
        msg = f"{path}: cell {index}: cell_type must be one of {expected}; got {value!r}"
        raise ValueError(msg)
    return value


def parse_cell_source(value: object, *, path: Path, index: int) -> str:
    """Parse a cell source into one source string."""
    if isinstance(value, str):
        return value
    if is_string_list(value):
        return "".join(value)
    if isinstance(value, list):
        msg = f"{path}: cell {index}: source list must contain only strings"
        raise TypeError(msg)
    msg = f"{path}: cell {index}: source must be a string or list of strings"
    raise TypeError(msg)


def parse_cell_id(value: object, *, path: Path, index: int) -> str:
    """Parse a stable nbformat 4.5 notebook cell ID."""
    if not isinstance(value, str) or not value.strip():
        msg = f"{path}: cell {index}: id must be a nonempty string"
        raise TypeError(msg)
    if CELL_ID_RE.fullmatch(value) is None:
        msg = f"{path}: cell {index}: id must contain 1-64 ASCII letters, digits, hyphens, or underscores; got {value!r}"
        raise ValueError(msg)
    return value


def parse_outputs(value: object, *, path: Path, index: int) -> int:
    """Parse a code cell's output collection into its count."""
    if not isinstance(value, list):
        msg = f"{path}: cell {index}: outputs must be a list"
        raise TypeError(msg)
    return len(value)


def parse_execution_count(value: object, *, path: Path, index: int) -> int | None:
    """Parse a code cell execution count."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{path}: cell {index}: execution_count must be an integer or null"
        raise TypeError(msg)
    if value < 0:
        msg = f"{path}: cell {index}: execution_count must be nonnegative or null"
        raise ValueError(msg)
    return value


def validate_attachments(value: object, *, path: Path, index: int) -> None:
    """Validate an nbformat attachment mapping and its MIME bundles."""
    if not is_json_object(value):
        msg = f"{path}: cell {index}: attachments must be a JSON object"
        raise TypeError(msg)
    for filename, raw_bundle in value.items():
        if not is_json_object(raw_bundle):
            msg = f"{path}: cell {index}: attachment {filename!r} must contain a JSON object MIME bundle"
            raise TypeError(msg)
        for mime_type, payload in raw_bundle.items():
            if JSON_MIME_RE.fullmatch(mime_type) is not None:
                continue
            if not isinstance(payload, str) and not is_string_list(payload):
                msg = f"{path}: cell {index}: attachment {filename!r} MIME value {mime_type!r} must be a string or list of strings"
                raise TypeError(msg)


def validate_cell_fields(value: dict[str, object], *, cell_type: CellType, path: Path, index: int) -> None:
    """Reject missing, unknown, and cell-type-forbidden nbformat fields."""
    present = frozenset(value)
    missing = sorted(CELL_REQUIRED_FIELDS[cell_type] - present)
    if missing:
        msg = f"{path}: cell {index}: missing required {cell_type} field(s): {', '.join(missing)}"
        raise TypeError(msg)
    unsupported = sorted(present - CELL_ALLOWED_FIELDS[cell_type])
    if unsupported:
        msg = f"{path}: cell {index}: unsupported field(s) for {cell_type} cell: {', '.join(unsupported)}"
        raise ValueError(msg)
    if "attachments" in value:
        validate_attachments(value["attachments"], path=path, index=index)


def parse_cell(value: object, *, path: Path, index: int) -> NotebookCell:
    """Parse one raw notebook cell into trusted data."""
    if not is_json_object(value):
        msg = f"{path}: cell {index}: expected a JSON object"
        raise TypeError(msg)
    cell_type = parse_cell_type(value.get("cell_type"), path=path, index=index)
    validate_cell_fields(value, cell_type=cell_type, path=path, index=index)
    if not is_json_object(value.get("metadata")):
        msg = f"{path}: cell {index}: metadata must be a JSON object"
        raise TypeError(msg)
    output_count = parse_outputs(value.get("outputs"), path=path, index=index) if cell_type == "code" else 0
    execution_count = parse_execution_count(value.get("execution_count"), path=path, index=index) if cell_type == "code" else None
    return NotebookCell(
        index=index,
        cell_id=parse_cell_id(value.get("id"), path=path, index=index),
        cell_type=cell_type,
        source=parse_cell_source(value.get("source"), path=path, index=index),
        output_count=output_count,
        execution_count=execution_count,
    )


def load_notebook(path: Path) -> LoadedNotebook:
    """Load and validate one nbformat 4 notebook JSON document."""
    if not path.is_file():
        msg = f"notebook does not exist or is not a file: {path}"
        raise FileNotFoundError(msg)
    with path.open(encoding="utf-8") as handle:
        raw_value: object = json.load(handle)
    if not is_json_object(raw_value):
        msg = f"{path}: notebook JSON must be an object"
        raise TypeError(msg)
    unsupported = sorted(frozenset(raw_value) - NOTEBOOK_ROOT_FIELDS)
    if unsupported:
        msg = f"{path}: unsupported notebook field(s): {', '.join(unsupported)}"
        raise ValueError(msg)
    nbformat = raw_value.get("nbformat")
    if isinstance(nbformat, bool) or not isinstance(nbformat, int):
        msg = f"{path}: nbformat must be integer 4, got {nbformat!r}"
        raise TypeError(msg)
    if nbformat != 4:
        msg = f"{path}: expected nbformat 4, got {nbformat!r}"
        raise ValueError(msg)
    nbformat_minor = raw_value.get("nbformat_minor")
    if isinstance(nbformat_minor, bool) or not isinstance(nbformat_minor, int):
        msg = f"{path}: nbformat_minor must be an integer"
        raise TypeError(msg)
    if nbformat_minor < 5:
        msg = f"{path}: expected nbformat 4.5 or newer for stable cell IDs, got 4.{nbformat_minor}"
        raise ValueError(msg)
    metadata = raw_value.get("metadata")
    if not isinstance(metadata, dict):
        msg = f"{path}: metadata must be a JSON object"
        raise TypeError(msg)
    raw_cells = raw_value.get("cells")
    if not isinstance(raw_cells, list):
        msg = f"{path}: cells must be a list"
        raise TypeError(msg)
    cells = tuple(parse_cell(cell, path=path, index=index) for index, cell in enumerate(raw_cells, start=1))
    cell_ids = [cell.cell_id for cell in cells]
    if len(cell_ids) != len(set(cell_ids)):
        msg = f"{path}: cell IDs must be unique"
        raise ValueError(msg)
    return LoadedNotebook(
        raw=raw_value,
        document=NotebookDocument(path=path, nbformat=4, nbformat_minor=nbformat_minor, cells=cells),
    )


def code_cells(notebook: NotebookDocument) -> tuple[NotebookCell, ...]:
    """Return the notebook's code cells."""
    return tuple(cell for cell in notebook.cells if cell.is_code)


def neutralize_ipython_syntax(source: str) -> str:
    """Replace IPython-only syntax while preserving Python lines and line numbers."""
    lines = source.splitlines(keepends=True)
    first_content = next((line.lstrip() for line in lines if line.strip()), "")
    if first_content.startswith("%%"):
        magic_name = first_content[2:].split(maxsplit=1)[0].lower()
        if magic_name not in PYTHON_CELL_MAGICS:
            return "".join(_neutralize_ipython_line(line, whole_cell=True) for line in lines)
    return "".join(_neutralize_ipython_line(line) for line in lines)


def _neutralize_ipython_line(line: str, *, whole_cell: bool = False) -> str:
    """Neutralize one IPython-only line without changing its line count."""
    stripped = line.lstrip()
    if not line.strip() or (not whole_cell and not stripped.startswith(("%", "!"))):
        return line
    ending = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else "\r" if line.endswith("\r") else ""
    content = line[: -len(ending)] if ending else line
    indent = content[: len(content) - len(content.lstrip())]
    replacement = f"{indent}# IPython syntax" if whole_cell or not indent else f"{indent}pass  # IPython syntax"
    return replacement + ending


def extract_code(notebook: NotebookDocument) -> CodeSnapshot:
    """Extract code cells and retain generated-line to source-cell mapping."""
    chunks: list[str] = []
    line_to_cell: dict[int, int] = {}
    current_line = 1
    cells = code_cells(notebook)
    for position, cell in enumerate(cells):
        marker = f"# %% notebook cell {cell.index} ({cell.cell_id})\n"
        chunks.append(marker)
        line_to_cell[current_line] = cell.index
        current_line += 1
        source_lines = neutralize_ipython_syntax(cell.source).splitlines(keepends=True)
        if not source_lines:
            source_lines = ["\n"]
        for source_line in source_lines:
            chunks.append(source_line)
            line_to_cell[current_line] = cell.index
            current_line += 1
        if not source_lines[-1].endswith(("\n", "\r")):
            chunks.append("\n")
        if position < len(cells) - 1:
            chunks.append("\n")
            line_to_cell[current_line] = cell.index
            current_line += 1
    return CodeSnapshot(source="".join(chunks), line_to_cell=line_to_cell)


def code_cell_diagnostics(notebook: NotebookDocument) -> list[Diagnostic]:
    """Compile code cells and reject committed outputs or execution counts."""
    diagnostics: list[Diagnostic] = []
    for cell in code_cells(notebook):
        try:
            compile(neutralize_ipython_syntax(cell.source), f"{notebook.path}:cell-{cell.index}:{cell.cell_id}", "exec")
        except SyntaxError as error:
            location = f"line {error.lineno}" if error.lineno is not None else "unknown line"
            diagnostics.append(Diagnostic(cell.index, f"syntax error at {location}: {error.msg}"))
        if cell.output_count:
            diagnostics.append(Diagnostic(cell.index, f"has {cell.output_count} output block(s); clear outputs before committing"))
        if cell.execution_count is not None:
            diagnostics.append(Diagnostic(cell.index, f"execution_count={cell.execution_count}; clear execution counts before committing"))
    return diagnostics


def tool_result(
    command: str,
    args: list[str],
    *,
    cwd: Path,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str] | Diagnostic:
    """Run a bounded external checker and convert process failures to diagnostics."""
    try:
        return run_safe_command(command, args, cwd=cwd, input=input_text, timeout=30, check=False)
    except ExecutableNotFoundError:
        return Diagnostic(None, f"{command} is required; run the checker through `uv run --locked`")
    except subprocess.TimeoutExpired as error:
        return Diagnostic(None, f"{command} timed out after {error.timeout} seconds")


def parse_line_diagnostics(
    output: str,
    *,
    tool: str,
    pattern: re.Pattern[str],
    line_to_cell: dict[int, int],
) -> list[Diagnostic]:
    """Parse concise tool diagnostics and map generated lines to notebook cells."""
    diagnostics: list[Diagnostic] = []
    for line in output.splitlines():
        if not line.strip() or line.startswith("Found ") or line == "All checks passed!":
            continue
        match = pattern.match(line)
        if match is None:
            diagnostics.append(Diagnostic(None, f"{tool}: {line}"))
            continue
        cell = line_to_cell.get(int(match.group("line")))
        diagnostics.append(Diagnostic(cell, f"{tool}: {match.group('message')}"))
    return diagnostics


def ruff_check_diagnostics(path: Path, snapshot: CodeSnapshot, repo_root: Path) -> list[Diagnostic]:
    """Run Ruff linting on extracted notebook code."""
    result = tool_result(
        "ruff",
        [
            "check",
            "--output-format",
            "concise",
            "--stdin-filename",
            f"{path.stem}_notebook.py",
            "--extend-ignore",
            "INP001,B018",
            "-",
        ],
        cwd=repo_root,
        input_text=snapshot.source,
    )
    if isinstance(result, Diagnostic):
        return [result]
    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    if result.returncode == 0:
        return []
    if result.returncode != 1:
        return [Diagnostic(None, f"ruff check failed with exit code {result.returncode}: {output.strip()}")]
    return parse_line_diagnostics(output, tool="ruff check", pattern=RUFF_LOCATION_RE, line_to_cell=snapshot.line_to_cell)


def ruff_format_diagnostics(path: Path, snapshot: CodeSnapshot, repo_root: Path) -> list[Diagnostic]:
    """Run Ruff formatting checks on extracted notebook code."""
    result = tool_result(
        "ruff",
        ["format", "--check", "--stdin-filename", f"{path.stem}_notebook.py", "-"],
        cwd=repo_root,
        input_text=snapshot.source,
    )
    if isinstance(result, Diagnostic):
        return [result]
    output = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
    if result.returncode == 0:
        return []
    if result.returncode != 1:
        return [Diagnostic(None, f"ruff format failed with exit code {result.returncode}: {output}")]
    return [Diagnostic(None, f"ruff format: extracted notebook code is not formatted: {output}")]


def ty_diagnostics(path: Path, snapshot: CodeSnapshot, repo_root: Path) -> list[Diagnostic]:
    """Run Ty on extracted notebook code."""
    with tempfile.TemporaryDirectory(prefix="notebook-check-") as temporary_directory:
        extracted_path = Path(temporary_directory) / f"{path.stem}_notebook.py"
        extracted_path.write_text(snapshot.source, encoding="utf-8")
        result = tool_result(
            "ty",
            ["check", "--project", str(repo_root), "--output-format", "concise", str(extracted_path)],
            cwd=repo_root,
        )
    if isinstance(result, Diagnostic):
        return [result]
    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    if result.returncode == 0:
        return []
    if result.returncode != 1:
        return [Diagnostic(None, f"ty failed with exit code {result.returncode}: {output.strip()}")]
    return parse_line_diagnostics(output, tool="ty", pattern=TY_LOCATION_RE, line_to_cell=snapshot.line_to_cell)


def lint_notebook(path: Path, options: CheckOptions) -> list[Diagnostic]:
    """Validate one notebook and return every discovered diagnostic."""
    notebook = load_notebook(path).document
    diagnostics = code_cell_diagnostics(notebook)
    snapshot = extract_code(notebook)
    if options.run_ruff:
        diagnostics.extend(ruff_check_diagnostics(path, snapshot, options.repo_root))
    if options.run_format:
        diagnostics.extend(ruff_format_diagnostics(path, snapshot, options.repo_root))
    if options.run_ty:
        diagnostics.extend(ty_diagnostics(path, snapshot, options.repo_root))
    return diagnostics


def lint_notebooks(paths: tuple[Path, ...], options: CheckOptions) -> int:
    """Lint every requested notebook and report cell-aware failures."""
    failed = False
    for path in paths:
        try:
            diagnostics = lint_notebook(path, options)
        except (OSError, TypeError, ValueError) as error:
            diagnostics = [Diagnostic(None, str(error))]
        for diagnostic in diagnostics:
            location = f"cell {diagnostic.cell}" if diagnostic.cell is not None else "notebook"
            print(f"{path}: {location}: error: {diagnostic.message}", file=sys.stderr)
        if diagnostics:
            failed = True
        else:
            print(f"OK linted {path}")
    return int(failed)


def output_path_for(path: Path, *, repo_root: Path, output_dir: Path) -> Path:
    """Return an executed-notebook path that cannot alias source notebooks."""
    source_path = path.resolve()
    notebook_root = (repo_root / "notebooks").resolve()
    try:
        relative = source_path.relative_to(notebook_root)
    except ValueError as error:
        msg = f"executed notebook must be under {notebook_root}: {source_path}"
        raise ValueError(msg) from error
    output_path = (output_dir.resolve() / relative).resolve()
    if output_path.is_relative_to(notebook_root):
        msg = f"executed notebook output must remain outside the source notebook directory {notebook_root}: {output_path}"
        raise ValueError(msg)
    return output_path


def destination_mode(path: Path) -> int:
    """Return an existing destination's mode or the explicit mode for a new artifact."""
    try:
        return stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        return DEFAULT_OUTPUT_MODE


def write_executed_notebook(nbformat: Any, notebook: Any, output_path: Path) -> None:
    """Atomically publish an executed notebook artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_mode = destination_mode(output_path)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.stem}-",
            suffix=".ipynb",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            nbformat.write(notebook, handle)
        temporary_path.chmod(output_mode)
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def execute_notebook(path: Path, options: CheckOptions) -> Path:
    """Execute one notebook and publish the result under the output directory."""
    output_dir = options.output_dir.resolve()
    output_path = output_path_for(path, repo_root=options.repo_root, output_dir=output_dir)
    nbclient = import_module("nbclient")
    nbformat = import_module("nbformat")
    ipython_dir = output_dir / ".ipython"
    matplotlib_dir = output_dir / ".matplotlib"
    ipython_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    isolated_environment = {
        "IPYTHONDIR": str(ipython_dir),
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(matplotlib_dir),
    }
    previous_environment = {name: os.environ.get(name) for name in isolated_environment}
    os.environ.update(isolated_environment)
    try:
        with path.open(encoding="utf-8") as handle:
            notebook = nbformat.read(handle, as_version=4)
        client = nbclient.NotebookClient(
            notebook,
            timeout=options.timeout,
            kernel_name="python3",
            resources={"metadata": {"path": str(options.repo_root)}},
        )
        client.execute()
        write_executed_notebook(nbformat, notebook, output_path)
    finally:
        for name, previous_value in previous_environment.items():
            if previous_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous_value
    print(f"OK executed {path} -> {output_path}")
    return output_path


def execute_notebooks(paths: tuple[Path, ...], options: CheckOptions) -> None:
    """Execute every requested notebook."""
    for path in paths:
        execute_notebook(path, options)


def write_json_atomic(path: Path, value: dict[str, object]) -> None:
    """Atomically replace a JSON document after complete serialization."""
    serialized = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    output_mode = destination_mode(path)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.stem}-",
            suffix=path.suffix,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(serialized)
        temporary_path.chmod(output_mode)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def clear_notebook(path: Path) -> bool:
    """Clear code-cell outputs and counts, returning whether the file changed."""
    loaded = load_notebook(path)
    raw_cells = loaded.raw["cells"]
    if not isinstance(raw_cells, list):
        msg = "validated notebook cells lost their list invariant"
        raise TypeError(msg)
    changed = False
    for raw_cell in raw_cells:
        if not is_json_object(raw_cell):
            msg = "validated notebook cell lost its object invariant"
            raise TypeError(msg)
        if raw_cell.get("cell_type") == "code":
            metadata = raw_cell.get("metadata")
            if not is_json_object(metadata):
                msg = "validated code-cell metadata lost its object invariant"
                raise TypeError(msg)
            if raw_cell["outputs"]:
                empty_outputs: list[object] = []
                raw_cell["outputs"] = empty_outputs
                changed = True
            if raw_cell["execution_count"] is not None:
                raw_cell["execution_count"] = None
                changed = True
            if "execution" in metadata:
                del metadata["execution"]
                changed = True
    if not changed:
        return False
    write_json_atomic(path, loaded.raw)
    return True


def clear_notebooks(paths: tuple[Path, ...]) -> None:
    """Clear generated state from every requested source notebook."""
    for path in paths:
        status = "cleared" if clear_notebook(path) else "already clean"
        print(f"OK {status}: {path}")


def parse_positive_int(value: str) -> int:
    """Parse a positive integer command-line value."""
    try:
        parsed = int(value)
    except ValueError as error:
        msg = f"expected a positive integer, got {value!r}"
        raise argparse.ArgumentTypeError(msg) from error
    if parsed <= 0:
        msg = f"expected a positive integer, got {value!r}"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def parse_args(argv: list[str]) -> CheckOptions:
    """Parse weak command-line values into trusted checker options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("lint", "execute", "clear"))
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="notebooks to check; execution requires paths under <repo-root>/notebooks",
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="repository root used for discovery and execution")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("target/notebooks"),
        help="executed-notebook artifact directory; must remain outside <repo-root>/notebooks",
    )
    parser.add_argument("--timeout", type=parse_positive_int, default=120, help="per-cell execution timeout in seconds")
    parser.add_argument("--no-ruff", action="store_true", help="skip Ruff lint checks")
    parser.add_argument("--no-format", action="store_true", help="skip Ruff format checks")
    parser.add_argument("--no-ty", action="store_true", help="skip Ty checks")
    namespace = parser.parse_args(argv)
    repo_root = cast("Path", namespace.repo_root).resolve()
    raw_paths = cast("list[Path]", namespace.paths)
    paths = tuple(path.resolve() if path.is_absolute() else (repo_root / path).resolve() for path in raw_paths)
    if not paths:
        paths = tuple(discover_notebooks(repo_root))
    output_dir = cast("Path", namespace.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    return CheckOptions(
        mode=cast("CheckMode", namespace.mode),
        paths=paths,
        repo_root=repo_root,
        output_dir=output_dir,
        timeout=cast("int", namespace.timeout),
        run_ruff=not cast("bool", namespace.no_ruff),
        run_format=not cast("bool", namespace.no_format),
        run_ty=not cast("bool", namespace.no_ty),
    )


def run(options: CheckOptions) -> int:
    """Run one validated checker action."""
    if not options.paths:
        print("No notebooks found.")
        return 0
    if options.mode == "lint":
        return lint_notebooks(options.paths, options)
    if options.mode == "execute":
        execute_notebooks(options.paths, options)
        return 0
    clear_notebooks(options.paths)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run notebook validation with concise boundary-error reporting."""
    try:
        return run(parse_args(sys.argv[1:] if argv is None else argv))
    except ModuleNotFoundError as error:
        dependency = error.name or "notebook dependency"
        print(
            f"error: {dependency} is required; install `markov-chain-monte-carlo-tooling[notebook]` or run the checker through `uv run --locked`",
            file=sys.stderr,
        )
        return 1
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
