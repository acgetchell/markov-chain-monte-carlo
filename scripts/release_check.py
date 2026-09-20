"""Check shared release metadata plus MCMC's publication and evidence policies."""

import argparse
import contextlib
import io
import os
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TypeGuard

SKIP_DIRS = frozenset({".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tmp_pycache", ".venv", "archive", "archives", "target", "tests"})
SKIP_MARKDOWN_FILES = frozenset({"CHANGELOG.md"})
ZENODO_CONCEPT_DOI = "10.5281/zenodo.20033111"
type ParsedObject = dict[str, object]


class ReleaseCheckError(ValueError):
    """Raised when release metadata cannot be parsed unambiguously."""


def _is_parsed_object(value: object) -> TypeGuard[ParsedObject]:
    """Return true when a parsed TOML value is an object with string keys."""
    return isinstance(value, dict) and all(isinstance(key, str) for key in value)


def _require_parsed_object(value: object, context: str) -> ParsedObject:
    """Return *value* as a TOML object or raise with context."""
    if not _is_parsed_object(value):
        msg = f"{context} is not a TOML object"
        raise ReleaseCheckError(msg)
    return value


def _read_toml(path: Path) -> ParsedObject:
    """Parse *path* as TOML and return its root table."""
    data: object = tomllib.loads(path.read_text(encoding="utf-8"))
    return _require_parsed_object(data, str(path))


def _require_table(data: ParsedObject, key: str, path: Path) -> ParsedObject:
    """Return a required child TOML table."""
    table = data.get(key)
    if not _is_parsed_object(table):
        msg = f"{path} is missing a [{key}] table"
        raise ReleaseCheckError(msg)
    return table


def _require_string(data: ParsedObject, key: str, context: str) -> str:
    """Return a required string field."""
    value = data.get(key)
    if not isinstance(value, str):
        msg = f"{context} is missing a string {key}"
        raise ReleaseCheckError(msg)
    return value


@dataclass(frozen=True, slots=True)
class PackageInfo:
    """Cargo package identity that defines the expected release version."""

    name: str
    version: str
    repository_slug: str


def _github_repository_slug(repository: str, path: Path) -> str:
    """Return the ``owner/repository`` slug from Cargo's repository URL."""
    match = re.fullmatch(r"https://github\.com/(?P<slug>[^/]+/[^/]+?)(?:\.git)?/?", repository)
    if match is None:
        msg = f"{path} [package] repository must be a GitHub HTTPS URL, found {repository!r}"
        raise ReleaseCheckError(msg)
    return match.group("slug")


def _read_cargo_package_info(cargo_toml: Path) -> PackageInfo:
    """Read the Cargo package name, version, and repository."""
    package = _require_table(_read_toml(cargo_toml), "package", cargo_toml)
    repository = _require_string(package, "repository", f"{cargo_toml} [package]")
    return PackageInfo(
        name=_require_string(package, "name", f"{cargo_toml} [package]"),
        version=_require_string(package, "version", f"{cargo_toml} [package]"),
        repository_slug=_github_repository_slug(repository, cargo_toml),
    )


def _iter_markdown_files(root: Path) -> list[Path]:
    """Return active Markdown files that can carry current release references."""
    markdown_files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        relative_dir = Path(dirpath).relative_to(root)
        dirnames[:] = [dirname for dirname in dirnames if not (set((relative_dir / dirname).parts) & SKIP_DIRS)]
        markdown_files.extend(Path(dirpath) / filename for filename in filenames if filename.endswith(".md") and filename not in SKIP_MARKDOWN_FILES)
    return sorted(markdown_files, key=lambda path: path.relative_to(root).as_posix())


def _readme_tag_link_pattern(repository_slug: str, *, include_main: bool = False) -> re.Pattern[str]:
    """Match owned README links while retaining their artifact paths."""
    escaped_slug = re.escape(repository_slug)
    branch = "main|" if include_main else ""
    return re.compile(
        rf"https://(?:github\.com/{escaped_slug}/(?:blob|raw|tree)/|raw\.githubusercontent\.com/{escaped_slug}/)"
        r"(?:v(?P<version>[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?)"
        rf"|(?P<revision>{branch}[0-9a-f]{{7,40}}))(?=/|$|[^0-9A-Za-z._+-])"
        r'(?P<path>/[^\s)\]>"?#]*)?'
    )


def is_performance_artifact_link(match: re.Match[str]) -> bool:
    """Keep measured artifacts pinned to their measured release during metadata updates."""
    path = match.group("path") or ""
    return path == "/docs/PERFORMANCE.md" or path.startswith("/docs/archive/performance/")


def concept_doi_problems(root: Path) -> list[str]:
    """Require the fixed DOI before shared metadata synchronization can repair references."""
    problems: list[str] = []
    doi_patterns = {
        "CITATION.cff": re.compile(r"^doi:\s*['\"]?(?P<doi>[^'\"\s]+)['\"]?\s*(?:#.*)?$", re.MULTILINE),
        "README.md": re.compile(r"\[!\[DOI\]\([^)]*\)\]\(https://doi\.org/(?P<doi>[^)]+)\)"),
        "REFERENCES.md": re.compile(r"^- DOI: <https://doi\.org/(?P<doi>[^>]+)>\s*$", re.MULTILINE),
    }
    for name, pattern in doi_patterns.items():
        values = [match["doi"] for match in pattern.finditer((root / name).read_text(encoding="utf-8"))]
        if values != [ZENODO_CONCEPT_DOI]:
            problems.append(f"{name}: requires exactly one concept DOI {ZENODO_CONCEPT_DOI}; found {values}")
    return problems


def consumer_problems(root: Path) -> list[str]:
    """Keep required publication surfaces and measured evidence under MCMC ownership."""
    for name in ("Cargo.lock", "pyproject.toml", "uv.lock", "CITATION.cff", "CHANGELOG.md", "README.md", "REFERENCES.md"):
        if not (root / name).is_file():
            raise ReleaseCheckError(f"required release surface is missing: {name}")
    package = _read_cargo_package_info(root / "Cargo.toml")
    problems = concept_doi_problems(root)
    benchmark = re.compile(r"just performance-release\s+v(?P<version>[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?)(?=\s|`|$)")
    links = _readme_tag_link_pattern(package.repository_slug)
    for path in _iter_markdown_files(root):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            values = [match["version"] for match in benchmark.finditer(line)]
            if path == root / "README.md":
                values.extend(match["version"] or match["revision"] for match in links.finditer(line) if not is_performance_artifact_link(match))
            for value in values:
                if value != package.version:
                    problems.append(f"{path.relative_to(root)}:{line_number}: active release reference {value}; expected {package.version}")
    return problems


def main(argv: list[str] | None = None) -> int:
    """Validate the shared contract, then the consumer's additional release policies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    root = parser.parse_args(argv).root.resolve()
    from research_repo_tools.cli import main as shared_main  # noqa: PLC0415 - optional development tooling, after --help.

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        status = shared_main(["--root", str(root), "release", "check", "--final-release"])
    if status:
        return status
    try:
        problems = consumer_problems(root)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as error:
        print(f"Could not check MCMC release policy: {error}", file=sys.stderr)
        return 1
    for problem in problems:
        print(problem, file=sys.stderr)
    if not problems:
        print(output.getvalue(), end="")
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
