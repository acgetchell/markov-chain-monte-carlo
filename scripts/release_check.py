"""Validate synchronized release metadata against the Cargo package version."""

import argparse
import os
import re
import sys
import tomllib
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, TypeGuard

if TYPE_CHECKING:
    from collections.abc import Sequence

SKIP_DIRS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tmp_pycache",
        ".venv",
        "archive",
        "target",
        "tests",
    }
)
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


@dataclass(frozen=True, slots=True)
class PythonProjectInfo:
    """Python support-package identity used to locate its uv lock entry."""

    name: str
    version: str


class ReferenceKind(StrEnum):
    """A release surface whose version must match Cargo.toml."""

    BENCHMARK_CURRENT_TAG = "release benchmark current tag"
    CARGO_ADD = "cargo add command"
    CARGO_LOCK = "Cargo.lock root package"
    CHANGELOG = "latest generated changelog release"
    CHANGELOG_COMPARISON = "current changelog comparison target"
    CITATION = "CITATION.cff version"
    DEPENDENCY_SNIPPET = "documentation dependency snippet"
    PYPROJECT = "pyproject.toml project"
    README_TAG_LINK = "README tag-pinned link"
    UV_LOCK = "uv.lock editable package"


@dataclass(frozen=True, slots=True)
class VersionReference:
    """A parsed release-version reference with source location."""

    path: Path
    line: int
    version: str
    kind: ReferenceKind
    text: str


@dataclass(frozen=True, slots=True)
class VersionMismatch:
    """A release-version reference that does not match Cargo.toml."""

    reference: VersionReference
    package: PackageInfo


class MetadataKind(StrEnum):
    """A release metadata field that must agree across publication surfaces."""

    CHANGELOG_DATE = "latest changelog release date"
    CITATION_DATE = "CITATION.cff date-released"
    CITATION_DOI = "CITATION.cff concept DOI"
    README_DOI = "README DOI badge target"
    REFERENCES_DOI = "REFERENCES.md concept DOI"


@dataclass(frozen=True, slots=True)
class MetadataReference:
    """A parsed non-version release metadata reference with source location."""

    path: Path
    line: int
    value: str
    kind: MetadataKind
    text: str


@dataclass(frozen=True, slots=True)
class MetadataMismatch:
    """A release metadata reference that differs from its canonical value."""

    reference: MetadataReference
    expected: str


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


def _read_python_project_info(pyproject_toml: Path) -> PythonProjectInfo:
    """Read the Python support package name and version."""
    project = _require_table(_read_toml(pyproject_toml), "project", pyproject_toml)
    return PythonProjectInfo(
        name=_require_string(project, "name", f"{pyproject_toml} [project]"),
        version=_require_string(project, "version", f"{pyproject_toml} [project]"),
    )


def _toml_table_key_line(path: Path, table_name: str, key: str) -> int:
    """Return the line number for *key* in a TOML table."""
    current_table: str | None = None
    key_re = re.compile(rf"^{re.escape(key)}\s*=")
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current_table = stripped.strip("[]")
        elif current_table == table_name and key_re.match(stripped):
            return line_number
    msg = f"{path} [{table_name}] is missing {key}"
    raise ReleaseCheckError(msg)


def _version_reference(path: Path, line: int, version: str, kind: ReferenceKind) -> VersionReference:
    """Build a version reference and include the source line text."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not 1 <= line <= len(lines):
        msg = f"{path} has no line {line} for {kind}"
        raise ReleaseCheckError(msg)
    return VersionReference(path=path, line=line, version=version, kind=kind, text=lines[line - 1].strip())


def _package_entries(path: Path) -> list[ParsedObject]:
    """Return TOML ``[[package]]`` entries from a lockfile."""
    packages = _read_toml(path).get("package")
    if not isinstance(packages, list):
        msg = f"{path} is missing [[package]] entries"
        raise ReleaseCheckError(msg)
    return [_require_parsed_object(package, f"{path} [[package]] entry {index}") for index, package in enumerate(packages, start=1)]


def _array_table_key_line(path: Path, table_name: str, table_index: int, key: str) -> int:
    """Return the line for *key* inside the requested array-table entry."""
    current_index = -1
    in_target_table = False
    key_re = re.compile(rf"^{re.escape(key)}\s*=")
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if stripped == f"[[{table_name}]]":
            current_index += 1
            in_target_table = current_index == table_index
        elif stripped.startswith("[["):
            in_target_table = False
        elif in_target_table and key_re.match(stripped):
            return line_number
    msg = f"{path} [[{table_name}]] entry {table_index + 1} is missing {key}"
    raise ReleaseCheckError(msg)


def _single_package_reference(
    path: Path,
    entries: list[ParsedObject],
    candidate_indices: list[int],
    package_name: str,
    kind: ReferenceKind,
) -> VersionReference:
    """Return the only matching package reference or raise on ambiguity."""
    if len(candidate_indices) != 1:
        msg = f"{path} must contain exactly one {kind} named {package_name!r}; found {len(candidate_indices)}"
        raise ReleaseCheckError(msg)
    index = candidate_indices[0]
    version = _require_string(entries[index], "version", f"{path} [[package]] entry {index + 1}")
    line = _array_table_key_line(path, "package", index, "version")
    return _version_reference(path, line, version, kind)


def _cargo_lock_reference(path: Path, package: PackageInfo) -> VersionReference:
    """Return the root package reference from Cargo.lock."""
    entries = _package_entries(path)
    candidates = [index for index, entry in enumerate(entries) if entry.get("name") == package.name and "source" not in entry]
    return _single_package_reference(path, entries, candidates, package.name, ReferenceKind.CARGO_LOCK)


def _pyproject_reference(path: Path, project: PythonProjectInfo) -> VersionReference:
    """Return the Python project version reference."""
    line = _toml_table_key_line(path, "project", "version")
    return _version_reference(path, line, project.version, ReferenceKind.PYPROJECT)


def _uv_lock_reference(path: Path, project: PythonProjectInfo) -> VersionReference:
    """Return the editable Python project reference from uv.lock."""
    entries = _package_entries(path)
    candidates: list[int] = []
    for index, entry in enumerate(entries):
        source = entry.get("source")
        if entry.get("name") == project.name and _is_parsed_object(source) and isinstance(source.get("editable"), str):
            candidates.append(index)
    return _single_package_reference(path, entries, candidates, project.name, ReferenceKind.UV_LOCK)


_CITATION_VERSION_RE = re.compile(r"^version:\s*(?P<quote>['\"]?)(?P<version>[0-9A-Za-z][0-9A-Za-z.+-]*)(?P=quote)\s*(?:#.*)?$")


def _citation_reference(path: Path) -> VersionReference:
    """Return the single top-level CITATION.cff version reference."""
    references: list[VersionReference] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.startswith("version:"):
            continue
        match = _CITATION_VERSION_RE.fullmatch(line)
        if match is None:
            msg = f"{path}:{line_number}: top-level version must be a non-empty scalar"
            raise ReleaseCheckError(msg)
        references.append(_version_reference(path, line_number, match.group("version"), ReferenceKind.CITATION))
    if len(references) != 1:
        msg = f"{path} must contain exactly one top-level version; found {len(references)}"
        raise ReleaseCheckError(msg)
    return references[0]


_VERSION_PATTERN = r"[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?"
_CHANGELOG_RELEASE_RE = re.compile(
    rf"^##\s+(?P<opening_bracket>\[)?v?(?P<version>{_VERSION_PATTERN})(?(opening_bracket)\])\s+-\s+(?P<date>\d{{4}}-\d{{2}}-\d{{2}})\s*$"
)


def _changelog_reference(path: Path) -> VersionReference:
    """Return the first generated release heading from CHANGELOG.md."""
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        match = _CHANGELOG_RELEASE_RE.match(line)
        if match is not None:
            return _version_reference(path, line_number, match.group("version"), ReferenceKind.CHANGELOG)
    msg = f"{path} has no generated release heading"
    raise ReleaseCheckError(msg)


def _metadata_reference(path: Path, line: int, value: str, kind: MetadataKind, text: str) -> MetadataReference:
    """Build a metadata reference while preserving its diagnostic context."""
    return MetadataReference(path=path, line=line, value=value, kind=kind, text=text.strip())


def _single_reference(references: list[MetadataReference], path: Path, description: str) -> MetadataReference:
    """Require exactly one parsed metadata reference."""
    if len(references) != 1:
        location = f":{references[0].line}" if references else ""
        lines = f" at lines {', '.join(str(reference.line) for reference in references)}" if references else ""
        msg = f"{path}{location} must contain exactly one {description}; found {len(references)}{lines}"
        raise ReleaseCheckError(msg)
    return references[0]


def _citation_metadata_reference(path: Path, field: str, value_pattern: str, kind: MetadataKind) -> MetadataReference:
    """Return one non-empty top-level scalar from CITATION.cff."""
    pattern = re.compile(rf"^{re.escape(field)}:\s*(?P<quote>['\"]?)(?P<value>{value_pattern})(?P=quote)\s*(?:#.*)?$")
    references: list[MetadataReference] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.startswith(f"{field}:"):
            continue
        match = pattern.fullmatch(line)
        if match is None:
            msg = f"{path}:{line_number}: top-level {field} must be a valid non-empty scalar"
            raise ReleaseCheckError(msg)
        references.append(_metadata_reference(path, line_number, match.group("value"), kind, line))
    return _single_reference(references, path, f"top-level {field}")


def _citation_doi_reference(path: Path) -> MetadataReference:
    """Return the stable concept DOI from CITATION.cff."""
    return _citation_metadata_reference(path, "doi", r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", MetadataKind.CITATION_DOI)


def _citation_date_reference(path: Path) -> MetadataReference:
    """Return and validate the release date from CITATION.cff."""
    reference = _citation_metadata_reference(path, "date-released", r"\d{4}-\d{2}-\d{2}", MetadataKind.CITATION_DATE)
    try:
        date.fromisoformat(reference.value)
    except ValueError as error:
        msg = f"{path}:{reference.line}: date-released is not a valid calendar date: {reference.value}"
        raise ReleaseCheckError(msg) from error
    return reference


def _current_changelog_heading_candidate_re(version: str) -> re.Pattern[str]:
    """Recognize even malformed level-two headings for one package version."""
    escaped = re.escape(version)
    return re.compile(rf"^##\s+\[?v?{escaped}\]?(?:\s|$)")


def _changelog_date_reference(path: Path, version: str) -> MetadataReference | None:
    """Return the validated date on the current package-version heading."""
    heading_candidate_re = _current_changelog_heading_candidate_re(version)
    references: list[MetadataReference] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        match = _CHANGELOG_RELEASE_RE.fullmatch(line)
        if match is not None and match.group("version") == version:
            reference = _metadata_reference(path, line_number, match.group("date"), MetadataKind.CHANGELOG_DATE, line)
            try:
                date.fromisoformat(reference.value)
            except ValueError as error:
                msg = f"{path}:{line_number}: changelog release date is not a valid calendar date: {reference.value}"
                raise ReleaseCheckError(msg) from error
            references.append(reference)
            continue
        if heading_candidate_re.match(line) is not None:
            msg = f"{path}:{line_number}: current-version heading must contain exactly one ISO release date"
            raise ReleaseCheckError(msg)
    if not references:
        return None
    return _single_reference(references, path, f"dated changelog heading for version {version}")


_README_DOI_RE = re.compile(r"\[!\[DOI\]\([^)]*\)\]\(https://doi\.org/(?P<doi>10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)\)")


def _readme_doi_reference(path: Path) -> MetadataReference:
    """Return the DOI targeted by the README badge."""
    references: list[MetadataReference] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        for match in _README_DOI_RE.finditer(line):
            references.append(_metadata_reference(path, line_number, match.group("doi"), MetadataKind.README_DOI, line))
    return _single_reference(references, path, "DOI badge target")


_REFERENCES_DOI_RE = re.compile(r"^- DOI: <https://doi\.org/(?P<doi>10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)>\s*$")


def _references_doi_reference(path: Path) -> MetadataReference:
    """Return the concept DOI entry from REFERENCES.md."""
    dois: list[MetadataReference] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        match = _REFERENCES_DOI_RE.fullmatch(line)
        if match is None:
            continue
        dois.append(_metadata_reference(path, line_number, match.group("doi"), MetadataKind.REFERENCES_DOI, line))
    return _single_reference(dois, path, "concept DOI entry")


def _changelog_comparison_references(path: Path, version: str) -> list[VersionReference]:
    """Return comparison targets whose link label is the current version."""
    comparison_re = re.compile(rf"^\[{re.escape(version)}\]:\s+\S+/compare/v{_VERSION_PATTERN}\.\.\.v(?P<version>{_VERSION_PATTERN})(?:\s|$)")
    references: list[VersionReference] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        match = comparison_re.match(line)
        if match is not None:
            references.append(_version_reference(path, line_number, match.group("version"), ReferenceKind.CHANGELOG_COMPARISON))
    return references


def _iter_markdown_files(root: Path) -> list[Path]:
    """Return active Markdown files that can carry current release references."""
    markdown_files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        relative_dir = Path(dirpath).relative_to(root)
        dirnames[:] = [dirname for dirname in dirnames if not (set((relative_dir / dirname).parts) & SKIP_DIRS)]
        markdown_files.extend(Path(dirpath) / filename for filename in filenames if filename.endswith(".md") and filename not in SKIP_MARKDOWN_FILES)
    return sorted(markdown_files, key=lambda path: path.relative_to(root).as_posix())


def _dependency_regex(package_name: str) -> re.Pattern[str]:
    """Build a regex for Cargo dependency snippets naming *package_name*."""
    escaped_name = re.escape(package_name)
    return re.compile(rf'(?<![\w.-]){escaped_name}\s*=\s*(?:"(?P<plain>[^"]+)"|\{{[^}}]*version\s*=\s*"(?P<table>[^"]+)"[^}}]*\}})')


def _dependency_references(path: Path, package_name: str) -> list[VersionReference]:
    """Return dependency snippet references in a Markdown file."""
    dependency_re = _dependency_regex(package_name)
    references: list[VersionReference] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        for match in dependency_re.finditer(line):
            references.append(
                VersionReference(
                    path=path,
                    line=line_number,
                    version=match.group("plain") or match.group("table"),
                    kind=ReferenceKind.DEPENDENCY_SNIPPET,
                    text=line.strip(),
                )
            )
    return references


def _cargo_add_regex(package_name: str) -> re.Pattern[str]:
    """Build a regex for versioned cargo-add commands naming *package_name*."""
    escaped_name = re.escape(package_name)
    return re.compile(rf"(?<![\w.-])cargo\s+add\b[^`\n]*?(?<![\w.-]){escaped_name}@(?P<version>[^\s`]+)")


def _cargo_add_references(path: Path, package_name: str) -> list[VersionReference]:
    """Return versioned cargo-add command references in a Markdown file."""
    cargo_add_re = _cargo_add_regex(package_name)
    references: list[VersionReference] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        references.extend(
            VersionReference(path, line_number, match.group("version"), ReferenceKind.CARGO_ADD, line.strip()) for match in cargo_add_re.finditer(line)
        )
    return references


def _readme_tag_references(path: Path, repository_slug: str) -> list[VersionReference]:
    """Return release-pinned README links that should track the package version."""
    escaped_slug = re.escape(repository_slug)
    tag_link_re = re.compile(
        rf"https://(?:github\.com/{escaped_slug}/(?:blob|raw|tree)/|raw\.githubusercontent\.com/{escaped_slug}/)"
        r"(?:v(?P<version>[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?)"
        r"|(?P<revision>[0-9a-f]{7,40}))(?=/|$|[^0-9A-Za-z._+-])"
    )
    references: list[VersionReference] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        references.extend(
            VersionReference(
                path,
                line_number,
                match.group("version") or match.group("revision"),
                ReferenceKind.README_TAG_LINK,
                line.strip(),
            )
            for match in tag_link_re.finditer(line)
        )
    return references


_BENCHMARK_CURRENT_TAG_RE = re.compile(
    r"just performance-release\s+v"
    r"(?P<version>[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?)(?=\s|`)"
)


def _benchmark_current_tag_references(path: Path) -> list[VersionReference]:
    """Return current-release tags from explicit curated-report commands."""
    references: list[VersionReference] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        references.extend(
            VersionReference(path, line_number, match.group("version"), ReferenceKind.BENCHMARK_CURRENT_TAG, line.strip())
            for match in _BENCHMARK_CURRENT_TAG_RE.finditer(line)
        )
    return references


def _version_references(root: Path, package: PackageInfo) -> list[VersionReference]:
    """Collect all current-release references that should match Cargo.toml."""
    pyproject_path = root / "pyproject.toml"
    project = _read_python_project_info(pyproject_path)
    changelog_path = root / "CHANGELOG.md"
    references = [
        _cargo_lock_reference(root / "Cargo.lock", package),
        _pyproject_reference(pyproject_path, project),
        _uv_lock_reference(root / "uv.lock", project),
        _citation_reference(root / "CITATION.cff"),
        _changelog_reference(changelog_path),
    ]
    references.extend(_changelog_comparison_references(changelog_path, package.version))
    for path in _iter_markdown_files(root):
        references.extend(_dependency_references(path, package.name))
        references.extend(_cargo_add_references(path, package.name))
        references.extend(_benchmark_current_tag_references(path))
    references.extend(_readme_tag_references(root / "README.md", package.repository_slug))
    return references


def find_version_mismatches(root: Path) -> list[VersionMismatch]:
    """Return release-version references that differ from Cargo.toml."""
    package = _read_cargo_package_info(root / "Cargo.toml")
    return [VersionMismatch(reference=reference, package=package) for reference in _version_references(root, package) if reference.version != package.version]


def find_release_metadata_mismatches(root: Path) -> list[MetadataMismatch]:
    """Return DOI and release-date references that disagree across release surfaces."""
    package = _read_cargo_package_info(root / "Cargo.toml")
    citation_doi = _citation_doi_reference(root / "CITATION.cff")
    citation_date = _citation_date_reference(root / "CITATION.cff")
    changelog_date = _changelog_date_reference(root / "CHANGELOG.md", package.version)
    readme_doi = _readme_doi_reference(root / "README.md")
    references_doi = _references_doi_reference(root / "REFERENCES.md")
    expected = [
        (citation_doi, ZENODO_CONCEPT_DOI),
        (readme_doi, ZENODO_CONCEPT_DOI),
        (references_doi, ZENODO_CONCEPT_DOI),
    ]
    if changelog_date is not None:
        expected.insert(0, (citation_date, changelog_date.value))
    return [MetadataMismatch(reference=reference, expected=value) for reference, value in expected if reference.value != value]


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="release-check",
        description="Check release metadata and active version references against Cargo.toml.",
        suggest_on_error=True,
        color=False,
    )
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd(), help="Repository root to check (default: current directory)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Validate release metadata and report the synchronized version."""
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    root = args.root.resolve()
    try:
        package = _read_cargo_package_info(root / "Cargo.toml")
        mismatches = find_version_mismatches(root)
        metadata_mismatches = find_release_metadata_mismatches(root)
    except (OSError, ReleaseCheckError, tomllib.TOMLDecodeError) as error:
        print(f"Could not check release-version synchronization: {error}", file=sys.stderr)
        return 1

    if mismatches:
        print("Release-version references are out of sync with Cargo.toml:", file=sys.stderr)
        for mismatch in mismatches:
            reference = mismatch.reference
            rel_path = reference.path.relative_to(root)
            print(
                f"  {rel_path}:{reference.line}: {reference.kind} found {reference.version}, expected {mismatch.package.version}: {reference.text}",
                file=sys.stderr,
            )
        return 1

    if metadata_mismatches:
        print("Release DOI or date references are out of sync:", file=sys.stderr)
        for mismatch in metadata_mismatches:
            reference = mismatch.reference
            rel_path = reference.path.relative_to(root)
            print(
                f"  {rel_path}:{reference.line}: {reference.kind} found {reference.value}, expected {mismatch.expected}: {reference.text}",
                file=sys.stderr,
            )
        return 1

    print(f"Release metadata is synchronized at {package.version}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
