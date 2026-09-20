"""Prepare release metadata transactionally from one stable GitHub tag."""

import argparse
import contextlib
import io
import re
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from archive_performance import _publish_texts, _published_releases, _tag_version, normalize_tag
from release_check import (
    _iter_markdown_files,
    _read_cargo_package_info,
    _readme_tag_link_pattern,
    concept_doi_problems,
    consumer_problems,
    is_performance_artifact_link,
)
from subprocess_utils import ExecutableNotFoundError, get_safe_executable

_PERFORMANCE_PAIR = re.compile(r"(just performance-release[ \t]+)v[0-9]+\.[0-9]+\.[0-9]+([ \t]+)v[0-9]+\.[0-9]+\.[0-9]+(?=\s|`|$)")


@dataclass(frozen=True, slots=True)
class UpdateSummary:
    """Release identities and files changed by a successful preparation."""

    tag: str
    previous_tag: str
    release_date: str
    changed_paths: tuple[Path, ...]


def parse_release_tag(tag: str) -> str:
    """Require the stable vX.Y.Z spelling, without normalizing user input."""
    if normalize_tag(tag) != tag:
        msg = f"release tag must use stable vX.Y.Z form: {tag!r}"
        raise ValueError(msg)
    return tag


def infer_previous_release(root: Path, target: str) -> str:
    """Discover the preceding stable published release, excluding drafts and prereleases."""
    get_safe_executable("gh")
    releases = _published_releases(root)
    target_version = _tag_version(target)
    if any(_tag_version(release.tag) > target_version for release in releases):
        msg = f"target {target} is older than an already published stable release"
        raise ValueError(msg)
    previous = [release.tag for release in releases if _tag_version(release.tag) < target_version]
    if not previous:
        msg = f"no published stable GitHub release precedes {target}"
        raise ValueError(msg)
    return max(previous, key=_tag_version)


def _read_text(path: Path) -> str:
    return path.read_bytes().decode("utf-8")


def _replace_version_match(match: re.Match[str], value: str, allowed: frozenset[str], group: str) -> str:
    original = match.group(group)
    if original not in allowed:
        msg = f"unexpected active release version {original!r}; expected one of {sorted(allowed)}"
        raise ValueError(msg)
    start, end = match.span(group)
    return match.group(0)[: start - match.start()] + value + match.group(0)[end - match.start() :]


def _prepare_updates(root: Path, tag: str, previous: str, release_date: str) -> dict[Path, str]:
    """Run the public shared CLI in a staging tree before applying consumer policies."""
    from research_repo_tools.cli import main as shared_main  # noqa: PLC0415 - optional development tooling.

    if problems := concept_doi_problems(root):
        raise ValueError("prepared release metadata failed validation: " + "; ".join(problems))
    version = tag.removeprefix("v")
    allowed = frozenset({version, previous.removeprefix("v")})
    package = _read_cargo_package_info(root / "Cargo.toml")
    paths = {root / name for name in ("Cargo.toml", "Cargo.lock", "pyproject.toml", "uv.lock", "CITATION.cff", "CHANGELOG.md")}
    paths.update(_iter_markdown_files(root))
    with tempfile.TemporaryDirectory(prefix="mcmc-release-validation-") as directory:
        staged = Path(directory)
        for path in paths:
            destination = staged / path.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            status = shared_main(
                [
                    "--root",
                    str(staged),
                    "release",
                    "update",
                    tag,
                    "--previous-release",
                    previous,
                    "--date",
                    release_date,
                ]
            )
        if status:
            raise ValueError(f"shared release preparation failed: {output.getvalue().strip()}")
        for path in _iter_markdown_files(staged):
            text = _read_text(path)
            text = _PERFORMANCE_PAIR.sub(lambda match: f"{match[1]}{tag}{match[2]}{previous}", text)
            if path == staged / "README.md":

                def replace_link(match: re.Match[str]) -> str:
                    if is_performance_artifact_link(match):
                        return match.group(0)
                    group = "version" if match.group("version") is not None else "revision"
                    accepted = allowed if group == "version" else frozenset({match.group(group)})
                    return _replace_version_match(match, version if group == "version" else tag, accepted, group)

                text = _readme_tag_link_pattern(package.repository_slug, include_main=True).sub(replace_link, text)
            path.write_bytes(text.encode("utf-8"))
        if problems := consumer_problems(staged):
            raise ValueError("prepared release metadata failed validation: " + "; ".join(problems))
        return {path: _read_text(staged / path.relative_to(root)) for path in paths}


def update_release_version(root: Path, tag: str, *, previous_tag: str | None = None, release_date: str | None = None, dry_run: bool = False) -> UpdateSummary:
    """Validate then atomically replace owned metadata; restore prior contents on failure."""
    root = root.resolve()
    tag = parse_release_tag(tag)
    owned = {root / name for name in ("Cargo.toml", "Cargo.lock", "pyproject.toml", "uv.lock", "CITATION.cff", "CHANGELOG.md")}
    owned.update(_iter_markdown_files(root))
    for path in owned:
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            msg = f"release metadata must be a repository-contained regular file, not a symbolic link: {path}"
            raise ValueError(msg)
    previous = parse_release_tag(previous_tag) if previous_tag is not None else infer_previous_release(root, tag)
    if _tag_version(previous) >= _tag_version(tag):
        msg = f"previous release {previous} must precede {tag}"
        raise ValueError(msg)
    today = release_date if release_date is not None else datetime.now(UTC).date().isoformat()
    if date.fromisoformat(today).isoformat() != today:
        msg = "release date must use YYYY-MM-DD form"
        raise ValueError(msg)
    updates = _prepare_updates(root, tag, previous, today)
    changed = tuple((path, updates[path]) for path in sorted(updates, key=lambda path: path.relative_to(root).as_posix()) if _read_text(path) != updates[path])
    if not dry_run:
        _publish_texts(changed)
    return UpdateSummary(tag, previous, today, tuple(path for path, _ in changed))


def main(argv: list[str] | None = None) -> int:
    """Prepare a release without dependency upgrades, changelog generation, or measurements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", help="Target stable release tag in vX.Y.Z form")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--previous-release")
    parser.add_argument("--date")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        summary = update_release_version(args.repo_root, args.tag, previous_tag=args.previous_release, release_date=args.date, dry_run=args.dry_run)
    except subprocess.CalledProcessError as error:
        print(f"Release preparation failed: {error.stderr or error.stdout or error}", file=sys.stderr)
        return 1
    except (ExecutableNotFoundError, OSError, RuntimeError, subprocess.TimeoutExpired, TypeError, ValueError, tomllib.TOMLDecodeError) as error:
        print(f"Release preparation failed: {error}", file=sys.stderr)
        return 1
    action = "Would prepare" if args.dry_run else "Prepared"
    print(f"{action} {summary.tag} against {summary.previous_tag}; UTC release date {summary.release_date}.")
    for path in summary.changed_paths:
        print(f"{'Would update' if args.dry_run else 'Updated'} {path.relative_to(args.repo_root.resolve())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
