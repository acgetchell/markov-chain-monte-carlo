#!/usr/bin/env python3
"""Create annotated git tags from CHANGELOG.md sections.

Handles GitHub's 125KB tag-annotation size limit by falling back to a short
reference message when the changelog section is too large.

Usage:
    tag-release v1.2.3          # create annotated tag from CHANGELOG.md
    tag-release v1.2.3 --force  # recreate tag if it already exists
    tag-release v1.2.3 --debug  # verbose output

Ported from the delaunay project's changelog_utils.py (tag-creation subset).
"""

import argparse
import logging
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NewType

from subprocess_utils import (
    ExecutableNotFoundError,
    run_git_command,
    run_git_command_with_input,
)

# GitHub's maximum size for git tag annotations (bytes)
_GITHUB_TAG_ANNOTATION_LIMIT = 125_000

# ANSI color codes for terminal output
_GREEN = "\033[0;32m"
_BLUE = "\033[0;34m"
_YELLOW = "\033[1;33m"
_RESET = "\033[0m"

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SemVer validation
# ---------------------------------------------------------------------------

# SemVer 2.0.0 strict with required 'v' prefix
# Alphanumeric prerelease identifier: any [0-9A-Za-z-]+ containing at least one
# non-digit.  This permits identifiers like "1a" that start with a digit but are
# not purely numeric (SemVer 2.0.0 §9).
_ALNUM_ID = r"(?:(?=[0-9A-Za-z-]*[A-Za-z-])[0-9A-Za-z-]+)"
_SEMVER_RE = re.compile(
    r"^v"
    r"(0|[1-9]\d*)\."
    r"(0|[1-9]\d*)\."
    r"(0|[1-9]\d*)"
    rf"(?:-(?:(?:0|[1-9]\d*)|{_ALNUM_ID})"
    rf"(?:\.(?:(?:0|[1-9]\d*)|{_ALNUM_ID}))*"
    r")?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)


GitHubRepositoryUrl = NewType("GitHubRepositoryUrl", str)


@dataclass(frozen=True, slots=True)
class ReleaseVersion:
    """Validated SemVer release tag and its unprefixed version number."""

    tag: str
    number: str

    def __post_init__(self) -> None:
        """Reject direct construction that would bypass SemVer parsing."""
        if not _SEMVER_RE.fullmatch(self.tag):
            msg = f"Tag version should follow SemVer format 'vX.Y.Z' (e.g., v0.3.5, v1.2.3-rc.1). Got: {self.tag}"
            raise ValueError(msg)
        if self.number != self.tag.removeprefix("v"):
            msg = f"Release version number {self.number!r} does not match tag {self.tag!r}"
            raise ValueError(msg)

    @classmethod
    def parse(cls, raw: str) -> ReleaseVersion:
        """Parse a strict ``vX.Y.Z`` SemVer release tag."""
        return cls(tag=raw, number=raw.removeprefix("v"))


@dataclass(frozen=True, slots=True)
class TagOptions:
    """Validated tag-release command-line options."""

    version: ReleaseVersion
    force: bool
    debug: bool


def validate_semver(tag_version: str) -> None:
    """Raise ``ValueError`` if *tag_version* is not valid ``vX.Y.Z`` SemVer."""
    ReleaseVersion.parse(tag_version)


def parse_version(tag_version: str) -> str:
    """Parse a release tag and return its version number without ``v``."""
    return ReleaseVersion.parse(tag_version).number


def parse_release_version_argument(raw: str) -> ReleaseVersion:
    """Parse a release version while preserving its diagnostic in argparse."""
    try:
        return ReleaseVersion.parse(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


# ---------------------------------------------------------------------------
# Changelog helpers
# ---------------------------------------------------------------------------


def find_changelog(start: Path | None = None) -> Path:
    """Locate ``CHANGELOG.md`` in *start* or its parent.

    Raises:
        FileNotFoundError: If ``CHANGELOG.md`` cannot be found.
    """
    base = start or Path.cwd()
    for candidate in (base / "CHANGELOG.md", base.parent / "CHANGELOG.md"):
        if candidate.is_file():
            return candidate
    msg = "CHANGELOG.md not found in current directory or parent directory."
    raise FileNotFoundError(msg)


def extract_changelog_section(changelog: Path, version: str) -> str:
    """Extract the changelog body for *version* (without ``v`` prefix).

    Raises:
        LookupError: If the version section is not found or empty.
    """
    content = changelog.read_text(encoding="utf-8")
    header_re = _version_header_re(version)

    lines = content.split("\n")
    section: list[str] = []
    collecting = False

    for line in lines:
        if re.match(r"^##\s", line):
            if collecting:
                break
            if header_re.match(line):
                collecting = True
                continue
        elif collecting:
            section.append(line)

    if not collecting:
        msg = f"No changelog section found for version {version}. Expected a heading like: ## [{version}] - YYYY-MM-DD"
        raise LookupError(msg)

    # Trim leading/trailing blank lines (O(n) index scan + slice)
    start = 0
    while start < len(section) and not section[start].strip():
        start += 1
    end = len(section)
    while end > start and not section[end - 1].strip():
        end -= 1
    section = section[start:end]

    body = "\n".join(section)
    if not body.strip():
        msg = f"Changelog section for version {version} is empty."
        raise LookupError(msg)
    return body


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def _tag_exists(tag_version: str) -> bool:
    """Return ``True`` if *tag_version* already exists as a git tag."""
    try:
        run_git_command(["rev-parse", "-q", "--verify", f"refs/tags/{tag_version}"])
    except subprocess.CalledProcessError:
        return False
    else:
        return True


def parse_github_repository_url(raw: str) -> GitHubRepositoryUrl:
    """Parse a supported GitHub remote into its canonical HTTPS URL."""
    patterns = [
        r"^git@github\.com:(?P<slug>[^/]+/[^/]+?)(?:\.git)?/?$",
        r"^https://github\.com/(?P<slug>[^/]+/[^/]+?)(?:\.git)?/?$",
        r"^ssh://git@github\.com[:/](?P<slug>[^/]+/[^/]+?)(?:\.git)?/?$",
    ]
    for pat in patterns:
        m = re.match(pat, raw)
        if m:
            return GitHubRepositoryUrl(f"https://github.com/{m.group('slug')}")
    msg = f"origin remote is not a supported GitHub URL: {raw!r}"
    raise ValueError(msg)


def _get_repo_url() -> GitHubRepositoryUrl:
    """Detect and parse the GitHub HTTPS URL from the ``origin`` remote."""
    result = run_git_command(["remote", "get-url", "origin"])
    return parse_github_repository_url(result.stdout.strip())


def _version_header_re(version: str) -> re.Pattern[str]:
    """Build the header regex for *version*, matching ``extract_changelog_section``."""
    return re.compile(rf"^##\s*\[?v?{re.escape(version)}\]?(?:$|\s|\()")


def _github_anchor(changelog: Path, version: str) -> str:
    """Build a GitHub-compatible heading anchor (matches ``github-slugger``)."""
    header_re = _version_header_re(version)
    try:
        for line in changelog.read_text(encoding="utf-8").splitlines():
            if header_re.match(line):
                heading = line.removeprefix("## ").strip()
                # Strip inline-link markup [text](url) → text
                heading = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", heading)
                # Strip reference-style brackets [text] → text
                heading = re.sub(r"\[([^\]]+)\]", r"\1", heading)
                heading = heading.lower()
                # Remove everything except letters, digits, spaces, hyphens
                heading = re.sub(r"[^a-z0-9\s-]", "", heading)
                # Replace whitespace runs with a single hyphen
                return re.sub(r"\s+", "-", heading)
    except OSError:
        pass
    return re.sub(r"[^a-z0-9-]", "", f"v{version}".lower())


# ---------------------------------------------------------------------------
# Core workflow
# ---------------------------------------------------------------------------


def create_tag(tag_version: str | ReleaseVersion, *, force: bool = False) -> None:
    """Create an annotated git tag with changelog content.

    If the changelog section exceeds GitHub's 125KB limit, creates the tag
    with a short reference message instead.
    """
    release = ReleaseVersion.parse(tag_version) if isinstance(tag_version, str) else tag_version
    tag = release.tag
    version = release.number

    # Check for existing tag (but don't delete yet — validate first)
    tag_existed = _tag_exists(tag)
    if tag_existed and not force:
        msg = f"Tag '{tag}' already exists; use --force to recreate it or delete it manually"
        raise FileExistsError(msg)

    # Extract changelog section (before any mutation)
    changelog = find_changelog()
    section = extract_changelog_section(changelog, version)
    section_bytes = len(section.encode("utf-8"))

    # Check size limit
    if section_bytes > _GITHUB_TAG_ANNOTATION_LIMIT:
        print(f"{_YELLOW}⚠ Changelog section ({section_bytes:,} bytes) exceeds GitHub's tag limit ({_GITHUB_TAG_ANNOTATION_LIMIT:,} bytes){_RESET}")
        anchor = _github_anchor(changelog, version)
        repo_url = _get_repo_url()
        tag_message = (
            f"Version {version}\n\n"
            f"This release contains extensive changes. See full changelog:\n"
            f"<{repo_url}/blob/{tag}/CHANGELOG.md#{anchor}>\n\n"
            f"For detailed release notes, refer to CHANGELOG.md in the repository.\n"
        )
        is_truncated = True
        print(f"{_BLUE}→ Creating annotated tag with CHANGELOG.md reference{_RESET}")
    else:
        tag_message = section
        is_truncated = False
        print(f"{_BLUE}Tag message preview ({section_bytes:,} bytes):{_RESET}")
        preview = section.split("\n")[:20]
        print("----------------------------------------")
        print("\n".join(preview))
        if len(section.split("\n")) > 20:
            print("... (truncated for preview)")
        print("----------------------------------------")

    # Create or atomically replace the annotated tag. Git prepares the tag
    # object before updating the ref, so a failed command preserves any
    # existing tag.
    label = "reference" if is_truncated else "full changelog"
    print(f"{_BLUE}Creating annotated tag '{tag}' with {label} content...{_RESET}")
    command = ["tag"]
    if tag_existed and force:
        command.append("--force")
    command.extend(["--annotate", tag, "-F", "-", "--cleanup=verbatim"])
    run_git_command_with_input(command, input_data=tag_message)

    # Success
    print(f"{_GREEN}✓ Successfully created tag '{tag}'{_RESET}")
    print()
    print("Next steps:")
    if force:
        print(f"  1. Force-push the tag: {_BLUE}git push --force origin {tag}{_RESET}")
    else:
        print(f"  1. Push the tag: {_BLUE}git push origin {tag}{_RESET}")
    print(f"  2. Create GitHub release: {_BLUE}gh release create {tag} --title {tag} --notes-from-tag{_RESET}")
    if is_truncated:
        print(f"\n{_YELLOW}Note: Tag annotation references CHANGELOG.md due to size (>125KB).{_RESET}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> TagOptions:
    """Parse command-line values into trusted tag-release options."""
    parser = argparse.ArgumentParser(
        prog="tag-release",
        description="Create an annotated git tag from a CHANGELOG.md section.",
    )
    parser.add_argument("version", type=parse_release_version_argument, help="Tag version (e.g. v1.2.3)")
    parser.add_argument("--force", action="store_true", help="Recreate tag if it already exists")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    namespace = parser.parse_args(argv)
    return TagOptions(version=namespace.version, force=namespace.force, debug=namespace.debug)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``tag-release``."""
    options = parse_args(argv)

    if options.debug:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    try:
        create_tag(options.version, force=options.force)
    except (
        ValueError,
        FileNotFoundError,
        FileExistsError,
        LookupError,
        ExecutableNotFoundError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
