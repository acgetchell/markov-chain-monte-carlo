#!/usr/bin/env python3
"""Post-process a git-cliff generated CHANGELOG.md.

Applies lightweight markdown hygiene that is difficult to express in
Tera templates:

  1. Strip trailing blank lines (git-cliff Tera templates emit an extra
     trailing newline that triggers Markdown rule MD012).
  2. Apply the repository's pinned rumdl Markdown rules before publishing.

Usage:
    postprocess-changelog                     # default: CHANGELOG.md
    postprocess-changelog path/to/CHANGELOG.md
"""

import argparse
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from subprocess_utils import ExecutableNotFoundError, run_safe_command

_FORMAT_TIMEOUT_SECONDS = 30


class MarkdownFormatError(RuntimeError):
    """Raised when rumdl cannot produce valid formatted Markdown."""


@dataclass(frozen=True, slots=True)
class PostprocessOptions:
    """Validated changelog post-processing options."""

    path: Path


def postprocess(path: Path) -> None:
    """Read *path*, produce valid Markdown, and replace it atomically with LF newlines."""
    text = path.read_text(encoding="utf-8")

    # 1. Strip trailing blank lines — keep nonblank content unchanged and one final newline.
    lines = text.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    text = "\n".join(lines) + "\n"
    text = _format_markdown(text, path)

    mode = stat.S_IMODE(path.stat().st_mode)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        temporary.chmod(mode)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _format_markdown(text: str, path: Path) -> str:
    """Apply configured rumdl fixes to *text* and return normalized LF Markdown."""
    config = _find_rumdl_config(path)
    try:
        result = run_safe_command(
            "rumdl",
            ["check", "--fix", "--stdin", "--stdin-filename", str(path), "--no-cache", "--config", str(config)],
            input=text,
            timeout=_FORMAT_TIMEOUT_SECONDS,
        )
    except subprocess.CalledProcessError as error:
        diagnostics = (error.stderr or error.stdout or "unknown formatter error").strip()
        raise MarkdownFormatError(f"rumdl could not format {path.name}: {diagnostics}") from error

    if text.strip() and not result.stdout.strip():
        raise MarkdownFormatError(f"rumdl returned empty output for nonempty {path.name}")
    return "\n".join(result.stdout.splitlines()).rstrip() + "\n"


def _find_rumdl_config(path: Path) -> Path:
    """Find the repository rumdl configuration from the output path or current directory."""
    searched: set[Path] = set()
    for root in (path.resolve().parent, Path.cwd().resolve()):
        for directory in (root, *root.parents):
            if directory in searched:
                continue
            searched.add(directory)
            config = directory / "rumdl.toml"
            if config.is_file():
                return config
    raise MarkdownFormatError(f"rumdl.toml not found for {path}")


def parse_args(argv: list[str] | None = None) -> PostprocessOptions:
    """Parse command-line values into trusted post-processing options."""
    parser = argparse.ArgumentParser(
        prog="postprocess-changelog",
        description="Apply markdown hygiene to a git-cliff generated CHANGELOG.md.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("CHANGELOG.md"),
        help="Path to CHANGELOG.md (default: CHANGELOG.md)",
    )
    namespace = parser.parse_args(argv)
    return PostprocessOptions(path=namespace.path)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``postprocess-changelog``."""
    options = parse_args(argv)

    changelog = options.path
    if not changelog.is_file():
        print(f"Error: {changelog} not found", file=sys.stderr)
        return 1

    try:
        postprocess(changelog)
    except (ExecutableNotFoundError, OSError, RuntimeError, subprocess.TimeoutExpired, UnicodeError) as error:
        print(f"Error: could not post-process {changelog}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
