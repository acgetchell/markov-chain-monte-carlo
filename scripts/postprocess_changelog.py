#!/usr/bin/env python3
"""Post-process a git-cliff generated CHANGELOG.md.

Applies lightweight markdown hygiene that is difficult to express in
Tera templates:

  1. Strip trailing blank lines (git-cliff Tera templates emit an extra
     trailing newline that triggers markdownlint MD012).

Usage:
    postprocess-changelog                     # default: CHANGELOG.md
    postprocess-changelog path/to/CHANGELOG.md
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PostprocessOptions:
    """Validated changelog post-processing options."""

    path: Path


def postprocess(path: Path) -> None:
    """Read *path*, apply hygiene fixes, and write it back."""
    text = path.read_text(encoding="utf-8")

    # 1. Strip trailing blank lines — keep exactly one trailing newline.
    text = text.rstrip("\n") + "\n"

    path.write_text(text, encoding="utf-8")


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

    postprocess(changelog)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
