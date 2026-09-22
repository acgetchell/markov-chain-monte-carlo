"""Prepare MCMC release metadata through the shared validated release plan."""

import argparse
import re
import subprocess
import sys
from pathlib import Path

from research_repo_tools.process import ExecutableNotFoundError, format_exception_diagnostics
from research_repo_tools.releases import ReleaseAdapter, ReleaseContext, ReleaseResult, apply_release, plan_release

_PERFORMANCE_PAIR = re.compile(r"(just performance-release[ \t]+)v[0-9]+\.[0-9]+\.[0-9]+([ \t]+)v[0-9]+\.[0-9]+\.[0-9]+(?=\s|\x60|$)")


def performance_command_edits(candidate: Path, context: ReleaseContext) -> dict[str, bytes]:
    """Advance the documented baseline without making offline checks discover releases."""
    edits = {}
    for path in sorted(candidate.rglob("*.md"), key=lambda path: path.relative_to(candidate).as_posix()):
        if path.name == "CHANGELOG.md":
            continue
        original = path.read_bytes()
        updated = _PERFORMANCE_PAIR.sub(lambda match: f"{match[1]}{context.tag}{match[2]}{context.previous_tag}", original.decode("utf-8")).encode("utf-8")
        if updated != original:
            edits[path.relative_to(candidate).as_posix()] = updated
    return edits


def update_release_version(root: Path, tag: str, *, previous_tag: str | None = None, release_date: str | None = None, dry_run: bool = False) -> ReleaseResult:
    """Preserve the stable-tag command contract and apply the exact validated candidate."""
    if re.fullmatch(r"v(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)", tag) is None:
        raise ValueError(f"release tag must use stable vX.Y.Z form: {tag!r}")
    plan = plan_release(root, tag, previous_tag=previous_tag, release_date=release_date, adapter=ReleaseAdapter(prepare=performance_command_edits))
    return ReleaseResult(plan.context, tuple(edit.path for edit in plan.edits)) if dry_run else apply_release(plan)


def main(argv: list[str] | None = None) -> int:
    """Prepare or preview a release without dependency upgrades or measurements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", help="Target stable release tag in vX.Y.Z form")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--previous-release")
    parser.add_argument("--date")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = update_release_version(args.repo_root, args.tag, previous_tag=args.previous_release, release_date=args.date, dry_run=args.dry_run)
    except (ExecutableNotFoundError, OSError, RuntimeError, subprocess.SubprocessError, TypeError, ValueError, ExceptionGroup) as error:
        print(f"Release preparation failed: {format_exception_diagnostics(error)}", file=sys.stderr)
        return 1
    context = result.context
    print(f"{'Would prepare' if args.dry_run else 'Prepared'} {context.tag} against {context.previous_tag}; UTC release date {context.release_date}.")
    for path in result.changed_paths:
        print(f"{'Would update' if args.dry_run else 'Updated'} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
