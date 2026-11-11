"""Prepare only an identical mutable draft from verified protected-main source.

Shared public interfaces own metadata, checks, signed evidence and release-target
validation. This MCMC orchestration neither measures nor publishes anything.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from functools import partial
from pathlib import Path

from research_repo_tools import changelog, config
from research_repo_tools.dependency_provenance import check_release_dependencies
from research_repo_tools.measurement import resolve_revision
from research_repo_tools.process import ExecutableNotFoundError, format_exception_diagnostics, run_command
from research_repo_tools.release_assets import preflight_release_target, require_release_checkout, revalidate_release_target
from research_repo_tools.release_evidence import verify_release_evidence
from research_repo_tools.release_interfaces import load_evidence_policy
from research_repo_tools.release_publishing import check_metadata, require_checks
from research_repo_tools.release_runtime import isolated_release_runtime
from research_repo_tools.release_validation import api, parse_release_json, read_pages, redact_release_diagnostic

REPOSITORY = "acgetchell/markov-chain-monte-carlo"


def require_tag_source(root: Path, tag: str) -> str:
    """Bind the stable tag event, checkout and remote protected-main source."""
    if re.fullmatch(r"v(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)", tag) is None:
        message = "draft preparation requires a canonical stable vX.Y.Z tag"
        raise ValueError(message)
    commit = resolve_revision(root, f"refs/tags/{tag}")
    if tuple(os.environ.get(key) for key in ("GITHUB_REPOSITORY", "GITHUB_EVENT_NAME", "GITHUB_REF", "GITHUB_SHA")) != (
        REPOSITORY,
        "push",
        f"refs/tags/{tag}",
        commit,
    ):
        message = "draft preparation requires the exact owning tag-push ref and SHA"
        raise ValueError(message)
    require_release_checkout(root, commit)
    repository = api("", repository=REPOSITORY)
    branch = api("branches/main", repository=REPOSITORY)
    comparison = api(f"compare/{commit}...main", repository=REPOSITORY)
    if repository.get("default_branch") != "main" or branch.get("protected") is not True or comparison.get("status") not in {"ahead", "identical"}:
        message = "release source must belong to protected default main"
        raise ValueError(message)
    refs = run_command("git", ["--no-pager", "ls-remote", f"https://github.com/{REPOSITORY}.git", f"refs/tags/{tag}", f"refs/tags/{tag}^{{}}"], cwd=root).stdout
    observed = dict(line.split()[::-1] for line in refs.splitlines())
    if observed.get(f"refs/tags/{tag}^{{}}", observed.get(f"refs/tags/{tag}")) != commit:
        message = "remote tag differs from the captured source SHA"
        raise ValueError(message)
    return commit


def prepare(root: Path, tag: str) -> int:
    """Create a draft only after all gates; retry without editing existing assets."""
    commit = require_tag_source(root, tag)
    settings = config.load(root=root)
    check_release_dependencies(root, packages=settings.release.published_dependencies)
    check_metadata(settings, tag)
    policy = load_evidence_policy(root / "tooling/release-evidence.toml")
    if policy.validation.repository != REPOSITORY or policy.package != "markov-chain-monte-carlo" or policy.guarantee != "signed-source":
        message = "draft preparation requires MCMC signed-source policy"
        raise ValueError(message)
    if settings.publishing is None or settings.publishing.repository != REPOSITORY:
        message = "draft preparation requires MCMC publishing policy"
        raise ValueError(message)
    publishing = settings.publishing
    read = partial(api, repository=REPOSITORY)
    selection = verify_release_evidence(policy, tag.removeprefix("v"), commit, read)

    def required_checks() -> None:
        pages = parse_release_json(
            run_command("gh", ["api", "--paginate", "--slurp", f"repos/{REPOSITORY}/commits/{commit}/check-runs?filter=latest&per_page=100"], cwd=root).stdout
        )
        require_checks(pages, commit, publishing.required_checks)

    required_checks()
    notes = changelog.notes(settings, tag)[0]
    releases = read_pages(lambda endpoint: {"releases": read(endpoint)}, "releases", "releases")
    matches = [release for release in releases if release.get("tag_name") == tag]
    if len(matches) > 1:
        message = "ambiguous draft release identity"
        raise ValueError(message)
    target = preflight_release_target(root, REPOSITORY, tag, commit, expected_title=tag, checkout=root) if matches else None

    def identical_notes(identifier: int) -> None:
        release = read(f"releases/{identifier}")
        if release.get("id") != identifier or release.get("body") != notes:
            message = "draft identity or generated release notes changed"
            raise ValueError(message)

    if target is not None:
        if target.release_id != matches[0].get("id"):
            message = "draft release identity changed during preparation"
            raise ValueError(message)
        identical_notes(target.release_id)
    verify_release_evidence(policy, tag.removeprefix("v"), commit, read, expected=selection)
    required_checks()
    if require_tag_source(root, tag) != commit:
        message = "tag source changed during preparation"
        raise ValueError(message)
    if target is None:
        with tempfile.TemporaryDirectory(prefix="mcmc-draft-notes-") as temporary:
            path = Path(temporary) / "notes.md"
            path.write_text(notes, encoding="utf-8", newline="\n")
            run_command(
                "gh",
                ["release", "create", tag, "--repo", REPOSITORY, "--verify-tag", "--target", commit, "--draft", "--title", tag, "--notes-file", str(path)],
                cwd=root,
            )
        target = preflight_release_target(root, REPOSITORY, tag, commit, expected_title=tag, checkout=root)
    revalidate_release_target(root, target, checkout=root)
    identical_notes(target.release_id)
    return target.release_id


def main() -> int:
    """Require the event's one stable tag; redact provider diagnostics on failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag")
    args = parser.parse_args()
    try:
        root = Path.cwd().resolve()
        with isolated_release_runtime(config.load(root=root)):
            identifier = prepare(root, args.tag)
        sys.stdout.write(f"Verified mutable draft {identifier} for {args.tag}; review before publication.\n")
    except (ExecutableNotFoundError, OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as error:
        sys.stderr.write(f"Draft preparation failed: {redact_release_diagnostic(format_exception_diagnostics(error))}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
