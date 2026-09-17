"""Regression tests for the public Just recipe surface."""

import json
import os
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest

import update_cargo_tool_pins

REPO_ROOT = Path(__file__).resolve().parents[2]
JUSTFILE = REPO_ROOT / "justfile"
RECIPE_DECLARATION = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*)(?:\s+.*?)?:(?=\s|$)", re.MULTILINE)
WORKFLOW_VERSION_LOOKUP = re.compile(r"(?:just --evaluate|resolve_version) [\"']?([a-z0-9_]+_version)")
RELEASE_PERFORMANCE_RECIPES = {
    "bench-compare",
    "bench-latest",
    "bench-latest-vs-last",
    "bench-save-baseline",
    "bench-save-last",
    "performance-doc",
    "performance-github-assets",
    "performance-local",
    "performance-readme",
    "performance-release",
}
UPDATE_RECIPES = {
    "update",
    "update-cargo-dependencies",
    "update-cargo-tools",
    "update-dependencies",
    "update-python-dependencies",
    "update-version",
}


def _run_just(*args: str) -> subprocess.CompletedProcess[str]:
    executable = shutil.which("just")
    assert executable is not None
    return subprocess.run(  # noqa: S603 - executable is resolved and arguments are test constants.
        [executable, *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        encoding="utf-8",
    )


def _recipes() -> dict[str, dict[str, Any]]:
    document = json.loads(_run_just("--dump", "--dump-format", "json").stdout)
    recipes = document["recipes"]
    assert isinstance(recipes, dict)
    return recipes


def test_recipe_declarations_are_lexicographically_sorted() -> None:
    names = RECIPE_DECLARATION.findall(JUSTFILE.read_text(encoding="utf-8"))

    assert names == sorted(names)


def test_bare_just_shows_curated_help() -> None:
    result = _run_just()

    assert result.stdout.startswith("Common Just workflows:\n")
    assert "Use 'just --list' for the complete grouped recipe reference." in result.stdout


def _run_review_probe(tmp_path: Path, *recipe_args: str, **overrides: str) -> subprocess.CompletedProcess[str]:
    """Exercise the real recipes with local stubs, never either remote service."""
    executable = shutil.which("just")
    assert executable is not None
    stub = tmp_path / "coderabbit"
    stub.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\0" "$@"\nexit "$REVIEW_STATUS"\n',
        encoding="utf-8",
        newline="\n",
    )
    stub.chmod(0o755)
    git_stub = tmp_path / "git"
    git_stub.write_text(
        '#!/usr/bin/env bash\ncase "$*" in\n'
        '  "--no-pager ls-remote --exit-code origin refs/heads/main")\n'
        '    printf "%s\\trefs/heads/main\\n" "$REMOTE_COMMIT"; exit "$REMOTE_STATUS" ;;\n'
        '  "--no-pager rev-parse --verify refs/remotes/origin/main^{commit}")\n'
        '    printf "%s\\n" "$LOCAL_COMMIT"; exit "$LOCAL_STATUS" ;;\n'
        '  *) echo "Unexpected Git command: $*" >&2; exit 99 ;;\nesac\n',
        encoding="utf-8",
        newline="\n",
    )
    git_stub.chmod(0o755)
    return subprocess.run(  # noqa: S603 - fixed Just recipes invoke local CLI stubs with no network effects.
        [executable, *recipe_args],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
            "REVIEW_STATUS": "0",
            "REMOTE_COMMIT": "a" * 40,
            "LOCAL_COMMIT": "a" * 40,
            "REMOTE_STATUS": "0",
            "LOCAL_STATUS": "0",
            **overrides,
        },
        check=False,
        capture_output=True,
        encoding="utf-8",
        timeout=30,
    )


@pytest.mark.parametrize(
    ("recipe_args", "scope_args"),
    [
        (("review",), ["--base", "origin/main"]),
        (("review", "main"), ["--base", "main"]),
        (("review", "topic/it's; printf injected"), ["--base", "topic/it's; printf injected"]),
        (("review-uncommitted",), ["--uncommitted"]),
    ],
)
@pytest.mark.parametrize("review_status", [0, 23])
def test_review_scope_and_failures_reach_the_cli(tmp_path: Path, recipe_args: tuple[str, ...], scope_args: list[str], review_status: int) -> None:
    result = _run_review_probe(tmp_path, *recipe_args, REVIEW_STATUS=str(review_status))

    assert result.returncode == review_status, result.stderr
    assert result.stdout.split("\0") == ["review", "--agent", "--include-untracked", "-c", "AGENTS.md", ".coderabbit.yml", *scope_args, ""]


@pytest.mark.parametrize("overrides", [{"LOCAL_COMMIT": "b" * 40}, {"LOCAL_STATUS": "1"}, {"REMOTE_STATUS": "1"}])
def test_default_review_requires_a_verified_current_remote_base(tmp_path: Path, overrides: dict[str, str]) -> None:
    result = _run_review_probe(tmp_path, "review", **overrides)

    assert result.returncode == 1
    assert result.stdout == ""
    if "REMOTE_STATUS" in overrides:
        assert "Cannot verify origin/main" in result.stderr
    else:
        assert "git fetch origin" in result.stderr


@pytest.mark.parametrize("recipe_args", [("review", "main"), ("review-uncommitted",)])
def test_local_review_scopes_do_not_require_remote_access(tmp_path: Path, recipe_args: tuple[str, ...]) -> None:
    result = _run_review_probe(tmp_path, *recipe_args, REMOTE_STATUS="1", LOCAL_STATUS="1")

    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("review\0--agent\0")


def test_review_is_discoverable_and_separate_from_validation() -> None:
    recipes = _recipes()
    help_text = _run_just("help-workflows").stdout
    for name in ("review", "review-uncommitted"):
        assert recipes[name]["private"] is False
        assert {"group": "review"} in recipes[name]["attributes"]
        assert f"just {name}" in help_text
    for name in ("check", "ci", "setup-tools", "update"):
        result = _run_just("--dry-run", name)
        assert "coderabbit" not in (result.stdout + result.stderr).lower()


def test_public_recipes_have_one_group_and_a_description() -> None:
    for name, recipe in _recipes().items():
        if recipe["private"]:
            continue
        groups = [attribute["group"] for attribute in recipe["attributes"] if "group" in attribute]
        assert recipe["doc"], f"public recipe {name!r} has no description"
        assert len(groups) == 1, f"public recipe {name!r} has groups {groups!r}"


def test_public_recipes_do_not_duplicate_exact_behavior() -> None:
    signatures: defaultdict[str, list[str]] = defaultdict(list)
    for name, recipe in _recipes().items():
        if recipe["private"]:
            continue
        signature = json.dumps(
            {
                "body": recipe["body"],
                "dependencies": recipe["dependencies"],
                "parameters": recipe["parameters"],
            },
            sort_keys=True,
        )
        signatures[signature].append(name)

    duplicates = [names for names in signatures.values() if len(names) > 1]
    assert duplicates == []


def test_workflow_tool_version_lookups_resolve_from_just() -> None:
    workflow_text = "\n".join(path.read_text(encoding="utf-8") for path in sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml")))
    version_names = sorted(set(WORKFLOW_VERSION_LOOKUP.findall(workflow_text)))

    assert version_names
    for name in version_names:
        result = _run_just("--evaluate", name)
        assert result.stdout.strip(), name


def test_pinned_tool_guards_reference_their_justfile_versions() -> None:
    recipes = _recipes()
    guards = {
        "_ensure-cargo-edit": "cargo_edit_version",
        "_ensure-cargo-llvm-cov": "cargo_llvm_cov_version",
        "_ensure-cargo-nextest": "cargo_nextest_version",
        "_ensure-dprint": "dprint_version",
        "_ensure-git-cliff": "git_cliff_version",
        "_ensure-rumdl": "rumdl_version",
        "_ensure-taplo": "taplo_version",
        "_ensure-typos": "typos_version",
        "_ensure-uv": "uv_version",
        "_ensure-zizmor": "zizmor_version",
    }

    for guard, version_name in guards.items():
        assert version_name in json.dumps(recipes[guard]["body"]), guard


def test_python_environment_has_one_canonical_ci_sync() -> None:
    recipes = _recipes()
    notebook_dependencies = {dependency["recipe"] for dependency in recipes["notebook-sync"]["dependencies"]}

    assert notebook_dependencies == {"python-sync"}
    assert recipes["notebook-sync"]["body"] == []
    assert recipes["python-sync"]["body"] == [["uv sync --locked"]]
    dry_run = _run_just("--dry-run", "ci")
    assert (dry_run.stdout + dry_run.stderr).count("uv sync --locked") == 1


def test_setup_checks_system_prerequisites_before_managed_installs() -> None:
    dependencies = {dependency["recipe"] for dependency in _recipes()["setup-tools"]["dependencies"]}

    assert dependencies == {"_ensure-jq", "_ensure-uv"}


def test_justfile_validation_is_wired_into_repository_gates() -> None:
    recipes = _recipes()

    repository_checks = {dependency["recipe"] for dependency in recipes["check-repository-tooling"]["dependencies"]}
    ci_checks = {dependency["recipe"] for dependency in recipes["ci"]["dependencies"]}
    tooling_ci = {dependency["recipe"] for dependency in recipes["ci-repository-tooling"]["dependencies"]}

    assert "justfile-fmt-check" in repository_checks
    assert {"justfile-fmt-check", "test-python"} <= ci_checks
    assert tooling_ci == {"check-repository-tooling", "test-python"}


def test_release_performance_recipes_are_public_and_documented() -> None:
    recipes = _recipes()
    assert "performance-rerender" not in recipes

    for name in RELEASE_PERFORMANCE_RECIPES:
        assert name in recipes
        assert recipes[name]["private"] is False
        assert recipes[name]["doc"], name


def test_release_commands_separate_preparation_measurement_and_publication() -> None:
    recipes = _recipes()
    preflight = {dependency["recipe"] for dependency in recipes["update-version"]["dependencies"]}
    assert "_ensure-gh" in preflight
    results = [_run_just("--dry-run", name) for name in ("performance-doc", "performance-readme")]
    commands = "\n".join(result.stdout + result.stderr for result in results)
    assert "archive-performance --rerender --promote" in commands
    assert "publish-performance-readme" in commands
    assert "cargo bench" not in commands
    assert "--infer-release" not in commands
    assert "research-repo-tools changelog generate" in json.dumps(recipes["changelog-release"]["body"])
    assert "--date" in json.dumps(recipes["changelog-release"]["body"])


def test_release_performance_recipes_are_discoverable_in_help() -> None:
    help_text = _run_just("help-workflows").stdout

    for name in RELEASE_PERFORMANCE_RECIPES:
        assert f"just {name}" in help_text


def test_update_workflow_is_public_documented_and_discoverable() -> None:
    recipes = _recipes()
    help_text = _run_just("help-workflows").stdout

    for name in UPDATE_RECIPES:
        assert recipes[name]["private"] is False
        assert recipes[name]["doc"], name
    assert "just update" in help_text


def test_update_workflow_composes_the_expected_phases() -> None:
    recipes = _recipes()
    aggregate = [dependency["recipe"] for dependency in recipes["update"]["dependencies"]]
    dependencies = [dependency["recipe"] for dependency in recipes["update-dependencies"]["dependencies"]]

    assert aggregate == ["_ensure-cargo-install-update", "_ensure-uv-stable", "update-dependencies", "update-cargo-tools"]
    assert dependencies == ["_ensure-cargo-edit", "_ensure-uv-stable", "update-cargo-dependencies", "update-python-dependencies"]


@pytest.mark.parametrize("recipe", ["update", "update-cargo-tools", "update-dependencies", "update-python-dependencies"])
def test_update_preflights_stable_uv_before_mutations(recipe: str) -> None:
    result = _run_just("--dry-run", recipe)
    rendered = result.stdout + result.stderr
    preflight = "uv run --locked --no-sync --no-python-downloads python scripts/update_cargo_tool_pins.py --check-uv"

    assert rendered.count(preflight) == 1
    assert rendered.index("uv --version") < rendered.index(preflight)
    if recipe in {"update", "update-cargo-tools"}:
        assert rendered.index(preflight) < rendered.index("cargo install-update --locked")
    if recipe in {"update", "update-dependencies"}:
        assert rendered.index(preflight) < rendered.index("cargo upgrade --incompatible allow")
    if recipe in {"update", "update-dependencies", "update-python-dependencies"}:
        assert rendered.index(preflight) < rendered.index("uv run --locked update-python-dev-pins")
        assert rendered.index(preflight) < rendered.index("uv lock --upgrade")
        assert rendered.index(preflight) < rendered.index("uv sync --locked --group dev")


def test_cargo_update_is_an_unpinned_bootstrap_helper() -> None:
    recipes = _recipes()
    guard = json.dumps(recipes["_ensure-cargo-install-update"]["body"])
    setup = json.dumps(recipes["setup-tools"]["body"])

    assert "command -v cargo-install-update" in guard
    assert "--version" not in guard
    assert "if ! have cargo-install-update; then" in setup
    assert "cargo install --locked cargo-update" in setup
    assert "cargo_update_version" not in JUSTFILE.read_text(encoding="utf-8")

    result = _run_just("--dry-run", "update-cargo-tools")
    rendered = result.stdout + result.stderr
    package_block = re.search(r"packages=\(\n(?P<packages>.*?)\n\)", rendered, re.DOTALL)
    assert package_block is not None
    packages = set(re.findall(r"^\s+([a-z0-9-]+)$", package_block.group("packages"), re.MULTILINE))
    assert packages == set(update_cargo_tool_pins.PIN_TO_PACKAGE.values())
    assert "cargo-update" not in packages
    assert "cargo install-update --all" not in rendered


def test_latest_vs_last_composes_measurement_and_report_steps() -> None:
    dependencies = {dependency["recipe"] for dependency in _recipes()["bench-latest-vs-last"]["dependencies"]}

    assert dependencies == {"bench-latest", "python-sync"}


def test_repository_file_commands_include_nonignored_untracked_files() -> None:
    recipes = _recipes()
    expected = {
        "action-lint",
        "markdown-check",
        "markdown-fix",
        "semgrep",
        "toml-fmt",
        "toml-fmt-check",
        "toml-lint",
        "validate-json",
        "yaml-check",
        "yaml-fix",
    }
    discovered = {name for name, recipe in recipes.items() if "git ls-files" in json.dumps(recipe["body"])}

    assert discovered == expected
    for name in expected:
        body = json.dumps(recipes[name]["body"])
        assert "git ls-files -co --exclude-standard -z --" in body, name


def test_release_workflow_uses_the_canonical_baseline_recipe() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "release-benchmarks.yml").read_text(encoding="utf-8")

    assert 'run: just bench-save-baseline "$RELEASE_TAG"' in workflow
    assert "workflow_dispatch:" in workflow
    assert "types:\n      - published" not in workflow
    assert "must exist as a mutable draft" in workflow
    assert 'gh release edit "$RELEASE_TAG" --draft=false' in workflow
    assert "--clobber" not in workflow


def test_audit_workflow_self_triggers_and_preserves_readable_failure_output() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "audit.yml").read_text(encoding="utf-8")

    assert "pull_request:\n    paths:\n      - .github/workflows/audit.yml" in workflow
    assert "cargo audit --json > audit-results.json\n          json_status=$?" in workflow
    assert "cargo audit\n          readable_status=$?" in workflow
    assert 'if (( json_status != 0 )); then\n            exit "$json_status"' in workflow
    assert 'exit "$readable_status"' in workflow


def test_release_performance_docs_record_the_prospective_asset_boundary() -> None:
    benchmarking = (REPO_ROOT / "docs" / "BENCHMARKING.md").read_text(encoding="utf-8")
    legacy_report = (REPO_ROOT / "docs" / "archive" / "performance" / "v0.4.1-vs-v0.4.0.md").read_text(encoding="utf-8")
    releasing = (REPO_ROOT / "docs" / "RELEASING.md").read_text(encoding="utf-8")

    assert "Legacy, non-reproducible report" in legacy_report
    assert "Repository-owned CSV measurements" in legacy_report
    assert "native Criterion sample archives are unavailable" in legacy_report
    assert "releases through `v0.4.2` have no Criterion baseline attachment" in benchmarking
    assert "`v0.4.3` release creates the first durable" in benchmarking
    assert "`v0.4.4` creates the first complete historical pair" in benchmarking
    assert "`v0.4.2` and earlier releases have no Criterion baseline attachment" in releasing
    assert "`v0.4.4`-against-`v0.4.3` pair" in releasing


def test_ci_runs_the_full_repository_gate_on_every_matrix_platform() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "- name: Run CI checks\n        run: just ci" in workflow
    assert "run: just ci-portability" not in workflow


def test_workflows_resolve_the_python_version_from_the_justfile() -> None:
    ci_workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    release_workflow = (REPO_ROOT / ".github" / "workflows" / "release-benchmarks.yml").read_text(encoding="utf-8")

    assert 'python_version="$(resolve_version python_version)"' in ci_workflow
    assert "python-version: ${{ steps.tool_versions.outputs.PYTHON_VERSION }}" in ci_workflow
    assert 'python_version="$(just --evaluate python_version)"' in release_workflow
    assert "python-version: ${{ steps.python_version.outputs.value }}" in release_workflow
