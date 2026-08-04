"""Regression tests for the public Just recipe surface."""

import json
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

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
    "performance-github-assets",
    "performance-local",
    "performance-release",
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

    for name in RELEASE_PERFORMANCE_RECIPES:
        assert name in recipes
        assert recipes[name]["private"] is False
        assert recipes[name]["doc"], name


def test_release_performance_recipes_are_discoverable_in_help() -> None:
    help_text = _run_just("help-workflows").stdout

    for name in RELEASE_PERFORMANCE_RECIPES:
        assert f"just {name}" in help_text


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


def test_workflows_resolve_the_python_version_from_the_justfile() -> None:
    ci_workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    release_workflow = (REPO_ROOT / ".github" / "workflows" / "release-benchmarks.yml").read_text(encoding="utf-8")

    assert 'python_version="$(resolve_version python_version)"' in ci_workflow
    assert "python-version: ${{ steps.tool_versions.outputs.PYTHON_VERSION }}" in ci_workflow
    assert 'python_version="$(just --evaluate python_version)"' in release_workflow
    assert "python-version: ${{ steps.python_version.outputs.value }}" in release_workflow
