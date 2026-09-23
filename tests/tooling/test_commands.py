"""Regression tests for the public Just recipe surface."""

import json
import re
import shlex
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
JUSTFILE = REPO_ROOT / "justfile"
RECIPE_DECLARATION = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*)(?:\s+.*?)?:(?=\s|$)", re.MULTILINE)


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


def test_review_recipes_select_shared_modes() -> None:
    prefix = ["uv", "run", "--locked", "--group", "dev", "research-repo-tools", "review"]
    assert shlex.split(_run_just("--dry-run", "review").stderr) == [*prefix, "branch", "--base=origin/main"]
    assert shlex.split(_run_just("--dry-run", "review-uncommitted").stderr) == [*prefix, "uncommitted"]


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


def test_dependency_only_environment_and_registry_pins() -> None:
    import tomllib

    manifest = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert manifest["tool"]["uv"]["package"] is False
    assert "build-system" not in manifest
    assert "scripts" not in manifest["project"]
    assert manifest["dependency-groups"]["tooling"] == ["research-repo-tools==0.1.5"]
    assert "research-repo-tools[notebooks]==0.1.5" in manifest["dependency-groups"]["notebook"]
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    shared = next(package for package in lock["package"] if package["name"] == "research-repo-tools")
    assert shared["version"] == "0.1.5"
    assert shared["source"] == {"registry": "https://pypi.org/simple"}


def test_offline_reporting_and_explicit_measurement_boundary() -> None:
    for name in ("performance-doc", "performance-readme"):
        result = _run_just("--dry-run", name)
        assert "performance " in result.stderr
        assert "--allow-git-mutations" not in result.stderr
        assert "cargo bench" not in result.stderr
    for name in ("performance-local", "performance-release"):
        assert "--allow-git-mutations" in _run_just("--dry-run", name).stderr
    assert "release update" in _run_just("--dry-run", "update-version", "v9.9.9", "--previous-release", "v9.9.8", "--dry-run").stderr


def test_scientific_checks_and_full_platform_ci_remain_wired() -> None:
    recipes = _recipes()
    ci = {item["recipe"] for item in recipes["ci"]["dependencies"]}
    assert {"test-python", "notebook-check", "validate-examples", "test-rust-ci"} <= ci
    notebook = {item["recipe"] for item in recipes["notebook-execute-fast"]["dependencies"]}
    assert "validate-ising-example" in notebook
    workflow = (REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "run: just ci" in workflow
    for platform in ("ubuntu-latest", "macos-latest", "windows-latest"):
        assert platform in workflow


def test_release_credentials_and_shared_commands_are_separated() -> None:
    workflow = (REPO_ROOT / ".github/workflows/release-benchmarks.yml").read_text(encoding="utf-8")
    validation, rest = workflow.split("  validate-release:\n", 1)[1].split("  release-baseline:\n", 1)
    baseline, publication = rest.split("  publish-baseline:\n", 1)
    for writer, command in ((validation, "release-draft"), (publication, "release-upload")):
        assert "contents: write" in writer
        assert "actions/checkout@" not in writer
        assert "research-repo-tools==0.1.5 research-repo-tools performance " + command in writer
    assert "--publish" in publication
    assert "GH_TOKEN:" not in baseline
    assert "persist-credentials: false" in baseline
    assert 'just performance-baseline "$RELEASE_TAG"' in baseline
    setup = (REPO_ROOT / ".github/actions/setup-toolchain/action.yml").read_text(encoding="utf-8")
    assert "research-repo-tools toolchain export" in setup
    assert "<<'PY'" not in workflow + setup


def test_file_validation_uses_shared_selection() -> None:
    recipes = _recipes()
    for name in ("_notebook-all", "action-lint", "markdown-check", "semgrep", "toml-lint", "validate-json", "yaml-check"):
        assert "research-repo-tools files run" in json.dumps(recipes[name]["body"])


def test_scan_excludes_nested_negative_fixtures_but_keeps_consumer_tests() -> None:
    from research_repo_tools.selection import select_files

    command = shlex.split(_run_just("--dry-run", "semgrep").stderr)
    selection = command[: command.index("--")]
    excludes = tuple(selection[index + 1] for index, argument in enumerate(selection) if argument == "--exclude")
    selected = select_files(REPO_ROOT, exclude=excludes)
    assert "tests/tooling/test_commands.py" in selected
    assert not any(name.startswith("tests/semgrep/") for name in selected)
    assert not any(name.startswith("docs/performance/") for name in selected)
