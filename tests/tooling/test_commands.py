"""Regression tests for the public Just recipe surface."""

import json
import re
import shlex
import shutil
import subprocess
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

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
    manifest = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert manifest["tool"]["uv"]["package"] is False
    assert "build-system" not in manifest
    assert "scripts" not in manifest["project"]
    assert manifest["dependency-groups"]["tooling"] == ["research-repo-tools==0.1.6"]
    assert "research-repo-tools[notebooks]==0.1.6" in manifest["dependency-groups"]["notebook"]
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    shared = next(package for package in lock["package"] if package["name"] == "research-repo-tools")
    assert shared["version"] == "0.1.6"
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
        assert "research-repo-tools==0.1.6 research-repo-tools performance " + command in writer
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


def test_python_gate_covers_fixtures_with_full_configured_native_checks() -> None:
    from research_repo_tools.selection import select_files

    commands = [shlex.split(line) for line in _run_just("--dry-run", "python-check").stderr.splitlines() if "research-repo-tools files run" in line]
    native = []
    for command in commands:
        selection, tool = command[: command.index("--")], command[command.index("--") + 1 :]
        assert "--exclude" not in selection
        includes = tuple(selection[index + 1] for index, value in enumerate(selection) if value == "--include")
        assert set(includes) == {"*.py", "*.pyi"}
        selected = select_files(REPO_ROOT, include=includes)
        assert "tests/semgrep/tests/tooling/python_exceptions.py" in selected
        assert "tests/tooling/test_commands.py" in selected
        native.append(tool)
    assert native == [
        ["ty", "check", "--no-force-exclude"],
        ["ruff", "format", "--check", "--no-force-exclude"],
        ["ruff", "check", "--no-fix", "--no-force-exclude"],
    ]
    recipes = _recipes()
    for gate in ("ci", "check-repository-tooling"):
        assert "python-check" in {item["recipe"] for item in recipes[gate]["dependencies"]}
    assert "check-repository-tooling" in {item["recipe"] for item in recipes["check"]["dependencies"]}
    assert "notebook-lint" in {item["recipe"] for item in recipes["check-repository-tooling"]["dependencies"]}
    assert "notebook-lint" in {item["recipe"] for item in recipes["notebook-check"]["dependencies"]}
    assert "--include '*.ipynb'" in _run_just("--dry-run", "notebook-lint").stderr


def test_python_annotation_policy_and_review_exclusions_remain_precise() -> None:
    manifest = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    lint = manifest["tool"]["ruff"]["lint"]
    assert {"ANN001", "ANN002", "ANN003", "ANN201", "ANN202", "ANN204", "ANN205", "ANN206"} <= set(lint["extend-select"])
    assert {"TC", "UP"} <= set(lint["select"])
    assert lint["flake8-type-checking"]["strict"] is True
    fixture = "tests/semgrep/tests/tooling/python_exceptions.py"
    assert {rule for rule in lint["per-file-ignores"][fixture] if rule.startswith(("ANN", "TC", "UP"))} == {"ANN201"}
    review = yaml.safe_load((REPO_ROOT / ".coderabbit.yml").read_bytes())["reviews"]
    assert "!tests/semgrep/**" in review["path_filters"]
    assert review["pre_merge_checks"]["docstrings"]["mode"] == "off"


def test_zizmor_local_and_sarif_workflow_share_the_declared_policy() -> None:
    manifest = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["research-repo-tools"]
    assert manifest["toolchain"]["cargo"]["zizmor"] == "1.30.1"
    assert manifest["zizmor"]["persona"] == "regular"
    prefix = ["uv", "run", "--locked", "--group", "dev", "research-repo-tools", "zizmor", "check"]
    assert shlex.split(_run_just("--dry-run", "zizmor").stderr) == [*prefix, "$@", ".github"]
    recipe = _recipes()["zizmor"]
    assert "positional-arguments" in recipe["attributes"]
    assert [(parameter["name"], parameter["kind"]) for parameter in recipe["parameters"]] == [("args", "star")]
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/zizmor.yml").read_bytes())
    job = workflow["jobs"]["analyze"]
    assert job["permissions"] == {"actions": "read", "contents": "read", "security-events": "write"}
    steps = job["steps"]
    assert any(step.get("uses") == "$/.github/actions/setup-toolchain" for step in steps)
    audit = next(step for step in steps if step.get("id") == "audit")
    sarif = next(step for step in steps if step.get("id") == "sarif")
    assert audit["run"] == "just zizmor --require-online"
    assert not audit.get("continue-on-error", False)
    assert sarif["run"] == 'just zizmor --require-online --format sarif > "$RUNNER_TEMP/zizmor.sarif"'
    assert sarif["if"] == "${{ !cancelled() && steps.audit.outcome != 'skipped' }}"
    for step in (audit, sarif):
        assert step["env"] == {"ZIZMOR_GITHUB_TOKEN": "${{ github.token }}"}
    upload = next(step for step in steps if step.get("uses", "").startswith("github/codeql-action/upload-sarif@"))
    assert upload["with"]["sarif_file"] == "${{ runner.temp }}/zizmor.sarif"
    assert upload["with"]["category"] == "zizmor"
    assert "!cancelled() && steps.sarif.outcome == 'success'" in upload["if"]
    assert "github.event.pull_request.head.repo.full_name == github.repository" in upload["if"]
    assert "github.actor != 'dependabot[bot]'" in upload["if"]


def test_scan_excludes_nested_negative_fixtures_but_keeps_consumer_tests() -> None:
    from research_repo_tools.selection import select_files

    command = shlex.split(_run_just("--dry-run", "semgrep").stderr)
    selection = command[: command.index("--")]
    excludes = tuple(selection[index + 1] for index, argument in enumerate(selection) if argument == "--exclude")
    selected = select_files(REPO_ROOT, exclude=excludes)
    assert "tests/tooling/test_commands.py" in selected
    assert not any(name.startswith("tests/semgrep/") for name in selected)
    assert not any(name.startswith("docs/performance/") for name in selected)
