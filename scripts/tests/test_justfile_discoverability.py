"""Regression tests for the public Just recipe surface."""

import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest

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
    """Run real recipes and the installed CLI with local process-boundary stubs."""
    executable = shutil.which("just")
    assert executable is not None
    probe = tmp_path / "review_probe.py"
    probe.write_text(
        textwrap.dedent(
            """
            import json
            import os
            import subprocess
            import sys
            from unittest.mock import patch

            from research_repo_tools.cli import main

            assert sys.argv[1:6] == ["run", "--locked", "--group", "dev", "research-repo-tools"]

            def run(args, **kwargs):
                if args[0] == "git":
                    assert "--no-pager" in args
                    remote = "ls-remote" in args
                    key = "REMOTE" if remote else "LOCAL"
                    status = int(os.environ[key + "_STATUS"])
                    if status:
                        raise subprocess.CalledProcessError(status, args)
                    output = os.environ[key + "_COMMIT"]
                    if remote:
                        output += "\\trefs/heads/main"
                    return subprocess.CompletedProcess(args, 0, output + "\\n")
                assert args[0] == "coderabbit", args
                assert kwargs["capture_output"] is False
                assert kwargs["timeout"] is None
                print(json.dumps(args[1:]))
                return subprocess.CompletedProcess(args, int(os.environ["REVIEW_STATUS"]))

            with patch("research_repo_tools.process.shutil.which", side_effect=lambda name: name):
                with patch("research_repo_tools.process.subprocess.run", side_effect=run):
                    raise SystemExit(main(sys.argv[6:]))
            """
        ),
        encoding="utf-8",
        newline="\n",
    )
    stub = tmp_path / "uv"
    stub.write_text(
        '#!/usr/bin/env bash\nexec "$REVIEW_PYTHON" "$REVIEW_PROBE" "$@"\n',
        encoding="utf-8",
        newline="\n",
    )
    stub.chmod(0o755)
    return subprocess.run(  # noqa: S603 - local stubs cannot contact Git or CodeRabbit.
        [executable, *recipe_args],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
            "REVIEW_PYTHON": sys.executable,
            "REVIEW_PROBE": str(probe),
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
        (("review",), ["--base=origin/main"]),
        (("review", "main"), ["--base=main"]),
        (("review", "topic/it's;printf-injected"), ["--base=topic/it's;printf-injected"]),
        (("review-uncommitted",), ["--uncommitted"]),
    ],
)
@pytest.mark.parametrize("review_status", [0, 23])
def test_review_scope_and_failures_reach_the_cli(tmp_path: Path, recipe_args: tuple[str, ...], scope_args: list[str], review_status: int) -> None:
    result = _run_review_probe(tmp_path, *recipe_args, REVIEW_STATUS=str(review_status))

    assert result.returncode == review_status, result.stderr
    assert json.loads(result.stdout) == [
        "review",
        "--agent",
        "--include-untracked",
        *scope_args,
        "--config",
        str(REPO_ROOT / "AGENTS.md"),
        str(REPO_ROOT / ".coderabbit.yml"),
    ]


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
    result = _run_review_probe(tmp_path, *recipe_args, REMOTE_STATUS="1")

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)[:2] == ["review", "--agent"]


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


def test_managed_tool_declarations_replace_legacy_guards() -> None:
    manifest = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    tools = manifest["tool"]["research-repo-tools"]["toolchain"]["cargo"]
    assert set(tools) == {
        "cargo-audit",
        "cargo-edit",
        "cargo-llvm-cov",
        "cargo-nextest",
        "dprint",
        "git-cliff",
        "rumdl",
        "taplo-cli",
        "typos-cli",
        "zizmor",
    }
    assert "just" not in tools
    assert manifest["tool"]["uv"]["required-version"].startswith("==")
    assert (REPO_ROOT / ".python-version").read_text(encoding="utf-8").strip() == "3.14"
    assert {name for name in _recipes() if name.startswith("_ensure-")} == {"_ensure-gh", "_ensure-jq", "_ensure-uv-stable"}
    rendered = _run_just("--dry-run", "ci")
    commands = rendered.stdout + rendered.stderr
    assert "toolchain run -- cargo" in commands
    assert "cargo install" not in commands
    assert "toolchain sync" not in commands
    assert "toolchain upgrade" not in commands


def test_environment_syncs_select_locked_groups_and_project_kernel() -> None:
    recipes = _recipes()
    notebook = json.dumps(recipes["notebook-sync"]["body"])
    assert "--locked --managed-python --only-group tooling research-repo-tools notebooks sync" in notebook
    python = _run_just("--dry-run", "python-sync")
    assert "toolchain run -- uv sync --locked --managed-python --group dev" in python.stderr


def test_setup_checks_system_prerequisites_before_shared_installs() -> None:
    dependencies = {dependency["recipe"] for dependency in _recipes()["setup"]["dependencies"]}
    assert dependencies == {"_ensure-jq"}
    result = _run_just("--dry-run", "setup-tools")
    assert "--managed-python --only-group tooling research-repo-tools setup" in result.stderr
    assert "cargo install" not in result.stderr


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
    for name, expected in {
        "update": ["update-tools", "update-dependencies"],
        "update-tools": ["update-uv", "update-cargo-tools", "setup"],
        "update-dependencies": ["update-cargo-dependencies", "update-python-dependencies"],
    }.items():
        assert [item["recipe"] for item in recipes[name]["dependencies"]] == expected


def test_update_bootstraps_uv_before_declared_tools_and_dependencies() -> None:
    result = _run_just("--dry-run", "update")
    rendered = result.stdout + result.stderr
    owner_upgrade = "uv run --no-config --no-sync --no-python-downloads research-repo-tools deps update-uv"
    cargo_upgrade = "research-repo-tools toolchain upgrade"
    setup = "research-repo-tools setup"
    requirements = "toolchain run -- cargo upgrade --incompatible allow"
    python_pins = "research-repo-tools deps update-python"
    assert rendered.index(owner_upgrade) < rendered.index(cargo_upgrade) < rendered.index(setup) < rendered.index(requirements)
    assert rendered.index(requirements) < rendered.index(python_pins) < rendered.index("uv lock --upgrade")
    assert "cargo install-update" not in rendered
    assert "deps update-tools" not in rendered


def test_python_only_updates_preflight_uv_and_keep_full_lock_refresh() -> None:
    result = _run_just("--dry-run", "update-python-dependencies")
    rendered = result.stdout + result.stderr
    assert rendered.index("deps check-uv") < rendered.index("deps update-python") < rendered.index("uv lock --upgrade")
    assert rendered.index("uv lock --upgrade") < rendered.index("toolchain run -- uv sync --locked --managed-python --group dev")


def test_tools_check_does_not_install_or_sync() -> None:
    result = _run_just("--dry-run", "tools-check")
    assert "--locked --no-sync --no-python-downloads research-repo-tools toolchain check" in result.stderr
    assert "uv sync" not in result.stderr
    assert "toolchain sync" not in result.stderr


def test_latest_vs_last_composes_measurement_and_report_steps() -> None:
    dependencies = {dependency["recipe"] for dependency in _recipes()["bench-latest-vs-last"]["dependencies"]}

    assert dependencies == {"bench-latest", "python-sync"}


def test_repository_file_commands_include_nonignored_untracked_files() -> None:
    recipes = _recipes()
    expected = {
        "_notebook-all",
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
    assert "toolchain run -- cargo audit --json > audit-results.json\n          json_status=$?" in workflow
    assert "toolchain run -- cargo audit\n          readable_status=$?" in workflow
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


def test_workflows_share_authoritative_setup_declarations() -> None:
    setup = (REPO_ROOT / ".github/actions/setup-toolchain/action.yml").read_text(encoding="utf-8")
    assert "version-file: pyproject.toml" in setup
    assert "--locked --managed-python --only-group tooling research-repo-tools setup" in setup
    assert "toolchain run -- python" in setup
    assert "runner.arch" in setup
    for name in ("ci", "codecov", "release-benchmarks", "rust-clippy", "audit", "codeql"):
        workflow = (REPO_ROOT / ".github/workflows" / f"{name}.yml").read_text(encoding="utf-8")
        assert "uses: $/.github/actions/setup-toolchain" in workflow
        assert "setup-rust-toolchain@" not in workflow
        assert "setup-python@" not in workflow


@pytest.mark.parametrize(
    ("invalid_name", "invalid_value"),
    [
        (None, None),
        ("RESEARCH_REPO_TOOLS_HOME", "/fixture/cache\nINJECTED=true"),
        ("RESEARCH_REPO_TOOLS_HOME", "/fixture/cache\rINJECTED=true"),
        ("PATH", "/bin\r\nINJECTED=true"),
        ("RUSTUP_NO_UPDATE_CHECK", "1\nINJECTED=true"),
        ("RUSTUP_NO_UPDATE_CHECK", None),
    ],
)
def test_setup_environment_export_validates_before_writing(tmp_path: Path, invalid_name: str | None, invalid_value: str | None) -> None:
    setup = (REPO_ROOT / ".github/actions/setup-toolchain/action.yml").read_text(encoding="utf-8")
    script = textwrap.dedent(setup.split("<<'PY'\n", 1)[1].rsplit("\n        PY", 1)[0])
    values = {
        "RESEARCH_REPO_TOOLS_HOME": str(tmp_path / "managed cache=one"),
        "PATH": r"C:\Tools\bin;C:\Program Files\Python",
        "CARGO_HOME": str(tmp_path / "cargo"),
        "RUSTUP_HOME": str(tmp_path / "rustup"),
        "RUSTUP_TOOLCHAIN": "1.98.1",
        "RUSTUP_AUTO_INSTALL": "0",
        "RUSTUP_NO_UPDATE_CHECK": "1",
    }
    destination = tmp_path / "github_env.txt"
    original = b"EXISTING=keep\r\n"
    destination.write_bytes(original)
    environment = {**os.environ, **values, "GITHUB_ENV": str(destination)}
    if invalid_name is not None:
        if invalid_value is None:
            environment.pop(invalid_name)
        else:
            environment[invalid_name] = invalid_value
    result = subprocess.run(  # noqa: S603 - executes the repository-owned exporter with controlled fixture values.
        [sys.executable, "-I", "-c", script],
        env=environment,
        check=False,
        capture_output=True,
        encoding="utf-8",
    )
    if invalid_name is None:
        assert result.returncode == 0, result.stderr
        expected = "".join(f"{name}={value}\n" for name, value in values.items()).encode("utf-8")
        assert destination.read_bytes() == original + expected
    else:
        assert result.returncode != 0
        assert invalid_name in result.stderr
        assert destination.read_bytes() == original
