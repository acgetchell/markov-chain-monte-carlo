"""MCMC command scope, validation coverage, and release credential boundaries."""

import json
import shlex
import shutil
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


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


def test_offline_reporting_and_explicit_measurement_boundary() -> None:
    for name in ("performance-doc", "performance-readme"):
        result = _run_just("--dry-run", name)
        assert "performance " in result.stderr
        assert "--allow-git-mutations" not in result.stderr
        assert "cargo bench" not in result.stderr
    for name in ("performance-local", "performance-release"):
        assert "--allow-git-mutations" in _run_just("--dry-run", name).stderr
    for gate in ("check", "ci", "setup-tools", "update"):
        assert "research-repo-tools review " not in _run_just("--dry-run", gate).stderr


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
    manifest = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    shared_pin = manifest["dependency-groups"]["tooling"][0]
    workflow = (REPO_ROOT / ".github/workflows/release-benchmarks.yml").read_text(encoding="utf-8")
    validation, rest = workflow.split("  validate-release:\n", 1)[1].split("  release-baseline:\n", 1)
    baseline, publication = rest.split("  publish-baseline:\n", 1)
    for writer, command in ((validation, "release-draft"), (publication, "release-upload")):
        assert "contents: write" in writer
        assert "actions/checkout@" not in writer
        assert f"{shared_pin} research-repo-tools performance {command}" in writer
    assert "--publish" in publication
    assert "GH_TOKEN:" not in baseline
    assert "persist-credentials: false" in baseline
    assert 'just performance-baseline "$RELEASE_TAG"' in baseline
    setup = (REPO_ROOT / ".github/actions/setup-toolchain/action.yml").read_text(encoding="utf-8")
    assert "research-repo-tools toolchain export" in setup
    assert "<<'PY'" not in workflow + setup


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


def test_dependabot_caller_uses_shared_approval_without_personal_tokens() -> None:
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/dependabot-auto-merge.yml").read_bytes())
    events = workflow.get("on", workflow.get(True))  # PyYAML's YAML 1.1 resolver treats unquoted "on" as True.
    assert set(events) == {"pull_request_target"}
    assert events["pull_request_target"]["branches"] == ["main"]
    assert workflow["permissions"] == {}
    assert len(workflow["jobs"]) == 1
    job = workflow["jobs"]["approve-and-enable-auto-merge"]
    assert job["uses"].split("@")[0] == "acgetchell/research-repo-tools/.github/workflows/dependabot-approve.yml"
    assert "steps" not in job
    assert "secrets" not in job
    assert job["permissions"] == {"contents": "write", "pull-requests": "write"}
    assert job["with"]["repository"] == "acgetchell/markov-chain-monte-carlo"
    policy = json.loads(job["with"]["policy"])
    assert set(policy) == {"cargo", "uv", "github_actions"}
    assert set(policy["cargo"]["files"]) == {"Cargo.toml", "Cargo.lock"}
    assert set(policy["uv"]["files"]) == {"pyproject.toml", "uv.lock"}
    actions = {
        path.relative_to(REPO_ROOT).as_posix()
        for directory in (REPO_ROOT / ".github/workflows", REPO_ROOT / ".github/actions")
        for path in directory.rglob("*")
        if path.suffix in {".yml", ".yaml"}
    }
    assert set(policy["github_actions"]["files"]) == actions
    for path in (REPO_ROOT / ".github").rglob("*"):
        if path.is_file():
            assert b"CODERABBIT_REVIEW_TOKEN" not in path.read_bytes()


def test_security_workflows_cover_owned_inputs_and_fail_on_findings() -> None:
    from research_repo_tools.selection import select_files

    osv = shlex.split(_run_just("--dry-run", "security-osv").stderr)
    lockfiles = osv[osv.index("osv") + 1 :]
    assert set(lockfiles) == set(select_files(REPO_ROOT, include=("Cargo.lock", "**/Cargo.lock", "uv.lock")))
    assert shlex.split(_run_just("--dry-run", "security-secrets").stderr)[-2:] == ["security", "secrets"]
    for scanner, recipe in (("osv", "security-osv"), ("gitleaks", "security-secrets")):
        workflow = yaml.safe_load((REPO_ROOT / f".github/workflows/{scanner}.yml").read_bytes())
        events = workflow.get("on", workflow.get(True))
        assert {"pull_request", "push", "schedule", "workflow_dispatch"} <= set(events)
        assert "pull_request_target" not in events
        assert workflow["permissions"] == {"contents": "read"}
        job = workflow["jobs"]["scan"]
        assert not job.get("continue-on-error", False)
        checkout = next(step for step in job["steps"] if step.get("uses", "").startswith("actions/checkout@"))
        assert checkout["with"]["persist-credentials"] is False
        if scanner == "gitleaks":
            assert checkout["with"]["fetch-depth"] == 0
        scan = next(step for step in job["steps"] if step.get("id") == "scan")
        assert scan["run"] == f"just {recipe}"
        assert not scan.get("continue-on-error", False)
        upload = next(step for step in job["steps"] if step.get("uses", "").startswith("actions/upload-artifact@"))
        assert upload["if"] == "${{ !cancelled() && steps.scan.outcome != 'skipped' }}"
        assert upload["with"]["path"] == f"target/security/{scanner}-*"
        assert upload["with"]["if-no-files-found"] == "error"


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
