"""MCMC command scope, validation coverage, and release credential boundaries."""

import base64
import json
import shlex
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest
import yaml
from packaging.requirements import Requirement
from research_repo_tools.cargo_examples import discover_examples
from research_repo_tools.just_inspect import dry_run, inspect_justfile
from research_repo_tools.process import run_command

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_offline_reporting_and_explicit_measurement_boundary() -> None:
    for name in ("performance-doc", "performance-readme"):
        result = dry_run(REPO_ROOT, name)
        assert "performance " in result.stderr
        assert "--allow-git-mutations" not in result.stderr
        assert "cargo bench" not in result.stderr
    assert "--allow-git-mutations" in dry_run(REPO_ROOT, "performance-local").stderr
    for gate in ("check", "ci", "setup", "update"):
        assert "research-repo-tools review " not in dry_run(REPO_ROOT, gate).stderr


def test_scientific_checks_and_full_platform_ci_remain_wired() -> None:
    recipes = inspect_justfile(REPO_ROOT).recipes
    ci = {item["recipe"] for item in recipes["ci"]["dependencies"]}
    assert {"test-python", "notebook-check", "validate-examples", "test-rust-ci"} <= ci
    notebook = dry_run(REPO_ROOT, "notebook-execute-fast").stderr
    assert "uv run --isolated --locked --managed-python --only-group tooling research-repo-tools notebooks sync" in notebook
    producer = "research-repo-tools validation cargo-examples"
    consumer = "research-repo-tools notebooks execute"
    assert notebook.count(producer) == 1
    assert "tooling/ising-example.toml" in notebook
    assert notebook.count(consumer) == 1
    assert notebook.index(producer) < notebook.index(consumer)
    workflow = (REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "run: just ci" in workflow
    for platform in ("ubuntu-latest", "macos-latest", "windows-latest"):
        assert platform in workflow


@pytest.mark.parametrize(("status_exit", "status_output", "expected_dirty"), [(0, "", "false"), (0, " M src/lib.rs", "true"), (128, "", None)])
def test_example_build_requires_successful_source_status(status_exit: int, status_output: str, expected_dirty: str | None) -> None:
    """Run the actual recipe with process-boundary stand-ins, without invoking Git or Cargo."""
    executable = shutil.which("bash")
    assert executable is not None
    recipe = dry_run(REPO_ROOT, "_run-examples", ["tooling/examples.toml"]).stderr
    stand_ins = f"""
git() {{
    if [ "$2" = rev-parse ]; then
        printf '%s\\n' fixture-revision
    else
        printf '%s' {shlex.quote(status_output)}
        return {status_exit}
    fi
}}
uv() {{ printf 'build:%s:%s\\n' "$MCMC_SOURCE_REVISION" "$MCMC_SOURCE_DIRTY"; }}
"""
    result = subprocess.run(  # noqa: S603 - resolved Bash executes a repository recipe with inert process stand-ins.
        [executable], input=stand_ins + recipe, check=False, capture_output=True, encoding="utf-8"
    )
    assert result.stderr == ""
    if expected_dirty is None:
        assert result.returncode == status_exit
        assert result.stdout == ""
    else:
        assert result.returncode == 0
        assert result.stdout == f"build:fixture-revision:{expected_dirty}\n"


def test_every_example_runs_once_in_full_ci_through_shared_validation() -> None:
    configuration = tomllib.loads((REPO_ROOT / "tooling/examples.toml").read_text(encoding="utf-8"))
    discovery = discover_examples(REPO_ROOT)
    examples = {example.name for example in discovery}
    selected = []
    features = set(tomllib.loads((REPO_ROOT / "Cargo.toml").read_bytes().decode("utf-8"))["features"])
    for line in dry_run(REPO_ROOT, "ci").stderr.splitlines():
        if "research-repo-tools validation cargo-examples " in line:
            plan = tomllib.loads((REPO_ROOT / shlex.split(line)[-1]).read_text(encoding="utf-8"))
            assert plan["profile"] == "dev"
            assert plan["timeout"] == 300
            assert plan["build-timeout"] == 1800
            names = set(plan.get("include", examples)) - set(plan.get("exclude", []))
            for name in names:
                policy = plan["examples"][name]
                assert policy["expect"]
                assert set(policy["features"]) == features
            selected.extend(names)
    assert set(configuration["examples"]) == examples - {"ising_1d"}
    assert sorted(selected) == sorted(examples)
    for example in discovery:
        assert set(example.required_features) <= features


def test_coverage_workflow_uses_shared_report_validation_after_generation() -> None:
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/codecov.yml").read_bytes())
    steps = workflow["jobs"]["coverage"]["steps"]
    run = next(step for step in steps if step.get("name") == "Run coverage")
    assert run["run"] == "just coverage-ci"
    assert not run.get("continue-on-error", False)
    commands = dry_run(REPO_ROOT, "coverage-ci").stderr
    assert commands.index("--cobertura --output-path coverage/cobertura.xml") < commands.index("just coverage-report")
    summary = shlex.split(dry_run(REPO_ROOT, "coverage-report").stderr)
    assert summary == ["uv", "run", "--locked", "--group", "dev", "research-repo-tools", "coverage", "report", "$@"]
    upload = next(step for step in steps if step.get("uses", "").startswith("codecov/codecov-action@"))
    assert upload["with"]["files"] == "coverage/cobertura.xml"


@pytest.mark.parametrize("failure", [None, "cargo", "converter", "formatter"])
def test_clippy_sarif_analysis_propagates_pipeline_failures(failure: str | None, tmp_path: Path) -> None:
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/rust-clippy.yml").read_bytes())
    steps = workflow["jobs"]["clippy-sarif"]["steps"]
    analysis = next(step for step in steps if step.get("name") == "Run clippy with SARIF output")
    assert analysis["shell"] == "bash"
    assert not analysis.get("continue-on-error", False)
    executable = shutil.which("bash")
    assert executable is not None
    providers = (
        f"uv() {{ printf '%s\\n' '{{}}'; return {7 if failure == 'cargo' else 0}; }}\n"
        f"clippy-sarif() {{ cat; return {7 if failure == 'converter' else 0}; }}\n"
        f"sarif-fmt() {{ cat; return {7 if failure == 'formatter' else 0}; }}\n"
    )
    result = subprocess.run(  # noqa: S603 - actual pipeline with inert providers under the Actions Bash policy.
        [executable, "--noprofile", "--norc", "-e", "-o", "pipefail"],
        input=(providers + analysis["run"]).encode("utf-8"),
        cwd=tmp_path,
        check=False,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == (0 if failure is None else 7), result.stderr
    assert (tmp_path / "rust-clippy-results.sarif").read_bytes() == b"{}\n"
    upload = next(step for step in steps if step.get("name") == "Upload SARIF results")
    assert "always()" in upload["if"]


def test_release_credentials_and_shared_commands_are_separated() -> None:
    manifest = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    shared_requirement = Requirement(manifest["dependency-groups"]["tooling"][0])
    shared_pin = f"{shared_requirement.name}{shared_requirement.specifier}"
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/release-benchmarks.yml").read_bytes())
    jobs = workflow["jobs"]
    assert set(jobs) == {"fetch", "verify", "attach"}
    for name in ("fetch", "attach"):
        writer = jobs[name]
        assert writer["permissions"]["contents"] == "write"
        assert all("actions/checkout@" not in step.get("uses", "") for step in writer["steps"])
        assert shared_pin in yaml.safe_dump(writer)
    verifier = jobs["verify"]
    assert verifier["permissions"] == {"contents": "read", "checks": "read", "actions": "read"}
    assert "persist-credentials: false" in yaml.safe_dump(verifier)
    assert 'just release-receive "$RUNNER_TEMP/accepted-candidate"' in yaml.safe_dump(verifier)
    assert "performance-baseline" not in yaml.safe_dump(jobs)
    assert "performance measure" not in yaml.safe_dump(jobs)
    setup = yaml.safe_load((REPO_ROOT / ".github/actions/setup-toolchain/action.yml").read_bytes())
    setup_steps = setup["runs"]["steps"]
    installer = next(step for step in setup_steps if step.get("run", "").endswith("research-repo-tools setup"))
    assert installer["run"] == "uv run --locked --managed-python --only-group tooling research-repo-tools setup"
    assert installer["env"]["GITHUB_TOKEN"] == "${{ github.token }}"  # noqa: S105 - literal Actions expression, not a credential.
    scanner = next(step for step in setup_steps if step.get("name") == "Set up the declared scanner through the shared API")
    selected = next(step for step in setup_steps if "SELECTED_TOOL" in step.get("env", {}))
    assert installer["if"] == "inputs.scanner == '' && inputs.tool == ''"
    assert selected["if"] == "inputs.scanner == '' && inputs.tool != ''"
    assert 'toolchain sync --tool "$SELECTED_TOOL"' in selected["run"]
    assert scanner["if"] == "inputs.scanner != ''"
    assert all("GITHUB_TOKEN" not in step.get("env", {}) for step in setup_steps if step not in (installer, scanner, selected))


def test_release_version_check_uses_shared_final_metadata_validation(capsys: pytest.CaptureFixture[str]) -> None:
    from research_repo_tools.cli import main

    version = tomllib.loads((REPO_ROOT / "Cargo.toml").read_text(encoding="utf-8"))["package"]["version"]
    assert main(["--root", str(REPO_ROOT), "release", "check", "--final-release"]) == 0
    assert f"Release metadata is synchronized at {version}." in capsys.readouterr().out
    assert "release check --final-release" in dry_run(REPO_ROOT, "release-version-check").stderr


@pytest.mark.parametrize("tag", ["v0.4.2", "0.5.0", "v0.5.0-rc.1", "v0.5.0+build.1", "v00.5.0", "v0.5.0\n"])
def test_shared_release_policy_rejects_wrong_or_noncanonical_tag(tag: str) -> None:
    from research_repo_tools.cli import main

    assert main(["--root", str(REPO_ROOT), "release", "check", tag]) == 1


def test_python_gates_delegate_to_shared_validation_and_keep_fixture_coverage() -> None:
    from research_repo_tools.selection import select_files

    for recipe, action in (("python-check", "check"), ("python-fix", "fix"), ("python-typecheck", "typecheck")):
        commands = [shlex.split(line) for line in dry_run(REPO_ROOT, recipe).stderr.splitlines() if "research-repo-tools python " in line]
        assert commands == [["uv", "run", "--locked", "--no-sync", "--no-python-downloads", "research-repo-tools", "python", action]]
    selected = select_files(REPO_ROOT, include=("*.py", "*.pyi"))
    assert "tests/semgrep/tests/tooling/subprocess_mocks.py" in selected
    assert "tests/tooling/test_commands.py" in selected
    recipes = inspect_justfile(REPO_ROOT).recipes
    for gate in ("ci", "check-repository-tooling"):
        assert "python-check" in {item["recipe"] for item in recipes[gate]["dependencies"]}
    assert "check-repository-tooling" in {item["recipe"] for item in recipes["check"]["dependencies"]}
    assert "notebook-lint" in {item["recipe"] for item in recipes["check-repository-tooling"]["dependencies"]}
    assert "notebook-lint" in {item["recipe"] for item in recipes["notebook-check"]["dependencies"]}
    assert "--include '*.ipynb'" in dry_run(REPO_ROOT, "notebook-lint").stderr


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
    # Credentialed publication requires maintainer review.
    assert set(policy["github_actions"]["files"]) == actions - {".github/workflows/publish-crates.yml"}
    for path in (REPO_ROOT / ".github").rglob("*"):
        if path.is_file():
            assert b"CODERABBIT_REVIEW_TOKEN" not in path.read_bytes()


def test_security_workflows_cover_owned_inputs_and_fail_on_findings() -> None:
    from research_repo_tools.selection import select_files

    osv = shlex.split(dry_run(REPO_ROOT, "audit").stderr)
    lockfiles = osv[osv.index("osv") + 1 :]
    assert set(lockfiles) == set(select_files(REPO_ROOT, include=("Cargo.lock", "**/Cargo.lock", "uv.lock", "**/uv.lock")))
    assert shlex.split(dry_run(REPO_ROOT, "security-secrets").stderr)[-2:] == ["security", "secrets"]
    for scanner, recipe in (("osv", "audit"), ("gitleaks", "security-secrets")):
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


@pytest.mark.parametrize(
    ("record", "archived", "expected_exit"),
    [
        ('  "tukey.json": "{fences}"', True, 0),
        ('  "tukey.json": "{prefix}', True, 0),
        ('key.json": "{fences}"', True, 0),
        ('  "key.json": "{fences}"', True, 3),
        ('  "api_key": "{other}"', True, 3),
        ('  "tukey.json": "{other}"', True, 3),
        ('  "tukey.json": "{fences}"', False, 3),
    ],
)
def test_gitleaks_benchmark_scope(tmp_path: Path, record: str, *, archived: bool, expected_exit: int) -> None:
    """Cover complete and chunked numeric fields and preserve unrelated detections."""
    from research_repo_tools import config, toolchain, toolchain_config

    archive = Path("docs/performance/runs/runs") / ("a" * 64)
    source = tmp_path / (archive if archived else Path("other")) / "run.json"
    source.parent.mkdir(parents=True)
    fences = base64.b64encode(b"[14181.6878609986,12.872,73.30,57.8]").decode("ascii")
    other = base64.b64encode(b'[14181.6878609986,12.872,73.30,"public fixture"]').decode("ascii")
    source.write_bytes(record.format(fences=fences, prefix=fences[:37], other=other).encode("ascii") + b"\n")

    runtime = toolchain.Runtime(toolchain_config.load(config.load(root=REPO_ROOT)), tools=["gitleaks"])
    scanner = next(status for status in runtime.inspect() if status.name == "gitleaks")
    assert scanner.ok, scanner
    result = run_command(
        scanner.path,
        ["dir", ".", "--config", str(REPO_ROOT / ".gitleaks.toml"), "--redact=100", "--ignore-gitleaks-allow", "--exit-code=3"],
        cwd=tmp_path,
        check=False,
    )
    assert result.returncode == expected_exit, result.stderr


def test_python_annotation_policy_and_review_exclusions_remain_precise() -> None:
    manifest = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    lint = manifest["tool"]["ruff"]["lint"]
    assert {"ANN001", "ANN002", "ANN003", "ANN201", "ANN202", "ANN204", "ANN205", "ANN206"} <= set(lint["extend-select"])
    assert {"TC", "UP"} <= set(lint["select"])
    assert lint["flake8-type-checking"]["strict"] is True
    fixture = "tests/semgrep/tests/tooling/subprocess_mocks.py"
    assert not {rule for rule in lint["per-file-ignores"][fixture] if rule.startswith(("ANN", "TC", "UP"))}
    review = yaml.safe_load((REPO_ROOT / ".coderabbit.yml").read_bytes())["reviews"]
    assert "!tests/semgrep/**" in review["path_filters"]
    assert review["pre_merge_checks"]["docstrings"]["mode"] == "off"


def test_managed_update_preserves_python_tool_policy_without_setup() -> None:
    manifest = tomllib.loads((REPO_ROOT / "pyproject.toml").read_bytes().decode("utf-8"))
    config = manifest["tool"]["research-repo-tools"]
    assert config["toolchain"]["inherit-python"] is True
    assert config["toolchain"]["inherit-python-tools"] is True
    assert all(tool in manifest["dependency-groups"]["dev"] for tool in ("pytest", "ruff", "ty"))
    for name in ("dprint", "rumdl"):
        assert name in config["toolchain"]["binaries"]
        assert name not in config["toolchain"]["cargo"]
    update = dry_run(REPO_ROOT, "update-tools").stderr
    assert "deps update-uv" in update
    assert "toolchain upgrade" in update
    assert "research-repo-tools setup" not in update
    python_update = dry_run(REPO_ROOT, "update-python-dependencies").stderr
    assert "uv run --isolated --locked --only-group tooling" in python_update
    assert "deps update-python --all-projects --adopt-supplier tooling" in python_update


@pytest.mark.parametrize("unapproved", [False, True])
def test_shared_allowlist_covers_the_actual_composite_action(tmp_path: Path, *, unapproved: bool) -> None:
    """Exercise the composite document through the same gate as CI."""
    from research_repo_tools.cli import main

    action = yaml.safe_load((REPO_ROOT / ".github/actions/setup-toolchain/action.yml").read_bytes())
    if unapproved:
        action["runs"]["steps"][0]["uses"] = "untrusted/setup-uv@" + "a" * 40
    composite = tmp_path / "action.yml"
    composite.write_text(yaml.safe_dump(action), encoding="utf-8", newline="\n")
    policy = tmp_path / "actions.json"
    policy.write_bytes((REPO_ROOT / ".github/settings/actions-selected.json").read_bytes())
    assert main(["--root", str(tmp_path), "actions", "allowlist", "--policy", str(policy), str(composite)]) == int(unapproved)
    recipes = inspect_justfile(REPO_ROOT).recipes
    for gate in ("ci", "check-repository-tooling"):
        assert "workflow-allowlist-check" in {item["recipe"] for item in recipes[gate]["dependencies"]}
    command = shlex.split(dry_run(REPO_ROOT, "workflow-allowlist-check").stderr)
    assert command[command.index("actions") :] == [
        "actions",
        "allowlist",
        "--policy",
        ".github/settings/actions-selected.json",
        ".github/workflows",
        ".github/actions/setup-toolchain/action.yml",
    ]


def test_zizmor_local_and_sarif_workflow_share_the_declared_policy() -> None:
    manifest = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["research-repo-tools"]
    assert manifest["zizmor"]["persona"] == "regular"
    prefix = ["uv", "run", "--locked", "--only-group", "tooling", "research-repo-tools", "zizmor", "check"]
    assert shlex.split(dry_run(REPO_ROOT, "zizmor").stderr) == [*prefix, "$@", ".github"]
    recipe = inspect_justfile(REPO_ROOT).recipes["zizmor"]
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


@pytest.mark.parametrize("scanner", ["zizmor", "gitleaks", "osv-scanner", "cargo"])
def test_scanner_setup_forwards_only_allowlisted_tools(scanner: str, monkeypatch: pytest.MonkeyPatch) -> None:
    action = yaml.safe_load((REPO_ROOT / ".github/actions/setup-toolchain/action.yml").read_bytes())
    steps = action["runs"]["steps"]
    step = next(step for step in steps if step.get("name") == "Set up the declared scanner through the shared API")
    monkeypatch.setenv("SCANNER", scanner)
    script = 'uv() { printf "%s\\n" "$*"; }\n' + step["run"]
    result = run_command("bash", ["--noprofile", "--norc", "-euc", script], cwd=REPO_ROOT, check=False)
    if scanner == "cargo":
        assert result.returncode == 2
        assert result.stdout == ""
        assert "Unsupported scanner: cargo" in result.stderr
        return
    assert result.returncode == 0, result.stderr
    base = "run --locked --managed-python --only-group tooling research-repo-tools toolchain"
    assert result.stdout.splitlines() == [f"{base} {operation} --tool {scanner}" for operation in ("sync", "export")]
    assert next(step for step in steps if step.get("name") == "Expose checked tools to workflow actions")["if"] == "inputs.scanner == ''"
    filename = "osv" if scanner == "osv-scanner" else scanner
    workflow = yaml.safe_load((REPO_ROOT / f".github/workflows/{filename}.yml").read_bytes())
    job = next(iter(workflow["jobs"].values()))
    setup = next(step for step in job["steps"] if step.get("uses") == "$/.github/actions/setup-toolchain")
    assert setup["with"] == {"scanner": scanner}


def test_scan_excludes_nested_negative_fixtures_but_keeps_consumer_tests() -> None:
    from research_repo_tools.selection import select_files

    command = shlex.split(dry_run(REPO_ROOT, "semgrep").stderr.replace("\\\n", " "))
    assert command[:8] == ["uv", "run", "--locked", "--group", "dev", "research-repo-tools", "semgrep", "scan"]
    includes = tuple(command[index + 1] for index, argument in enumerate(command) if argument == "--include")
    excludes = tuple(command[index + 1] for index, argument in enumerate(command) if argument == "--exclude")
    rules = yaml.safe_load((REPO_ROOT / "semgrep.yaml").read_bytes())["rules"]
    rule_paths = {path.removeprefix("/") for rule in rules for path in rule["paths"]["include"] if not path.startswith("/tests/semgrep/")}
    assert set(includes) == {f":(glob){path}" if "*" in path else path for path in rule_paths}
    selected = select_files(REPO_ROOT, include=includes, exclude=excludes)
    assert {"src/lib.rs", "tests/public_api.rs", "tests/tooling/test_commands.py", "benches/diagnostic_backends/src/lib.rs", "docs/BENCHMARKING.md"} <= set(
        selected
    )
    assert not any(name.startswith("tests/semgrep/") for name in selected)
    assert not any(name.startswith("docs/performance/") for name in selected)


def test_semgrep_sarif_uses_canonical_scan_and_only_current_reports() -> None:
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/semgrep-sarif.yml").read_bytes())
    steps = workflow["jobs"]["semgrep-sarif"]["steps"]
    scan = next(step for step in steps if step.get("id") == "semgrep")
    assert scan["run"] == "uv run --locked --managed-python --group dev just semgrep"
    assert scan["continue-on-error"] is True
    canonical = shlex.split(dry_run(REPO_ROOT, "semgrep").stderr.replace("\\\n", " "))
    destination = canonical[canonical.index("--output") + 1]
    reports = next(step for step in steps if step.get("id") == "reports")
    assert reports["run"] == f"uv run --locked --managed-python --group dev research-repo-tools semgrep check-reports --output {destination}"
    assert reports["if"] == "always() && !cancelled()"
    assert reports["continue-on-error"] is True
    settings = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["research-repo-tools"]["semgrep"]
    assert settings["report-layout"] == "aggregate"
    assert settings["target-timeout"] == 30
    assert settings["inline-suppressions"] is True
    upload = next(step for step in steps if step.get("uses", "").startswith("github/codeql-action/upload-sarif@"))
    assert upload["with"]["sarif_file"] == f"{destination}/semgrep.sarif"
    assert upload["with"]["category"] == settings["report-category"]
    assert "always()" in upload["if"]
    assert "steps.reports.outcome == 'success'" in upload["if"]
    assert "github.event.pull_request.head.repo.full_name == github.repository" in upload["if"]
    gate = next(step for step in steps if step.get("name") == "Fail on repository rule findings or invalid reports")
    assert gate["if"] == "always() && (steps.semgrep.outcome != 'success' || steps.reports.outcome != 'success')"
    assert gate["run"] == "exit 1"
