"""MCMC command scope, validation coverage, and release credential boundaries."""

import json
import shlex
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest
import yaml
from research_repo_tools.cargo_examples import discover_examples
from research_repo_tools.just_inspect import dry_run, inspect_justfile

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_offline_reporting_and_explicit_measurement_boundary() -> None:
    for name in ("performance-doc", "performance-readme"):
        result = dry_run(REPO_ROOT, name)
        assert "performance " in result.stderr
        assert "--allow-git-mutations" not in result.stderr
        assert "cargo bench" not in result.stderr
    for name in ("performance-local", "performance-release"):
        assert "--allow-git-mutations" in dry_run(REPO_ROOT, name).stderr
    for gate in ("check", "ci", "setup", "update"):
        assert "research-repo-tools review " not in dry_run(REPO_ROOT, gate).stderr


def test_scientific_checks_and_full_platform_ci_remain_wired() -> None:
    recipes = inspect_justfile(REPO_ROOT).recipes
    ci = {item["recipe"] for item in recipes["ci"]["dependencies"]}
    assert {"test-python", "notebook-check", "validate-examples", "test-rust-ci"} <= ci
    notebook = dry_run(REPO_ROOT, "notebook-execute-fast").stderr
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


def test_release_credentials_and_shared_commands_are_separated() -> None:
    manifest = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    shared_pin = manifest["dependency-groups"]["tooling"][0]
    workflow = (REPO_ROOT / ".github/workflows/release-benchmarks.yml").read_text(encoding="utf-8")
    validation, rest = workflow.split("  validate-release:\n", 1)[1].split("  release-baseline:\n", 1)
    baseline, publication = rest.split("  publish-baseline:\n", 1)
    for writer, command in ((validation, "release-draft"), (publication, "release-upload")):
        assert "contents: write" in writer
        assert "actions/checkout@" not in writer
        assert f"{shared_pin} research-repo-tools --root" in writer
        assert f"performance {command}" in writer
    assert "--publish" not in publication
    assert "GH_TOKEN:" not in baseline
    assert "persist-credentials: false" in baseline
    assert 'just performance-baseline "$RELEASE_TAG"' in baseline
    setup = (REPO_ROOT / ".github/actions/setup-toolchain/action.yml").read_text(encoding="utf-8")
    assert "research-repo-tools toolchain export" in setup
    assert "<<'PY'" not in workflow
    setup_steps = yaml.safe_load(setup)["runs"]["steps"]
    installer = next(step for step in setup_steps if step.get("run", "").endswith("research-repo-tools setup"))
    assert installer["env"]["GITHUB_TOKEN"] == "${{ github.token }}"  # noqa: S105 - literal Actions expression, not a credential.
    scanner = next(step for step in setup_steps if step.get("name") == "Set up the declared scanner through the shared API")
    assert installer["if"] == "inputs.scanner == ''"
    assert scanner["if"] == "inputs.scanner != ''"
    assert all("GITHUB_TOKEN" not in step.get("env", {}) for step in setup_steps if step not in (installer, scanner))


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
    assert set(policy["uv"]["files"]) == {"pyproject.toml", "uv.lock", "tooling/semgrep/pyproject.toml", "tooling/semgrep/uv.lock"}
    actions = {
        path.relative_to(REPO_ROOT).as_posix()
        for directory in (REPO_ROOT / ".github/workflows", REPO_ROOT / ".github/actions")
        for path in directory.rglob("*")
        if path.suffix in {".yml", ".yaml"}
    }
    # Credentialed publishing changes require maintainer review.
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
    assert not config["toolchain"].get("inherit-python-tools", False)
    assert not any("python-tools" in str(requirement) for group in manifest["dependency-groups"].values() for requirement in group)
    for name in ("dprint", "rumdl"):
        assert name in config["toolchain"]["binaries"]
        assert name not in config["toolchain"]["cargo"]
    update = dry_run(REPO_ROOT, "update-tools").stderr
    assert "deps update-uv" in update
    assert "toolchain upgrade" in update
    assert "research-repo-tools setup" not in update
    assert "deps update-python --all-projects" in dry_run(REPO_ROOT, "update-python-dependencies").stderr


@pytest.mark.parametrize("unapproved", [False, True])
def test_shared_allowlist_covers_the_actual_composite_action(tmp_path: Path, *, unapproved: bool) -> None:
    """The shared CLI accepts workflows; adapt this consumer's composite steps only."""
    from research_repo_tools.cli import main

    action = yaml.safe_load((REPO_ROOT / ".github/actions/setup-toolchain/action.yml").read_bytes())
    if unapproved:
        action["runs"]["steps"][0]["uses"] = "untrusted/setup-uv@" + "a" * 40
    workflow = tmp_path / "setup.yml"
    workflow.write_text(yaml.safe_dump({"jobs": {"setup": {"steps": action["runs"]["steps"]}}}), encoding="utf-8", newline="\n")
    policy = tmp_path / "actions.json"
    policy.write_bytes((REPO_ROOT / ".github/settings/actions-selected.json").read_bytes())
    assert main(["--root", str(tmp_path), "actions", "allowlist", "--policy", str(policy), str(workflow)]) == int(unapproved)
    recipes = inspect_justfile(REPO_ROOT).recipes
    for gate in ("ci", "check-repository-tooling"):
        assert "workflow-allowlist-check" in {item["recipe"] for item in recipes[gate]["dependencies"]}
    assert "actions allowlist --policy .github/settings/actions-selected.json .github/workflows" in dry_run(REPO_ROOT, "workflow-allowlist-check").stderr


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


@pytest.mark.parametrize("scanner", ["zizmor", "gitleaks", "osv-scanner"])
def test_scanner_setup_installs_only_its_declared_plan(scanner: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from research_repo_tools import toolchain

    action = yaml.safe_load((REPO_ROOT / ".github/actions/setup-toolchain/action.yml").read_bytes())
    steps = action["runs"]["steps"]
    step = next(step for step in steps if step.get("name") == "Set up the declared scanner through the shared API")
    script = step["run"].split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    plans = []

    def capture(runtime: toolchain.Runtime) -> None:
        plans.append(runtime.plan)

    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setenv("SCANNER", scanner)
    monkeypatch.setenv("RESEARCH_REPO_TOOLS_HOME", str(tmp_path / "managed"))
    destination = tmp_path / "environment"
    monkeypatch.setenv("GITHUB_ENV", str(destination))
    monkeypatch.setattr(toolchain.Runtime, "sync", capture)
    monkeypatch.setattr(toolchain.Runtime, "inspect", lambda _runtime: [])
    names = ("PATH", "UV_PYTHON_INSTALL_DIR", "CARGO_HOME", "RUSTUP_HOME", "RUSTUP_TOOLCHAIN", "RUSTUP_AUTO_INSTALL", "RUSTUP_NO_UPDATE_CHECK")
    monkeypatch.setattr(toolchain.Runtime, "environment", lambda _runtime: dict.fromkeys(names, "fixture"))
    exec(compile(script, str(REPO_ROOT / ".github/actions/setup-toolchain/action.yml"), "exec"), {})  # noqa: S102 - execute the checked-in setup adapter.

    assert len(plans) == 1
    plan = plans[0]
    assert {tool.package for tool in plan.cargo} == ({"zizmor"} if scanner == "zizmor" else set())
    assert {tool.name for tool in plan.binaries} == (set() if scanner == "zizmor" else {scanner})
    if scanner == "zizmor":
        assert plan.rust is not None
        assert plan.rust.profile == "minimal"
        assert plan.rust.components == plan.rust.targets == ()
    else:
        assert plan.rust is None
    exported = dict(line.split("=", 1) for line in destination.read_text(encoding="utf-8").splitlines())
    assert exported["RESEARCH_REPO_TOOLS_HOME"] == str(tmp_path / "managed")
    assert ("RUSTUP_HOME" in exported) == (scanner == "zizmor")
    assert next(step for step in steps if step.get("name") == "Expose checked tools to workflow actions")["if"] == "inputs.scanner == ''"
    filename = "osv" if scanner == "osv-scanner" else scanner
    workflow = yaml.safe_load((REPO_ROOT / f".github/workflows/{filename}.yml").read_bytes())
    job = next(iter(workflow["jobs"].values()))
    setup = next(step for step in job["steps"] if step.get("uses") == "$/.github/actions/setup-toolchain")
    assert setup["with"] == {"scanner": scanner}


def test_scan_excludes_nested_negative_fixtures_but_keeps_consumer_tests() -> None:
    from research_repo_tools.selection import select_files

    command = shlex.split(dry_run(REPO_ROOT, "semgrep").stderr)
    assert command[:7] == ["uv", "run", "--locked", "--project", "tooling/semgrep", "uv", "run"]
    selection = command[: command.index("--")]
    excludes = tuple(selection[index + 1] for index, argument in enumerate(selection) if argument == "--exclude")
    selected = select_files(REPO_ROOT, exclude=excludes)
    assert "tests/tooling/test_commands.py" in selected
    assert not any(name.startswith("tests/semgrep/") for name in selected)
    assert not any(name.startswith("docs/performance/") for name in selected)


def test_semgrep_sarif_uses_canonical_inventory_in_one_aggregate_scan() -> None:
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/semgrep-sarif.yml").read_bytes())
    steps = workflow["jobs"]["semgrep-sarif"]["steps"]
    scan = next(step for step in steps if step.get("id") == "semgrep")
    script = scan["run"]
    commands = [shlex.split(line) for line in script.replace("\\\n", " ").splitlines()]
    inventory = next(command for command in commands if "files" in command)
    canonical = shlex.split(dry_run(REPO_ROOT, "semgrep").stderr)
    canonical_selection = canonical[: canonical.index("--")]
    assert inventory[:8] == ["uv", "run", "--locked", "--managed-python", "--group", "dev", "research-repo-tools", "files"]
    assert inventory[8] == "list"
    assert tuple(inventory[index + 1] for index, argument in enumerate(inventory) if argument == "--exclude") == tuple(
        canonical_selection[index + 1] for index, argument in enumerate(canonical_selection) if argument == "--exclude"
    )
    assert "--include" not in inventory
    assert inventory[-3:] == ["--null", ">", "$RUNNER_TEMP/semgrep-inputs"]
    assert ["mapfile", "-d", "", "-t", "scan_paths", "<", "$RUNNER_TEMP/semgrep-inputs"] in commands
    assert script.index("files list") < script.index("mapfile") < script.index("set +e")
    native = [command for command in commands if "semgrep" in command]
    assert len(native) == 1
    assert native[0][:7] == ["uv", "run", "--locked", "--managed-python", "--project", "tooling/semgrep", "semgrep"]
    assert native[0][6:] == [
        *canonical[canonical.index("--") + 1 :],
        "--sarif",
        "--output",
        "semgrep-results.sarif",
        "--",
        "${scan_paths[@]}",
    ]
    assert script.index('"${scan_paths[@]}"') < script.index("status=$?") < script.index('echo "exit_code=$status"')
    upload = next(step for step in steps if step.get("uses", "").startswith("github/codeql-action/upload-sarif@"))
    assert upload["with"]["sarif_file"] == "semgrep-results.sarif"
    assert "always()" in upload["if"]
    gate = next(step for step in steps if step.get("name") == "Fail on repository rule findings")
    assert gate["if"] == "steps.semgrep.outputs.exit_code != '0'"
    assert gate["run"] == "exit 1"
