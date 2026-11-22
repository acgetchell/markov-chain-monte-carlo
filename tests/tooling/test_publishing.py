"""Exercise markov-chain-monte-carlo's published release contract without credentials or uploads."""

import json
import os
import shlex
import subprocess
import sys
import tomllib
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml
from research_repo_tools import changelog, config, dependency_provenance, release_publishing
from research_repo_tools.cli import main
from research_repo_tools.just_inspect import dry_run, inspect_justfile
from research_repo_tools.process import run_command
from research_repo_tools.registry import RegistryVersion
from research_repo_tools.release_assets import ReleaseTarget, serialize_release_target
from research_repo_tools.release_publishing import require_checks

from tests.tooling.support_evidence import TOOLING_VERSION, install_clean_checkout

REPO_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = "acgetchell/markov-chain-monte-carlo"
COMMIT = "a" * 40
WORKFLOW = REPO_ROOT / ".github/workflows/publish-crates.yml"
POLICY = config.load(root=REPO_ROOT)
VERSION = tomllib.loads((REPO_ROOT / "Cargo.toml").read_text(encoding="utf-8"))["package"]["version"]
TAG = f"v{VERSION}"
CHECKS = (
    "repository-tooling",
    "build (ubuntu-latest)",
    "build (macos-latest)",
    "build (windows-latest)",
    "Clippy SARIF Analysis",
    "Repository Rule SARIF Analysis",
)


def test_publishing_policy_requires_native_scientific_gates_and_the_criterion_asset() -> None:
    assert POLICY.release.published_dependencies == ("research-repo-tools",)
    assert POLICY.publishing is not None
    assert (POLICY.publishing.registry, POLICY.publishing.package, POLICY.publishing.repository) == (
        "crates-io",
        "markov-chain-monte-carlo",
        "acgetchell/markov-chain-monte-carlo",
    )
    assert tuple((check.name, check.app_id) for check in POLICY.publishing.required_checks) == tuple((name, 15368) for name in CHECKS)
    assert POLICY.publishing.required_assets == ("markov-chain-monte-carlo-v{version}-criterion-baseline.tar.gz",)
    assert POLICY.publishing.expected_title == "tag"
    assert POLICY.publishing.generated_notes
    assert POLICY.publishing.require_immutable


@pytest.fixture
def release_event(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Supply only provider/read-only Git evidence to the installed release CLI."""
    event = {
        "action": "published",
        "repository": {"full_name": "acgetchell/markov-chain-monte-carlo"},
        "release": {"id": 42, "tag_name": TAG, "draft": False, "prerelease": False},
    }
    event_file = tmp_path / "event.json"
    event_file.write_text(json.dumps(event), encoding="utf-8", newline="\n")
    for key, value in {
        "GITHUB_EVENT_NAME": "release",
        "GITHUB_REPOSITORY": "acgetchell/markov-chain-monte-carlo",
        "GITHUB_SHA": COMMIT,
        "GITHUB_REF": f"refs/tags/{TAG}",
        "GITHUB_EVENT_PATH": str(event_file),
    }.items():
        monkeypatch.setenv(key, value)

    def git(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        assert args in (["--no-pager", "rev-parse", "HEAD"], ["--no-pager", "status", "--porcelain", "--untracked-files=all"])
        return subprocess.CompletedProcess(args, 0, stdout=COMMIT if "rev-parse" in args else "", stderr="")

    checks = [
        {"id": index, "name": name, "head_sha": COMMIT, "app": {"id": 15368}, "status": "completed", "conclusion": "success"}
        for index, name in enumerate(CHECKS, 1)
    ]
    responses: dict[str, object] = {
        "": {"full_name": "acgetchell/markov-chain-monte-carlo", "default_branch": "main"},
        "branches/main": {"name": "main", "protected": True},
        f"compare/{COMMIT}...main": {"status": "identical"},
        f"commits/{COMMIT}/check-runs?filter=latest&per_page=100": [{"total_count": len(checks), "check_runs": checks}],
    }
    monkeypatch.setattr(release_publishing, "run_git_command", git)

    # These fixtures model a published supplier while testing native release
    # identities. Separate consumer tests exercise the real development-pin gate.
    def published(root: Path, *, packages: tuple[str, ...] | None = None) -> tuple[dependency_provenance.DependencyIdentity, ...]:
        assert root == REPO_ROOT
        assert packages == ("research-repo-tools",)
        return (dependency_provenance.DependencyIdentity("research-repo-tools", TOOLING_VERSION, "https://pypi.org/simple"),)

    monkeypatch.setattr(dependency_provenance, "check_release_dependencies", published)
    install_clean_checkout(monkeypatch, REPO_ROOT, COMMIT)
    monkeypatch.setattr(release_publishing, "_api", lambda _root, _repo, endpoint, **_kwargs: responses[endpoint])
    monkeypatch.setattr(release_publishing, "_remote_commit", lambda *_args: COMMIT)
    release = release_publishing.GitHubRelease(
        repository="acgetchell/markov-chain-monte-carlo",
        tag=TAG,
        identifier=42,
        draft=False,
        immutable=True,
        prerelease=False,
        assets=((f"markov-chain-monte-carlo-{TAG}-criterion-baseline.tar.gz", 7, 100, "digest"),),
        title=TAG,
        body=changelog.notes(POLICY, TAG)[0],
    )
    monkeypatch.setattr(release_publishing, "lookup_release", lambda *_args: release)
    return {"checks": checks, "release": release}


@pytest.mark.usefixtures("release_event")
def test_installed_gate_accepts_actual_metadata_and_required_native_results() -> None:
    assert main(["--root", str(REPO_ROOT), "release", "gate", TAG]) == 0


@pytest.mark.parametrize("name", CHECKS)
def test_each_configured_native_check_is_required(release_event: dict[str, Any], capsys: pytest.CaptureFixture[str], name: str) -> None:
    check = next(item for item in release_event["checks"] if item["name"] == name)
    check["name"] = "unrelated check"
    assert main(["--root", str(REPO_ROOT), "release", "gate", TAG]) == 1
    diagnostic = capsys.readouterr().err
    assert f"Unmet release checks on exact commit {COMMIT}:" in diagnostic
    assert f"required check {name!r} (app ID 15368): missing" in diagnostic


@pytest.mark.parametrize("immutable", [True, False, None])
def test_installed_gate_requires_immutable_publication(
    release_event: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], *, immutable: bool | None
) -> None:
    release = replace(release_event["release"], immutable=immutable)
    monkeypatch.setattr(release_publishing, "lookup_release", lambda *_args: release)
    assert main(["--root", str(REPO_ROOT), "release", "gate", TAG]) == (0 if immutable is True else 1)
    if immutable is not True:
        assert ("must be immutable" if immutable is False else "lifecycle fields") in capsys.readouterr().err


@pytest.mark.parametrize("mismatch", ["none", "title", "notes", "crlf"])
def test_installed_gate_checks_generated_body_and_title(
    release_event: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], mismatch: str
) -> None:
    release = release_event["release"]
    body = release.body.replace("\n", "\r\n") if mismatch == "crlf" else "wrong" if mismatch == "notes" else release.body
    release = replace(release, title="wrong" if mismatch == "title" else TAG, body=body)
    monkeypatch.setattr(release_publishing, "lookup_release", lambda *_args: release)
    assert main(["--root", str(REPO_ROOT), "release", "gate", TAG]) == (0 if mismatch in {"none", "crlf"} else 1)
    if mismatch in {"title", "notes"}:
        assert ("release title differs" if mismatch == "title" else "release body differs") in capsys.readouterr().err


@pytest.mark.parametrize("drift", ["none", "notes", "tag"])
def test_post_upload_verification_rechecks_the_reviewed_release(
    release_event: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], drift: str
) -> None:
    release = replace(release_event["release"], asset_states=((7, "uploaded"),))
    commit = COMMIT
    target = tmp_path / "owner's approved target.json"
    target.write_bytes(serialize_release_target(ReleaseTarget(REPOSITORY, TAG, release.identifier, commit, TAG)))

    def wait(registry: str, package: str, version: str, *, attempts: int, interval: int) -> RegistryVersion:
        nonlocal release, commit
        assert (registry, package, version, attempts, interval) == ("crates-io", "markov-chain-monte-carlo", VERSION, 7, 10)
        if drift == "notes":
            release = replace(release, body="Edited after upload\n")
        elif drift == "tag":
            commit = "b" * 40
        return RegistryVersion("crates-io", package, version, present=True)

    monkeypatch.setattr(release_publishing, "lookup_release", lambda *_args: release)
    monkeypatch.setattr(release_publishing, "_remote_commit", lambda *_args: commit)
    monkeypatch.setattr(release_publishing, "wait_for_version", wait)
    workflow = yaml.safe_load(WORKFLOW.read_bytes())
    recipe = shlex.split(workflow["jobs"]["publish"]["steps"][-1]["run"])[1]
    command = shlex.split(dry_run(REPO_ROOT, recipe, [TAG, str(target)]).stderr)
    assert main(["--root", str(REPO_ROOT), *command[command.index("release") :]]) == (0 if drift == "none" else 1)
    if drift != "none":
        diagnostic = "release body differs from generated changelog notes" if drift == "notes" else "published tag differs from captured source"
        assert diagnostic in capsys.readouterr().err


def test_release_notes_stdout_excludes_setup_diagnostics() -> None:
    # The real notes recipe must produce only the installed generator's notes.
    script = dry_run(REPO_ROOT, "release-notes", [TAG]).stderr
    runner = (
        "uv() {\n"
        '  if [[ "$*" == "run --locked --no-sync --no-python-downloads research-repo-tools deps check-uv" ]]; then\n'
        "    printf '%s\\n' 'fixture setup diagnostic'\n"
        '  elif [[ "$*" == "sync --locked --group dev" ]]; then\n'
        "    return 0\n"
        "  else\n"
        '    [[ "$1 $2 $3 $4 $5" == "run --locked --group dev research-repo-tools" ]] || return 90\n'
        "    shift 5\n"
        f'    {shlex.quote(sys.executable)} -m research_repo_tools --root {shlex.quote(str(REPO_ROOT))} "$@"\n'
        "  fi\n"
        "}\n"
    )
    result = run_command("bash", ["--noprofile", "--norc", "-euc", runner + script], cwd=REPO_ROOT, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout == changelog.notes(POLICY, TAG)[0]


def test_publish_recipe_uses_the_captured_candidate() -> None:
    dependencies = inspect_justfile(REPO_ROOT).recipes["release-publish"]["dependencies"]
    assert [(item["recipe"], item["arguments"][0]) for item in dependencies] == [("_release-candidate", "publish")]


def test_workflow_requires_approval_revalidation_and_absence_before_oidc() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_bytes())
    assert set(workflow[True]) == {"workflow_dispatch"}
    assert set(workflow[True]["workflow_dispatch"]["inputs"]) == {"tag", "candidate", "acceptance"}
    assert workflow["concurrency"] == {"group": "publish-crates-markov-chain-monte-carlo", "cancel-in-progress": False}
    jobs = workflow["jobs"]
    assert jobs["publish"]["needs"] == "verify"
    assert jobs["publish"]["environment"] == "crates-io"
    assert jobs["verify"]["permissions"] == {"contents": "read", "checks": "read", "actions": "read"}
    assert jobs["publish"]["permissions"] == {"contents": "read", "checks": "read", "actions": "read", "id-token": "write"}
    assert "github.repository == 'acgetchell/markov-chain-monte-carlo'" in jobs["verify"]["if"]
    assert "github.triggering_actor == 'acgetchell'" in jobs["verify"]["if"]
    assert "CARGO_REGISTRY_TOKEN" not in yaml.safe_dump(jobs["verify"])
    for job in jobs.values():
        assert job["steps"][0]["with"] == {"ref": "${{ github.sha }}", "fetch-depth": 0, "persist-credentials": False}
        assert job["steps"][1]["with"]["cache"] == "false"
        gate = next(step for step in job["steps"] if "just release-receive" in step.get("run", ""))
        assert gate["env"]["CANDIDATE_TRANSFER"] == "${{ inputs.candidate }}"
        assert gate["env"]["CANDIDATE_ACCEPTANCE"] == "${{ inputs.acceptance }}"
        assert 'just release-receive "$RUNNER_TEMP/accepted-candidate" --published' in gate["run"]
        assert job["steps"][1]["with"]["tool"] == "cargo"
    steps = jobs["publish"]["steps"]
    assert jobs["verify"]["steps"][3]["id"] == "approval"
    assert jobs["verify"]["steps"][3]["with"]["path"].endswith("/acceptance.json")
    assert steps[2]["env"]["ARTIFACT_DIGEST"] == "${{ needs.verify.outputs.artifact-digest }}"
    assert steps[3]["with"]["artifact-ids"] == "${{ needs.verify.outputs.artifact-id }}"
    assert '--expected "$RUNNER_TEMP/approved-acceptance/acceptance.json"' in steps[4]["run"]
    assert 'release registry "$RELEASE_TAG" --expect absent' in steps[4]["run"]
    assert steps[5]["id"] == "auth"
    assert steps[6]["run"].endswith("toolchain run --tool cargo -- cargo publish --locked --package markov-chain-monte-carlo --registry crates-io --no-verify")
    assert [index for index, step in enumerate(steps) if "CARGO_REGISTRY_TOKEN" in step.get("env", {})] == [6]
    assert steps[7]["if"] == "${{ !cancelled() && steps.auth.outcome == 'success' }}"
    assert steps[7]["run"] == 'just release-verify "$RELEASE_TAG" "$RUNNER_TEMP/accepted-candidate/target.json"'
    assert "continue-on-error" not in WORKFLOW.read_text(encoding="utf-8")
    commands = dry_run(REPO_ROOT, "publish-check").stderr
    assert commands.count("--package markov-chain-monte-carlo --registry crates-io --dry-run") == 2
    assert "--all-features" in commands
    assert "--allow-dirty" not in commands
    assert "--no-verify" not in commands
    preparation = dry_run(REPO_ROOT, "release-dependency-builds").stderr
    managed = [shlex.split(line) for line in preparation.splitlines() if "toolchain run" in line]
    assert len(managed) == 3
    assert all(command[command.index("toolchain") :][:5] == ["toolchain", "run", "--tool", "cargo", "--"] for command in managed)


@pytest.mark.parametrize(
    ("failure", "expected_exit"), [("none", 0), ("dependency-builds", 1), ("evidence", 1), ("candidate", 1), ("present", 1), ("unavailable", 2)]
)
def test_actual_preupload_shell_requires_evidence_and_proven_absence(tmp_path: Path, failure: str, expected_exit: int) -> None:
    workflow = yaml.safe_load(WORKFLOW.read_bytes())
    script = next(
        step["run"] for step in workflow["jobs"]["publish"]["steps"] if step.get("name") == "Recheck evidence and registry absence after deployment approval"
    )
    marker = tmp_path / "upload-reached"
    gate = "just() { return 1; }" if failure == "evidence" else "just() { return 0; }"
    if failure == "dependency-builds":
        gate = 'just() { [[ "$1" != release-dependency-builds ]]; }'
    if failure == "candidate":
        gate = 'just() { [[ "$1" != release-receive ]]; }'
    registry_exit = expected_exit if failure in {"present", "unavailable"} else 0
    registry = f"uv() {{ return {registry_exit}; }}"
    script = f"{gate}\n{registry}\n{script}\ntouch {shlex.quote(marker.as_posix())}"
    result = run_command(
        "bash",
        ["--noprofile", "--norc", "-euc", script],
        env={
            **{key: value for key, value in os.environ.items() if key not in {"BASH_ENV", "SHELLOPTS", "BASHOPTS"}},
            "RELEASE_TAG": TAG,
            "CANDIDATE_TRANSFER": '{"schema": 1}',
            "RUNNER_TEMP": tmp_path.as_posix(),
        },
        check=False,
    )
    assert result.returncode == expected_exit, result.stderr
    assert marker.exists() == (failure == "none")


@pytest.mark.parametrize("failure", ["none", "attempt", "identifier", "digest", "expired", "different-digest", "different-run"])
def test_actual_approval_artifact_guard_rejects_drift(tmp_path: Path, failure: str) -> None:
    workflow = yaml.safe_load(WORKFLOW.read_bytes())
    script = next(step["run"] for step in workflow["jobs"]["publish"]["steps"] if step.get("name") == "Require this exact approval receipt")
    metadata = {
        "expired": failure == "expired",
        "digest": "sha256:" + ("b" if failure == "different-digest" else "a") * 64,
        "workflow_run": {"id": 99 if failure == "different-run" else 42},
    }
    supplied = tmp_path / "artifact metadata.json"
    supplied.write_text(json.dumps(metadata), encoding="utf-8", newline="\n")
    marker = tmp_path / "approval-reached"
    provider = f'gh() {{ [[ "$*" == "api repos/acgetchell/markov-chain-monte-carlo/actions/artifacts/7" ]] && cat {shlex.quote(supplied.as_posix())}; }}'
    result = run_command(
        "bash",
        ["--noprofile", "--norc", "-euc", f"{provider}\n{script}\ntouch {shlex.quote(marker.as_posix())}"],
        env={
            **{key: value for key, value in os.environ.items() if key not in {"BASH_ENV", "SHELLOPTS", "BASHOPTS"}},
            "VALIDATED_ATTEMPT": "1" if failure == "attempt" else "2",
            "GITHUB_RUN_ATTEMPT": "2",
            "ARTIFACT_ID": "07" if failure == "identifier" else "7",
            "ARTIFACT_DIGEST": "invalid" if failure == "digest" else "a" * 64,
            "GITHUB_RUN_ID": "42",
            "GITHUB_REPOSITORY": "acgetchell/markov-chain-monte-carlo",
        },
        check=False,
    )
    assert result.returncode == (0 if failure == "none" else 1), result.stderr
    assert marker.exists() == (failure == "none")


def test_shared_release_interface_matches_released_reference() -> None:
    recipes = inspect_justfile(REPO_ROOT).recipes
    assert {"update-version", "release-version-check", "tag", "release-package", "release-publish", "release-verify", "release-visibility"} <= recipes.keys()
    assert not {"release-update", "release-check", "tag-release", "publish"} & recipes.keys()
    update = dry_run(REPO_ROOT, "update-version", ["v9.0.0", "--previous-release", "v8.0.0", "--date", "2099-01-01", "--offline", "--dry-run"]).stderr
    assert 'research-repo-tools release update "$@"' in update
    assert "cargo publish" not in update
    verification = shlex.split(dry_run(REPO_ROOT, "release-verify", ["v9.0.0"]).stderr)
    assert verification[verification.index("research-repo-tools") :] == [
        "research-repo-tools",
        "release",
        "verify",
        "v9.0.0",
        "--attempts",
        "7",
        "--interval",
        "10",
    ]
    for gate in ("ci", "check-repository-tooling"):
        assert "release-version-check" in {item["recipe"] for item in recipes[gate]["dependencies"]}
    guide = (REPO_ROOT / "docs/RELEASING.md").read_text(encoding="utf-8")
    headings = [
        "## Prepare the environment",
        "## Step 1: Prepare the release PR",
        "### 2. Update release metadata",
        "### 3. Generate the release changelog",
        "### 4. Generate the release performance comparison",
        "### 5. Publish the README performance snapshot",
        "## Step 2: Publish after the PR is merged",
        "### 5. Wait for the draft GitHub release",
        "### 6. Attach accepted benchmarks and review the draft",
        "### 7. Publish the reviewed draft and approve Cargo publication",
        "### 8. Verify registry availability and downstream resolution",
    ]
    positions = [guide.index(heading) for heading in headings]
    assert positions == sorted(positions)


def test_visibility_uses_isolated_supplier_and_explicit_release_identities() -> None:
    settings = POLICY.publishing
    assert settings is not None
    target = "owner's approved target.json"
    command = shlex.split(dry_run(REPO_ROOT, "release-visibility", [TAG, target]).stderr)
    assert "--no-project" in command
    assert "--isolated" in command
    assert command[command.index("--with") + 1] == f"research-repo-tools=={TOOLING_VERSION}"
    assert command[command.index("visibility") + 1 :] == [
        "--repository",
        settings.repository,
        "--registry",
        settings.registry,
        "--package",
        settings.package,
        "--required-asset",
        settings.required_assets[0].format(version=VERSION),
        "--attempts",
        "7",
        "--interval",
        "10",
        "--target",
        target,
        "--",
        TAG,
    ]


@pytest.mark.parametrize(("recipe", "flags"), [("tag", ["--force"]), ("tag-preview", ["--dry-run"])])
@pytest.mark.parametrize("review_ok", [True, False])
def test_tag_binds_the_same_commit_to_review_and_tag(recipe: str, flags: list[str], *, review_ok: bool) -> None:
    args = [TAG, "owner path", "alternate"]
    args += ["--commit", COMMIT, "--force"] if recipe == "tag" else []
    script = dry_run(REPO_ROOT, recipe, args).stderr
    head = "b" * 40 if recipe == "tag" else COMMIT
    stubs = f"""\
git() {{ printf '%s\\n' {head}; }}
just() {{ printf '%s\\n' "$@"; return {0 if review_ok else 1}; }}
uv() {{ printf '%s\\n' "$@"; }}
"""
    result = run_command("bash", ["--noprofile", "--norc", "-euc", stubs + script], check=False)
    calls = result.stdout.splitlines()
    assert calls[:8] == ["_release-candidate", "reviewed", f"--tag={TAG}", "owner path", "--selection", "alternate", "--commit", COMMIT]
    assert result.returncode == (0 if review_ok else 1)
    if review_ok:
        command = calls.index("toolchain")
        assert calls[command : command + 6] == ["toolchain", "run", "--tool", "cargo", "--", "research-repo-tools"]
        assert calls[calls.index("changelog") :] == ["changelog", "tag", TAG, "--commit", COMMIT, *flags]
    else:
        assert len(calls) == 8


@pytest.mark.parametrize("failure", ["pending", "failed", "provider", "commit", "ambiguous"])
def test_actual_required_scientific_checks_fail_closed(failure: str) -> None:
    settings = POLICY.publishing
    assert settings is not None
    runs: list[dict[str, Any]] = [
        {"id": index, "name": check.name, "head_sha": COMMIT, "status": "completed", "conclusion": "success", "app": {"id": check.app_id}}
        for index, check in enumerate(settings.required_checks, 1)
    ]
    if failure == "pending":
        runs[-1].update(status="in_progress", conclusion=None)
    elif failure == "failed":
        runs[-1]["conclusion"] = "failure"
    elif failure == "provider":
        runs[-1]["app"] = {"id": 1}
    elif failure == "commit":
        runs[-1]["head_sha"] = "b" * 40
    else:
        runs.append({**runs[-1], "id": len(runs) + 1})
    message = {
        "pending": "pending/running",
        "failed": "unsuccessful",
        "provider": "same name observed from app IDs",
        "commit": "different release commit",
        "ambiguous": "ambiguous",
    }[failure]
    with pytest.raises(ValueError, match=message):
        require_checks([{"total_count": len(runs), "check_runs": runs}], COMMIT, settings.required_checks)


@pytest.mark.parametrize("change", [{"draft": True}, {"prerelease": True}, {"id": 0}, {"tag_name": "v0.5.1-rc.1"}])
def test_consumer_gate_rejects_invalid_event_before_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: dict[str, object]) -> None:
    release = {"id": 1, "draft": False, "prerelease": False, "tag_name": "v0.5.1", **change}
    event = tmp_path / "release.json"
    event.write_text(json.dumps({"action": "published", "repository": {"full_name": REPOSITORY}, "release": release}), encoding="utf-8", newline="\n")
    for key, value in {
        "GITHUB_EVENT_NAME": "release",
        "GITHUB_REPOSITORY": REPOSITORY,
        "GITHUB_SHA": COMMIT,
        "GITHUB_REF": "refs/tags/v0.5.1",
        "GITHUB_EVENT_PATH": str(event),
    }.items():
        monkeypatch.setenv(key, value)
    assert main(["--root", str(REPO_ROOT), "release", "gate", "v0.5.1"]) == 1


def test_settings_payloads_restrict_deployments_and_allow_only_pinned_authentication() -> None:
    settings = REPO_ROOT / ".github/settings"
    environment = json.loads((settings / "crates-io-environment.json").read_bytes())
    assert environment["reviewers"] == [{"type": "User", "id": 434849}]
    assert environment["can_admins_bypass"] is False
    assert environment["deployment_branch_policy"] == {"protected_branches": False, "custom_branch_policies": True}
    assert json.loads((settings / "crates-io-tags.json").read_bytes()) == {"name": "v*", "type": "tag"}
    actions = json.loads((settings / "actions-selected.json").read_bytes())
    auth = [entry for entry in actions["patterns_allowed"] if entry.startswith("rust-lang/")]
    workflow = yaml.safe_load(WORKFLOW.read_bytes())
    step = next(step for step in workflow["jobs"]["publish"]["steps"] if step.get("id") == "auth")
    # Permit main and the independently reviewed #219 upgrade during adoption.
    approved = {
        "rust-lang/crates-io-auth-action@c6f97d42243bad5fab37ca0427f495c86d5b1a18",
        "rust-lang/crates-io-auth-action@3268eb413133e2960fa4ae0e963ca76380c54cd9",
    }
    assert len(auth) == len(approved)
    assert set(auth) == approved
    assert step["uses"] in approved
