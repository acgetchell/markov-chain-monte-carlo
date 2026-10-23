"""MCMC's shared release interface, scientific gates, and credential boundaries."""

import json
import shlex
import shutil
import subprocess
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING, Any

import pytest
import yaml
from research_repo_tools.cli import main
from research_repo_tools.config import load
from research_repo_tools.just_inspect import dry_run, inspect_justfile
from research_repo_tools.release_publishing import require_checks

if TYPE_CHECKING:
    from collections.abc import Sequence

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = "acgetchell/markov-chain-monte-carlo"
COMMIT = "a" * 40


def test_shared_release_interface_matches_delaunay() -> None:
    recipes = inspect_justfile(ROOT).recipes
    assert {"update-version", "release-version-check", "tag", "release-package", "release-publish", "release-verify"} <= recipes.keys()
    assert not {"release-update", "release-check", "tag-release", "tag-preview", "publish"} & recipes.keys()
    update = dry_run(ROOT, "update-version", ["v9.0.0", "--previous-release", "v8.0.0", "--date", "2099-01-01", "--offline", "--dry-run"]).stderr
    assert 'research-repo-tools release update "$@"' in update
    assert "cargo publish" not in update
    verification = shlex.split(dry_run(ROOT, "release-verify", ["v9.0.0"]).stderr)
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
    guide = (ROOT / "docs/RELEASING.md").read_text(encoding="utf-8")
    headings = [
        "## Prepare the environment",
        "## Step 1: Prepare the release PR",
        "### 2. Update release metadata",
        "### 3. Generate the release changelog",
        "### 4. Generate release performance measurements",
        "### 5. Publish the README performance snapshot",
        "## Step 2: Publish after the PR is merged",
        "### 5. Create the draft GitHub release",
        "### 6. Run benchmarks and review the draft",
        "### 7. Publish the reviewed draft and approve Cargo publication",
        "### 8. Verify registry availability and downstream resolution",
    ]
    positions = [guide.index(heading) for heading in headings]
    assert positions == sorted(positions)


def test_publication_builds_without_credentials_then_rechecks_after_approval() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/publish-crates.yml").read_bytes())
    assert workflow.get("on", workflow.get(True)) == {"release": {"types": ["published"]}}
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["concurrency"] == {"group": "publish-crates-markov-chain-monte-carlo", "cancel-in-progress": False}
    jobs = workflow["jobs"]
    assert set(jobs) == {"verify", "publish"}
    assert REPOSITORY in jobs["verify"]["if"]
    assert "!github.event.release.draft" in jobs["verify"]["if"]
    assert "!github.event.release.prerelease" in jobs["verify"]["if"]
    assert jobs["verify"]["permissions"] == {"contents": "read", "checks": "read"}
    assert jobs["publish"]["needs"] == "verify"
    assert jobs["publish"]["environment"] == "crates-io"
    assert jobs["publish"]["permissions"] == {"contents": "read", "checks": "read", "id-token": "write"}
    for job in jobs.values():
        steps = job["steps"]
        checkout = next(step for step in steps if step.get("uses", "").startswith("actions/checkout@"))
        assert checkout["with"] == {"ref": "${{ github.sha }}", "fetch-depth": 0, "persist-credentials": False}
        setup = next(step for step in steps if step.get("uses") == "$/.github/actions/setup-toolchain")
        assert setup["with"]["cache"] == "false"
        assert any('just release-gate "$RELEASE_TAG"' in step.get("run", "") for step in steps)
    assert any(step.get("run") == "just release-package" for step in jobs["verify"]["steps"])
    packages = dry_run(ROOT, "release-package").stderr.splitlines()
    cargo = [line for line in packages if "cargo publish" in line]
    assert len(cargo) == 2
    for command in cargo:
        arguments = shlex.split(command)
        assert {"--locked", "--dry-run", "--package", "markov-chain-monte-carlo", "--registry", "crates-io"} <= set(arguments)
        assert not {"--allow-dirty", "--no-verify"} & set(arguments)
    assert sum("--all-features" in command for command in cargo) == 1
    steps = jobs["publish"]["steps"]
    absence = next(index for index, step in enumerate(steps) if "--expect absent" in step.get("run", ""))
    auth = next(index for index, step in enumerate(steps) if step.get("id") == "auth")
    upload = next(index for index, step in enumerate(steps) if "cargo publish" in step.get("run", ""))
    assert absence < auth < upload
    assert "--no-verify" in steps[upload]["run"]
    assert "--allow-dirty" not in steps[upload]["run"]
    token = steps[upload]["env"]["CARGO_REGISTRY_TOKEN"]
    assert token == "${{ steps.auth.outputs.token }}"  # noqa: S105 - literal Actions expression, not a credential.
    assert all("CARGO_REGISTRY_TOKEN" not in step.get("env", {}) for step in steps[:upload])
    verification = steps[upload + 1]
    assert verification["if"] == "${{ !cancelled() && steps.auth.outcome == 'success' }}"
    assert verification["run"] == 'just release-verify "$RELEASE_TAG"'


def test_benchmark_workflow_binds_checkout_and_writer_to_trusted_preflight() -> None:
    jobs = yaml.safe_load((ROOT / ".github/workflows/release-benchmarks.yml").read_bytes())["jobs"]
    preflight, baseline, writer = (jobs[name] for name in ("validate-release", "release-baseline", "publish-baseline"))
    assert preflight["outputs"] == {"commit": "${{ steps.preflight.outputs.commit }}", "target": "${{ steps.preflight.outputs.target }}"}
    capture = next(step for step in preflight["steps"] if step.get("id") == "preflight")
    assert 'gh api "repos/$GH_REPO/commits/$RELEASE_TAG"' in capture["run"]
    assert '--commit "$commit"' in capture["run"]
    assert '--expected-title "$RELEASE_TAG"' in capture["run"]
    assert '--output "$RUNNER_TEMP/release-target.json"' in capture["run"]
    checkout = next(step for step in baseline["steps"] if step.get("uses", "").startswith("actions/checkout@"))
    assert checkout["with"] == {"ref": "${{ needs.validate-release.outputs.commit }}", "fetch-depth": 0, "persist-credentials": False}
    guard = next(step for step in baseline["steps"] if step.get("name") == "Require the captured checkout and unchanged tag")
    assert guard["env"] == {"EXPECTED_COMMIT": "${{ needs.validate-release.outputs.commit }}"}
    assert baseline["steps"].index(guard) < next(index for index, step in enumerate(baseline["steps"]) if "just performance-baseline" in step.get("run", ""))
    assert set(writer["needs"]) == {"validate-release", "release-baseline"}
    upload = writer["steps"][-1]
    assert upload["env"]["RELEASE_TARGET"] == "${{ needs.validate-release.outputs.target }}"
    assert 'test -n "$RELEASE_TARGET"' in upload["run"]
    assert '--target "$RUNNER_TEMP/release-target.json"' in upload["run"]
    assert "--publish" not in upload["run"]
    assert all("actions/checkout@" not in step.get("uses", "") for job in (preflight, writer) for step in job["steps"])


@pytest.mark.parametrize("change", ["missing", "malformed", "repository", "tag", "moved-tag", "replaced-release"])
def test_real_benchmark_handoff_commands_reject_changed_or_missing_target(change: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from research_repo_tools import release_assets

    tag = "v0.5.2"
    release = {
        "id": 73,
        "url": f"https://api.github.com/repos/{REPOSITORY}/releases/73",
        "tag_name": tag,
        "name": tag,
        "draft": True,
        "immutable": False,
        "prerelease": False,
        "assets": [],
    }
    requests: list[list[str]] = []
    changed = False

    def provider(executable: str | Path, arguments: Sequence[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        assert str(executable) == "gh"
        args = list(arguments)
        requests.append(args)
        assert "--method" not in args
        endpoint = args[-1]
        if endpoint == f"repos/{REPOSITORY}/commits/{tag}":
            payload = COMMIT
        elif endpoint == f"repos/{REPOSITORY}/releases?per_page=100":
            payload = [[release]]
        elif endpoint.endswith("/assets?per_page=100"):
            payload = [[]]
        elif endpoint == f"repos/{REPOSITORY}/releases/73":
            payload = {**release, "id": 74} if changed and change == "replaced-release" else release
        else:
            assert endpoint == f"repos/{REPOSITORY}/git/ref/tags/{tag}"
            payload = {"ref": f"refs/tags/{tag}", "object": {"type": "commit", "sha": "b" * 40 if changed and change == "moved-tag" else COMMIT}}
        return subprocess.CompletedProcess([str(executable), *args], 0, json.dumps(payload).encode("utf-8"), b"")

    monkeypatch.setattr(release_assets, "run_command_bytes", provider)
    jobs = yaml.safe_load((ROOT / ".github/workflows/release-benchmarks.yml").read_bytes())["jobs"]
    capture = next(step for step in jobs["validate-release"]["steps"] if step.get("id") == "preflight")
    writer = jobs["publish-baseline"]["steps"][-1]
    values = {"GH_REPO": REPOSITORY, "RELEASE_TAG": tag, "RUNNER_TEMP": str(tmp_path), "commit": COMMIT, "RELEASE_ASSET": "baseline.tar.gz"}
    commands = []
    for step in (capture, writer):
        command = next(line for line in step["run"].replace("\\\n", " ").splitlines() if line.strip().startswith("uvx "))
        arguments = shlex.split(Template(command).substitute(values))
        commands.append(arguments[arguments.index("research-repo-tools") + 1 :])
    assert main(commands[0]) == 0
    target = tmp_path / "release-target.json"
    original = json.loads(target.read_bytes())
    assert (original["repository"], original["tag"], original["release_id"], original["commit"]) == (REPOSITORY, tag, 73, COMMIT)
    if change == "missing":
        target.unlink()
    elif change == "malformed":
        target.write_bytes(b"{}")
    elif change in {"repository", "tag"}:
        original[change] = "other/repository" if change == "repository" else "v0.5.1"
        target.write_text(json.dumps(original), encoding="utf-8", newline="\n")
    changed = True
    requests.clear()
    (tmp_path / "baseline").mkdir()
    (tmp_path / "baseline/baseline.tar.gz").write_bytes(b"inert archive fixture")
    assert main(commands[1]) == 1
    assert all("--method" not in request for request in requests)
    if change in {"missing", "malformed", "repository", "tag"}:
        assert not requests


@pytest.mark.parametrize(("head", "tag_commit"), [(COMMIT, COMMIT), ("b" * 40, COMMIT), (COMMIT, "b" * 40)])
def test_benchmark_checkout_guard_rejects_unexpected_commit(head: str, tag_commit: str) -> None:
    jobs = yaml.safe_load((ROOT / ".github/workflows/release-benchmarks.yml").read_bytes())["jobs"]
    guard = next(step for step in jobs["release-baseline"]["steps"] if step.get("name") == "Require the captured checkout and unchanged tag")
    executable = shutil.which("bash")
    assert executable is not None
    providers = (
        f"set -euo pipefail\nEXPECTED_COMMIT={COMMIT}\nRELEASE_TAG=v0.5.2\n"
        f"git() {{ if [ \"$3\" = HEAD ]; then printf '%s' {head}; else printf '%s' {tag_commit}; fi; }}\n"
    )
    result = subprocess.run([executable], input=providers + guard["run"], check=False, capture_output=True, encoding="utf-8")  # noqa: S603 - actual guard with inert Git observations.
    assert (result.returncode == 0) == (head == tag_commit == COMMIT)


@pytest.mark.parametrize("failure", ["missing", "pending", "failed", "provider", "commit", "ambiguous"])
def test_actual_required_scientific_checks_fail_closed(failure: str) -> None:
    settings = load(root=ROOT).publishing
    assert settings is not None
    assert settings.required_assets == ("markov-chain-monte-carlo-v{version}-criterion-baseline.tar.gz",)
    assert {check.name for check in settings.required_checks} == {
        "build (ubuntu-latest)",
        "build (macos-latest)",
        "build (windows-latest)",
        "Clippy SARIF Analysis",
        "Repository Rule SARIF Analysis",
    }
    runs: list[dict[str, Any]] = [
        {"id": index, "name": check.name, "head_sha": COMMIT, "status": "completed", "conclusion": "success", "app": {"id": check.app_id}}
        for index, check in enumerate(settings.required_checks, 1)
    ]
    require_checks([{"total_count": len(runs), "check_runs": runs}], COMMIT, settings.required_checks)
    if failure == "missing":
        runs.pop()
    elif failure == "pending":
        runs[-1].update(status="in_progress", conclusion=None)
    elif failure == "failed":
        runs[-1]["conclusion"] = "failure"
    elif failure == "provider":
        runs[-1]["app"] = {"id": 1}
    elif failure == "commit":
        runs[-1]["head_sha"] = "b" * 40
    else:
        runs.append({**runs[-1], "id": len(runs) + 1})
    message = "different release commit" if failure == "commit" else "lacks one unambiguous successful result"
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
    assert main(["--root", str(ROOT), "release", "gate", "v0.5.1"]) == 1


@pytest.mark.parametrize(
    ("body", "title", "expected"),
    [("Reviewed notes\n", "v0.5.1", 0), ("Reviewed notes\r\n", "v0.5.1", 0), ("Changed notes", "v0.5.1", 1), ("Reviewed notes\n", "Other title", 1)],
)
def test_notes_check_executes_real_recipe_with_inert_providers(body: str, title: str, expected: int) -> None:
    executable = shutil.which("bash")
    assert executable is not None
    payload = json.dumps({"tagName": "v0.5.1", "name": title, "body": body})
    providers = f"uv() {{ printf '%s\\n' 'Reviewed notes'; }}\ngh() {{ printf '%s' {shlex.quote(payload)}; }}\n"
    script = dry_run(ROOT, "_release-notes-check", ["v0.5.1"]).stderr
    result = subprocess.run(  # noqa: S603 - resolved Bash runs the real recipe with inert process providers.
        [executable], input=providers + script, check=False, capture_output=True, encoding="utf-8"
    )
    assert result.returncode == expected


def test_settings_payloads_restrict_deployments_and_allow_only_pinned_authentication() -> None:
    settings = ROOT / ".github/settings"
    environment = json.loads((settings / "crates-io-environment.json").read_bytes())
    assert environment["reviewers"] == [{"type": "User", "id": 434849}]
    assert environment["can_admins_bypass"] is False
    assert environment["deployment_branch_policy"] == {"protected_branches": False, "custom_branch_policies": True}
    assert json.loads((settings / "crates-io-tags.json").read_bytes()) == {"name": "v*", "type": "tag"}
    actions = json.loads((settings / "actions-selected.json").read_bytes())
    auth = [entry for entry in actions["patterns_allowed"] if entry.startswith("rust-lang/")]
    workflow = yaml.safe_load((ROOT / ".github/workflows/publish-crates.yml").read_bytes())
    step = next(step for step in workflow["jobs"]["publish"]["steps"] if step.get("id") == "auth")
    assert auth == ["rust-lang/crates-io-auth-action@*"]
    assert step["uses"] == "rust-lang/crates-io-auth-action@c6f97d42243bad5fab37ca0427f495c86d5b1a18"
