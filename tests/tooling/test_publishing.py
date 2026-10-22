"""MCMC's shared release interface, scientific gates, and credential boundaries."""

import json
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml
from research_repo_tools.cli import main
from research_repo_tools.config import load
from research_repo_tools.release_publishing import require_checks

from .test_commands import _recipes, _run_just

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = "acgetchell/markov-chain-monte-carlo"
COMMIT = "a" * 40


def test_shared_release_interface_matches_delaunay() -> None:
    recipes = _recipes()
    assert {"update-version", "release-version-check", "tag", "release-package", "release-publish", "release-verify"} <= recipes.keys()
    assert not {"release-update", "release-check", "tag-release", "tag-preview", "publish"} & recipes.keys()
    update = _run_just("--dry-run", "update-version", "v9.0.0", "--previous-release", "v8.0.0", "--date", "2099-01-01", "--offline", "--dry-run").stderr
    assert 'research-repo-tools release update "$@"' in update
    assert "cargo publish" not in update
    verification = shlex.split(_run_just("--dry-run", "release-verify", "v9.0.0").stderr)
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
    packages = _run_just("--dry-run", "release-package").stderr.splitlines()
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
    script = _run_just("--dry-run", "_release-notes-check", "v0.5.1").stderr
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
    assert auth == [step["uses"]]
    assert "*" not in auth[0]
