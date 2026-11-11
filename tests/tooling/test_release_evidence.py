"""Consumer gates exercised through the installed exact-commit evidence verifier."""

import json
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml
from research_repo_tools import release_evidence
from research_repo_tools.just_inspect import inspect_justfile
from research_repo_tools.release_interfaces import load_evidence_policy

if TYPE_CHECKING:
    from collections.abc import Callable

ROOT = Path(__file__).resolve().parents[2]
COMMIT = "a" * 40
REPOSITORY = "acgetchell/markov-chain-monte-carlo"
POLICY = load_evidence_policy(ROOT / "tooling/release-evidence.toml")


@pytest.fixture
def retained_evidence(monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, Any], Callable[[str], Any]]:
    """Inert GitHub transport; use the real manifest and selection implementations."""
    repository = {"id": 17, "full_name": REPOSITORY}
    run = {
        "id": 29,
        "run_attempt": 1,
        "head_sha": COMMIT,
        "head_branch": "main",
        "workflow_id": 31,
        "path": ".github/workflows/ci.yml",
        "name": "CI",
        "event": "push",
        "repository": repository,
        "head_repository": repository,
        "pull_requests": [],
        "status": "completed",
        "conclusion": "success",
    }
    assert POLICY.signing_job is not None
    jobs = [
        {
            "id": index,
            "name": name,
            "run_id": run["id"],
            "run_attempt": 1,
            "head_sha": COMMIT,
            "head_branch": "main",
            "status": "completed",
            "conclusion": "success",
            "labels": list(dict(POLICY.validation.runner_labels).get(name, ("ubuntu-latest",))),
            "steps": [{"name": step, "status": "completed", "conclusion": "success"} for step in steps],
        }
        for index, (name, steps) in enumerate((*POLICY.validation.required_steps, POLICY.signing_job), 1)
    ]
    artifact = {
        "id": 41,
        "name": "release-evidence-29-1",
        "digest": "sha256:" + "b" * 64,
        "size_in_bytes": 2000,
        "expired": False,
        "workflow_run": {"id": 29, "head_sha": COMMIT, "head_branch": "main", "repository_id": 17, "head_repository_id": 17},
    }
    state: dict[str, Any] = {"run": run, "runs": [run], "jobs": jobs, "artifact": artifact, "signatures": []}

    def read(endpoint: str) -> Any:
        if endpoint == "actions/workflows/ci.yml":
            return {"id": 31, "path": ".github/workflows/ci.yml", "name": "CI", "state": "active"}
        if endpoint.startswith("actions/workflows/31/runs?"):
            return {"total_count": len(state["runs"]), "workflow_runs": state["runs"]}
        if endpoint == "actions/runs/29":
            return state["run"]
        if endpoint.startswith("actions/runs/29/attempts/"):
            return {"total_count": len(state["jobs"]), "jobs": state["jobs"]}
        if endpoint.startswith("actions/runs/29/artifacts"):
            return {"total_count": 1, "artifacts": [state["artifact"]]}
        assert endpoint == "actions/artifacts/41"
        return state["artifact"]

    state["manifest"] = release_evidence.create_evidence_manifest(POLICY, "0.5.2", COMMIT, 29, 1, read)

    def download(repository_name: str, identity: dict[str, Any], directory: Path, names: tuple[str, ...], **limits: int) -> None:
        assert repository_name == REPOSITORY
        assert identity["id"] == 41
        assert names == ("validation.json", "validation.sigstore.json")
        assert limits == {"max_bytes": release_evidence.MAX_BYTES, "max_expanded_bytes": release_evidence.MAX_BYTES}
        directory.mkdir()
        (directory / names[0]).write_bytes(state["manifest"])
        (directory / names[1]).write_bytes(b"inert signature fixture")

    def signature(subject: Path, bundle: Path, *args: object, **kwargs: object) -> None:
        assert subject.read_bytes() == state["manifest"]
        assert bundle.read_bytes() == b"inert signature fixture"
        assert args == (COMMIT, ".github/workflows/ci.yml", "refs/heads/main", 29, 1)
        assert kwargs == {"repository": REPOSITORY}
        state["signatures"].append(args)
        if state.get("bad_signature"):
            message = "untrusted signer"
            raise ValueError(message)

    monkeypatch.setattr(release_evidence, "download_artifact", download)
    monkeypatch.setattr(release_evidence, "verify_attestation", signature)
    return state, read


def test_installed_verifier_accepts_pre_tag_evidence_and_frozen_recheck(retained_evidence: tuple[dict[str, Any], Callable[[str], Any]]) -> None:
    state, read = retained_evidence
    selected = release_evidence.verify_release_evidence(POLICY, "0.5.2", COMMIT, read)
    parsed = json.loads(selected)
    assert parsed["package"] == {"name": "markov-chain-monte-carlo", "version": "0.5.2"}
    assert parsed["run"]["run_attempt"] == 1
    assert parsed["artifact"]["id"] == 41
    assert set(parsed["jobs"]) == {name for name, _ in POLICY.validation.required_steps} | {"release-evidence"}
    assert release_evidence.verify_release_evidence(POLICY, "0.5.2", COMMIT, read, expected=selected) == selected
    assert len(state["signatures"]) == 2


@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "pending",
        "failed",
        "cancelled",
        "ambiguous",
        "expired",
        "wrong-commit",
        "dispatch",
        "partial-rerun",
        "skipped-step",
        "wrong-host",
        "signature",
    ],
)
def test_installed_verifier_rejects_ineligible_main_evidence(retained_evidence: tuple[dict[str, Any], Callable[[str], Any]], failure: str) -> None:
    state, read = retained_evidence
    if failure == "missing":
        state["runs"] = []
    elif failure == "pending":
        state["run"]["status"] = "in_progress"
    elif failure in {"failed", "cancelled"}:
        state["run"]["conclusion"] = "failure" if failure == "failed" else failure
    elif failure == "ambiguous":
        state["runs"].append({**state["run"], "id": 30})
    elif failure == "expired":
        state["artifact"]["expired"] = True
    elif failure == "wrong-commit":
        state["run"]["head_sha"] = "c" * 40
    elif failure == "dispatch":
        state["run"]["event"] = "workflow_dispatch"
    elif failure == "partial-rerun":
        state["run"]["run_attempt"] = 2
    elif failure == "skipped-step":
        state["jobs"][0]["steps"][0]["conclusion"] = "skipped"
    elif failure == "wrong-host":
        state["jobs"][2]["labels"] = ["ubuntu-latest"]
    else:
        state["bad_signature"] = True
    messages = {
        "missing": "missing or failed main-push",
        "pending": "still pending",
        "failed": "missing or failed main-push",
        "cancelled": "missing or failed main-push",
        "ambiguous": "ambiguous successful main-push",
        "expired": "expired or mismatched",
        "wrong-commit": "exact commit's trusted main-push",
        "dispatch": "exact commit's trusted main-push",
        "partial-rerun": "missing or ambiguous evidence artifact",
        "skipped-step": "did not validate required step",
        "wrong-host": "required hosted platform",
        "signature": "untrusted signer",
    }
    with pytest.raises(ValueError, match=messages[failure]):
        release_evidence.verify_release_evidence(POLICY, "0.5.2", COMMIT, read)


@pytest.mark.parametrize("change", ["attempt", "artifact", "version"])
def test_approval_cannot_select_replacement_evidence(retained_evidence: tuple[dict[str, Any], Callable[[str], Any]], change: str) -> None:
    state, read = retained_evidence
    selected = release_evidence.verify_release_evidence(POLICY, "0.5.2", COMMIT, read)
    version = "0.5.2"
    if change == "attempt":
        state["run"]["run_attempt"] = 2
    elif change == "artifact":
        state["artifact"]["digest"] = "sha256:" + "d" * 64
    else:
        version = "0.5.3"
    messages = {"attempt": "selected run or attempt changed", "artifact": "approved artifact identity changed", "version": "approved evidence policy"}
    with pytest.raises(ValueError, match=messages[change]):
        release_evidence.verify_release_evidence(POLICY, version, COMMIT, read, expected=selected)


def test_declared_policy_covers_real_workflow_steps_and_native_hosts() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_bytes())
    jobs = workflow["jobs"]
    assert POLICY.guarantee == "signed-source"
    assert POLICY.signing_job is not None
    for name, steps in (*POLICY.validation.required_steps, POLICY.signing_job):
        job = jobs["build" if name.startswith("build (") else name]
        assert set(steps) <= {step.get("name") for step in job["steps"]}
    assert jobs["build"]["steps"][-1]["run"] == "just ci-native"
    assert jobs["repository-tooling"]["steps"][-1]["run"] == "just ci-repository-tooling"
    assert set(jobs["release-evidence"]["needs"]) == {"build", "repository-tooling", "package"}
    assert jobs["package"]["needs"] == "repository-tooling"
    for name in ("package", "release-evidence"):
        assert "github.event_name == 'push'" in jobs[name]["if"]
        assert "github.ref == 'refs/heads/main'" in jobs[name]["if"]
    signer = jobs["release-evidence"]
    assert signer["permissions"] == {"contents": "read", "actions": "read", "attestations": "write", "id-token": "write"}
    assert not any(step.get("uses", "").startswith("$/") for step in signer["steps"])
    upload = signer["steps"][-1]["with"]
    assert upload["name"] == "release-evidence-${{ github.run_id }}-${{ github.run_attempt }}"
    assert upload["retention-days"] == 30


def test_publication_rechecks_frozen_selection_before_registry_credentials() -> None:
    jobs = yaml.safe_load((ROOT / ".github/workflows/publish-crates.yml").read_bytes())["jobs"]
    assert jobs["verify"]["outputs"]["evidence"] == "${{ steps.evidence.outputs.identity }}"
    freeze = next(step for step in jobs["verify"]["steps"] if step.get("id") == "evidence")
    assert freeze["run"] == 'just _release-evidence verify "$RUNNER_TEMP/release-selection.json"'
    steps = jobs["publish"]["steps"]
    recheck = next(step for step in steps if "APPROVED_EVIDENCE" in step.get("env", {}))
    assert recheck["env"]["APPROVED_EVIDENCE"] == "${{ needs.verify.outputs.evidence }}"
    assert 'test -n "$APPROVED_EVIDENCE"' in recheck["run"]
    assert '"$RUNNER_TEMP/rechecked-selection.json" "$RUNNER_TEMP/approved-selection.json"' in recheck["run"]
    assert recheck["run"].index("just _release-evidence") < recheck["run"].index("--expect absent")
    assert steps.index(recheck) < next(index for index, step in enumerate(steps) if step.get("id") == "auth")


def test_required_native_checks_fail_when_repository_tooling_does_not_succeed() -> None:
    job = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_bytes())["jobs"]["build"]
    assert job["needs"] == "repository-tooling"
    # A skipped required check can pass branch protection, including after cancellation.
    assert job["if"] == "${{ always() }}"
    assert not job.get("continue-on-error", False)
    guard = job["steps"][0]
    assert guard["name"] == "Require repository tooling success"
    assert guard["if"] == "${{ cancelled() || needs.repository-tooling.result != 'success' }}"
    assert guard["shell"] == "bash"
    assert not guard.get("continue-on-error", False)
    executable = shutil.which("bash")
    assert executable is not None
    result = subprocess.run(  # noqa: S603 - actual inert workflow failure guard, with no external providers.
        [executable], input=guard["run"], check=False, capture_output=True, encoding="utf-8"
    )
    assert result.returncode == 1
    assert "Repository tooling did not succeed" in result.stdout
    native = next(step for step in job["steps"] if step.get("run") == "just ci-native")
    assert "if" not in native
    assert not native.get("continue-on-error", False)


def test_remote_recipe_union_preserves_every_local_ci_validator() -> None:
    recipes = inspect_justfile(ROOT).recipes

    def validators(name: str) -> set[str]:
        result: set[str] = set()
        for dependency in recipes[name]["dependencies"]:
            child = dependency["recipe"]
            result.add(child)
            result.update(validators(child))
        return result

    native = validators("ci-native")
    repository = validators("ci-repository-tooling")
    assert native | repository == validators("ci")
    assert {
        "python-check",
        "notebook-lint",
        "test-python",
        "notebook-check",
        "clippy-all-targets",
        "doc",
        "test-rust-ci",
        "test-doc",
        "bench-compile",
        "validate-examples",
    } <= native
    assert not {"python-check", "notebook-lint", "test-python"} & repository
