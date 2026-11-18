"""Independent release evidence and provider fixtures shared by consumer tests."""

import json
import shlex
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from research_repo_tools import release_assets, release_evidence, release_interfaces, release_validation
from research_repo_tools.release_interfaces import load_evidence_policy

if TYPE_CHECKING:
    import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = "acgetchell/markov-chain-monte-carlo"
COMMIT = "a" * 40
DIGEST = "sha256:" + "b" * 64
RUN_ENDPOINT = "actions/runs/101"
JOBS_ENDPOINT = "actions/runs/101/attempts/2/jobs?per_page=100&page=1"
ARTIFACTS_ENDPOINT = "actions/runs/101/artifacts?per_page=100&page=1"
ARTIFACT_ENDPOINT = "actions/artifacts/301"
POLICY = load_evidence_policy(REPO_ROOT / "tooling/release-evidence.toml")
NATIVE_GATES = tuple((name, steps, 201 + index) for index, (name, steps) in enumerate(POLICY.validation.required_steps))
NATIVE_RUNNERS = {name: list(labels) for name, labels in POLICY.validation.runner_labels}
assert POLICY.signing_job is not None
SIGNING_STEPS = POLICY.signing_job[1]


def install_clean_checkout(patch: pytest.MonkeyPatch, checkout: Path, commit: str) -> None:
    """Supply read-only Git responses to the installed checkout verifier."""
    options = ("--no-pager", "--no-optional-locks", "--no-replace-objects", "-c", "core.fsmonitor=false", "-c", "diff.autoRefreshIndex=false")

    def git(executable: str | Path, args: list[str], *, cwd: Path, timeout: int, **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        assert cwd.samefile(checkout)
        assert timeout == 30
        assert Path(executable).is_absolute()
        assert tuple(args[: len(options)]) == options
        command = tuple(args[len(options) :])
        responses = {
            ("rev-parse", "HEAD"): f"{commit}\n".encode("ascii"),
            ("ls-files", "-v", "-z"): b"H Cargo.toml\0",
            ("submodule", "foreach", "--quiet", "--recursive", shlex.join((Path(executable).as_posix(), *options, "ls-files", "-v", "-z"))): b"",
            ("status", "--porcelain", "-z", "--untracked-files=all", "--ignore-submodules=none"): b"",
        }
        assert command in responses, f"unexpected read-only Git command: {command}"
        return subprocess.CompletedProcess(args, 0, stdout=responses[command], stderr=b"")

    patch.setattr(release_assets, "run_command_bytes", git)


class EvidenceProvider:
    """Keep signed inputs independent from live provider responses."""

    def __init__(self, commit: str = COMMIT) -> None:
        """Create independent signed and provider identities for one CI attempt."""
        self.run: dict[str, Any] = {
            "id": 101,
            "run_attempt": 2,
            "workflow_id": 9,
            "path": ".github/workflows/ci.yml",
            "name": "CI",
            "event": "push",
            "head_branch": "main",
            "head_sha": commit,
            "pull_requests": [],
            "repository": {"id": 11, "full_name": REPOSITORY},
            "head_repository": {"id": 11, "full_name": REPOSITORY},
            "status": "completed",
            "conclusion": "success",
        }
        gates = [*NATIVE_GATES, ("release-evidence", SIGNING_STEPS, 206)]
        self.jobs: list[dict[str, Any]] = [
            {
                "id": identifier,
                "name": name,
                "run_id": 101,
                "run_attempt": 2,
                "head_branch": "main",
                "head_sha": commit,
                "status": "completed",
                "conclusion": "success",
                "labels": NATIVE_RUNNERS.get(name, ["ubuntu-latest"]).copy(),
                "steps": [{"number": number, "name": step, "status": "completed", "conclusion": "success"} for number, step in enumerate(steps, 1)],
            }
            for name, steps, identifier in gates
        ]
        self.artifact: dict[str, Any] = {
            "id": 301,
            "name": "release-evidence-101-2",
            "expired": False,
            "digest": DIGEST,
            "size_in_bytes": 512,
            "workflow_run": {"id": 101, "head_sha": commit, "head_branch": "main", "repository_id": 11, "head_repository_id": 11},
        }
        self.document: dict[str, Any] = {
            "schema": "research-repo-tools/release-evidence/v1",
            "policy": {
                "repository": REPOSITORY,
                "workflow": ".github/workflows/ci.yml",
                "workflow_name": "CI",
                "ref": "refs/heads/main",
                "event": "push",
                "required_steps": {name: list(steps) for name, steps, _ in NATIVE_GATES},
                "runner_labels": deepcopy(NATIVE_RUNNERS),
                "artifact_prefix": "release-evidence",
                "guarantee": "signed-source",
                "signing_job": {"release-evidence": list(SIGNING_STEPS)},
                "path_in_vcs": "",
            },
            "run": {
                "id": 101,
                "run_attempt": 2,
                "head_sha": commit,
                "head_branch": "main",
                "workflow_id": 9,
                "path": ".github/workflows/ci.yml",
                "name": "CI",
                "event": "push",
                "repository_id": 11,
                "head_repository_id": 11,
            },
            "jobs": {name: identifier for name, _, identifier in NATIVE_GATES},
            "package": {"name": "markov-chain-monte-carlo", "version": "0.5.42"},
        }
        self.certificate = {"runInvocationURI": f"https://github.com/{REPOSITORY}/actions/runs/101/attempts/2", "buildTrigger": "push"}
        self.responses: dict[str, Any] = {
            "actions/workflows/ci.yml": {"id": 9, "path": ".github/workflows/ci.yml", "name": "CI", "state": "active"},
            f"actions/workflows/9/runs?event=push&branch=main&head_sha={commit}&per_page=100&page=1": {"workflow_runs": [self.run]},
            RUN_ENDPOINT: self.run,
            JOBS_ENDPOINT: {"jobs": self.jobs},
            ARTIFACTS_ENDPOINT: {"artifacts": [self.artifact]},
            ARTIFACT_ENDPOINT: self.artifact,
        }

    def api(self, endpoint: str, *, repository: str) -> Any:
        """Return an isolated response to an authenticated request."""
        assert repository == REPOSITORY
        assert endpoint in self.responses, f"unexpected authenticated request: {endpoint}"
        return deepcopy(self.responses[endpoint])

    def gh(self, *args: str) -> str:
        """Return the selected attestation response without a provider process."""
        assert args[:2] == ("attestation", "verify")
        return json.dumps([{"verificationResult": {"signature": {"certificate": self.certificate}}}])

    def download(self, repository: str, artifact: dict[str, Any], directory: Path, names: tuple[str, ...], **bounds: int) -> None:
        """Write the independently authored evidence and signature fixture."""
        # ZIP authentication and extraction are covered by the shared package.
        # This boundary supplies only the adapter's two extracted input files.
        assert repository == REPOSITORY
        assert artifact["id"] == self.artifact["id"]
        assert names == ("validation.json", "validation.sigstore.json")
        assert bounds == {"max_bytes": 32 * 1024 * 1024, "max_expanded_bytes": 32 * 1024 * 1024}
        directory.mkdir()
        (directory / "validation.json").write_text(json.dumps(self.document), encoding="utf-8", newline="\n")
        (directory / "validation.sigstore.json").write_text('{"fixture":"signed bundle"}', encoding="utf-8", newline="\n")

    def install(self, patch: pytest.MonkeyPatch) -> None:
        """Fake authenticated provider services, leaving policy and Git checks real."""
        patch.setattr(release_interfaces, "api", self.api)
        patch.setattr(release_validation, "gh", self.gh)
        patch.setattr(release_evidence, "download_artifact", self.download)
