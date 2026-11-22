"""Real MCMC candidate capture, publication, final binding and archive reuse.

The fixture supplies synthetic Criterion bytes and inert provider responses;
it never times Rust or represents the data as scientific measurements.
"""

import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from research_repo_tools import common_measurement, criterion_inventory, dependency_provenance, measurement, performance_release
from research_repo_tools.candidate_reuse import parse_candidate_policy, verify_candidate_archive
from research_repo_tools.cli import main as shared_main
from research_repo_tools.common_measurement import DiscoveryPolicy
from research_repo_tools.evidence import parse_evidence, serialize_evidence
from research_repo_tools.host_metadata import HostMetadata
from research_repo_tools.measurement import load_measurement, resolve_revision
from research_repo_tools.performance_progress import ProgressReporter
from research_repo_tools.process import run_git_bytes
from research_repo_tools.publication import publish_publication
from research_repo_tools.releases import PublishedRelease
from research_repo_tools.scheduled_runs import parse_scheduled_run
from research_repo_tools.suite_measurement import SuiteMeasurementPlan
from tooling import release_candidate as candidate

from tests.tooling.support_evidence import EvidenceProvider

if TYPE_CHECKING:
    from collections.abc import Sequence

ROOT = Path(__file__).resolve().parents[2]


def git(root: Path, *args: str) -> None:
    """Mutate only the disposable fixture repository, never the user's checkout."""
    run_git_bytes(["--no-pager", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "-c", "commit.gpgsign=false", *args], cwd=root)


def commit(root: Path) -> str:
    git(root, "add", ".")
    git(root, "commit", "--quiet", "--allow-empty", "-m", "fixture")
    return resolve_revision(root, "HEAD")


def configure_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "absent-gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path / "fixture-home"))


def fixture_checkout(tmp_path: Path) -> Path:
    """Create baseline and prospective source in a disposable Git repository."""
    root = tmp_path / "consumer checkout"
    root.mkdir()
    for name in (
        "tooling/performance-release.toml",
        "tooling/measurement-release-signal.toml",
        "tooling/performance-report.toml",
        "tooling/performance-publication.toml",
        "tooling/performance-release-interpretation.md",
        "tooling/release-evidence.toml",
        "benches/stepping.rs",
        "rust-toolchain.toml",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    (root / "src").mkdir()
    (root / "src/lib.rs").write_bytes(b"pub fn fixture() -> u8 { 1 }\n")
    (root / "Cargo.lock").write_bytes(b'version = 4\n[[package]]\nname = "criterion"\nversion = "0.8.2"\n')
    (root / "pyproject.toml").write_bytes(b'[project]\nname = "fixture"\nversion = "0.0.0"\nrequires-python = ">=3.15"\n')
    (root / ".gitignore").write_bytes(b"/target/\n")
    (root / "README.md").write_bytes(b"# Fixture\n\n<!-- PERFORMANCE:BEGIN -->\nPending\n<!-- PERFORMANCE:END -->\n")
    package = root / "Cargo.toml"
    package.write_bytes(b'[package]\nname = "markov-chain-monte-carlo"\nversion = "0.5.2"\n')
    git(root, "init", "--quiet")
    commit(root)
    git(root, "-c", "tag.gpgsign=false", "tag", "v0.5.2")
    package.write_bytes(package.read_bytes().replace(b'"0.5.2"', b'"0.5.42"'))
    commit(root)
    return root


def acquire_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    configure_git(tmp_path, monkeypatch)
    root = fixture_checkout(tmp_path)
    monkeypatch.setattr(performance_release, "published_releases", lambda *_args, **_kwargs: (PublishedRelease("v0.5.2", datetime(2026, 10, 1, tzinfo=UTC)),))
    monkeypatch.setattr(dependency_provenance, "check_release_dependencies", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(candidate, "check_release_dependencies", lambda *_args, **_kwargs: ())
    plan = load_measurement(root, "tooling/measurement-release-signal.toml")
    assert isinstance(plan, SuiteMeasurementPlan)
    policy = plan.suites[0].measurement.current.policy
    assert isinstance(policy, DiscoveryPolicy)
    cases = policy.discovery.required
    assert len(cases) == 22
    calls: list[tuple[str, ...]] = []
    timed = 0
    interrupted = False

    def process(executable: str | Path, args: Sequence[str], *, cwd: Path, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal timed, interrupted
        command = (str(executable), *args)
        assert command[:6] == ("cargo", "bench", "--locked", "--bench", "stepping", "--")
        calls.append(command)
        if "--list" in args:
            return subprocess.CompletedProcess(command, 0, "".join(f"{name}: benchmark\n" for name in cases), "")
        if "--test" not in args:
            assert "--save-baseline=retained" in args
            if timed == 2 and not interrupted:
                interrupted = True
                message = "fixture interruption before the second trial"
                raise RuntimeError(message)
            for index, name in enumerate(cases):
                raw = {
                    "benchmark.json": {"full_id": name},
                    "sample.json": {"iters": [1] * 100, "times": [5] * 100},
                    "estimates.json": {
                        statistic: {"point_estimate": 5, "confidence_interval": {"confidence_level": 0.95, "lower_bound": 4, "upper_bound": 6}}
                        for statistic in ("mean", "median")
                    },
                    "tukey.json": [1, 2, 3, 4],
                }
                for sample in ("new", "retained"):
                    directory = cwd / "target/criterion" / str(index) / sample
                    directory.mkdir(parents=True)
                    for filename, data in raw.items():
                        (directory / filename).write_bytes((json.dumps(data) + "\r\n").encode())
            timed += 1
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(common_measurement, "run_command_live", process)
    monkeypatch.setattr(criterion_inventory, "run_command", process)
    host = HostMetadata(
        "fixture-os",
        "fixture-arch",
        "fixture-cpu",
        4,
        8,
        1024,
        tuple((name, f"fixture {name}") for name, _command in plan.suites[0].measurement.measurement.probes),
    )
    monkeypatch.setattr(measurement, "capture_host", lambda *_args, **_kwargs: host)
    settings = performance_release.load_performance_release(root, candidate.SETTINGS)
    performance_release.prepare_performance(root, settings)
    owner = tmp_path / "independent owner"
    with pytest.raises(RuntimeError, match="fixture interruption"):
        candidate.acquire(root, owner, "target/checkpoint", resume=False, progress=ProgressReporter(mode="none"))
    assert timed == 2
    candidate.acquire(root, owner, "target/checkpoint", resume=True, progress=ProgressReporter(mode="none"))
    assert timed == 4
    assert sum("--save-baseline=retained" in command for command in calls) == 5
    evidence = parse_evidence((owner / "run.json").read_bytes(), (owner / "evidence.json").read_bytes())
    assert len(parse_scheduled_run(evidence).trials) == 2
    assert candidate.selected_owner(root) == owner.resolve()

    forbid_acquisition(monkeypatch)
    publish_publication(performance_release.load_readme_plan(root, settings))
    commit(root)
    return root, owner


def forbid_acquisition(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Step 2/retry invoked acquisition, discovery, gate, or a Cargo build")

    monkeypatch.setattr(common_measurement, "run_command_live", forbidden)
    monkeypatch.setattr(criterion_inventory, "run_command", forbidden)
    monkeypatch.setattr(candidate, "measure_release", forbidden)
    monkeypatch.setattr(measurement, "measure_pair", forbidden)


@pytest.fixture(scope="module")
def retained_acquisition(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """One real capture supplies read-only templates for independent verification cases."""
    with pytest.MonkeyPatch.context() as patch:
        return acquire_fixture(tmp_path_factory.mktemp("candidate template"), patch)


@pytest.fixture
def acquired(retained_acquisition: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    source, record = retained_acquisition
    configure_git(tmp_path, monkeypatch)
    root = shutil.copytree(source, tmp_path / source.name, ignore=shutil.ignore_patterns("target"))
    owner = shutil.copytree(record, tmp_path / record.name)
    monkeypatch.setattr(dependency_provenance, "check_release_dependencies", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(candidate, "check_release_dependencies", lambda *_args, **_kwargs: ())
    forbid_acquisition(monkeypatch)
    EvidenceProvider(resolve_revision(root, "HEAD")).install(monkeypatch)
    return root, owner


def test_real_candidate_reuses_one_acquisition_for_report_readme_and_archive(acquired: tuple[Path, Path]) -> None:
    root, owner = acquired
    original = {name: (owner / name).read_bytes() for name in ("run.json", "evidence.json", "policy.json", "publication.json")}
    accepted = candidate.verify(root, owner)
    original_acceptance = accepted.read_bytes()
    assert candidate.verify_existing(root, owner, "reviewed") == accepted
    assert accepted.read_bytes() == original_acceptance
    final = resolve_revision(root, "HEAD")
    archive = candidate.archive(owner, final)
    original_archive = archive.read_bytes()
    assert candidate.archive(owner, final).read_bytes() == original_archive
    policy = parse_candidate_policy((accepted.parent / "policy.json").read_bytes())
    retained = verify_candidate_archive(archive, policy, accepted.read_bytes())
    assert serialize_evidence(retained) == (original["run.json"], original["evidence.json"])
    assert policy.candidate.revision != final
    assert policy.commit == final
    assert (root / "docs/performance/current.md").is_file()
    assert (root / "docs/assets/release-performance.csv").is_file()
    assert (root / "docs/assets/release-performance.svg").is_file()
    assert {name: (owner / name).read_bytes() for name in original} == original


def test_shared_tagger_leaves_ref_absent_until_dependency_preflight_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_git(tmp_path, monkeypatch)
    root = tmp_path / "tag checkout"
    root.mkdir()
    (root / "Cargo.toml").write_bytes(b'[package]\nname = "fixture"\nversion = "1.2.3"\n')
    (root / "Cargo.lock").write_bytes(b'version = 4\n[[package]]\nname = "fixture"\nversion = "1.2.3"\n')
    (root / "pyproject.toml").write_bytes(b'[project]\nname = "fixture"\nversion = "1.2.3"\n\n[tool.research-repo-tools.release]\ndate-policy = "declared"\n')
    (root / "CHANGELOG.md").write_bytes(b"# Changelog\n\n## [1.2.3] - 2026-11-21\n\n- Fixture release.\n")
    git(root, "init", "--quiet", "--template=")
    git(root, "config", "user.name", "Fixture")
    git(root, "config", "user.email", "fixture@example.invalid")
    git(root, "config", "tag.gpgSign", "false")
    revision = commit(root)
    arguments = ["--root", str(root), "changelog", "tag", "v1.2.3", "--commit", revision]

    def unavailable(checkout: Path, **_kwargs: object) -> None:
        assert checkout.samefile(root)
        message = "fixture dependency unavailable in the managed Cargo cache"
        raise ValueError(message)

    monkeypatch.setattr(dependency_provenance, "check_release_dependencies", unavailable)
    assert shared_main(arguments) == 1
    assert "dependency unavailable in the managed Cargo cache" in capsys.readouterr().err
    assert run_git_bytes(["--no-pager", "tag", "--list"], cwd=root).stdout == b""

    monkeypatch.setattr(dependency_provenance, "check_release_dependencies", lambda *_args, **_kwargs: ())
    assert shared_main(arguments) == 0
    assert f"Created local annotated tag v1.2.3 at {revision}" in capsys.readouterr().out
    assert run_git_bytes(["--no-pager", "cat-file", "-t", "refs/tags/v1.2.3"], cwd=root).stdout.strip() == b"tag"
    assert resolve_revision(root, "v1.2.3") == revision


@pytest.mark.parametrize(
    "changed", ["src/lib.rs", "benches/stepping.rs", "Cargo.lock", "tooling/measurement-release-signal.toml", "docs/performance/current.md"]
)
def test_real_candidate_rejects_changed_scientific_inputs_or_generated_report(
    acquired: tuple[Path, Path], changed: str, capsys: pytest.CaptureFixture[str]
) -> None:
    root, owner = acquired
    path = root / changed
    path.write_bytes(path.read_bytes() + b"\n# changed input\n")
    final = commit(root)
    with pytest.raises(RuntimeError, match="candidate verification failed"):
        candidate.verify(root, owner)
    diagnostic = capsys.readouterr().err
    expected = "committed generated output differs" if changed.startswith("docs/") else "performance-relevant or undeclared final-tree differences"
    assert expected in diagnostic
    assert changed in diagnostic
    assert not (owner / "final" / final / "reviewed/acceptance.json").exists()
