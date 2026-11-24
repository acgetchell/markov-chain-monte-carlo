"""Real MCMC candidate capture, publication, final binding and archive reuse.

The fixture supplies synthetic Criterion bytes and inert provider responses;
it never times Rust or represents the data as scientific measurements.
"""

import json
import runpy
import shutil
import subprocess
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import tomli_w
from research_repo_tools import common_measurement, criterion_inventory, dependency_provenance, measurement, performance_release, release_validation
from research_repo_tools.benchmark_series import SeriesBaseline, SeriesPair
from research_repo_tools.candidate_reuse import parse_candidate_policy, verify_candidate_archive
from research_repo_tools.cli import main as shared_main
from research_repo_tools.common_measurement import DiscoveryPolicy
from research_repo_tools.evidence import parse_evidence, serialize_evidence
from research_repo_tools.host_metadata import HostMetadata
from research_repo_tools.measurement import load_measurement, resolve_revision
from research_repo_tools.performance_progress import ProgressReporter
from research_repo_tools.process import run_git_bytes
from research_repo_tools.publication import publish_publication
from research_repo_tools.release_assets import ReleaseTarget
from research_repo_tools.releases import PublishedRelease
from research_repo_tools.scheduled_runs import parse_scheduled_run
from research_repo_tools.suite_measurement import SuiteMeasurementPlan
from research_repo_tools.suite_runs import parse_suite_run
from tooling import release_candidate as candidate

from tests.tooling.support_evidence import EvidenceProvider, install_transfer, workflow_python

if TYPE_CHECKING:
    from collections.abc import Sequence

ROOT = Path(__file__).resolve().parents[2]


def git(root: Path, *args: str) -> None:
    """Mutate only the disposable fixture repository, never the user's checkout."""
    run_git_bytes(["--no-pager", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "-c", "commit.gpgsign=false", *args], cwd=root)


def commit(root: Path) -> str:
    """Stage the disposable checkout and return the new HEAD, allowing an empty commit."""
    git(root, "add", ".")
    git(root, "commit", "--quiet", "--allow-empty", "-m", "fixture")
    return resolve_revision(root, "HEAD")


def configure_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate the fixture's home and exclude the user's global and system Git settings."""
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


def install_synthetic_criterion(root: Path, monkeypatch: pytest.MonkeyPatch, *, interrupt: bool = False) -> list[tuple[str, ...]]:
    """Supply deterministic raw Criterion files for the actual 22 MCMC workloads."""
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
            if interrupt and timed == 2 and not interrupted:
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
    return calls


def acquire_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Return a disposable checkout and its independent owner after a resumed acquisition.

    Synthetic Criterion samples drive the real capture path. Interrupt the second
    trial, resume it, and commit the resulting publication in the checkout.
    """
    configure_git(tmp_path, monkeypatch)
    root = fixture_checkout(tmp_path)
    calls = install_synthetic_criterion(root, monkeypatch, interrupt=True)
    monkeypatch.setattr(performance_release, "published_releases", lambda *_args, **_kwargs: (PublishedRelease("v0.5.2", datetime(2026, 10, 1, tzinfo=UTC)),))
    settings = performance_release.load_performance_release(root, candidate.SETTINGS)
    performance_release.prepare_performance(root, settings)
    owner = tmp_path / "independent owner"
    with pytest.raises(RuntimeError, match="fixture interruption"):
        candidate.acquire(root, owner, "target/checkpoint", resume=False, progress=ProgressReporter(mode="none"))
    assert sum("--save-baseline=retained" in command for command in calls) == 3
    candidate.acquire(root, owner, "target/checkpoint", resume=True, progress=ProgressReporter(mode="none"))
    assert sum("--save-baseline=retained" in command for command in calls) == 5
    evidence = parse_evidence((owner / "run.json").read_bytes(), (owner / "evidence.json").read_bytes())
    assert len(parse_scheduled_run(evidence).trials) == 2
    assert candidate.selected_owner(root) == owner.resolve()

    forbid_acquisition(monkeypatch)
    with pytest.raises(ValueError, match="raw acquisition already retained; use just performance-capture"):
        candidate.acquire(root, owner, "target/checkpoint", resume=True, progress=ProgressReporter(mode="none"))
    publish_publication(performance_release.load_readme_plan(root, settings))
    commit(root)
    return root, owner


def forbid_acquisition(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail on discovery, timing or build subprocesses during verification and recovery."""

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Step 2/retry invoked acquisition, discovery, gate, or a Cargo build")

    monkeypatch.setattr(common_measurement, "run_command_live", forbidden)
    monkeypatch.setattr(criterion_inventory, "run_command", forbidden)
    monkeypatch.setattr(candidate, "measure_and_retain", forbidden)
    monkeypatch.setattr(measurement, "measure_pair", forbidden)


def configure_series(root: Path, *, initial: bool) -> None:
    """Apply the operator's series declaration only inside the disposable checkout."""
    directory = root / "tooling"
    name = "benchmark-series/stepping.toml"
    source = "tooling/measurement-release-signal.toml"
    plan = tomllib.loads((root / source).read_text(encoding="utf-8"))
    cases = plan["suites"][0]["measurement"]["phases"]["current"]["discovery"]["required"]
    plan["benchmark-series"] = {"id": "stepping", "first-release": "v0.5.42", "workloads": {"stepping": cases}}
    publication = tomllib.loads((ROOT / "tooling/performance-publication.toml").read_text(encoding="utf-8"))
    publication["measurement"] = name
    if initial:
        plan["schema"] = 5
        del plan["schedule"]
        for key in ("trial", "comparisons", "svg", "prose-file"):
            del publication[key]
        publication["series"] = [series for series in publication["series"] if series["name"] == "stepping/current"]
        publication["provenance"] = {"current": {}}
        publication["references"] = [ref for ref in publication["references"] if ref["source"] != "previous-tag"]
        publication["references"][1]["count"] = 1
        publication["table"] = {"intervals": "below", "timing-legend": True}
    (root / name).parent.mkdir(exist_ok=True)
    (root / name).write_bytes(tomli_w.dumps(plan).encode())
    (directory / "performance-publication.toml").write_bytes(tomli_w.dumps(publication).encode())
    for filename in ("performance-release.toml", "performance-report.toml"):
        config = tomllib.loads((ROOT / "tooling" / filename).read_text(encoding="utf-8"))
        config["measurement"] = name
        if filename == "performance-release.toml":
            config["order"] = "version"
            if not initial:
                for key in ("preparation", "payload", "manifest"):
                    config[key] = config[key].replace("bench-reports/", "bench-reports/successor/")
        (directory / filename).write_bytes(tomli_w.dumps(config).encode())


def verify_series_attachment(root: Path, owner: Path, final: str, review: pytest.MonkeyPatch) -> None:
    """Revalidate the original evidence and execute both workflow jobs against the inert provider."""
    provider = EvidenceProvider(final)
    for endpoint in ("releases?per_page=100&page=1", "git/ref/tags/v0.5.2"):
        provider.responses[endpoint] = candidate.api(endpoint, repository=candidate.REPOSITORY)
    provider.install(review)
    review.setattr(candidate, "api", provider.api)
    tmp_path = owner.parent
    original = (owner / "run.json").read_bytes(), (owner / "evidence.json").read_bytes()
    candidate.preflight(root, owner, final)
    accepted = candidate.verify(root, owner)
    assert candidate.verify_existing(root, owner, "reviewed") == accepted
    archive = candidate.archive(owner, final)
    policy = parse_candidate_policy((accepted.parent / "policy.json").read_bytes())
    assert serialize_evidence(verify_candidate_archive(archive, policy, accepted.read_bytes())) == original
    target = ReleaseTarget(candidate.REPOSITORY, policy.tag, 42, final, policy.tag)
    files = {"policy": accepted.parent / "policy.json", "publication": owner / "publication.json", "acceptance": accepted, "archive": archive}
    transfer = install_transfer(
        tmp_path,
        review,
        target,
        {role: path.read_bytes() for role, path in files.items()},
        ((owner / "acquisition.sha256").read_text(encoding="ascii").strip(), (owner / "publication.sha256").read_text(encoding="ascii").strip()),
    )
    payload = json.dumps(transfer["data"])
    review.setenv("CANDIDATE_TRANSFER", payload)
    runpy.run_path(str(workflow_python(tmp_path, "fetch")), run_name="__main__")
    attached = candidate.receive(root, tmp_path / "accepted-candidate", payload.encode(), source=tmp_path / "incoming-candidate")
    assert attached.read_bytes() == archive.read_bytes()
    runpy.run_path(str(workflow_python(tmp_path, "attach")), run_name="__main__")
    assert transfer["calls"].count("upload") == 1


def test_approved_series_initialization_then_counterbalanced_successor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    configure_git(tmp_path, monkeypatch)
    root = fixture_checkout(tmp_path)
    calls = install_synthetic_criterion(root, monkeypatch)
    published = {"v0.5.2": resolve_revision(root, "v0.5.2")}

    def history(endpoint: str, *, repository: str) -> object:
        assert repository == candidate.REPOSITORY
        if endpoint == "releases?per_page=100&page=1":
            return [
                {"id": i, "tag_name": tag, "draft": False, "prerelease": False, "published_at": "2026-10-01T00:00:00Z"} for i, tag in enumerate(published, 1)
            ]
        tag = endpoint.removeprefix("git/ref/tags/")
        return {"ref": f"refs/tags/{tag}", "object": {"type": "commit", "sha": published[tag]}}

    monkeypatch.setattr(release_validation, "api", history)
    monkeypatch.setattr(candidate, "api", history)
    configure_series(root, initial=True)
    settings = performance_release.load_performance_release(root, candidate.SETTINGS)
    prepared = performance_release.prepare_performance(root, settings)
    assert isinstance(prepared.pair, SeriesBaseline)
    owner = tmp_path / "initial owner"
    candidate.acquire(root, owner, "target/initial-checkpoint", resume=False, progress=ProgressReporter(mode="none"))
    original = (owner / "run.json").read_bytes(), (owner / "evidence.json").read_bytes()
    run = parse_suite_run(parse_evidence(*original))
    assert run.series_baseline
    assert run.current_only
    assert sum("--save-baseline=retained" in command for command in calls) == 1
    with monkeypatch.context() as review:
        forbid_acquisition(review)
        publish_publication(performance_release.load_readme_plan(root, settings))
        readme = (root / "README.md").read_bytes()
        assert b"| By-value step | 5<br>[4, 6] |" in readme
        assert b"Confidence level: 95%" in readme
        assert b"Baseline / current" not in readme
        final = commit(root)
        verify_series_attachment(root, owner, final, review)

    git(root, "-c", "tag.gpgsign=false", "tag", "v0.5.42")
    published["v0.5.42"] = final
    package = root / "Cargo.toml"
    package.write_bytes(package.read_bytes().replace(b"0.5.42", b"0.5.43"))
    configure_series(root, initial=False)
    successor = performance_release.load_performance_release(root, candidate.SETTINGS)
    pair = performance_release.prepare_performance(root, successor).pair
    assert isinstance(pair, SeriesPair)
    assert pair.history.predecessor == ("v0.5.42", final)
    next_owner = tmp_path / "successor owner"
    candidate.acquire(root, next_owner, "target/successor-checkpoint", resume=False, progress=ProgressReporter(mode="none"))
    paired = parse_scheduled_run(parse_evidence((next_owner / "run.json").read_bytes(), (next_owner / "evidence.json").read_bytes()))
    assert len(paired.trials) == 2
    publish_publication(performance_release.load_readme_plan(root, successor))
    assert b"Baseline / current" in (root / "README.md").read_bytes()
    assert (root / "docs/assets/release-performance.svg").is_file()
    assert sum("--save-baseline=retained" in command for command in calls) == 5
    assert ((owner / "run.json").read_bytes(), (owner / "evidence.json").read_bytes()) == original


@pytest.fixture(scope="module")
def retained_acquisition(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """One real capture supplies read-only templates for independent verification cases."""
    with pytest.MonkeyPatch.context() as patch:
        return acquire_fixture(tmp_path_factory.mktemp("candidate template"), patch)


@pytest.fixture
def acquired(retained_acquisition: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Copy the checkout and owner for one test, omitting target and substituting provider responses.

    Discovery and timing are disabled so recovery must use retained evidence.
    """
    source, record = retained_acquisition
    configure_git(tmp_path, monkeypatch)
    root = shutil.copytree(source, tmp_path / source.name, ignore=shutil.ignore_patterns("target"))
    owner = shutil.copytree(record, tmp_path / record.name)
    monkeypatch.setattr(dependency_provenance, "check_release_dependencies", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(candidate, "check_release_dependencies", lambda *_args, **_kwargs: ())
    forbid_acquisition(monkeypatch)
    EvidenceProvider(resolve_revision(root, "HEAD")).install(monkeypatch)
    return root, owner


def test_recommended_capture_recovers_missing_scratch_from_original_owner(acquired: tuple[Path, Path]) -> None:
    """Recover at the measured revision with all timing and discovery disabled by the fixture."""
    root, owner = acquired
    original = {path.name: path.read_bytes() for path in owner.iterdir() if path.is_file()}
    preparation = performance_release.parse_preparation(original["preparation.json"])
    git(root, "checkout", "--quiet", "--detach", preparation.current_revision)
    settings = performance_release.load_performance_release(root, candidate.SETTINGS)
    assert not (root / settings.preparation).exists()
    command = candidate.retained_capture_command(root, owner, settings.payload, settings.manifest)
    assert command == ("just", "performance-capture", str(owner))
    candidate.run_owner_operation(root, candidate.argument_parser().parse_args(["capture", *command[2:]]))
    assert (root / settings.preparation).read_bytes() == original["preparation.json"]
    assert (root / settings.payload).read_bytes() == original["run.json"]
    assert (root / settings.manifest).read_bytes() == original["evidence.json"]
    assert candidate.selected_owner(root) == owner.resolve()
    assert {path.name: path.read_bytes() for path in owner.iterdir() if path.is_file()} == original


def test_capture_rejects_stale_owner_before_restoring_scratch(acquired: tuple[Path, Path]) -> None:
    """The final publication commit differs from the original measurement revision."""
    root, owner = acquired
    settings = performance_release.load_performance_release(root, candidate.SETTINGS)
    with pytest.raises(ValueError, match="stale preparation"):
        candidate.capture(root, owner)
    assert not any((root / path).exists() for path in (settings.preparation, settings.payload, settings.manifest))


def test_capture_preserves_conflicting_scratch_preparation(acquired: tuple[Path, Path]) -> None:
    root, owner = acquired
    original = (owner / "preparation.json").read_bytes()
    git(root, "checkout", "--quiet", "--detach", performance_release.parse_preparation(original).current_revision)
    settings = performance_release.load_performance_release(root, candidate.SETTINGS)
    scratch = root / settings.preparation
    scratch.parent.mkdir(parents=True)
    conflicting = original + b"\r\n"
    scratch.write_bytes(conflicting)
    with pytest.raises(ValueError, match="independent preparation changed"):
        candidate.capture(root, owner)
    assert scratch.read_bytes() == conflicting
    assert (owner / "preparation.json").read_bytes() == original
    assert not any((root / path).exists() for path in (settings.payload, settings.manifest))


def test_real_candidate_reuses_one_acquisition_for_report_readme_and_archive(acquired: tuple[Path, Path], capsys: pytest.CaptureFixture[str]) -> None:
    root, owner = acquired
    original = {name: (owner / name).read_bytes() for name in ("run.json", "evidence.json", "policy.json", "publication.json")}
    assert shared_main(["--root", str(root), "performance", "promote", "tooling/performance-report.toml", "--check"]) == 0
    assert "Report inventory: current." in capsys.readouterr().out
    final = resolve_revision(root, "HEAD")
    candidate.remember_owner(root, owner)
    candidate.run_owner_operation(root, candidate.argument_parser().parse_args(["preflight", final]))
    preflight = json.loads(capsys.readouterr().out)
    assert preflight["schema"] == "research-repo-tools/candidate-preflight/v1"
    assert preflight["commit"] == final
    assert preflight["identity"] == parse_candidate_policy(original["policy.json"]).identity
    assert "docs/performance/current.md" in preflight["changed_paths"]
    assert not (owner / "final").exists()
    accepted = candidate.verify(root, owner)
    original_acceptance = accepted.read_bytes()
    assert candidate.verify_existing(root, owner, "reviewed") == accepted
    assert accepted.read_bytes() == original_acceptance
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
    "changed",
    ["src/lib.rs", "benches/stepping.rs", "Cargo.lock", "tooling/measurement-release-signal.toml", "docs/performance/current.md", "docs/RELEASING.md"],
)
def test_real_candidate_rejects_changed_scientific_inputs_or_generated_report(
    acquired: tuple[Path, Path], changed: str, capsys: pytest.CaptureFixture[str]
) -> None:
    root, owner = acquired
    path = root / changed
    with path.open("ab") as stream:
        stream.write(b"\n# changed input\n")
    final = commit(root)
    expected = "committed generated output differs" if changed == "docs/performance/current.md" else "performance-relevant or undeclared final-tree differences"
    with pytest.raises(RuntimeError, match="candidate preflight failed"):
        candidate.preflight(root, owner, final)
    diagnostic = capsys.readouterr().err
    assert expected in diagnostic
    assert changed in diagnostic
    assert not (owner / "final").exists()
    if changed == "docs/performance/current.md":
        assert shared_main(["--root", str(root), "performance", "promote", "tooling/performance-report.toml", "--check"]) == 1
    with pytest.raises(RuntimeError, match="candidate verification failed"):
        candidate.verify(root, owner)
    diagnostic = capsys.readouterr().err
    assert expected in diagnostic
    assert changed in diagnostic
    assert not (owner / "final" / final / "reviewed/acceptance.json").exists()


def test_preflight_reproduces_committed_readme_csv_from_original_acquisition(acquired: tuple[Path, Path]) -> None:
    root, owner = acquired
    (root / "docs/assets/release-performance.csv").write_bytes(b"stale CSV\n")
    with pytest.raises(ValueError, match=r"README publication differs.*docs/assets/release-performance\.csv"):
        candidate.preflight(root, owner, commit(root))
    assert not (owner / "final").exists()


def test_active_inventory_rejects_malformed_retained_manifest(acquired: tuple[Path, Path]) -> None:
    root, owner = acquired
    report = root / "docs/performance/current.md"
    original = report.read_bytes()
    (manifest,) = root.glob("docs/performance/runs/runs/*/evidence.json")
    assert manifest.read_bytes() == (owner / "evidence.json").read_bytes()
    manifest.write_bytes(b"{\r\n")
    assert shared_main(["--root", str(root), "performance", "promote", "tooling/performance-report.toml", "--check"]) == 1
    assert report.read_bytes() == original
    assert manifest.read_bytes() == b"{\r\n"
