"""Exercise acquisition commands without timing, fetching or publication."""

import os
import shlex
import stat
from pathlib import Path

import pytest
from research_repo_tools.just_inspect import dry_run
from research_repo_tools.process import run_command

ROOT = Path(__file__).resolve().parents[2]


def test_release_prepares_before_measurement_and_readme_remains_separate() -> None:
    preparation = dry_run(ROOT, "performance-prepare").stderr
    assert "performance prepare tooling/performance-release.toml" in preparation
    assert "git fetch" not in preparation
    measurement = dry_run(ROOT, "performance-release").stderr
    assert measurement.count("performance prepare tooling/performance-release.toml --fetch-tags --allow-git-mutations") == 1
    assert measurement.count("--release-intent") == 1
    assert measurement.count("tooling/release_candidate.py acquire") == 1
    assert measurement.index("performance prepare") < measurement.index("tooling/release_candidate.py acquire")
    assert "performance readme" not in measurement
    assert "--progress 'human' --progress-cadence '300'" in measurement
    captured = dry_run(ROOT, "performance-release", ["json", "60"]).stderr
    assert "--progress 'json' --progress-cadence '60'" in captured
    assert captured.count("tooling/release_candidate.py acquire") == 1
    publication = dry_run(ROOT, "performance-readme").stderr
    assert "performance readme tooling/performance-release.toml" in publication
    assert "performance measure" not in publication
    assert "performance prepare" not in publication
    assert "performance release" not in publication


@pytest.mark.parametrize(
    ("mode", "failed_phase"),
    [("fresh", "none"), ("fresh", "prepare"), ("fresh", "release"), ("resume", "none"), ("resume", "release")],
)
def test_release_recipe_stops_on_failure_without_publishing_readme(tmp_path: Path, failed_phase: str, mode: str) -> None:
    """Execute the real Bash recipe with a bounded CLI substitute, never fetching or timing."""
    root = tmp_path / "release with spaces"
    root.mkdir()
    directory = tmp_path / "fixture bin"
    directory.mkdir()
    launcher = directory / "uv"
    launcher.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf "%s\\n" "$*" >> "$PERFORMANCE_COMMAND_LOG"\n'
        'if [[ " $* " == *" performance $PERFORMANCE_FAIL_PHASE "* || ( "$PERFORMANCE_FAIL_PHASE" == release && " $* " == *" acquire "* ) ]]; then\n'
        '    printf "fixture %s failure\\n" "$PERFORMANCE_FAIL_PHASE" >&2\n'
        "    exit 7\n"
        "fi\n",
        encoding="utf-8",
        newline="\n",
    )
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR)
    log = root / "commands.log"
    result = run_command(
        "just",
        ["--no-deps", "--justfile", str(ROOT / "justfile"), "--working-directory", str(root), "performance-release", "human", "300", "target/checkpoint", mode],
        cwd=root,
        env=os.environ
        | {
            "PATH": f"{directory}{os.pathsep}{os.environ['PATH']}",
            "PERFORMANCE_COMMAND_LOG": str(log),
            "PERFORMANCE_FAIL_PHASE": failed_phase,
        },
        check=False,
    )
    commands = [shlex.split(line) for line in log.read_text(encoding="utf-8").splitlines()]
    # Preparation and acquisition must observe the same managed Cargo cache.
    for command in commands:
        assert command[:8] == ["run", "--locked", "--group", "dev", "research-repo-tools", "toolchain", "run", "--"]
    phases = [
        command[command.index("performance") + 1 :] if "performance" in command else command[command.index("tooling/release_candidate.py") + 1 :]
        for command in commands
    ]
    expected = [["prepare", "tooling/performance-release.toml", "--fetch-tags", "--allow-git-mutations", "--release-intent"]] if mode == "fresh" else []
    if failed_phase != "prepare":
        expected.append(
            [
                "acquire",
                "--progress",
                "human",
                "--progress-cadence",
                "300",
                "--checkpoint",
                "target/checkpoint",
            ]
            + (["--resume"] if mode == "resume" else [])
        )
    assert phases == expected
    status = 0 if failed_phase == "none" else 7
    assert result.returncode == status, result.stderr
    if status:
        assert f"fixture {failed_phase} failure" in result.stderr
    assert f"exit {status}" in (root / "target/bench-reports/release-signal/runtime.txt").read_text(encoding="utf-8")
