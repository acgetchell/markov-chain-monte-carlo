"""Exercise published scanner selection and the native consumer fixture gate."""

import json
import os
import sysconfig
from importlib.metadata import version
from pathlib import Path

import pytest
from research_repo_tools.cli import main
from research_repo_tools.process import run_command

ROOT = Path(__file__).resolve().parents[2]


def test_published_setup_selects_the_declared_scanner(capsys: pytest.CaptureFixture[str]) -> None:
    """Shared setup verifies the locked native scanner without a local override."""
    assert main(["--root", str(ROOT), "semgrep", "setup"]) == 0
    expected = Path(sysconfig.get_path("scripts")) / ("semgrep.exe" if os.name == "nt" else "semgrep")
    assert f"Semgrep {version('semgrep')}: {expected}" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("source", "diagnostic"),
    [("print('blocked')", ""), ("print('allowed')", "expected finding not reported"), ("print('blocked')\nprint('blocked')", "unexpected finding")],
)
def test_native_fixture_gate_preserves_success_and_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str, diagnostic: str) -> None:
    """The installed command rejects missing and extra findings through native scans."""
    configuration = tmp_path / "scanner.toml"
    configuration.write_text(
        '[semgrep]\nconfig = "rules.yaml"\nfixtures = "fixture.py"\nnamespace = "mcmc."\ncwd = "."\n',
        encoding="utf-8",
        newline="\n",
    )
    (tmp_path / "rules.yaml").write_text(
        json.dumps(
            {"rules": [{"id": "mcmc.fixture", "languages": ["python"], "message": "fixture violation", "severity": "ERROR", "pattern": "print('blocked')"}]}
        ),
        encoding="utf-8",
        newline="\n",
    )
    (tmp_path / "fixture.py").write_text(f"# ruleid: mcmc.fixture\n{source}\n# ok: mcmc.fixture\nprint('allowed')\n", encoding="utf-8", newline="\n")
    ambient = tmp_path / "ambient"
    ambient.mkdir()
    for name in ("semgrep", "pysemgrep"):
        binary = ambient / name
        binary.write_text("#!/bin/sh\nexit 97\n", encoding="utf-8", newline="\n")
        binary.chmod(0o755)
        if os.name == "nt":
            (ambient / f"{name}.cmd").write_text("@echo off\nexit /b 97\n", encoding="utf-8", newline="\n")
    monkeypatch.setenv("PATH", os.pathsep.join([str(ambient), os.environ["PATH"]]))
    result = run_command(
        "uv",
        ["run", "--locked", "--group", "dev", "research-repo-tools", "--config", str(configuration), "--root", str(tmp_path), "semgrep", "check-fixtures"],
        cwd=ROOT,
        env=dict(os.environ),
        timeout=120,
        check=False,
    )
    assert result.returncode == (1 if diagnostic else 0), result.stderr
    if diagnostic:
        assert "mcmc.fixture" in result.stderr
        assert diagnostic in result.stderr
    else:
        assert not result.stderr
