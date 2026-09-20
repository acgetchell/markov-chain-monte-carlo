"""Installed consumer tooling packaging contracts."""

import configparser
import json
import os
import shutil
import subprocess
import sys
import zipfile
from email.parser import Parser
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CONSOLE_SCRIPTS = {
    "archive-performance": "archive_performance",
    "bench-compare": "bench_compare",
    "publish-performance-readme": "publish_performance_readme",
    "release-check": "release_check",
    "update-release-version": "update_release_version",
}


@pytest.fixture(scope="session")
def tooling_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the published wheel once for outside-checkout consumer tests."""
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required to build the tooling wheel")
    output_dir = tmp_path_factory.mktemp("tooling-wheel")
    source_dir = output_dir / "source"
    source_dir.mkdir()
    for filename in ("LICENSE", "pyproject.toml"):
        shutil.copy2(REPO_ROOT / filename, source_dir / filename)
    shutil.copytree(REPO_ROOT / "scripts", source_dir / "scripts", ignore=shutil.ignore_patterns("__pycache__", "tests"))
    subprocess.run(  # noqa: S603 - uv is resolved and every argument is a test constant or fixture path.
        [
            uv,
            "build",
            "--wheel",
            "--no-build-logs",
            "--out-dir",
            str(output_dir),
            str(source_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = tuple(output_dir.glob("*.whl"))
    assert len(wheels) == 1
    return wheels[0]


class TestBuiltToolingPackage:
    def test_wheel_declares_notebook_extra_and_console_scripts(self, tooling_wheel: Path) -> None:
        with zipfile.ZipFile(tooling_wheel) as archive:
            metadata_name = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
            entry_points_name = next(name for name in archive.namelist() if name.endswith(".dist-info/entry_points.txt"))
            metadata = Parser().parsestr(archive.read(metadata_name).decode("utf-8"))
            parser = configparser.ConfigParser()
            parser.read_string(archive.read(entry_points_name).decode("utf-8"))

        requirements = metadata.get_all("Requires-Dist") or []
        assert metadata.get_all("Provides-Extra") == ["notebook"]
        assert "# Tooling Scripts" in metadata.get_payload()
        for dependency in ("research-repo-tools[notebooks]", "matplotlib", "polars"):
            assert any(requirement.startswith(dependency) and 'extra == "notebook"' in requirement for requirement in requirements)
        assert {name: value.partition(":")[0] for name, value in parser["console_scripts"].items()} == CONSOLE_SCRIPTS

    @pytest.mark.parametrize("module_name", CONSOLE_SCRIPTS.values(), ids=CONSOLE_SCRIPTS)
    def test_wheel_console_script_help_works_outside_checkout(self, tooling_wheel: Path, tmp_path: Path, module_name: str) -> None:
        command = (
            "import importlib, pathlib, sys; "
            "module = importlib.import_module(sys.argv[1]); "
            "assert str(module.__file__).startswith(str(pathlib.Path(sys.argv[2]).resolve())); "
            "raise SystemExit(module.main(['--help']))"
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(tooling_wheel)

        result = subprocess.run(  # noqa: S603 - the current interpreter and fixed module names exercise the built wheel.
            [sys.executable, "-c", command, module_name, str(tooling_wheel)],
            cwd=tmp_path,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout.lower()

    def test_wheel_bench_compare_defaults_to_invocation_repository(self, tooling_wheel: Path, tmp_path: Path) -> None:
        criterion = tmp_path / "target" / "criterion" / "chain" / "step_by_value"
        for sample, point in (("release-a", 100.0), ("new", 80.0)):
            estimate = criterion / sample / "estimates.json"
            estimate.parent.mkdir(parents=True, exist_ok=True)
            statistic = {"point_estimate": point}
            estimate.write_text(json.dumps({"median": statistic, "mean": statistic}), encoding="utf-8")
        command = (
            "import bench_compare, pathlib, sys; "
            "assert str(bench_compare.__file__).startswith(str(pathlib.Path(sys.argv[1]).resolve())); "
            "raise SystemExit(bench_compare.main(['release-a', '--revision', 'fixture']))"
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(tooling_wheel)

        result = subprocess.run(  # noqa: S603 - the current interpreter exercises the built wheel's operational default paths.
            [sys.executable, "-c", command, str(tooling_wheel)],
            cwd=tmp_path,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        assert (tmp_path / "target" / "bench-reports" / "performance.md").is_file()

    def test_wheel_archive_performance_defaults_to_invocation_repository(self, tooling_wheel: Path, tmp_path: Path) -> None:
        command = (
            "import archive_performance, pathlib, sys; "
            "assert str(archive_performance.__file__).startswith(str(pathlib.Path(sys.argv[1]).resolve())); "
            "raise SystemExit(archive_performance.main(['--rerender', 'missing.csv']))"
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(tooling_wheel)

        result = subprocess.run(  # noqa: S603 - the current interpreter exercises the built wheel's operational default paths.
            [sys.executable, "-c", command, str(tooling_wheel)],
            cwd=tmp_path,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 2
        assert str(tmp_path / "missing.csv") in result.stderr
