"""Tests for atomic repository tool-pin reconciliation."""

import os
import subprocess
from typing import TYPE_CHECKING, Never

import pytest

import update_cargo_tool_pins
from subprocess_utils import ExecutableNotFoundError

if TYPE_CHECKING:
    from pathlib import Path

EXPECTED_PIN_TO_PACKAGE = {
    "cargo_edit_version": "cargo-edit",
    "cargo_llvm_cov_version": "cargo-llvm-cov",
    "cargo_nextest_version": "cargo-nextest",
    "dprint_version": "dprint",
    "git_cliff_version": "git-cliff",
    "just_version": "just",
    "rumdl_version": "rumdl",
    "taplo_version": "taplo-cli",
    "typos_version": "typos-cli",
    "zizmor_version": "zizmor",
}
EXPECTED_PIN_TO_TOOL = {**EXPECTED_PIN_TO_PACKAGE, "uv_version": "uv"}


def installed_output(*, override: tuple[str, str] | None = None) -> str:
    """Return representative ``cargo install --list`` output for every managed tool."""
    versions = dict.fromkeys(EXPECTED_PIN_TO_PACKAGE.values(), "1.2.3")
    if override is not None:
        versions[override[0]] = override[1]
    return "".join(f"{package} v{version}:\n    {package}\n" for package, version in versions.items())


def justfile_text(version: str = "1.2.3") -> str:
    """Return one assignment for every managed Just pin."""
    return "".join(f'{pin} := "{version}"\n' for pin in EXPECTED_PIN_TO_TOOL)


def test_managed_pin_inventory_matches_independent_contract() -> None:
    assert update_cargo_tool_pins.PIN_TO_PACKAGE == EXPECTED_PIN_TO_PACKAGE
    assert update_cargo_tool_pins.PIN_TO_TOOL == EXPECTED_PIN_TO_TOOL


def test_reconcile_pins_updates_cargo_and_uv_versions_atomically(tmp_path: Path) -> None:
    justfile = tmp_path / "justfile"
    justfile.write_text(justfile_text(), encoding="utf-8")

    changes = update_cargo_tool_pins.reconcile_pins(
        justfile,
        installed_output(override=("rumdl", "2.0.0")),
        "uv 3.0.0",
    )

    assert changes == {
        "rumdl_version": ("1.2.3", "2.0.0"),
        "uv_version": ("1.2.3", "3.0.0"),
    }
    updated = justfile.read_text(encoding="utf-8")
    assert 'rumdl_version := "2.0.0"' in updated
    assert 'uv_version := "3.0.0"' in updated
    assert list(tmp_path.glob(".justfile.*")) == []


def test_reconcile_pins_preserves_symlink_and_crlf_bytes(tmp_path: Path) -> None:
    target = tmp_path / "tool-pins.just"
    original = justfile_text().replace("\n", "\r\n").encode()
    target.write_bytes(original)
    justfile = tmp_path / "justfile"
    try:
        justfile.symlink_to(target.name)
    except OSError:
        if os.name != "nt":
            raise
        pytest.skip("symlink creation is unavailable on this Windows runner")

    changes = update_cargo_tool_pins.reconcile_pins(
        justfile,
        installed_output(override=("rumdl", "2.0.0")),
        "uv 3.0.0",
    )

    expected = original.replace(b'rumdl_version := "1.2.3"', b'rumdl_version := "2.0.0"').replace(b'uv_version := "1.2.3"', b'uv_version := "3.0.0"')
    assert changes == {
        "rumdl_version": ("1.2.3", "2.0.0"),
        "uv_version": ("1.2.3", "3.0.0"),
    }
    assert justfile.is_symlink()
    assert target.read_bytes() == expected
    assert list(tmp_path.glob(".tool-pins.just.*")) == []


def test_reconcile_pins_rejects_missing_package_without_writing(tmp_path: Path) -> None:
    justfile = tmp_path / "justfile"
    original = justfile_text()
    justfile.write_text(original, encoding="utf-8")
    incomplete = installed_output().replace("rumdl v1.2.3:\n    rumdl\n", "")

    with pytest.raises(ValueError, match="managed tool is not installed: rumdl"):
        update_cargo_tool_pins.reconcile_pins(justfile, incomplete, "uv 1.2.3")

    assert justfile.read_text(encoding="utf-8") == original


def test_update_pin_text_rejects_duplicate_assignment() -> None:
    duplicated = justfile_text() + 'rumdl_version := "1.2.3"\n'
    installed = update_cargo_tool_pins.parse_installed_packages(installed_output())
    installed["uv"] = "1.2.3"

    with pytest.raises(ValueError, match="expected exactly one rumdl_version assignment, found 2"):
        update_cargo_tool_pins.update_pin_text(duplicated, installed)


def test_parse_installed_packages_accepts_prerelease_with_build_metadata() -> None:
    version = "1.2.3-rc.1+build.5"

    installed = update_cargo_tool_pins.parse_installed_packages(installed_output(override=("rumdl", version)))

    assert installed["rumdl"] == version


def test_update_pin_text_rejects_prerelease_managed_tool() -> None:
    installed = update_cargo_tool_pins.parse_installed_packages(installed_output(override=("rumdl", "1.2.3-rc.1+build.5")))
    installed["uv"] = "1.2.3"

    with pytest.raises(ValueError, match=r"managed tool version must be stable X\.Y\.Z: rumdl"):
        update_cargo_tool_pins.update_pin_text(justfile_text(), installed)


@pytest.mark.parametrize("output", ["uv 1.2.3-rc.1", "uv 1.2.3.4", "uv release-1.2.3"])
def test_parse_tool_version_rejects_nonstable_or_embedded_versions(output: str) -> None:
    with pytest.raises(ValueError, match="expected exactly one uv version"):
        update_cargo_tool_pins.parse_tool_version(output, "uv")


@pytest.mark.parametrize(
    "output",
    ["", "uv unknown", "uv 1.2", "uv 1.2.3-rc.1", "uv 1.2.3+build.1", "uv 1.2.3.4", "uv release-1.2.3", "uv 1.2.3 2.0.0", "uv 1.2.3\nuv 1.2.3"],
)
def test_uv_preflight_and_reconciler_reject_invalid_versions_without_writing(
    output: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    justfile = tmp_path / "justfile"
    original = justfile_text().encode("utf-8")
    justfile.write_bytes(original)

    def uv_version(command: str, args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert command == "uv"
        assert args == ["--version"]
        assert kwargs == {"timeout": 30}
        return subprocess.CompletedProcess([command, *args], 0, stdout=output, stderr="")

    monkeypatch.setattr(update_cargo_tool_pins, "run_safe_command", uv_version)

    assert update_cargo_tool_pins.main(["--check-uv", "--justfile", str(justfile)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "failed uv preflight (requires stable X.Y.Z): expected exactly one uv version" in captured.err
    assert justfile.read_bytes() == original

    with pytest.raises(ValueError, match="expected exactly one uv version"):
        update_cargo_tool_pins.reconcile_pins(justfile, installed_output(), output)

    assert justfile.read_bytes() == original
    assert list(tmp_path.iterdir()) == [justfile]


@pytest.mark.parametrize("output", ["uv 99.0.0", "uv v99.0.0", "uv 99.0.0 (Homebrew 2026-09-04 aarch64-apple-darwin)\n"])
def test_uv_preflight_accepts_stable_versions_without_reading_pins_or_cargo(
    output: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing_justfile = tmp_path / "justfile"

    def uv_version(command: str, args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert command == "uv"
        assert args == ["--version"]
        assert kwargs == {"timeout": 30}
        return subprocess.CompletedProcess([command, *args], 0, stdout=output, stderr="")

    monkeypatch.setattr(update_cargo_tool_pins, "run_safe_command", uv_version)

    assert update_cargo_tool_pins.main(["--check-uv", "--justfile", str(missing_justfile)]) == 0
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
    assert not missing_justfile.exists()


@pytest.mark.parametrize(
    "error",
    [
        ExecutableNotFoundError("uv missing"),
        OSError("uv unavailable"),
        subprocess.CalledProcessError(2, ["uv", "--version"]),
        subprocess.TimeoutExpired("uv", 30),
    ],
)
def test_uv_preflight_reports_command_failure_without_traceback(
    error: Exception,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def failed_uv(_command: str, _args: list[str], **_kwargs: object) -> Never:
        raise error

    monkeypatch.setattr(update_cargo_tool_pins, "run_safe_command", failed_uv)

    assert update_cargo_tool_pins.main(["--check-uv"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"failed uv preflight (requires stable X.Y.Z): {error}\n"


def test_main_reports_missing_cargo_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def missing_cargo(_command: str, _args: list[str], **_kwargs: object) -> Never:
        msg = "Required executable 'cargo' not found in PATH"
        raise ExecutableNotFoundError(msg)

    monkeypatch.setattr(update_cargo_tool_pins, "run_safe_command", missing_cargo)

    assert update_cargo_tool_pins.main([]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "failed to update tool pins: Required executable 'cargo' not found in PATH\n"


def test_main_queries_cargo_then_uv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    justfile = tmp_path / "justfile"
    justfile.write_text(justfile_text(), encoding="utf-8")
    calls: list[tuple[str, list[str]]] = []

    def fake_run(command: str, args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, args))
        output = installed_output() if command == "cargo" else "uv 1.2.3"
        return subprocess.CompletedProcess([command, *args], 0, stdout=output, stderr="")

    monkeypatch.setattr(update_cargo_tool_pins, "run_safe_command", fake_run)

    assert update_cargo_tool_pins.main(["--justfile", str(justfile)]) == 0
    assert calls == [("cargo", ["install", "--list"]), ("uv", ["--version"])]
