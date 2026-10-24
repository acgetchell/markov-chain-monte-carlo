"""Prepared citation dates keep MCMC's prospective changelog commands repeatable."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from research_repo_tools.just_inspect import dry_run, inspect_justfile

if TYPE_CHECKING:
    from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def adapter() -> ModuleType:
    """Load the standalone adapter without creating a local tooling package."""
    spec = spec_from_file_location("mcmc_release_changelog", ROOT / "tooling/release_changelog.py")
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("quotation", ["", "'", '"'])
def test_omitted_date_reuses_prepared_leap_date_for_generation_and_preview(
    adapter: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, quotation: str
) -> None:
    citation = tmp_path / "CITATION.cff"
    contents = f"version: 0.5.2\ndate-released: {quotation}2000-02-29{quotation}\n".encode()
    citation.write_bytes(contents)
    monkeypatch.chdir(tmp_path)
    calls: list[list[str]] = []

    def shared_cli(argv: list[str]) -> int:
        calls.append(argv)
        return 0

    monkeypatch.setattr(adapter, "tooling_main", shared_cli)
    for extra in ([], ["--date", ""], ["--dry-run"]):
        assert adapter.main(["--tag", "v0.5.2", *extra]) == 0
        expected = ["changelog", "generate", "--tag", "v0.5.2", "--date", "2000-02-29"]
        if extra == ["--dry-run"]:
            expected.append("--dry-run")
        assert calls[-1] == expected
    assert citation.read_bytes() == contents


@pytest.mark.parametrize("arguments", [["--tag", "v0.5.2", "--date", "2000-02-29"], []])
def test_explicit_override_and_untagged_preview_do_not_require_citation(
    adapter: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arguments: list[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    calls: list[list[str]] = []

    def shared_cli(argv: list[str]) -> int:
        calls.append(argv)
        return 7

    monkeypatch.setattr(adapter, "tooling_main", shared_cli)
    assert adapter.main([*arguments, "--dry-run"]) == 7
    assert calls == [["changelog", "generate", *arguments, "--dry-run"]]
    assert not (tmp_path / "CITATION.cff").exists()


@pytest.mark.parametrize(
    "invalid",
    [
        (None, "CITATION.cff"),
        (b"[", "while parsing"),
        (b"- item\n", "top-level mapping"),
        (b"version: 0.5.2\n", "top-level date-released"),
        (b"date-released: 2000-02-29\n", "top-level version"),
        (b"version: 0.5.1\ndate-released: 2000-02-29\n", "does not match v0.5.2"),
        (b"version: 0.5.2\ndate-released: 2001-02-29\n", "day"),
        (b"version: 0.5.2\ndate-released: 20000229\n", "YYYY-MM-DD"),
        (b"version: 0.5.2\ndate-released: \n", "nonempty top-level date-released"),
        (b"version: 0.5.2\ndate-released: [2000-02-29]\n", "top-level date-released scalar"),
        (b"version: 0.5.2\ndate-released: 2000-02-29\ndate-released: 2000-03-01\n", "exactly one"),
        (b"version: 0.5.2\nversion: 0.5.3\ndate-released: 2000-02-29\n", "exactly one"),
    ],
)
def test_invalid_default_metadata_preserves_release_artifacts(
    adapter: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], invalid: tuple[bytes | None, str]
) -> None:
    contents, diagnostic = invalid
    citation = tmp_path / "CITATION.cff"
    if contents is not None:
        citation.write_bytes(contents)
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_bytes(b"retained changelog\r\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(adapter, "tooling_main", lambda _argv: pytest.fail("invalid metadata must not generate anything"))
    assert adapter.main(["--tag", "v0.5.2"]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "Release changelog rejected:" in output.err
    assert diagnostic in output.err
    assert "just update-version" in output.err
    assert changelog.read_bytes() == b"retained changelog\r\n"
    assert not (tmp_path / "docs").exists()
    if contents is not None:
        assert citation.read_bytes() == contents


def test_normal_alias_and_preview_support_an_optional_explicit_date() -> None:
    assert inspect_justfile(ROOT).aliases["changelog-unreleased"] == "changelog-release"
    implicit = dry_run(ROOT, "changelog-unreleased", ["v0.5.2"]).stderr
    explicit = dry_run(ROOT, "changelog-release", ["v0.5.2", "2000-02-29"]).stderr
    assert "tooling/release_changelog.py --tag 'v0.5.2' --date ''" in implicit
    assert "tooling/release_changelog.py --tag 'v0.5.2' --date '2000-02-29'" in explicit
    assert 'tooling/release_changelog.py --dry-run "$@"' in dry_run(ROOT, "changelog-preview", ["--tag", "v0.5.2"]).stderr
