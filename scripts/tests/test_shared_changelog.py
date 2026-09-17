"""Consumer boundaries for the published changelog toolkit; parser tests live upstream."""

import tomllib
from pathlib import Path

import pytest

from tag_release import extract_changelog_section
from update_python_dev_pins import _resolution_requirements, parse_project

ROOT = Path(__file__).resolve().parents[2]


def test_tooling_pin_survives_development_pin_resolution() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    groups = tomllib.loads(text)["dependency-groups"]
    assert {"include-group": "tooling"} in groups["dev"]
    assert groups["tooling"] == ["research-repo-tools==0.1.0"]
    _, pins = parse_project(text)
    assert all(pin.name != "research-repo-tools" for pin in pins)
    assert "research-repo-tools==0.1.0" in _resolution_requirements(text, pins).splitlines()


@pytest.mark.parametrize("archived", [False, True])
def test_tag_wrapper_uses_shared_notes_with_reference_links(tmp_path: Path, archived: bool) -> None:
    root = tmp_path / "CHANGELOG.md"
    root.write_text("# Changelog\n", encoding="utf-8", newline="\n")
    target = tmp_path / "docs/archives/changelog/0.1.md" if archived else root
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "# Changelog\n\n## [0.1.0] - 2026-03-24\n\n### Added\n\n- Preserve `Chain<S>` and [details][api].\n\n[api]: https://example.com/api\n",
        encoding="utf-8",
        newline="\n",
    )
    notes = extract_changelog_section(root, "0.1.0")
    assert "`Chain<S>`" in notes
    assert "[details][api]" in notes
    assert "[api]: https://example.com/api" in notes
    with pytest.raises(LookupError, match="not found"):
        extract_changelog_section(root, "9.9.9")
