"""Tests for tag_release.py — annotated tag creation with size-limit handling."""

import subprocess
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

import tag_release
from tag_release import (
    _GITHUB_TAG_ANNOTATION_LIMIT,
    ReleaseVersion,
    _github_anchor,
    extract_changelog_section,
    find_changelog,
    parse_github_repository_url,
    parse_version,
    validate_semver,
)

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# SemVer validation
# ---------------------------------------------------------------------------


class TestValidateSemver:
    @pytest.mark.parametrize(
        "version",
        [
            "v0.1.0",
            "v1.0.0",
            "v12.34.56",
            "v1.2.3-rc.1",
            "v1.2.3-alpha",
            "v1.2.3+build.42",
            "v1.2.3-beta.1+build.123",
            "v1.0.0-1a",  # digit-prefixed alphanumeric prerelease
            "v1.0.0-0a",  # leading zero OK when not purely numeric
            "v1.0.0-1a.2b",  # dot-separated digit-prefixed IDs
        ],
    )
    def test_valid_versions(self, version: str) -> None:
        validate_semver(version)  # should not raise

    @pytest.mark.parametrize(
        "version",
        [
            "0.1.0",  # missing v prefix
            "v1",  # incomplete
            "v1.2",  # missing patch
            "v01.2.3",  # leading zero
            "v1.02.3",  # leading zero
            "v1.2.03",  # leading zero
            "v1.0.0-01",  # leading zero in purely numeric prerelease
            "vfoo",  # garbage
            "",  # empty
        ],
    )
    def test_invalid_versions(self, version: str) -> None:
        with pytest.raises(ValueError, match="SemVer format"):
            validate_semver(version)


class TestParseVersion:
    def test_strips_v_prefix(self) -> None:
        assert parse_version("v1.2.3") == "1.2.3"

    def test_no_prefix(self) -> None:
        with pytest.raises(ValueError, match="SemVer format"):
            parse_version("1.2.3")

    def test_direct_construction_rejects_invalid_tag(self) -> None:
        with pytest.raises(ValueError, match="SemVer format"):
            ReleaseVersion(tag="release-1.2.3", number="1.2.3")

    def test_direct_construction_rejects_mismatched_number(self) -> None:
        with pytest.raises(ValueError, match="does not match tag"):
            ReleaseVersion(tag="v1.2.3", number="9.9.9")

    def test_cli_preserves_semver_guidance(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as error:
            tag_release.parse_args(["1.2.3"])

        assert error.value.code == 2
        assert "Tag version should follow SemVer format 'vX.Y.Z'" in capsys.readouterr().err


class TestParseGitHubRepositoryUrl:
    @pytest.mark.parametrize(
        "remote",
        [
            "git@github.com:acgetchell/markov-chain-monte-carlo.git",
            "https://github.com/acgetchell/markov-chain-monte-carlo.git",
            "ssh://git@github.com/acgetchell/markov-chain-monte-carlo.git",
        ],
    )
    def test_normalizes_supported_github_remotes(self, remote: str) -> None:
        assert parse_github_repository_url(remote) == "https://github.com/acgetchell/markov-chain-monte-carlo"

    def test_rejects_non_github_remote(self) -> None:
        with pytest.raises(ValueError, match="not a supported GitHub URL"):
            parse_github_repository_url("https://example.com/owner/repo.git")


# ---------------------------------------------------------------------------
# Changelog helpers
# ---------------------------------------------------------------------------


_SAMPLE_CHANGELOG = """\
# Changelog

## [0.2.0] - 2025-03-01

### Added

    - Streaming statistics via `OnlineStats`

### Changed

- Bump version to 0.2.0

## [0.1.3] - 2025-02-15

### Fixed

- Minor doc typo
"""


class TestFindChangelog:
    def test_finds_in_current_dir(self, tmp_path: Path) -> None:
        (tmp_path / "CHANGELOG.md").write_text("# Changelog\n", encoding="utf-8")
        result = find_changelog(tmp_path)
        assert result.name == "CHANGELOG.md"

    def test_finds_in_parent_dir(self, tmp_path: Path) -> None:
        (tmp_path / "CHANGELOG.md").write_text("# Changelog\n", encoding="utf-8")
        child = tmp_path / "scripts"
        child.mkdir()
        result = find_changelog(child)
        assert result.name == "CHANGELOG.md"

    def test_raises_when_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match=r"CHANGELOG\.md not found"):
            find_changelog(tmp_path)


class TestExtractChangelogSection:
    def test_extracts_section(self, tmp_path: Path) -> None:
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text(_SAMPLE_CHANGELOG, encoding="utf-8")

        section = extract_changelog_section(changelog, "0.2.0")
        assert "OnlineStats" in section
        assert "Bump version" in section
        # Should not include content from 0.1.3
        assert "Minor doc typo" not in section

    def test_extracts_older_section(self, tmp_path: Path) -> None:
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text(_SAMPLE_CHANGELOG, encoding="utf-8")

        section = extract_changelog_section(changelog, "0.1.3")
        assert "Minor doc typo" in section
        assert "OnlineStats" not in section

    def test_raises_for_missing_version(self, tmp_path: Path) -> None:
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text(_SAMPLE_CHANGELOG, encoding="utf-8")

        with pytest.raises(LookupError, match="No changelog section found"):
            extract_changelog_section(changelog, "9.9.9")

    def test_raises_for_empty_section(self, tmp_path: Path) -> None:
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text("# Changelog\n\n## [1.0.0] - 2025-01-01\n\n## [0.9.0] - 2024-12-01\n", encoding="utf-8")

        with pytest.raises(LookupError, match="empty"):
            extract_changelog_section(changelog, "1.0.0")


# ---------------------------------------------------------------------------
# GitHub anchor generation
# ---------------------------------------------------------------------------


class TestGitHubAnchor:
    """Verify _github_anchor matches github-slugger output."""

    def test_bracketed_heading(self, tmp_path: Path) -> None:
        """Heading ``## [1.0.0] - 2025-01-01`` should strip brackets and dots."""
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text(
            "# Changelog\n\n## [1.0.0] - 2025-01-01\n\n- Item\n",
            encoding="utf-8",
        )
        assert _github_anchor(changelog, "1.0.0") == "100---2025-01-01"

    def test_plain_v_heading(self, tmp_path: Path) -> None:
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text(
            "# Changelog\n\n## v0.2.0\n\n- Item\n",
            encoding="utf-8",
        )
        assert _github_anchor(changelog, "0.2.0") == "v020"

    def test_fallback_when_not_found(self, tmp_path: Path) -> None:
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text("# Changelog\n", encoding="utf-8")
        assert _github_anchor(changelog, "9.9.9") == "v999"

    def test_does_not_match_prerelease_heading(self, tmp_path: Path) -> None:
        """Looking for 1.0.0 must not match ## [1.0.0-rc.1]."""
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text(
            "# Changelog\n\n## [1.0.0-rc.1] - 2025-01-01\n\n- Item\n",
            encoding="utf-8",
        )
        # Should fall back since no exact 1.0.0 heading exists
        assert _github_anchor(changelog, "1.0.0") == "v100"


# ---------------------------------------------------------------------------
# Tag size limit handling
# ---------------------------------------------------------------------------


class TestTagSizeLimit:
    def test_small_section_uses_full_content(self, tmp_path: Path) -> None:
        """A normal-sized changelog section should be used as the tag message."""
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text(_SAMPLE_CHANGELOG, encoding="utf-8")

        section = extract_changelog_section(changelog, "0.2.0")
        assert len(section.encode("utf-8")) < _GITHUB_TAG_ANNOTATION_LIMIT

    def test_oversized_section_detected(self, tmp_path: Path) -> None:
        """Synthetic oversized changelog should exceed the limit."""
        # Build content > 125KB
        lines = [f"- Item number {i}" for i in range(20_000)]
        big_section = "\n".join(lines)

        changelog_text = f"# Changelog\n\n## [1.0.0] - 2025-01-01\n\n{big_section}\n\n## [0.9.0] - 2024-12-01\n\n- Old item\n"
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text(changelog_text, encoding="utf-8")

        section = extract_changelog_section(changelog, "1.0.0")
        assert len(section.encode("utf-8")) > _GITHUB_TAG_ANNOTATION_LIMIT


class TestCargoPackageVersion:
    def test_reads_the_authoritative_package_version(self, tmp_path: Path) -> None:
        cargo_toml = tmp_path / "Cargo.toml"
        cargo_toml.write_text('[package]\nname = "fixture"\nversion = "1.2.3"\n', encoding="utf-8")

        assert tag_release._cargo_package_version(cargo_toml) == "1.2.3"

    def test_rejects_a_missing_package_version(self, tmp_path: Path) -> None:
        cargo_toml = tmp_path / "Cargo.toml"
        cargo_toml.write_text('[package]\nname = "fixture"\n', encoding="utf-8")

        with pytest.raises(TypeError, match=r"Cargo\.toml \[package\] version"):
            tag_release._cargo_package_version(cargo_toml)


# ---------------------------------------------------------------------------
# create_tag workflow (mocked git)
# ---------------------------------------------------------------------------


class TestCreateTag:
    @pytest.fixture(autouse=True)
    def _matching_cargo_metadata(self, tmp_path: Path) -> None:
        (tmp_path / "Cargo.toml").write_text('[package]\nname = "fixture"\nversion = "1.0.0"\n', encoding="utf-8")

    def test_mismatched_package_version_fails_before_tag_lookup(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        changelog = tmp_path / "CHANGELOG.md"
        (tmp_path / "Cargo.toml").write_text('[package]\nname = "fixture"\nversion = "1.2.3"\n', encoding="utf-8")
        tag_exists = MagicMock()
        create_git_tag = MagicMock()
        monkeypatch.setattr(tag_release, "find_changelog", lambda: changelog)
        monkeypatch.setattr(tag_release, "_tag_exists", tag_exists)
        monkeypatch.setattr(tag_release, "run_git_command_with_input", create_git_tag)

        with pytest.raises(ValueError, match=r"requested tag v1\.0\.0 does not match .* version 1\.2\.3"):
            tag_release.create_tag("v1.0.0")

        tag_exists.assert_not_called()
        create_git_tag.assert_not_called()

    @patch("tag_release.run_git_command_with_input")
    @patch("tag_release._tag_exists", return_value=False)
    @patch("tag_release.find_changelog")
    @patch("tag_release.extract_changelog_section", return_value="### Added\n\n- Something new")
    def test_next_step_sets_release_title(
        self,
        _mock_extract: MagicMock,
        mock_find: MagicMock,
        _mock_exists: MagicMock,
        _mock_git_input: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_find.return_value = tmp_path / "CHANGELOG.md"

        tag_release.create_tag("v1.0.0")

        assert "gh release create v1.0.0 --title v1.0.0 --notes-from-tag" in capsys.readouterr().out

    @patch("tag_release.run_git_command_with_input")
    @patch("tag_release._tag_exists", return_value=False)
    @patch("tag_release.find_changelog")
    @patch("tag_release.extract_changelog_section", return_value="### Added\n\n- Something new")
    def test_creates_annotated_tag(
        self,
        _mock_extract: MagicMock,
        mock_find: MagicMock,
        _mock_exists: MagicMock,
        mock_git_input: MagicMock,
        tmp_path: Path,
    ) -> None:
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text(_SAMPLE_CHANGELOG, encoding="utf-8")
        mock_find.return_value = changelog

        tag_release.create_tag("v1.0.0")

        mock_git_input.assert_called_once()
        call_args = mock_git_input.call_args
        assert call_args[0][0] == ["tag", "--annotate", "v1.0.0", "-F", "-", "--cleanup=verbatim"]
        assert "### Added" in call_args[1]["input_data"]
        assert "Something new" in call_args[1]["input_data"]

    @patch("tag_release.run_git_command_with_input")
    @patch("tag_release._get_repo_url", return_value="https://github.com/acgetchell/markov-chain-monte-carlo")
    @patch("tag_release._tag_exists", return_value=False)
    @patch("tag_release.find_changelog")
    def test_oversized_creates_reference_tag(
        self,
        mock_find: MagicMock,
        _mock_exists: MagicMock,
        _mock_url: MagicMock,
        mock_git_input: MagicMock,
        tmp_path: Path,
    ) -> None:
        """When changelog exceeds 125KB, tag message should be a short reference."""
        lines = [f"- Item number {i}" for i in range(20_000)]
        big_section = "\n".join(lines)
        changelog_text = f"# Changelog\n\n## [1.0.0] - 2025-01-01\n\n{big_section}\n\n## [0.9.0] - 2024-12-01\n\n- Old\n"
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text(changelog_text, encoding="utf-8")
        mock_find.return_value = changelog

        # Patch extract to return the real oversized content
        with patch("tag_release.extract_changelog_section", return_value=big_section):
            tag_release.create_tag("v1.0.0")

        mock_git_input.assert_called_once()
        tag_message = mock_git_input.call_args[1]["input_data"]
        assert "See full changelog" in tag_message
        assert "CHANGELOG.md" in tag_message
        assert len(tag_message) < 1000

    @patch("tag_release.find_changelog")
    @patch("tag_release._tag_exists", return_value=True)
    def test_existing_tag_without_force_is_rejected(
        self,
        _mock_exists: MagicMock,
        mock_find: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_find.return_value = tmp_path / "CHANGELOG.md"

        with pytest.raises(FileExistsError, match="already exists"):
            tag_release.create_tag("v1.0.0", force=False)

    @patch("tag_release.run_git_command_with_input")
    @patch("tag_release._tag_exists", return_value=True)
    @patch("tag_release.find_changelog")
    @patch("tag_release.extract_changelog_section", return_value="### Fixed\n\n- Bug fix")
    def test_force_replaces_tag_with_one_git_command(
        self,
        _mock_extract: MagicMock,
        mock_find: MagicMock,
        _mock_exists: MagicMock,
        mock_git_input: MagicMock,
        tmp_path: Path,
    ) -> None:
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text(_SAMPLE_CHANGELOG, encoding="utf-8")
        mock_find.return_value = changelog

        tag_release.create_tag("v1.0.0", force=True)

        mock_git_input.assert_called_once_with(
            ["tag", "--force", "--annotate", "v1.0.0", "-F", "-", "--cleanup=verbatim"],
            input_data="### Fixed\n\n- Bug fix",
        )

    @patch("tag_release.run_git_command_with_input")
    @patch("tag_release._tag_exists", return_value=True)
    @patch("tag_release.find_changelog")
    @patch("tag_release.extract_changelog_section", side_effect=LookupError("not found"))
    def test_force_does_not_replace_tag_if_changelog_fails(
        self,
        _mock_extract: MagicMock,
        mock_find: MagicMock,
        _mock_exists: MagicMock,
        mock_git_input: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Tag must not be replaced if changelog extraction fails."""
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text("# Changelog\n", encoding="utf-8")
        mock_find.return_value = changelog

        with pytest.raises(LookupError):
            tag_release.create_tag("v1.0.0", force=True)

        mock_git_input.assert_not_called()


class TestMain:
    def test_reports_captured_git_stderr_without_traceback(self, capsys: pytest.CaptureFixture[str]) -> None:
        error = subprocess.CalledProcessError(
            128,
            ["git", "tag", "--annotate", "v1.0.0"],
            output="less useful output\n",
            stderr="fatal: unable to create tag object\n",
        )

        with patch("tag_release.create_tag", side_effect=error):
            status = tag_release.main(["v1.0.0"])

        captured = capsys.readouterr()
        assert status == 1
        assert captured.out == ""
        assert captured.err == ("Error: command failed with exit 128: git tag --annotate v1.0.0\nstderr:\nfatal: unable to create tag object\n")
        assert "Traceback" not in captured.err
