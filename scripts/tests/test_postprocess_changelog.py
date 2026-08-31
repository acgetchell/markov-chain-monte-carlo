"""Tests for postprocess_changelog.py — trailing blank line hygiene."""

import subprocess
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

import postprocess_changelog
from postprocess_changelog import postprocess

if TYPE_CHECKING:
    from pathlib import Path


class TestPostprocess:
    def test_strips_trailing_blank_lines(self, tmp_path: Path) -> None:
        f = tmp_path / "CHANGELOG.md"
        f.write_text("# Changelog\n\n- Item\n\n\n\n", encoding="utf-8")

        postprocess(f)

        assert f.read_text(encoding="utf-8") == "# Changelog\n\n- Item\n"

    @pytest.mark.parametrize("trailing", ["   \n\n", "\t \r\n\r\n"])
    def test_strips_whitespace_only_trailing_lines_and_invalid_trailing_spaces(self, tmp_path: Path, trailing: str) -> None:
        f = tmp_path / "CHANGELOG.md"
        f.write_bytes(f"# Changelog\n\n- Item  \n{trailing}".encode())

        postprocess(f)

        assert f.read_bytes() == b"# Changelog\n\n- Item\n"

    @pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"], ids=["lf", "crlf", "cr"])
    def test_output_uses_lf_even_with_windows_text_file_defaults(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        newline: str,
    ) -> None:
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_bytes(f"# Changelog{newline}{newline}- Item  {newline}  {newline}".encode())
        original_temporary_file = postprocess_changelog.tempfile.NamedTemporaryFile

        def windows_temporary_file(*args: Any, **kwargs: Any) -> Any:
            mode = args[0] if args else kwargs.get("mode", "w+b")
            if "b" not in mode and kwargs.get("newline") is None:
                kwargs["newline"] = "\r\n"
            return original_temporary_file(*args, **kwargs)

        monkeypatch.setattr(postprocess_changelog.tempfile, "NamedTemporaryFile", windows_temporary_file)

        postprocess(changelog)

        assert changelog.read_bytes() == b"# Changelog\n\n- Item\n"
        assert tuple(tmp_path.iterdir()) == (changelog,)

    def test_preserves_single_trailing_newline(self, tmp_path: Path) -> None:
        f = tmp_path / "CHANGELOG.md"
        f.write_text("# Changelog\n\n- Item\n", encoding="utf-8")

        postprocess(f)

        assert f.read_text(encoding="utf-8") == "# Changelog\n\n- Item\n"

    def test_adds_trailing_newline_if_missing(self, tmp_path: Path) -> None:
        f = tmp_path / "CHANGELOG.md"
        f.write_text("# Changelog\n\n- Item", encoding="utf-8")

        postprocess(f)

        assert f.read_text(encoding="utf-8") == "# Changelog\n\n- Item\n"

    def test_preserves_internal_blank_lines(self, tmp_path: Path) -> None:
        content = "# Changelog\n\n## [1.0.0]\n\n### Added\n\n- Item\n\n\n\n"
        f = tmp_path / "CHANGELOG.md"
        f.write_text(content, encoding="utf-8")

        postprocess(f)

        result = f.read_text(encoding="utf-8")
        # Internal blank lines preserved, only trailing ones stripped
        assert result == "# Changelog\n\n## [1.0.0]\n\n### Added\n\n- Item\n"

    def test_wraps_generated_markdown_to_the_configured_limit(self, tmp_path: Path) -> None:
        f = tmp_path / "CHANGELOG.md"
        f.write_text(f"# Changelog\n\n- {'release note ' * 20}\n", encoding="utf-8")

        postprocess(f)

        assert max(map(len, f.read_text(encoding="utf-8").splitlines())) <= 160

    def test_single_newline_file(self, tmp_path: Path) -> None:
        f = tmp_path / "CHANGELOG.md"
        f.write_text("\n", encoding="utf-8")

        postprocess(f)

        assert f.read_text(encoding="utf-8") == "\n"

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "CHANGELOG.md"
        f.write_text("", encoding="utf-8")

        postprocess(f)

        assert f.read_text(encoding="utf-8") == "\n"

    def test_preserves_original_if_atomic_replace_fails(self, tmp_path: Path) -> None:
        f = tmp_path / "CHANGELOG.md"
        original = "# Changelog\n\n- Existing entry\n\n"
        f.write_text(original, encoding="utf-8")

        with (
            patch.object(type(f), "replace", side_effect=OSError("injected replace failure")),
            pytest.raises(OSError, match="injected replace failure"),
        ):
            postprocess(f)

        assert f.read_text(encoding="utf-8") == original
        assert tuple(tmp_path.iterdir()) == (f,)

    def test_preserves_original_if_markdown_formatting_fails(self, tmp_path: Path) -> None:
        f = tmp_path / "CHANGELOG.md"
        original = "# Changelog\n\n- Existing entry\n\n"
        f.write_text(original, encoding="utf-8")
        failure = subprocess.CalledProcessError(1, ["rumdl"], stderr="unfixable Markdown")

        with (
            patch("postprocess_changelog.run_safe_command", side_effect=failure),
            pytest.raises(postprocess_changelog.MarkdownFormatError, match="unfixable Markdown"),
        ):
            postprocess(f)

        assert f.read_text(encoding="utf-8") == original
        assert tuple(tmp_path.iterdir()) == (f,)

    def test_main_reports_atomic_replace_failure_without_traceback(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text("# Changelog\n", encoding="utf-8")

        with patch("postprocess_changelog.postprocess", side_effect=OSError("injected replace failure")):
            result = postprocess_changelog.main([str(changelog)])

        captured = capsys.readouterr()
        assert result == 1
        assert captured.out == ""
        assert captured.err == f"Error: could not post-process {changelog}: injected replace failure\n"
        assert "Traceback" not in captured.err
