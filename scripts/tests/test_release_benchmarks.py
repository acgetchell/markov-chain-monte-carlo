"""Exercise release workflow shell steps without contacting GitHub."""

import os
import re
import shutil
import subprocess
from pathlib import Path
from textwrap import dedent

import pytest

WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/release-benchmarks.yml"
VALIDATE = "Validate draft GitHub Release"
PUBLISH = "Attach baseline and publish GitHub Release"
VIEW = "release view v1.0.0 --json isDraft,isImmutable --jq [.isDraft, .isImmutable] | @tsv"
ASSETS = "release view v1.0.0 --json assets --jq .assets[].name"
DOWNLOAD = "release download v1.0.0 --pattern baseline.tar.gz"
UPLOAD = "release upload v1.0.0 baseline.tar.gz"
EDIT = "release edit v1.0.0 --draft=false"


def _step_script(name: str) -> str:
    step = WORKFLOW.read_text(encoding="utf-8").split(f"      - name: {name}\n", 1)[1]
    match = re.search(r"        run: \|\n((?:          [^\n]*\n|\n)+)", step)
    assert match is not None
    return dedent(match[1])


def _run_step(tmp_path: Path, name: str, **overrides: str) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    bash = shutil.which("bash")
    assert bash is not None, "Bash is required by the repository's workflow tests"
    environment = {
        **os.environ,
        "RELEASE_TAG": "v1.0.0",
        "RELEASE_ASSET": "baseline.tar.gz",
        "RELEASE_STATE": "true\tfalse",
        "VIEW_STATUS": "0",
        "ASSET_NAMES": "",
        "ASSETS_STATUS": "0",
        "DOWNLOAD_CONTENT": "baseline",
        "DOWNLOAD_STATUS": "0",
        "UPLOAD_STATUS": "0",
        "EDIT_STATUS": "0",
        **overrides,
    }
    fake_gh = dedent("""\
        gh() {
          if [[ "$1 $2" == 'release download' ]]; then
            printf 'release download %s --pattern %s\\n' "$3" "$5" >> gh-calls.txt
          else
            printf '%s\\n' "$*" >> gh-calls.txt
          fi
          case "$1 $2" in
            'release view')
              if [[ "$5" == assets ]]; then
                printf '%s\\n' "$ASSET_NAMES"; return "$ASSETS_STATUS"
              fi
              printf '%s\\n' "$RELEASE_STATE"; return "$VIEW_STATUS" ;;
            'release download')
              printf '%s' "$DOWNLOAD_CONTENT" > "$7/$RELEASE_ASSET"
              return "$DOWNLOAD_STATUS" ;;
            'release upload') return "$UPLOAD_STATUS" ;;
            'release edit') return "$EDIT_STATUS" ;;
            *) return 99 ;;
          esac
        }
        """)
    (tmp_path / "baseline.tar.gz").write_bytes(b"baseline")
    result = subprocess.run(  # noqa: S603 - fixed workflow source and fake GitHub CLI, with no network effects.
        [bash, "--noprofile", "--norc", "-c", fake_gh + _step_script(name)],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        encoding="utf-8",
        timeout=10,
    )
    log = tmp_path / "gh-calls.txt"
    return result, log.read_text(encoding="utf-8").splitlines() if log.exists() else []


@pytest.mark.parametrize("step", [VALIDATE, PUBLISH])
def test_mutable_draft_is_accepted(tmp_path: Path, step: str) -> None:
    result, calls = _run_step(tmp_path, step)

    assert result.returncode == 0, result.stderr
    assert calls == ([VIEW] if step == VALIDATE else [VIEW, ASSETS, UPLOAD, EDIT])


@pytest.mark.parametrize("step", [VALIDATE, PUBLISH])
@pytest.mark.parametrize("state", ["false\tfalse", "false\ttrue", "true\ttrue", "", "null\tnull"])
def test_non_draft_or_invalid_state_prevents_publication(tmp_path: Path, step: str, state: str) -> None:
    result, calls = _run_step(tmp_path, step, RELEASE_STATE=state)

    assert result.returncode != 0
    assert "mutable draft" in result.stderr
    assert calls == [VIEW]


@pytest.mark.parametrize("step", [VALIDATE, PUBLISH])
def test_failed_lookup_prevents_publication_even_with_partial_output(tmp_path: Path, step: str) -> None:
    result, calls = _run_step(tmp_path, step, VIEW_STATUS="1")

    assert result.returncode == 1
    assert calls == [VIEW]


@pytest.mark.parametrize("tag", ["", "main", "v1.0.0-rc.1", "v1.0.0+build.42"])
def test_invalid_tag_fails_before_github_lookup(tmp_path: Path, tag: str) -> None:
    result, calls = _run_step(tmp_path, VALIDATE, RELEASE_TAG=tag)

    assert result.returncode != 0
    assert "stable vX.Y.Z" in result.stderr
    assert calls == []


def test_upload_failure_leaves_release_unpublished(tmp_path: Path) -> None:
    result, calls = _run_step(tmp_path, PUBLISH, UPLOAD_STATUS="1")

    assert result.returncode == 1
    assert calls == [VIEW, ASSETS, UPLOAD]


def test_publication_failure_is_reported(tmp_path: Path) -> None:
    result, calls = _run_step(tmp_path, PUBLISH, EDIT_STATUS="1")

    assert result.returncode == 1
    assert calls == [VIEW, ASSETS, UPLOAD, EDIT]


def test_publication_retry_reuses_identical_uploaded_asset(tmp_path: Path) -> None:
    result, calls = _run_step(tmp_path, PUBLISH, ASSET_NAMES="unrelated.zip\nbaseline.tar.gz")

    assert result.returncode == 0, result.stderr
    assert calls == [VIEW, ASSETS, DOWNLOAD, EDIT]


def test_conflicting_uploaded_asset_keeps_release_unpublished(tmp_path: Path) -> None:
    result, calls = _run_step(tmp_path, PUBLISH, ASSET_NAMES="baseline.tar.gz", DOWNLOAD_CONTENT="different baseline")

    assert result.returncode == 1
    assert "Existing release asset differs" in result.stderr
    assert calls == [VIEW, ASSETS, DOWNLOAD]


def test_failed_asset_lookup_prevents_upload_and_publication(tmp_path: Path) -> None:
    result, calls = _run_step(tmp_path, PUBLISH, ASSET_NAMES="baseline.tar.gz", ASSETS_STATUS="1")

    assert result.returncode == 1
    assert calls == [VIEW, ASSETS]


def test_failed_asset_download_prevents_publication_even_with_matching_bytes(tmp_path: Path) -> None:
    result, calls = _run_step(tmp_path, PUBLISH, ASSET_NAMES="baseline.tar.gz", DOWNLOAD_STATUS="1")

    assert result.returncode == 1
    assert calls == [VIEW, ASSETS, DOWNLOAD]


def test_unrelated_asset_names_do_not_skip_upload(tmp_path: Path) -> None:
    result, calls = _run_step(tmp_path, PUBLISH, ASSET_NAMES="baselineXtar.gz\nprefix-baseline.tar.gz\nbaseline.tar.gz.sha256")

    assert result.returncode == 0, result.stderr
    assert calls == [VIEW, ASSETS, UPLOAD, EDIT]


def test_draft_access_is_isolated_from_tagged_benchmark_code() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    validation, remainder = workflow.split("  validate-release:\n", 1)[1].split("  release-baseline:\n", 1)
    baseline, publication = remainder.split("  publish-baseline:\n", 1)

    assert "permissions:\n  contents: read\n" in workflow
    assert "permissions:\n      contents: write\n" in validation
    assert "uses:" not in validation
    assert "needs: validate-release\n" in baseline
    assert "ref: refs/tags/${{ inputs.release_tag }}\n" in baseline
    assert "permissions:" not in baseline
    assert "GH_TOKEN:" not in baseline
    assert "needs: release-baseline\n" in publication
    assert "permissions:\n      contents: write\n" in publication
    assert "actions/checkout@" not in publication
