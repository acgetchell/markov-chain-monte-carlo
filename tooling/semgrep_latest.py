"""Temporarily select MCMC's exact latest scanner for shared v0.2.5 checks.

Remove this adapter through MCMC #222 after the published v0.2.6 fix for
acgetchell/research-repo-tools#178 is adopted.
Only the process-local expected version changes. The shared helper still checks
the Python baseline, installed distribution, absolute executable and native
scanner version; no installed package file or shared validation is replaced.
"""

import sys
from importlib.metadata import version

from research_repo_tools import semgrep_tool
from research_repo_tools.cli import main as tooling_main

SHARED_VERSION = "0.2.5"
SHARED_SCANNER_VERSION = "1.182.1"
SCANNER_VERSION = "1.183.0"


def main(argv: list[str] | None = None) -> int:
    """Run the unchanged shared gate with an exact, verified consumer version."""
    arguments = sys.argv[1:] if argv is None else argv
    if arguments not in (["setup"], ["semgrep", "check-fixtures"]):
        message = "Temporary Semgrep adapter supports only shared setup and fixture checks"
        raise ValueError(message)
    if version("research-repo-tools") != SHARED_VERSION:
        message = "Review and remove the temporary Semgrep adapter when adopting a new shared package"
        raise RuntimeError(message)
    if semgrep_tool.VERSION != SHARED_SCANNER_VERSION or (arguments != ["setup"] and version("semgrep") != SCANNER_VERSION):
        message = f"Temporary adapter requires Semgrep {SCANNER_VERSION} and the unmodified shared {SHARED_SCANNER_VERSION} expectation"
        raise RuntimeError(message)
    original = semgrep_tool.VERSION
    try:
        semgrep_tool.VERSION = SCANNER_VERSION
        return tooling_main(arguments)
    finally:
        semgrep_tool.VERSION = original


if __name__ == "__main__":
    sys.exit(main())
