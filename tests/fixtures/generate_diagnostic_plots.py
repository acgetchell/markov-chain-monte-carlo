# /// script
# requires-python = "==3.13.7"
# dependencies = ["arviz==0.22.0", "numpy==2.2.6", "scipy==1.16.2"]
# ///
"""Verify plot ranks and reranked prefix diagnostics with isolated pinned oracles.

``uv run --script tests/fixtures/generate_diagnostic_plots.py`` verifies retained
evidence; ``--emit`` prints new evidence for deliberate retention. ``--report PATH``
cross-checks an exported Rust report, including every prefix's actual inputs.
"""

import hashlib
import json
import math
import platform
import runpy
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

FIXTURES = Path(__file__).resolve().parent
ESS = runpy.run_path(str(FIXTURES / "generate_ess.py"))
CASE_NAMES = ("independent", "positive_correlation", "location_disagreement", "scale_disagreement", "counts")
PREFIXES = (7, 16, 31)
REFERENCE = {
    "python": "3.13.7",
    "arviz": "0.22.0",
    "numpy": "2.2.6",
    "scipy": "1.16.2",
    "rank_source": "https://docs.scipy.org/doc/scipy-1.16.2/reference/generated/scipy.stats.rankdata.html",
    "ess_source": "https://github.com/arviz-devs/arviz/blob/v0.22.0/arviz/stats/diagnostics.py",
    "source_fixture": "ess.json",
    "source_sha256": hashlib.sha256((FIXTURES / "ess.json").read_bytes()).hexdigest(),
    "rank_method": "average_all_original_prefix_draws",
    "relative_tolerance": 2e-11,
    "absolute_tolerance": 2e-13,
    "command": "uv run --script tests/fixtures/generate_diagnostic_plots.py --emit",
}


def reference(chains: list[list[float]]) -> dict[str, Any]:
    """Use SciPy ranks and the existing public-ArviZ reference harness."""
    from scipy.stats import rankdata  # ty: ignore[unresolved-import]  # noqa: PLC0415 - isolated pinned oracle, not a notebook dependency.

    draws = np.asarray(chains, dtype=np.float64)
    estimates = ESS["references"](chains)
    return {
        "ranks": rankdata(draws, method="average").reshape(draws.shape).tolist(),
        "estimates": estimates,
        "unavailable": ESS["unavailable"](chains, estimates),
    }


def fixture() -> dict[str, Any]:
    """Reuse retained exact inputs; independently rerank and recompute each prefix."""
    source = json.loads((FIXTURES / "ess.json").read_bytes())
    cases = []
    for name in CASE_NAMES:
        original = next(case["chains"] for case in source["cases"] if case["name"] == name)
        cases.append(
            {
                "name": name,
                "prefixes": [{"draws_per_chain": length, **reference([chain[:length] for chain in original])} for length in PREFIXES],
            }
        )
    return {"reference": REFERENCE, "cases": cases}


def check_number(actual: float | None, expected: float | None, label: str) -> None:
    """Fail on availability changes as well as numeric drift."""
    if actual is None and expected is None:
        return
    if (
        actual is None
        or expected is None
        or isinstance(actual, bool)
        or not math.isfinite(actual)
        or not math.isclose(actual, expected, rel_tol=2e-11, abs_tol=2e-13)
    ):
        message = f"{label}: actual={actual}, expected={expected}"
        raise ValueError(message)


def verify_fixture(saved: dict[str, Any]) -> None:
    """Keep provenance, case/prefix inventories, ranks, errors, and metrics fixed."""
    expected = fixture()
    if saved["reference"] != expected["reference"] or [case["name"] for case in saved["cases"]] != list(CASE_NAMES):
        message = "Plot fixture provenance or case inventory changed"
        raise ValueError(message)
    for case, fresh in zip(saved["cases"], expected["cases"], strict=True):
        if [prefix["draws_per_chain"] for prefix in case["prefixes"]] != list(PREFIXES):
            message = "Plot fixture prefix inventory changed"
            raise ValueError(message)
        for prefix, oracle in zip(case["prefixes"], fresh["prefixes"], strict=True):
            if prefix["ranks"] != oracle["ranks"] or prefix["unavailable"] != oracle["unavailable"] or prefix["estimates"].keys() != oracle["estimates"].keys():
                message = f"{case['name']}: rank, availability, or metric inventory changed"
                raise ValueError(message)
            for metric, value in prefix["estimates"].items():
                check_number(value, oracle["estimates"][metric], f"{case['name']} prefix {prefix['draws_per_chain']} {metric}")
        sys.stdout.write(f"Verified prefixes for {case['name']}\n")


def verify_report_inventory(report: dict[str, Any]) -> None:
    """Require a complete, supported report before any oracle comparison."""
    conventions = {
        "schema_version": 2,
        "workflow": "rank_ess_efficiency_v1",
        "rank_convention": "one_based_average_ties_all_original_prefix_draws_signed_zero_equal",
        "split_convention": "first_and_last_floor_N_over_2",
        "quantile_convention": "linear_type_7_all_original_prefix_draws",
    }
    if any(report.get(key) != value for key, value in conventions.items()):
        message = "Unsupported plot report schema or conventions"
        raise ValueError(message)
    lengths = report.get("prefixes")
    if not isinstance(lengths, list) or not lengths or any(not isinstance(length, int) or isinstance(length, bool) or length < 4 for length in lengths):
        message = "Report requires a nonempty positive prefix inventory"
        raise ValueError(message)
    if lengths != sorted(set(lengths)):
        message = "Report prefix inventory must be unique and increasing"
        raise ValueError(message)
    runs = report.get("runs")
    if not isinstance(runs, list) or not runs or len({run["run_id"] for run in runs}) != len(runs):
        message = "Report requires nonempty uniquely identified runs"
        raise ValueError(message)
    for run in runs:
        chains = run["chains"]
        if len(chains) < 2 or run["chain_identity"] != "original_unsplit" or len({chain["chain_id"] for chain in chains}) != len(chains):
            message = "Report requires distinct original chains"
            raise ValueError(message)
        if any(len(chain["draws"]) != lengths[-1] for chain in chains) or any(
            not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) for chain in chains for value in chain["draws"]
        ):
            message = "Report chains must be finite and match the full prefix length"
            raise ValueError(message)
        if [prefix["samples_per_chain"] for prefix in run["prefixes"]] != lengths:
            message = "Run prefix inventory disagrees with report"
            raise ValueError(message)
        for prefix in run["prefixes"]:
            if [row["chain_id"] for row in prefix["ranks"]] != [chain["chain_id"] for chain in chains]:
                message = "Rank rows do not preserve original chain identity"
                raise ValueError(message)
            if [quantile["probability"] for quantile in prefix["quantiles"]] != [0.05, 0.5, 0.95]:
                message = "Unsupported report quantile inventory"
                raise ValueError(message)


def verify_report_metric(result: dict[str, Any], oracle: dict[str, Any], metric: str, context: str) -> None:
    """Check numeric results and unavailable reasons against one oracle metric."""
    expected_error = oracle["unavailable"].get(metric)
    if expected_error is not None:
        if result.get("status") != "unavailable" or result["value"] is not None or result["reason"] != expected_error:
            message = f"{context} {metric}: unavailable reason disagrees"
            raise ValueError(message)
    else:
        if result.get("status") != "estimated":
            message = f"{context} {metric}: expected estimated status"
            raise ValueError(message)
        check_number(result["value"], oracle["estimates"][metric], f"{context} {metric}")


def verify_report(report: dict[str, Any]) -> None:
    """Cross-check the exported Rust estimates, not a notebook reimplementation."""
    verify_report_inventory(report)
    for run in report["runs"]:
        for prefix in run["prefixes"]:
            length = prefix["samples_per_chain"]
            chains = [chain["draws"][:length] for chain in run["chains"]]
            oracle = reference(chains)
            if any(name in oracle["unavailable"] for name in ("q05_ess", "q95_ess")):
                oracle["unavailable"]["tail_ess"] = "component_unavailable"
            ranks = [chain["ranks"] for chain in prefix["ranks"]]
            if ranks != oracle["ranks"]:
                message = f"{run['scenario']} prefix {length}: ranks disagree"
                raise ValueError(message)
            metrics = {
                "mean_ess": prefix["ess"]["mean"]["estimate"],
                "bulk_ess": prefix["ess"]["bulk"]["estimate"],
                "tail_ess": prefix["ess"]["tail"]["estimate"],
                "mean_mcse": prefix["mean_mcse"],
            }
            for label, quantile in zip(("q05", "q50", "q95"), prefix["quantiles"], strict=True):
                metrics[f"{label}_ess"] = quantile["ess"]["estimate"]
                metrics[f"{label}_mcse"] = quantile["mcse"]
                if f"{label}_mcse" not in oracle["unavailable"]:
                    check_number(quantile["quantile"], oracle["estimates"][f"{label}_value"], f"{run['scenario']} {label} value")
                elif quantile["quantile"] is not None or quantile["interval"] is not None:
                    message = f"{run['scenario']} {label}: unavailable quantile retains a point or interval"
                    raise ValueError(message)
            for metric, result in metrics.items():
                verify_report_metric(result, oracle, metric, f"{run['scenario']} prefix {length}")
            for label, component in (("q05", "lower"), ("q95", "upper")):
                if prefix["ess"]["tail"][component]["estimate"] != metrics[f"{label}_ess"]:
                    message = f"{run['scenario']} prefix {length}: tail component disagrees with quantile ESS"
                    raise ValueError(message)
        sys.stdout.write(f"Verified exported ranks and precision: {run['scenario']}\n")


def main() -> None:
    """Check exact oracle versions before verifying or emitting evidence."""
    if platform.python_version() != REFERENCE["python"] or any(version(name) != REFERENCE[name] for name in ("arviz", "numpy", "scipy")):
        message = "Plot oracle requires Python 3.13.7, ArviZ 0.22.0, NumPy 2.2.6, SciPy 1.16.2"
        raise ValueError(message)
    if sys.argv[1:] == ["--emit"]:
        sys.stdout.write(json.dumps(fixture(), indent=2, allow_nan=False) + "\n")
    elif len(sys.argv) == 3 and sys.argv[1] == "--report":
        verify_report(json.loads(Path(sys.argv[2]).read_bytes()))
    elif len(sys.argv) == 1:
        verify_fixture(json.loads((FIXTURES / "diagnostic_plots.json").read_bytes()))
    else:
        message = "Use no arguments, --emit, or --report PATH"
        raise ValueError(message)


if __name__ == "__main__":
    main()
