"""ESS evidence integrity checks, independent of the historical oracle environment."""

import json
import runpy
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture
def verifier(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, dict[str, Any]]:
    """Keep a fixed oracle response while mutating only the evidence under test."""
    namespace = runpy.run_path(str(FIXTURES / "generate_ess.py"))
    fixture = json.loads((FIXTURES / "ess.json").read_bytes())
    original = deepcopy(fixture)

    def reference_values(chains: list[list[float]]) -> dict[str, float | None]:
        return next(case["reference"].copy() for case in original["cases"] if case["chains"] == chains)

    verify = namespace["verify_fixture"]
    monkeypatch.setitem(verify.__globals__, "references", reference_values)
    return verify, fixture


def test_complete_evidence_is_verified(verifier: tuple[Any, dict[str, Any]], capsys: pytest.CaptureFixture[str]) -> None:
    verify, fixture = verifier
    before = deepcopy(fixture)
    verify(fixture)
    assert fixture == before
    assert capsys.readouterr().out.splitlines() == [f"Verified {case['name']}" for case in fixture["cases"]]


@pytest.mark.parametrize("metric_change", ["missing", "unknown"])
def test_metric_inventory_is_complete_even_for_unavailable_results(verifier: tuple[Any, dict[str, Any]], metric_change: str) -> None:
    verify, fixture = verifier
    constant = next(case for case in fixture["cases"] if case["name"] == "constant")
    if metric_change == "missing":
        del constant["reference"]["mean_ess"]
    else:
        constant["reference"]["unknown_ess"] = 1.0
    with pytest.raises(ValueError, match="constant: reference metric inventory changed"):
        verify(fixture)


@pytest.mark.parametrize("field", ["python_version", "numpy_version", "scipy_version", "version", "relative_tolerance", "absolute_tolerance", "source"])
def test_provenance_and_tolerances_are_fixed(verifier: tuple[Any, dict[str, Any]], field: str) -> None:
    verify, fixture = verifier
    fixture["reference"][field] = "changed"
    with pytest.raises(ValueError, match="provenance or tolerances changed"):
        verify(fixture)


@pytest.mark.parametrize("case_change", ["missing", "duplicate"])
def test_case_inventory_is_complete(verifier: tuple[Any, dict[str, Any]], case_change: str) -> None:
    verify, fixture = verifier
    if case_change == "missing":
        fixture["cases"].pop()
    else:
        fixture["cases"].append(deepcopy(fixture["cases"][0]))
    with pytest.raises(ValueError, match="case inventory changed"):
        verify(fixture)


def test_reference_drift_still_fails(verifier: tuple[Any, dict[str, Any]]) -> None:
    verify, fixture = verifier
    fixture["cases"][0]["reference"]["mean_ess"] *= 2.0
    with pytest.raises(ValueError, match="independent mean_ess: actual="):
        verify(fixture)


@pytest.mark.parametrize("corruption", [None, "provenance", "rank", "prefix", "metric", "value", "availability"])
def test_plot_prefix_evidence_rejects_drift(monkeypatch: pytest.MonkeyPatch, corruption: str | None) -> None:
    """Preserve the oracle response while altering one retained-evidence contract."""
    namespace = runpy.run_path(str(FIXTURES / "generate_diagnostic_plots.py"))
    fixture = json.loads((FIXTURES / "diagnostic_plots.json").read_bytes())
    original = deepcopy(fixture)
    verify = namespace["verify_fixture"]
    monkeypatch.setitem(verify.__globals__, "fixture", lambda: deepcopy(original))
    prefix = fixture["cases"][0]["prefixes"][0]
    if corruption == "provenance":
        fixture["reference"]["source_sha256"] = "changed"
    elif corruption == "rank":
        prefix["ranks"][0][0] += 1
    elif corruption == "prefix":
        prefix["draws_per_chain"] += 1
    elif corruption == "metric":
        del prefix["estimates"]["bulk_ess"]
    elif corruption == "value":
        prefix["estimates"]["mean_mcse"] *= 2
    elif corruption == "availability":
        prefix["unavailable"]["bulk_ess"] = "ConstantSamples"
    if corruption is None:
        verify(fixture)
        assert fixture == original
    else:
        expected = {
            "provenance": "provenance or case inventory changed",
            "rank": "rank, availability, or metric inventory changed",
            "prefix": "prefix inventory changed",
            "metric": "rank, availability, or metric inventory changed",
            "value": "mean_mcse: actual=",
            "availability": "rank, availability, or metric inventory changed",
        }
        with pytest.raises(ValueError, match=expected[corruption]):
            verify(fixture)


@pytest.fixture
def report_verifier(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, dict[str, Any]]:
    """Use retained independent responses to test the report verification boundary."""
    namespace = runpy.run_path(str(FIXTURES / "generate_diagnostic_plots.py"))
    original = json.loads((FIXTURES / "ess.json").read_bytes())["cases"][0]["chains"]
    references = json.loads((FIXTURES / "diagnostic_plots.json").read_bytes())["cases"][0]["prefixes"]

    def reference(chains: list[list[float]]) -> dict[str, Any]:
        length = len(chains[0])
        assert chains == [chain[:length] for chain in original]
        return deepcopy(next(prefix for prefix in references if prefix["draws_per_chain"] == length))

    def estimate(value: float) -> dict[str, Any]:
        return {"status": "estimated", "value": value}

    prefixes = []
    for oracle in references:
        values = oracle["estimates"]
        quantiles: list[dict[str, Any]] = [
            {
                "probability": probability,
                "ess": {"estimate": estimate(values[f"{label}_ess"])},
                "mcse": estimate(values[f"{label}_mcse"]),
                "quantile": values[f"{label}_value"],
                "interval": None,
            }
            for label, probability in (("q05", 0.05), ("q50", 0.5), ("q95", 0.95))
        ]
        ess = {name: {"estimate": estimate(values[f"{name}_ess"])} for name in ("mean", "bulk", "tail")}
        ess["tail"].update(lower=deepcopy(quantiles[0]["ess"]), upper=deepcopy(quantiles[2]["ess"]))
        prefixes.append(
            {
                "samples_per_chain": oracle["draws_per_chain"],
                "ranks": [{"chain_id": f"chain-{index}", "ranks": ranks} for index, ranks in enumerate(oracle["ranks"])],
                "ess": ess,
                "mean_mcse": estimate(values["mean_mcse"]),
                "quantiles": quantiles,
            }
        )
    report = {
        "schema_version": 2,
        "workflow": "rank_ess_efficiency_v1",
        "rank_convention": "one_based_average_ties_all_original_prefix_draws_signed_zero_equal",
        "split_convention": "first_and_last_floor_N_over_2",
        "quantile_convention": "linear_type_7_all_original_prefix_draws",
        "prefixes": [7, 16, 31],
        "runs": [
            {
                "run_id": "independent",
                "scenario": "independent",
                "chain_identity": "original_unsplit",
                "chains": [{"chain_id": f"chain-{index}", "draws": chain[:31]} for index, chain in enumerate(original)],
                "prefixes": prefixes,
            }
        ],
    }
    verify = namespace["verify_report"]
    monkeypatch.setitem(verify.__globals__, "reference", reference)
    return verify, report


def test_exported_report_verification_checks_every_prefix(report_verifier: tuple[Any, dict[str, Any]], capsys: pytest.CaptureFixture[str]) -> None:
    verify, report = report_verifier
    before = deepcopy(report)
    verify(report)
    assert report == before
    assert capsys.readouterr().out == "Verified exported ranks and precision: independent\n"


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("schema", "Unsupported plot report"),
        ("no_runs", "nonempty uniquely identified runs"),
        ("no_prefixes", "nonempty positive prefix inventory"),
        ("unordered_prefixes", "unique and increasing"),
        ("missing_run_prefixes", "Run prefix inventory"),
        ("dropped_prefix", "Run prefix inventory"),
        ("short_chain", "full prefix length"),
        ("nonfinite_chain", "finite"),
        ("wrong_quantile", "quantile inventory"),
        ("unavailable_status", "expected estimated status"),
        ("missing_status", "expected estimated status"),
        ("hidden_quantile_drift", "q05 value"),
    ],
)
def test_exported_report_verification_rejects_incomplete_or_inconsistent_evidence(
    report_verifier: tuple[Any, dict[str, Any]], corruption: str, message: str
) -> None:
    verify, report = report_verifier
    run = report["runs"][0]
    prefix = run["prefixes"][0]
    if corruption == "schema":
        report["schema_version"] = 1
    elif corruption == "no_runs":
        report["runs"] = []
    elif corruption == "no_prefixes":
        report["prefixes"] = []
    elif corruption == "unordered_prefixes":
        report["prefixes"] = [16, 7, 31]
    elif corruption == "missing_run_prefixes":
        run["prefixes"] = []
    elif corruption == "dropped_prefix":
        run["prefixes"].pop(0)
    elif corruption == "short_chain":
        run["chains"][0]["draws"].pop()
    elif corruption == "nonfinite_chain":
        run["chains"][0]["draws"][0] = float("nan")
    elif corruption == "wrong_quantile":
        prefix["quantiles"][0]["probability"] = 0.1
    elif corruption == "unavailable_status":
        prefix["ess"]["mean"]["estimate"]["status"] = "unavailable"
    elif corruption == "missing_status":
        del prefix["ess"]["mean"]["estimate"]["status"]
    else:
        prefix["quantiles"][0]["mcse"]["status"] = "unavailable"
        prefix["quantiles"][0]["quantile"] = 100.0
    with pytest.raises(ValueError, match=message):
        verify(report)
