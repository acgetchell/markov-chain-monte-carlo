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
