# /// script
# requires-python = "==3.13.7"
# dependencies = ["arviz==0.22.0", "numpy==2.2.6", "scipy==1.16.2"]
# ///
"""Reproduce retained posterior values in Python without overwriting the fixture.

Run with ``uv run --script tests/fixtures/generate_combined_rhat.py``.
The isolated historical Python/dependency pins are separate from repository tooling.
"""

import json
import math
import platform
import sys
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np

EXPECTED_CASES = frozenset(
    {
        "binary_folded_degeneracy",
        "discrete_counts",
        "heavy_tails",
        "location",
        "odd_median",
        "odd_pooled_count",
        "one_folded_half_constant",
        "scale_only",
        "well_mixed_control",
    }
)


@dataclass(frozen=True, slots=True)
class _RhatCase:
    """Owned input accepted by parse_cases; no mutable JSON aliases reach the oracle."""

    name: str
    chains: tuple[tuple[float, ...], ...]
    components: tuple[float | None, ...]


def parse_cases(cases: object) -> tuple[_RhatCase, ...]:
    """Parse the complete nine-regime corpus before importing the oracle."""
    message = "Combined R-hat case inventory changed or is malformed"
    if not isinstance(cases, list):
        raise TypeError(message)
    parsed: list[_RhatCase] = []
    required = {"name", "chains", "location", "folded", "combined"}
    for case in cases:
        if not isinstance(case, dict) or not required <= set(case) <= required | {"unavailable", "policy_difference"} or not isinstance(case["name"], str):
            raise ValueError(message)
        chains = case["chains"]
        if not isinstance(chains, list) or len(chains) < 2:
            raise ValueError(message)
        draws: list[tuple[float, ...]] = []
        for chain in chains:
            if not isinstance(chain, list) or len(chain) < 4 or (draws and len(chain) != len(draws[0])):
                raise ValueError(message)
            values = tuple(parse_number(value) for value in chain)
            draws.append(values)
        components = tuple(None if case[key] is None else parse_number(case[key], positive=True) for key in ("location", "folded", "combined"))
        parsed.append(_RhatCase(case["name"], tuple(draws), components))
    names = [case.name for case in parsed]
    if len(names) != len(EXPECTED_CASES) or frozenset(names) != EXPECTED_CASES:
        raise ValueError(message)
    return tuple(parsed)


def parse_number(value: object, *, positive: bool = False) -> float:
    """Accept finite scientific numbers, excluding JSON booleans and overflow."""
    message = "Combined R-hat case inventory changed or is malformed: expected finite numeric draws and positive references"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(message)
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(message) from error
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(message)
    return result


def reference_tolerance(value: object) -> float:
    """Bound fixture-controlled tolerance before evaluating the reference."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 5e-13:
        message = "Combined R-hat relative tolerance must be a number in (0, 5e-13]"
        raise ValueError(message)
    return float(value)


def verify_component(name: str, component: str, actual: float, expected: float | None, tolerance: float) -> None:
    """Require numerical agreement, including explicit unavailable references."""
    matches = math.isnan(actual) if expected is None else math.isfinite(actual) and math.isclose(actual, expected, rel_tol=tolerance, abs_tol=0.0)
    if not matches:
        message = f"{name} {component}: actual={actual!r}, reference={expected!r}, relative_tolerance={tolerance}"
        raise ValueError(message)


def main() -> None:
    """Print both components, their maximum, and the pinned runtime versions."""
    fixture: object = json.loads(Path(__file__).with_name("combined_rhat.json").read_bytes())
    if not isinstance(fixture, dict):
        message = "Combined R-hat fixture must be an object"
        raise TypeError(message)
    cases = parse_cases(fixture.get("cases"))
    reference = fixture.get("reference")
    reproduction = fixture.get("reproduction")
    if not isinstance(reference, dict) or not isinstance(reproduction, dict):
        message = "Combined R-hat fixture requires reference and reproduction objects"
        raise TypeError(message)
    tolerance = reference_tolerance(reference.get("relative_tolerance"))
    python_version = platform.python_version()
    if python_version != reproduction.get("python_version"):
        message = f"Python {reproduction.get('python_version')} required; found {python_version}"
        raise ValueError(message)
    for package in ("arviz", "numpy", "scipy"):
        expected = reproduction.get("version" if package == "arviz" else f"{package}_version")
        actual = version(package)
        if actual != expected:
            message = f"{package} {expected} required; found {actual}"
            raise ValueError(message)

    # Inline dependencies are checked in the isolated script environment as well.
    import arviz as az  # ty: ignore[unresolved-import]  # noqa: PLC0415 - isolated historical oracle, not needed for boundary tests.

    for case in cases:
        draws = np.asarray(case.chains, dtype=np.float64)
        location = float(az.rhat(draws, method="z_scale"))
        # Fold all original draws before ArviZ splits/ranks, preserving odd middles.
        folded_draws = np.abs(draws - np.median(draws))
        # An entirely constant folded pool is undefined; avoid evaluating 0/0.
        folded = math.nan if np.all(folded_draws == folded_draws.flat[0]) else float(az.rhat(folded_draws, method="z_scale"))
        combined = max(location, folded) if math.isfinite(location) and math.isfinite(folded) else math.nan
        for (component, actual), expected in zip((("location", location), ("folded", folded), ("combined", combined)), case.components, strict=True):
            verify_component(case.name, component, actual, expected, tolerance)
        sys.stdout.write(f"{case.name} location={location:.17g} folded={folded:.17g} combined={combined:.17g}\n")

    sys.stdout.write(f"Python {python_version}; ArviZ {version('arviz')}; NumPy {version('numpy')}; SciPy {version('scipy')}\n")


if __name__ == "__main__":
    main()
