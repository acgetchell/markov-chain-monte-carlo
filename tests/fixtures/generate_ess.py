# /// script
# requires-python = "==3.13.7"
# dependencies = ["arviz==0.22.0", "numpy==2.2.6", "scipy==1.16.2"]
# ///
"""Generate/verify independent ESS and MCSE references without rewriting evidence.

Run ``uv run --script tests/fixtures/generate_ess.py`` to verify stored inputs.
Use ``--emit`` to print a newly generated fixture as JSON for deliberate retention.
The isolated historical environment is separate from repository Python tooling.
"""

import json
import math
import platform
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

REFERENCE = {
    "package": "arviz",
    "version": "0.22.0",
    "python_version": "3.13.7",
    "numpy_version": "2.2.6",
    "scipy_version": "1.16.2",
    "source": "https://github.com/arviz-devs/arviz/blob/v0.22.0/arviz/stats/diagnostics.py",
    "command": "uv run --script tests/fixtures/generate_ess.py --emit",
    "verify_command": "uv run --script tests/fixtures/generate_ess.py",
    "relative_tolerance": 2e-11,
    "absolute_tolerance": 2e-13,
    "seed": 188,
    "split": "first_and_last_floor_N_over_2",
    "quantile_scope": "all_original_draws_before_splitting",
    "quantile_method": "linear_type_7",
    "mcse_quantile_method": "beta_0.1586553_0.8413447_order_statistics",
    "policy": (
        "Reference numbers are retained even when this crate reports the named unavailable reason; constant sentinels and collapsed MCSE are not accepted."
    ),
}
CASE_NAMES = (
    "independent",
    "positive_correlation",
    "antithetic",
    "location_disagreement",
    "scale_disagreement",
    "heavy_tails",
    "counts",
    "binary",
    "odd_lengths",
    "constant",
    "stuck_halves",
    "minimum_length",
    "one_constant_half",
)


def references(chains: list[list[float]]) -> dict[str, float | None]:
    """Evaluate public ArviZ estimators on the original, separately identified chains."""
    import arviz as az  # ty: ignore[unresolved-import]  # noqa: PLC0415 - isolated historical oracle, not needed for contract tests.

    draws = np.asarray(chains, dtype=np.float64)
    result = {
        "mean_ess": float(az.ess(draws, method="mean")),
        "bulk_ess": float(az.ess(draws, method="bulk")),
        "tail_ess": float(az.ess(draws, method="tail")),
        "mean_mcse": float(az.mcse(draws, method="mean")),
    }
    for label, probability in (("q05", 0.05), ("q50", 0.5), ("q95", 0.95)):
        result[f"{label}_ess"] = float(az.ess(draws, method="quantile", prob=probability))
        result[f"{label}_mcse"] = float(az.mcse(draws, method="quantile", prob=probability))
        result[f"{label}_value"] = float(np.quantile(draws, probability, method="linear"))
    return {name: value if math.isfinite(value) else None for name, value in result.items()}


def unavailable(chains: list[list[float]], results: dict[str, float | None]) -> dict[str, str]:
    """Record deliberate constant/indicator/collapsed-interval policy differences."""
    draws = np.asarray(chains, dtype=np.float64)
    half = draws.shape[1] // 2
    split = np.concatenate((draws[:, :half], draws[:, -half:]), axis=0)
    errors: dict[str, str] = {}
    if np.all(split == split.flat[0]):
        errors.update(mean_ess="ConstantSamples", bulk_ess="ConstantSamples")
    elif np.all(np.ptp(split, axis=1) == 0):
        errors.update(mean_ess="NoWithinChainVariation", bulk_ess="NoWithinChainVariation")
    if "mean_ess" in errors:
        errors["mean_mcse"] = errors["mean_ess"]
    for label, probability in (("q05", 0.05), ("q50", 0.5), ("q95", 0.95)):
        indicators = split <= np.quantile(draws, probability, method="linear")
        if np.all(indicators == indicators.flat[0]):
            errors[f"{label}_ess"] = "DegenerateIndicator"
        elif np.all(np.ptp(indicators.astype(int), axis=1) == 0):
            errors[f"{label}_ess"] = "NoWithinChainVariation"
        if f"{label}_ess" in errors:
            errors[f"{label}_mcse"] = errors[f"{label}_ess"]
        elif results[f"{label}_mcse"] == 0:
            errors[f"{label}_mcse"] = "CollapsedQuantileInterval"
    return errors


def inputs() -> list[tuple[str, list[list[float]]]]:
    """Build fixed-seed regimes; retained exact inputs make RNG reproduction optional."""
    rng = np.random.default_rng(188)
    independent = rng.normal(size=(2, 64))
    cases = [("independent", independent.tolist())]
    for name, correlation in (("positive_correlation", 0.85), ("antithetic", -0.85)):
        innovations = rng.normal(size=(2, 64))
        draws = innovations.copy()
        for index in range(1, 64):
            draws[:, index] += correlation * draws[:, index - 1]
        cases.append((name, draws.tolist()))
    cases.extend(
        [
            ("location_disagreement", (independent + np.asarray([[0.0], [6.0]])).tolist()),
            ("scale_disagreement", (independent * np.asarray([[0.25], [8.0]])).tolist()),
            ("heavy_tails", rng.standard_cauchy(size=(2, 64)).tolist()),
            ("counts", rng.integers(0, 5, size=(3, 64)).astype(float).tolist()),
            ("binary", rng.integers(0, 2, size=(2, 64)).astype(float).tolist()),
            ("odd_lengths", np.insert(independent, 32, [100.0, -100.0], axis=1).tolist()),
            ("constant", np.ones((2, 8)).tolist()),
            ("stuck_halves", [[0.0] * 8 + [1.0] * 8, [2.0] * 8 + [3.0] * 8]),
            ("minimum_length", [[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0]]),
        ]
    )
    one_constant = independent.copy()
    one_constant[0, :32] = 7.0
    cases.append(("one_constant_half", one_constant.tolist()))
    return cases


def verify_fixture(fixture: dict[str, Any]) -> None:
    """Reject incomplete evidence or changed provenance before comparing numbers."""
    if fixture["reference"] != REFERENCE:
        message = "ESS reference provenance or tolerances changed"
        raise ValueError(message)
    if tuple(case["name"] for case in fixture["cases"]) != CASE_NAMES:
        message = "ESS fixture case inventory changed"
        raise ValueError(message)
    for case in fixture["cases"]:
        actual = references(case["chains"])
        if case["reference"].keys() != actual.keys():
            message = f"{case['name']}: reference metric inventory changed"
            raise ValueError(message)
        if unavailable(case["chains"], actual) != case["unavailable"]:
            message = f"{case['name']}: unavailable policy changed"
            raise ValueError(message)
        for metric, expected in case["reference"].items():
            value = actual[metric]
            if value is None and expected is None:
                continue
            if value is None or expected is None or not math.isclose(value, expected, rel_tol=2e-11, abs_tol=2e-13):
                message = f"{case['name']} {metric}: actual={value}, reference={expected}"
                raise ValueError(message)
        sys.stdout.write(f"Verified {case['name']}\n")


def main() -> None:
    """Verify exact environment pins, then emit or check every reference quantity."""
    pins = {"arviz": REFERENCE["version"], "numpy": REFERENCE["numpy_version"], "scipy": REFERENCE["scipy_version"]}
    if platform.python_version() != REFERENCE["python_version"] or any(version(package) != pin for package, pin in pins.items()):
        message = "ESS fixtures require Python 3.13.7, ArviZ 0.22.0, NumPy 2.2.6, SciPy 1.16.2"
        raise ValueError(message)
    if sys.argv[1:] == ["--emit"]:
        cases = []
        for name, chains in inputs():
            expected = references(chains)
            cases.append({"name": name, "chains": chains, "reference": expected, "unavailable": unavailable(chains, expected)})
        fixture = {"reference": REFERENCE, "cases": cases}
        sys.stdout.write(json.dumps(fixture, indent=2, allow_nan=False) + "\n")
        return
    if sys.argv[1:]:
        message = "Only --emit is supported"
        raise ValueError(message)
    verify_fixture(json.loads(Path(__file__).with_name("ess.json").read_bytes()))


if __name__ == "__main__":
    main()
