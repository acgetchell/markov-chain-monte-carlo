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
from importlib.metadata import version
from pathlib import Path

# Inline dependencies are checked in the isolated script environment as well.
import arviz as az  # ty: ignore[unresolved-import]
import numpy as np


def verify_component(name: str, component: str, actual: float, expected: float | None, tolerance: float) -> None:
    """Require numerical agreement, including explicit unavailable references."""
    matches = math.isnan(actual) if expected is None else math.isfinite(actual) and math.isclose(actual, expected, rel_tol=tolerance, abs_tol=0.0)
    if not matches:
        message = f"{name} {component}: actual={actual!r}, reference={expected!r}, relative_tolerance={tolerance}"
        raise ValueError(message)


def main() -> None:
    """Print both components, their maximum, and the pinned runtime versions."""
    fixture = json.loads(Path(__file__).with_name("combined_rhat.json").read_bytes())
    reproduction = fixture["reproduction"]
    python_version = platform.python_version()
    if python_version != reproduction["python_version"]:
        message = f"Python {reproduction['python_version']} required; found {python_version}"
        raise ValueError(message)
    for package in ("arviz", "numpy", "scipy"):
        expected = reproduction["version"] if package == "arviz" else reproduction[f"{package}_version"]
        actual = version(package)
        if actual != expected:
            message = f"{package} {expected} required; found {actual}"
            raise ValueError(message)

    tolerance = fixture["reference"]["relative_tolerance"]
    for case in fixture["cases"]:
        draws = np.asarray(case["chains"], dtype=np.float64)
        location = float(az.rhat(draws, method="z_scale"))
        # Fold all original draws before ArviZ splits/ranks, preserving odd middles.
        folded_draws = np.abs(draws - np.median(draws))
        # An entirely constant folded pool is undefined; avoid evaluating 0/0.
        folded = math.nan if np.all(folded_draws == folded_draws.flat[0]) else float(az.rhat(folded_draws, method="z_scale"))
        combined = max(location, folded) if math.isfinite(location) and math.isfinite(folded) else math.nan
        for component, actual in (("location", location), ("folded", folded), ("combined", combined)):
            verify_component(case["name"], component, actual, case[component], tolerance)
        sys.stdout.write(f"{case['name']} location={location:.17g} folded={folded:.17g} combined={combined:.17g}\n")

    sys.stdout.write(f"Python {python_version}; ArviZ {version('arviz')}; NumPy {version('numpy')}; SciPy {version('scipy')}\n")


if __name__ == "__main__":
    main()
