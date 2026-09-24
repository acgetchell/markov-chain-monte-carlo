# Scalar diagnostic backend decision

Issue #73, evaluated on September 23, 2026. **Keep the native ACF and Geyer initial
monotone sequence (IMS) implementations for the current public API.** The
published candidate crates were actually compiled, called, tested, and timed.
No candidate dependency was added to the main library graph.

The [independent comparison workspace](../../../../benches/diagnostic_backends/README.md)
contains the pinned dependencies, prototype adapter, exact oracles, and
reproduction commands. [Retained measurement data](diagnostic-backends.json)
contains both runs, confidence intervals, raw Criterion samples, and source hashes.
This is local experimental evidence, not a curated release performance claim.

## Correctness

### ACF: native versus arima 0.3.0

Both compute the biased, sample-mean-centered ACF with common divisor N. The
comparison called the published `arima::acf::acf` in two ways: directly and
through an adapter preserving our validation and origin-shift/scale preparation.

- All 6,558 nonconstant eight-element traces over {-1, 0, 1} matched independent
  integer raw-moment ratios. Multiple lag prefixes were checked for both arima
  paths, and the native full ACF was checked against the same oracle.
- Both native and adapted arima passed 500 additional full-range binary64 traces
  against exact rational arithmetic. Maximum observed absolute errors were
  1.11e-16 and 2.22e-16, respectively. These observations are not universal bounds.
- Raw arima failed the 1e-13 tolerance or returned non-finite correlations in
  394 of those 500 deliberately hostile traces. This corpus is not representative
  of ordinary data and does not estimate a real-world failure rate.
- One-ULP outliers near `f64::MAX` and at subnormal scale have the analytic ACF
  rho[k] = -k/56 for k > 0. Native and adapted arima passed; raw arima returned
  NaNs at the nonzero lags. Opposite-sign maximum magnitudes also passed through
  the adapter's half-difference normalization.
- The adapter must retain checks for insufficient samples, invalid lag, non-finite
  values, and constant traces. Arima narrows N to `u32`, requiring an additional
  size limit or a native fallback above `u32::MAX`. This limit was established by
  source inspection, not by allocating a multi-billion-sample trace.

**Result:** adapted arima is numerically viable on the tested domain. Raw arima
does not satisfy our finite-input contract. Neither testing nor finite observed
errors establish correctness for every representable input.

### Integrated time: IMS versus ferromorphic 0.19.0 IPS

The published `integrated_autocorrelation_time_within` implements the initial
positive sequence (IPS), without IMS's cumulative-minimum adjustment. It also
requires at least eight samples, caps the lag at N/2, and flags capped/floored
results instead of returning our truncation and nonpositive-time errors.

The integer oracle checked each estimator against its own formula. Among the
eight-element fixtures, 632 had an exactly zero pair and were excluded from
strict cutoff comparisons because rounded signs at zero are not guaranteed;
their ACF values were still checked. Exactly zero final times were similarly
excluded from strict IPS floor-flag assertions. Both implementations matched
the checked nonboundary arithmetic and decision cases.

A further 1,000 fixed-seed, sixteen-element integer fixtures produced 18 cases
where both methods found a cutoff, both estimates were positive, and the
estimates differed. One exact example is:

```text
samples = [-3, -5, 12, -4, -12, -9, 17, 9, -10, 15, -16, 6, 7, -13, 15, 0]
maximum lag = 8; retained window = 5 for both
IPS = 146066/472048 ≈ 0.30943
IMS = 130618/472048 ≈ 0.27671
```

This difference is not an implementation defect. Both estimators also passed
AR(1) regression checks at coefficients 0, 0.9, and -0.5, using seeds 73, 74,
and 75 and 100,000 retained samples. The tolerances are deterministic regression
allowances, not calibrated confidence intervals or evidence of convergence.

Ferromorphic's unscaled arithmetic does not cover our extreme-input contract:
the maximum-magnitude outlier produced a flagged value of 1 rather than the
analytic 27/28; the subnormal outlier returned `NoVariation` despite nonzero
variation. Normalization could address this numerical range issue, but would
not change IPS into IMS or supply an API operating on an already computed ACF.

**Result:** valid alternative method on the checked ordinary inputs, but not a
replacement for the current IMS API. Do not translate its flags into unqualified
estimates or treat its finite-sample cap flag as a certified bound.

## Performance

Apple M4 Max, ARM macOS 27.0 (26A428), managed Rust 1.98.1. Both backends ran in
the same Criterion 0.8.2 executable with the default Cargo bench profile, no
native BLAS features, 30 samples, one-second warm-up, and three-second measurement
windows. Two runs used identical fixtures, commands, workload order, and timed
code. No concurrent build or test process was deliberately run during timing.
Fixture generation, independent preflight checks, and logging were outside
timing; input validation, allocation, computation, and result destruction were
inside. The host was not reserved or frequency-pinned.

### Equivalent ACF work

Numbers are Criterion **mean point estimates**, with run 1 / run 2 shown
separately. Full intervals and samples are in the adjacent JSON.

| Samples | Maximum lag | Native | Adapted arima | Interpretation |
|---:|---:|---:|---:|---|
| 4,096 | 0 | 19.14 / 19.87 µs | 8.15 / 8.31 µs | Adapter takes about 58% less time |
| 4,096 | 1 | 27.96 / 27.84 µs | 10.40 / 10.34 µs | Adapter takes about 63% less time |
| 4,096 | 32 | 89.83 / 88.50 µs | 78.95 / 74.97 µs | Adapter takes 12–15% less time |
| 20,000 | 2,000 | 20.80 / 20.71 ms | 26.25 / 31.37 ms | Native takes 21–34% less time |
| 100,000 | 400 | 22.15 / 22.18 ms | 27.48 / 28.28 ms | Native takes 19–22% less time |

The second adapted-arima 20,000/2,000 measurement drifted substantially; report
both runs rather than choosing the favorable one. The direction of the large
workload result repeated, including the raw arima lower-overhead comparator
(approximately 26 ms and 27 ms for the two large workloads). The smaller
workloads favor arima. There is no universal performance winner, cross-platform
claim, or aggregate speedup inferred from this table.

### Different time-estimation workflows

| Samples / maximum lag / AR coefficient | Native full ACF then IMS | Ferromorphic IPS from samples |
|---|---:|---:|
| 20,000 / 2,000 / 0 | 20.65 / 21.32 ms | 0.113 / 0.112 ms |
| 20,000 / 2,000 / 0.95 | 20.87 / 20.75 ms | 0.994 / 1.017 ms |
| 100,000 / 400 / -0.5 | 22.45 / 22.64 ms | 0.772 / 0.776 ms |

Ferromorphic stops computing autocovariances as soon as the cutoff is found.
The native path intentionally constructs every requested ACF lag before scanning
it. These timings therefore compare different work and, in general, different
estimates. They must not be presented as equivalent IMS speedups.

For the 20,000-sample coefficient-0.95 fixture, native IMS on an **existing ACF**
took 60.45 / 61.78 ns. Ferromorphic exposes no corresponding ACF-consuming API.
The early-stop strategy is a useful future direction for a dedicated native
time-from-samples operation, independently of dependency adoption.

## Simplicity and decision

Arima would remove the local covariance loops and four-lag batching, but retain
input validation and robust normalization. Compensated summation remains useful
to IMS. We would also need a policy for its `u32` sample-count restriction and
mapping any backend failure into our typed errors. Its resolved normal-edge
dependency graph contains 31 other packages, including forecasting/optimization,
an older rand family, and proc-macro support. This is graph size, not a claim that
all 31 would be newly added relative to every feature configuration. The
downloaded package contains an Apache-2.0 license file.

Ferromorphic has no dependencies and declares Apache-2.0 and MSRV 1.98. However,
it is a broad neuromorphic-computing library, and its diagnostic entry point owns
the raw-sample calculation. Our `Autocorrelation` deliberately stores only
computed correlations and releases its borrow of the original samples.
Delegating `integrated_time` would require retaining/requiring original data,
changing the estimator, or reimplementing the missing IMS operation anyway.

**Keep both native implementations.** Correctness of the normalized arima
prototype is encouraging, but its lower setup cost does not compensate for
slower representative large ACF workloads, remaining adapter responsibilities,
and dependency breadth here. Ferromorphic's time-only performance is compelling
for that distinct workflow; its method and data requirements do not fit our
existing API. The decision is specific to these versions and this use case.

## Validation and provenance

The independent package's seven test functions passed, as did all 22 benchmark
preflights and all-target Clippy with warnings denied. `PROPTEST_RNG_SEED=73 just check ci`
passed on macOS: 291 Rust tests, 192 doctests, 32 Python tests, notebook execution,
examples, and benchmark compilation. The retained evidence audit verified all
44 workload results, 1,320 raw timing samples, and final source fingerprints.

The production autocorrelation source was unchanged by this comparison:
SHA-256 `02bf15d1b3a9443869773e9ed8a33b49d55e745e965cfaef32edc1a0695d0277`.
The timed harness SHA-256 was
`f7403e28cb92fd969bdf3fae834931dacc66d7fb86e18c7df8c7a391a8e26793`.
After timing, an untimed integer oracle's pair traversal was updated to
`as_chunks` for Clippy, and fixture failures received explicit messages to meet
repository lint policy. Measured adapters, fixtures, and timed closures did not
change. The evidence records timed and final source hashes. Tests and all
benchmark preflights were rerun after that cleanup. Native execution was on macOS only.

Published implementation references:
[arima ACF](https://docs.rs/arima/0.3.0/src/arima/acf.rs.html),
[ferromorphic Bayesian diagnostics](https://docs.rs/ferromorphic/0.19.0/ferromorphic/bayes/index.html).
The local registry sources were inspected directly when documentation fetching
was unavailable. No version-control state was mutated.
