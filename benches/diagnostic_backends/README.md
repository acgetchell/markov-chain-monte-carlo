# Diagnostic backend comparison

An unpublished, independent Cargo workspace comparing the library's scalar ACF
and Geyer initial-monotone-sequence (IMS) time with arima 0.3.0 and ferromorphic
0.19.0. Candidate dependencies stay outside the main crate's dependency graph.
The lockfile pins the experiment, including its reference-arithmetic dependencies.

`arima_acf` is a prototype adapter that retains validation and overflow-safe
normalization before calling arima's actual published ACF. Its strings describe
experimental failures; adopting it would require mapping those failures into
the library's typed errors. It explicitly rejects counts above `u32::MAX`,
because arima narrows the sample count internally. The native API has no such
restriction. The raw arima benchmark is a lower-overhead comparison on bounded
fixtures, not a contract-compatible replacement.

Ferromorphic's API accepts original samples and implements the initial positive
sequence (IPS), with a separate `N/2` lag ceiling and flags for capped/floored
results. It cannot replace `Autocorrelation::integrated_time`, which consumes
an existing ACF and implements IMS. Its time-from-samples timings therefore
describe a different workflow and estimator; they are not equivalent-work
speedup measurements. Early stopping avoids computing an entire requested ACF.

Run from the repository root:

```bash
uv run --locked --group dev research-repo-tools toolchain run -- cargo nextest run --locked --release \
  --manifest-path benches/diagnostic_backends/Cargo.toml --target-dir target/diagnostic-backends --no-capture
uv run --locked --group dev research-repo-tools toolchain run -- cargo bench --locked \
  --manifest-path benches/diagnostic_backends/Cargo.toml --target-dir target/diagnostic-backends --bench comparison -- --test
uv run --locked --group dev research-repo-tools toolchain run -- cargo bench --locked \
  --manifest-path benches/diagnostic_backends/Cargo.toml --target-dir target/diagnostic-backends --bench comparison -- --save-baseline backend-run-1
uv run --locked --group dev research-repo-tools toolchain run -- cargo bench --locked \
  --manifest-path benches/diagnostic_backends/Cargo.toml --target-dir target/diagnostic-backends --bench comparison -- --save-baseline backend-run-2
```

Correctness comes first: the tests use exact integer and rational arithmetic,
analytic extreme-value fixtures, and fixed-seed AR(1) checks. They distinguish
rounded decisions near zero from estimator differences. The broad AR(1)
tolerances are deterministic regression allowances, not confidence intervals.
The extreme corpus deliberately overrepresents hostile magnitudes; failure
counts are not population failure-rate estimates.

Criterion uses 30 samples, one second of warm-up, and three seconds of measurement
per workload. Every ACF fixture gets an independent raw-moment preflight before
timing. Fixture generation and checks are excluded; allocations and output
destruction are included. Both implementations run in the same executable with
the same release profile and inputs. The two repeated runs preserve command,
features, and workload order; local timings do not establish cross-platform
performance. Avoid concurrent builds or tests while measuring.

Executables go to the requested root `target/diagnostic-backends`; Criterion
stores its results under this package's ignored `target/criterion` directory.
The repository's regular `just ci` does not run this independent experiment.
Format with `cargo fmt --manifest-path benches/diagnostic_backends/Cargo.toml --all`
and lint with the same manifest/target options and `cargo clippy --all-targets -- -D warnings`,
through the managed toolchain wrapper shown above.

See the retained [decision and measurements](../../docs/performance/v1/experiments/diagnostic-backends.md).
