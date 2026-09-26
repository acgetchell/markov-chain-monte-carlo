# Benchmarking

The fixed-seed Criterion suite measures public MCMC transition and observation overhead.
Timings are empirical and host-dependent. Scientific workload contracts remain here;
the pinned shared package owns execution, evidence, release assets, and publication.

## Command Guide

| Command | Purpose |
| --- | --- |
| `just bench-latest` | Measure the stepping suite into `target/criterion/` |
| `just bench-save-last` | Save the conventional local baseline |
| `just bench-save-baseline NAME` | Save an explicitly named local baseline |
| `just bench-compare [NAME]` | Compare saved samples; default baseline is `last` |
| `just bench-latest-vs-last [NAME]` | Measure, then compare the saved baseline |
| `just performance-local` | Measure the working tree against the latest published stable release |
| `just performance-release [CURRENT BASELINE]` | Measure a release pair and promote shared evidence |
| `just performance-github-assets [CURRENT BASELINE]` | Compare authenticated release assets without measuring |
| `just performance-doc [--check or --preview]` | Rerender the current shared report offline |
| `just performance-readme [--check or --preview]` | Publish the explicitly configured README selection |
| `just performance-baseline TAG` | Measure a clean tagged checkout and package a shared release asset |
| `just performance OPERATION ...` | Forward an explicit operation to the shared performance CLI |
| `just bench` | Run the complete stepping harness |

Saved comparison Markdown is written to `target/bench-reports/performance.md`.
Local and asset comparisons write `local` and `github-assets` files with
`.comparison.json`, `.evidence.json`, and `.md` suffixes in that directory.
Release measurements write the `release` JSON pair before promotion.
Both explicit release tags must be supplied together. Selection uses publication
chronology, excluding drafts and prereleases.

## Release-Signal Workloads

`benches/stepping.rs` uses fixed seeds and public APIs. The release-signal set covers:

- by-value stepping and in-place acceptance/rollback paths;
- delayed-proposal accepted, rejected, and no-plan paths;
- fixed 100-step `Sampler` runs for by-value, in-place, and delayed proposals;
- fixed 100-step thinned runs for by-value, in-place, and delayed proposals at thinning intervals 1, 2, and 16;
- buffered observation, manual online accumulation, `OnlineStats`, and `BinningAnalysis`.

The harness measures transition and observation overhead, not distribution convergence. It has two deliberate fixture-lifecycle contracts:

- Chain-step, 100-step sampler, and buffered-observation benchmarks construct the chain, sampler, proposal, and seeded RNG once outside `b.iter`. Warmup and
  measured iterations advance that same state and RNG. These are steady-state latency or throughput measurements that exclude fixture construction.
- The thinned 100-step sampler groups follow that same persistent sampler/RNG lifecycle. Each timed iteration includes creation of the retained-state
  `Vec`, cloning of retained states, and destruction of the returned buffer.
- Manual online accumulation, `OnlineStats`, and `BinningAnalysis` use `iter_batched` to create a fresh chain and fixed-seed RNG outside each timed 100-step
  batch. Their timed work includes construction performed by the workflow itself, such as `Sampler` and accumulator construction.

The buffered workflow includes allocation and destruction of its returned `Vec`, because owning that buffer is part of the public operation. Preflight
checks establish that benchmarks named for accepted, rejected, rollback, or no-plan paths enter that path; the rejection fixture makes rejection
deterministic. Timing samples remain empirical and will vary with the host and surrounding load.

### Benchmark Contract Discipline

A benchmark name is a workload contract, not merely a display label. Keep a name stable only while its state lifecycle, RNG policy, setup boundary, step
count, target and proposal, expected outcome path, and output ownership remain comparable. Rename the benchmark when any of those dimensions changes
materially. The Python regression tests protect the two current lifecycle patterns.

The local report displays both benchmark-harness hash prefixes. Different hashes are an audit signal, not automatic proof that a comparison is invalid:
source may need to change to follow a compatible API. When hashes differ, review every shared name against this contract before accepting the report.

## Saved Samples and Local Measurement

```bash
just bench-save-last
# Make a change, then measure and compare.
just bench-latest-vs-last
just bench-compare last
just performance-local
```

Saved Criterion samples are local scratch data. Shared comparison keeps both complete
inventories and reports current-only and baseline-only rows. It uses median nanoseconds;
ratios are baseline/current and positive percentage reduction means lower current time.

`tooling/benchmark.toml` declares the stepping command, source/harness inventories,
tool probes, and compatibility fields. Local pairs require matching known OS,
architecture, and CPU identities. Different harnesses require review of shared names
against the workload contracts above. Unknown CPU identity fails this configured local
compatibility check; do not invent a value to make it pass.

Measurement explicitly permits temporary Git worktree creation/removal. The shared
runner captures exact binary patches and nonignored untracked files, rejects stale
inputs, executes with live output, and compares separate Criterion roots. Tags must
already exist locally; it never fetches. Commands execute trusted benchmark code.
An assistant subject to this repository's prohibition on Git mutations must leave
live worktree measurement to the maintainer.

## Shared Reports and Historical Evidence

New reports use `docs/performance/v1/current.md`; retained pairs and the generated
archive index live beside it. Evidence uses `research-repo-tools/criterion-comparison/v1`
inside `research-repo-tools/evidence/v1` envelopes. The JSON pair is authoritative;
shared CSV exports preserve points, bounds, confidence levels when known, units, and
coverage. Source/harness inventories use the shared versioned fingerprint framing.

`docs/PERFORMANCE.md`, `docs/archive/performance/`, and the existing README performance
section retain their historical bytes. The v0.4.2 CSV and provenance have verified
shared companions under the new path. Conversion records the original payload,
manifest, and configuration hashes; old source digests stay opaque under `legacy.*`.
They are never relabeled as newly captured shared fingerprints. No legacy writer or
renderer remains. The v0.4.1 report predates retained evidence and remains a historical
report without fabricated companions.

```bash
just performance-doc --check
just performance-doc
```

Rerendering reads retained evidence and configured prose without GitHub, Cargo, or
measurement. Promotion archives the previous shared report, rebases its evidence
links, refreshes the index, and publishes all outputs in one recoverable transaction.
Archived pairs are immutable; active evidence may change during release preparation.
Generated shared reports are checked by reproduction, not rewritten by formatters.

To promote explicitly saved shared evidence:

```bash
just performance-doc --payload target/bench-reports/release.comparison.json --manifest target/bench-reports/release.evidence.json
```

The v0.4.2 transition is complete: all reporting consumers use the verified shared
companions directly, and the one-time CSV conversion configuration is retired.
Historical originals stay immutable; consumer tests compare retained values and
provenance with those originals.

## Release Preparation and README Publication

`just performance-release` reads the current Cargo version. An unpublished newer
version uses working files against the latest published release. A published version
uses its tag against its predecessor. Explicit pairs measure their exact tags.
Same-label development comparisons cannot be promoted.

Before `just performance-readme`, edit `tooling/performance-readme.toml` to select the
reviewed evidence paths, pair-specific SVG and links, both independently verified
revisions/releases, and workload rows. Review the full report, host/toolchain metadata,
and coverage before choosing a README subset. This explicit selection replaces local
discovery and publisher code.

For a future release, keep `tag-policy = "prepare"`. It requires newer working-tree
evidence, matching source/harness fingerprints, and the configured inventories.
For tagged measurements, select `tag-policy = "existing"`. Existing tags must contain
the exact retained, referenced, linked, and generated bytes; filters and newline
conversion cannot establish equality. Both modes recheck inputs before publication.

```bash
just performance-readme --preview
just performance-readme
```

The checked-in selection records the historical pair as an audit reference and
deliberately refuses to publish converted legacy data as a freshly measured release.
The old README and SVG remain valid unchanged. A new measurement and reviewed selection
are required to publish the next README table and SVG. Keep linked artifacts in the
release commit before creating its tag. The publisher validates the Cargo version
and both shared-report identities and updates the marked section and SVG together.

## GitHub Release Assets

The manually dispatched `Release Benchmarks` workflow requires a mutable stable
draft. A read-only job measures a clean tagged checkout and packages
`markov-chain-monte-carlo-TAG-criterion-baseline.tar.gz`. Separate jobs install the exact
registry package for draft validation and attachment/publication; they never check out
benchmark code. The benchmark job has no write credentials.

Shared archives contain the complete selected Criterion sample and provenance in
`sample.json` and `provenance.json`. They are compact retained estimates, not raw
Criterion time-series archives. Preserve local raw Criterion output separately when
needed for independent reanalysis. An identical attachment retry succeeds; different
bytes fail without overwrite. Publication follows attachment verification and a fresh
draft-state check. The Actions artifact lasts 30 days; the release asset is durable.

`just performance-github-assets` downloads via authenticated GitHub CLI and checks
provider hashes when available, plus bounded extraction. Historical assets without
digests rely on GitHub HTTPS and repository access for authenticity. The read-only
`tooling/legacy-baseline.toml` supports old schema-1/2 archives without relabeling their
unknown provenance. Retire it when all selected release pairs have shared assets.

Releases through v0.4.2 have no baseline attachment. The first release containing this
workflow creates the first shared baseline; its successor enables the first complete
asset pair. Verify that pair before claiming live asset adoption. No backfill runs
implicitly. Runner hardware varies, so asset comparisons require manual compatibility
review; the CLI does not certify them as controlled same-host measurements.

Timing ratios and marginal bounds do not establish significance, convergence, mixing,
or effective sample size. Research comparisons need invariant-distribution or equilibrium
checks and observable-specific effective samples per second, autocorrelation, and
uncertainty for conventional and learned proposals.

## Broader Profiling

For distribution-level validation and mixing experiments, enable the `benchmarks` feature and use `BenchmarkTarget`.
The [reference distribution guide](benchmark_distributions.md) defines all four fixed two-dimensional targets, derives their analytical moments, and explains
the seeded example's ESS/second measurement scope. These experiments are separate from the Criterion stepping release signal described above.

`benches/autocorrelation.rs` provides focused diagnostic workloads, separate from the stepping release-signal suite. Its fixed-seed scalar AR(1) inputs
use coefficient 0.95, uniform innovations, and a 2,000-sample warm-up. ACF workloads vary sample count and inclusive maximum lag, including zero and one
to expose preparation cost. Timed calls include the public boundary checks, workspace/result allocation, computation, and destruction. Input generation
and fixture assertions run outside measurement. The integrated-time workload reuses one immutable ACF and measures only its window scan and result.
Every fixture's ACF is checked outside timing against an expanded raw-moment reference that does not reuse the production normalization or compensated sums.
The reference checks its conditioning and uses a conservative roundoff tolerance for these bounded fixtures.
These are computational workloads, not evidence of convergence or estimator accuracy.

The independent [backend comparison workspace](../benches/diagnostic_backends/README.md) evaluates arima ACF and ferromorphic IPS against the native
diagnostics. Its [decision report](performance/v1/experiments/diagnostic-backends.md) separates correctness, equivalent ACF timings, differing time-estimation
workflows, and dependency costs. These experimental results are separate from the curated release-signal reports.

Use the managed toolchain for a focused before/after comparison on the same host:

```bash
uv run --locked --group dev research-repo-tools toolchain run -- cargo bench --locked --bench autocorrelation -- --save-baseline before
# After changing the implementation, use identical Criterion options and features.
uv run --locked --group dev research-repo-tools toolchain run -- cargo bench --locked --bench autocorrelation -- --baseline before
```

`just bench` currently runs the same fixed-seed harness without selecting a release baseline. Filter Criterion benchmarks when investigating one path:

```bash
cargo bench --locked --bench stepping chain/step_mut
cargo bench --locked --bench stepping observing/
```

Use a profiler when the question is where time or allocations are spent. Release comparisons answer whether a stable workload changed; they do not identify
the cause.
