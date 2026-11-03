# Benchmarking

The fixed-seed Criterion suite measures public MCMC transition and observation overhead.
Timings are empirical and host-dependent. Scientific workload contracts remain here;
the pinned shared package owns execution, evidence, release assets, and publication.

## Command Guide

| Command | Purpose |
| --- | --- |
| `just bench` | Measure the stepping suite into `target/criterion/` |
| `just bench-save-baseline NAME` | Save a named local baseline; use `last` for the conventional baseline |
| `just bench-compare [NAME]` | Compare saved samples; default baseline is `last` |
| `just bench-latest-vs-last [NAME]` | Measure, then compare the saved baseline |
| `just performance-local` | Measure the working tree against the latest published stable release |
| `just performance-prepare` | Infer release identities and capture the plan without measuring |
| `just performance-release` | Measure the prepared version against the latest published release and retain its complete run |
| `just performance-github-assets [CURRENT BASELINE]` | Compare authenticated release assets without measuring |
| `just performance-doc [--check or --preview]` | Rerender an existing shared report offline |
| `just performance-readme [--check or --preview]` | Publish the declared README selection from the retained complete run |
| `just performance-baseline TAG` | Measure a clean tagged checkout and package a shared release asset |
| `just performance OPERATION ...` | Forward an explicit operation to the shared performance CLI |

Saved comparison Markdown is written to `target/bench-reports/performance.md`.
Local and asset comparisons write `local` and `github-assets` files with
`.comparison.json`, `.evidence.json`, and `.md` suffixes in that directory.
Release preparation infers the prospective version from package metadata and
its baseline from published stable release chronology, excluding drafts and prereleases.
`just performance-prepare` captures that pair; `just performance-release` retains
`performance.run.json` and `performance.evidence.json` in `target/bench-reports/`
before promotion. Neither command requires explicit release tags.

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
just bench-save-baseline last
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

<a id="shared-reports-and-historical-evidence"></a>

## Shared reports and the first baseline

The performance-series reset retired earlier reports, converted evidence, README timings, and the legacy release-asset adapter. The current series starts at
v0.5.0 and retains the prepared v0.5.1 comparison described below. The diagnostic-backend experiment remains under `docs/performance/v1/experiments/` because it
records a current implementation decision, separate from the stepping release signal.

For local development, save the current checkout without modifying Git state:

```bash
just bench-save-baseline current-checkout
```

After making an implementation change, run `just bench-latest-vs-last current-checkout`. These samples live under `target/criterion/` and are local
Criterion baselines, not tagged release evidence. Keep the raw samples and record the host/toolchain and working-tree changes if using them in research.
The shared release-baseline command requires a clean checkout at an existing stable tag; it cannot certify this untagged checkout as that release.

The shared release series starts at v0.5.0. The first retained comparison measures the prepared v0.5.1 working tree against that tagged baseline;
the README labels the current timings as a prepared tree. Do not backfill retired releases. `just performance-check` accepts an empty inventory before
the first comparison and rejects release evidence or a publication selection without its current report.

The retained v1 reports and measurements stay byte-identical under `docs/performance/v1/`. New reports use `docs/performance/current.md`;
retained pairs and the generated archive index use the dedicated `docs/performance/runs/` directory. Evidence uses
the shared complete-run schema inside `research-repo-tools/evidence/v1` envelopes. The JSON pair is authoritative; shared CSV exports
preserve points, bounds, confidence levels when known, units, and coverage. Source/harness inventories use shared versioned fingerprints.

The strict shared inventory accepts only an empty new namespace before its first comparison. The v1 README and diagnostic experiments remain outside it.
The legacy guard and existing README publication selection remain until the verified v0.5.3 release supplies new-namespace evidence; see
[#221](https://github.com/acgetchell/markov-chain-monte-carlo/issues/221). Keep the README plan independently reviewed against the retained pair.

Once the first new-namespace comparison exists, reproduce it without measuring:

```bash
just performance-doc --check
just performance-doc
```

For the first promotion, supply the measured pair explicitly so the renderer can discover its evidence:

```bash
just performance-doc --payload target/bench-reports/performance.run.json --manifest target/bench-reports/performance.evidence.json
```

Promotion archives the previous shared report when present, rebases evidence links, refreshes the generated index, and publishes its outputs in one
recoverable transaction. Rerendering uses retained evidence and configured prose without GitHub or Cargo. Generated reports are checked by reproduction,
not rewritten by Markdown formatters; new archived pairs are immutable.

## Release Preparation and README Publication

Set `TAG` once and prepare package metadata with `just update-version "$TAG"` as described in [releasing](RELEASING.md).
Then use the same three-stage flow as Delaunay:

```bash
just performance-prepare
just performance-release
```

`tooling/performance-release.toml` infers the prospective package version and its latest published stable predecessor. Preparation records independent
source, harness and release assertions in ignored scratch storage without measuring. The schema-4 `tooling/measurement-release-signal.toml` uses the
current public stepping harness for both source exports, with a minimum release of v0.5.2. Each phase discovers all 22 cases, runs an untimed native
gate, and acquires one complete run with 100 samples per case, mean/median intervals and retained raw Criterion files. Fixture setup stays outside
timing according to the harness's existing steady-state and fresh-batch contracts. Changed source or harness identities require fresh preparation.

The legacy `tooling/benchmark.toml` remains the independent single-tag baseline-archive declaration used by GitHub release assets.
The frozen `tooling/performance-readme.toml` still describes the old README evidence; neither it nor the old measurements are rewritten during adoption.
For new publication, `tooling/performance-publication.toml` independently selects all 13 unthinned stepping, sampler and observation workflows regardless
of timing direction. The complete report also retains all nine thinning cases. Labels, revisions and fingerprints come from the verified preparation
and retained run, while the declaration owns row eligibility, units, references and the prospective `prepare` tag policy.

Review the full report, host/toolchain metadata, and coverage before selecting a README subset. Then run:

```bash
just performance-readme --preview
just performance-readme
```

The shared publisher checks the Cargo version, report identities, measured fingerprints, independently reviewed provenance, and exact existing-tag blobs.
It rechecks inputs before updating the marked README section, CSV and SVG together. It does not measure again. Keep linked artifacts in the release
commit before creating its tag.

### Measurement Order and Interpretation

The pinned shared 0.2.5 runner measures the whole baseline phase before the current phase. Its public schema and CLI do not support alternating or
randomizing phase order, and `order = "published"` controls release selection only. The retained phase samples and source/host provenance do not include
an observed acquisition-order or phase-timestamp receipt. Do not infer order control from matching CPU, OS, compiler or harness identities.

Treat ratios as descriptive sequential observations. Host temperature, frequency, power state and background load can drift between phases;
Criterion's within-phase intervals do not account for that systematic drift. Preserve absolute timings and raw evidence, and withhold comparisons
when actual conditions are incompatible. A causal release-performance or significance claim requires stronger experimental evidence.
Counterbalanced trials, retained realized order and session timestamps require published shared-runner support and consumer validation.
`tooling/performance-release-interpretation.md` carries these limitations into the generated README publication; the new report title states its order.

[Shared #180](https://github.com/acgetchell/research-repo-tools/issues/180) targets that capability for v0.2.6.
[MCMC #225](https://github.com/acgetchell/markov-chain-monte-carlo/issues/225) targets next-release adoption after verified publication.
Until that adoption passes, preserve the current sequential observations and their qualified interpretation.

## GitHub Release Assets

The manually dispatched `Release Benchmarks` workflow requires a mutable stable
draft. A read-only job measures a clean tagged checkout and packages
`markov-chain-monte-carlo-TAG-criterion-baseline.tar.gz`. Separate jobs install the exact
registry package for draft validation and attachment; they never check out
benchmark code. The benchmark job has no write credentials.

Shared archives contain the complete selected Criterion sample and provenance in
`sample.json` and `provenance.json`. They are compact retained estimates, not raw
Criterion time-series archives. Preserve local raw Criterion output separately when
needed for independent reanalysis. An identical attachment retry succeeds; different
bytes fail without overwrite. The workflow leaves the release in draft for review.
After attachment verification, the maintainer runs `just release-publish "$TAG"`
to trigger the separately approved crates.io upload; see [RELEASING.md](RELEASING.md).
The Actions artifact lasts 30 days; the release asset is durable.

`just performance-github-assets CURRENT BASELINE` downloads native shared assets via authenticated GitHub CLI, checks provider hashes when available,
and uses bounded extraction. It no longer accepts the former repository-specific legacy layout.

The first two retained native shared release assets create the first complete asset pair. Verify that pair before claiming
live asset adoption. No backfill runs implicitly. Runner hardware varies, so asset comparisons require manual compatibility review; the CLI does not certify
them as controlled same-host measurements.

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

`just bench` runs the fixed-seed stepping harness without selecting a release baseline. Filter Criterion benchmarks when investigating one path:

```bash
cargo bench --locked --bench stepping chain/step_mut
cargo bench --locked --bench stepping observing/
```

Use a profiler when the question is where time or allocations are spent. Release comparisons answer whether a stable workload changed; they do not identify
the cause.
