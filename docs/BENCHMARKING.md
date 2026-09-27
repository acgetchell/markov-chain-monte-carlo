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
| `just performance-release CURRENT BASELINE` | Measure two releases from the new performance series and promote shared evidence |
| `just performance-github-assets [CURRENT BASELINE]` | Compare authenticated release assets without measuring |
| `just performance-doc [--check or --preview]` | Rerender an existing shared report offline |
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

<a id="shared-reports-and-historical-evidence"></a>

## Shared reports and the first baseline

Release performance tracking starts with the current development version. Earlier reports, converted evidence, README timings, and the legacy
release-asset adapter have been removed. The diagnostic-backend experiment remains under `docs/performance/v1/experiments/` because it records a
current implementation decision, separate from the stepping release signal.

For local development, save the current checkout without modifying Git state:

```bash
just bench-save-baseline current-checkout
```

After making an implementation change, run `just bench-latest-vs-last current-checkout`. These samples live under `target/criterion/` and are local
Criterion baselines, not tagged release evidence. Keep the raw samples and record the host/toolchain and working-tree changes if using them in research.
The shared release-baseline command requires a clean checkout at an existing stable tag; it cannot certify this untagged checkout as that release.

The next tagged release establishes the first shared baseline. Its successor supplies the first comparison in the new series. Do not measure an older
release just to fill the empty report slot. Until a pair exists, the README states that no comparison is available and `just performance-check` reports
the pending baseline. That check fails if release evidence or a publication selection exists without its current report.

New reports use `docs/performance/v1/performance.md`; retained pairs and the generated archive index live beside it. Evidence uses
`research-repo-tools/criterion-comparison/v1` inside `research-repo-tools/evidence/v1` envelopes. The JSON pair is authoritative; shared CSV exports
preserve points, bounds, confidence levels when known, units, and coverage. Source/harness inventories use shared versioned fingerprints.

Once the first comparison exists, reproduce it without measuring:

```bash
just performance-doc --check
just performance-doc
```

For the first promotion, supply the measured pair explicitly so the renderer can discover its evidence:

```bash
just performance-doc --payload target/bench-reports/release.comparison.json --manifest target/bench-reports/release.evidence.json
```

Promotion archives the previous shared report when present, rebases evidence links, refreshes the generated index, and publishes its outputs in one
recoverable transaction. Rerendering uses retained evidence and configured prose without GitHub or Cargo. Generated reports are checked by reproduction,
not rewritten by Markdown formatters; new archived pairs are immutable.

## Release Preparation and README Publication

For the first release after the reset, omit comparison and README publication commands; its tagged workflow records the baseline. For subsequent releases,
pass both tags explicitly to `just performance-release CURRENT BASELINE`, choosing the previous baseline from the new series. Automatic release discovery
does not know this repository's reset boundary. An unpublished newer Cargo version can use the working tree against the latest published baseline with
`just performance-release` once that latest release belongs to the new series. Same-label development comparisons cannot be promoted.

After reviewing a real pair, create `tooling/performance-readme.toml` for the shared `performance publish` command. No selection with placeholder
revisions is checked in. Configure:

- `schema = 1`, `document = "README.md"`, `unit = "ns"`, and the existing `<!-- PERFORMANCE:BEGIN -->` / `<!-- PERFORMANCE:END -->` markers.
- Retained comparison `payload` and evidence `manifest`, a pair-specific `svg`, `repository = "acgetchell/markov-chain-monte-carlo"`, and
  `prose-file = "tooling/performance-interpretation.md"`.
- Reviewed `rows` with benchmark names and readable labels, and `links` to the report, comparison, and provenance.
- Both `provenance.baseline` and `provenance.current`, each with an independently verified `revision` and `release`.
- `references` asserting the Cargo version and the report's current/baseline marker values against `version`, `tag`, and `previous-tag`.
- `tag-policy = "prepare"` for prospective release evidence, with `current-sources` and `current-harness` matching `tooling/benchmark.toml`.
  Use `"existing"` for tagged measurements; existing tags must contain the exact retained, referenced, linked, and generated bytes.

Review the full report, host/toolchain metadata, and coverage before selecting a README subset. Then run:

```bash
just performance-readme --preview
just performance-readme
```

The shared publisher checks the Cargo version, report identities, measured fingerprints, independently reviewed provenance, and exact existing-tag blobs.
It rechecks inputs before updating the marked README section and SVG together. Keep linked artifacts in the release commit before creating its tag.

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

`just performance-github-assets CURRENT BASELINE` downloads native shared assets via authenticated GitHub CLI, checks provider hashes when available,
and uses bounded extraction. It no longer accepts the former repository-specific legacy layout.

The next release containing this reset creates the first shared baseline; its successor enables the first complete asset pair. Verify that pair before claiming
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

`just bench` currently runs the same fixed-seed harness without selecting a release baseline. Filter Criterion benchmarks when investigating one path:

```bash
cargo bench --locked --bench stepping chain/step_mut
cargo bench --locked --bench stepping observing/
```

Use a profiler when the question is where time or allocations are spent. Release comparisons answer whether a stable workload changed; they do not identify
the cause.
