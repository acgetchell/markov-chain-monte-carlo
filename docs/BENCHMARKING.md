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
| `just performance-github-assets [CURRENT BASELINE]` | Compare historical two-file release assets without measuring |
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

The historical comparisons, their exact evidence, and the diagnostic-backend experiment are preserved in the immutable
[v0.5.2 archive](https://github.com/acgetchell/markov-chain-monte-carlo/tree/v0.5.2/docs/performance/v1).

For local development, save the current checkout without modifying Git state:

```bash
just bench-save-baseline current-checkout
```

After making an implementation change, run `just bench-latest-vs-last current-checkout`. These samples live under `target/criterion/` and are local
Criterion baselines, not tagged release evidence. Keep the raw samples and record the host/toolchain and working-tree changes if using them in research.
The shared release-baseline command requires a clean checkout at an existing stable tag; it cannot certify this untagged checkout as that release.

New reports use `docs/performance/current.md`. Retained runs and the generated archive index use `docs/performance/runs/`. Evidence uses
the shared complete-run schema inside `research-repo-tools/evidence/v1` envelopes. The JSON pair is authoritative; shared CSV exports
preserve points, bounds, confidence levels when known, units, and coverage. Source/harness inventories use shared versioned fingerprints.

For a working-tree measurement, the recorded Git revision identifies the base
commit. The run's `evidence.json` records the SHA-256 of each exported file in
`sources.current.context["source-export"]`. From the release preparation commit,
compare the Git blob hashes below with the `Cargo.toml` and `Cargo.lock` entries
in that export.

```bash
git cat-file blob HEAD:Cargo.toml | shasum -a 256
git cat-file blob HEAD:Cargo.lock | shasum -a 256
```

Before the first active report, `just performance-check` requires an empty archive and no companion outputs. After acquisition it reproduces the report.
Review the README publication plan against the selected evidence.

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
Prepare the release dependencies and the measurement plan before acquiring samples.

```bash
just release-dependency-builds
just performance-prepare --release-intent
just performance-release
```

`tooling/performance-release.toml` infers the prospective package version and its latest published stable predecessor. Preparation records independent
source, harness and release assertions in ignored scratch storage without measuring. The schema-6 `tooling/measurement-release-signal.toml` declares
two paired trials using the current public stepping harness, with a minimum release of v0.5.2. Each phase discovers all 22 cases, runs an untimed
native gate, and retains 100 samples per case, mean/median intervals and raw Criterion files. Fixture setup stays outside timing according to the
harness's steady-state and fresh-batch contracts. Changed source or harness identities require fresh preparation and acquisition.

The release-intent commands verify published dependency provenance. After an interruption, run `just performance-resume` with the same checkpoint.
The shared runner validates source, harness, host, inventory and schedule before timing. It preserves completed pairs and reacquires
the interrupted pair in its declared order. Use the retained report and README commands for rendering retries after successful acquisition.

The legacy `tooling/benchmark.toml` remains available for explicit single-tag experiments. Release attachment derives the accepted Step 1 archive.
For new publication, `tooling/performance-publication.toml` selects trial 1 and all 13 unthinned stepping, sampler and observation workflows regardless
of timing direction. The complete report retains both trials and all nine thinning cases. Labels, revisions and fingerprints come from the verified
preparation and retained run. The declaration owns trial and row selection, units, references and the prospective `prepare` tag policy.

Review the full report, host/toolchain metadata, and coverage before selecting a README subset. Then run:

```bash
just performance-readme --preview
just performance-readme
```

The shared publisher checks the Cargo version, report identities, measured fingerprints, independently reviewed provenance, and exact existing-tag blobs.
It rechecks inputs before updating the marked README section, CSV and SVG together. It does not measure again. Keep linked artifacts in the release
commit before creating its tag.

### Measurement Order and Interpretation

The v0.5.3 declaration uses the existing paired protocol. A new benchmark series
requires a separately reviewed scientific reason, such as a workload with no
compatible published predecessor.

#### Starting an approved series

1. Before measurement, put the suite declaration at `benchmark-series/ID.toml`.
   Use schema 5 and omit the schedule for the inaugural release. Set
   `benchmark-series.id`, `first-release`, and the exact native case IDs per
   suite in `workloads`. Retain the correctness gate, source and harness inputs,
   case discovery, sample count, intervals, and external acquisition owner.
2. Point the release, report, and README declarations at that measurement file.
   Set the release workflow to `mode = "infer-release"` and `order = "version"`.
   Select current-phase absolute timings with uncertainty in the inaugural README.
   Use current provenance and version/tag references. Omit the trial selector,
   baseline references, ratios, ratio SVG, and paired-trial interpretation.
3. Review `just performance-prepare` and its historical selection before running
   the release acquisition. Use fresh checkpoint, scratch, and owner paths for
   each release. Run the normal report, README, source preflight, and signed
   candidate checks. They use the original samples through attachment.
4. For the compatible successor, keep the series ID, declaration path, first
   release, suite contracts and case inventory. Restore schema 6, the two-trial
   alternating schedule, and the paired publication with trial 1 selected.
   Verify that preparation selects the inaugural tag and its exact commit.

RRT 0.3.2 rejects an accepted declared series once its own tag appears in
published history. Keep live series initialization pending a published fix
and consumer verification of the registry approval path.
[RRT #274](https://github.com/acgetchell/research-repo-tools/issues/274)

#### Interpreting the paired trials

Trial 1 acquires baseline then current, and trial 2 reverses that order. Each trial uses fresh source exports and output trees. The report retains
realized order and phase timestamps alongside each trial's source, host, toolchain and sample identities. The performance release configuration's
`order = "published"` still selects the predecessor from release history. Earlier sequential observations retain their original interpretation.

Compare both trials before drawing conclusions from the README's trial 1 ratios. Host temperature, frequency, power state and background load
can drift within or between pairs. Criterion's within-phase intervals describe timing variation within a phase. Two counterbalanced pairs can
show sensitivity to order but provide limited evidence about between-trial uncertainty or causal release effects. Preserve absolute timings
and raw evidence, and withhold comparisons when observed conditions are incompatible.

## GitHub Release Assets

Step 1's `performance-release` freezes the original acquisition, policy and
publisher receipt under an external owner before promoting its report. README
publication consumes that retained run separately. The owner and independently
reviewed full acceptance are described in [releasing](RELEASING.md#reusable-ci-evidence).

Before merge, run `just release-preflight "$(git rev-parse HEAD)"` on the clean
preparation commit. It compares the source and committed outputs with the
external acquisition and reports undeclared paths. See [release preparation](RELEASING.md)
for the README reproduction and package checks.

After merge, `just release-evidence` accepts the original measured source against
the final clean commit and signed native/default/all-feature CI. Changed relevant
source, harness, dependencies, policy, provenance or retained bytes block reuse.
Declared report/README outputs are accepted only through the shared publication
contract; original measured and final accepted source identities remain distinct.

`just release-stage` stages four inert assets against the authenticated mutable
draft. `just release-benchmarks "$TAG"` dispatches `Release Benchmarks` at that tag
with the independently recorded transfer. Checkout-free jobs fetch and attach
exact original IDs/hashes; the read-only checkout verifies and derives the archive
without sampling. The immutable verified handoff is selected by artifact ID,
digest and original attempt. Identical retries retain existing assets; conflicts
stop without overwrite.

The candidate archive retains the complete paired schedule, every original
Criterion raw file, source/host/toolchain provenance, and the final acceptance
binding. It preserves both trials and all 22 workloads rather than reducing the
release asset to the README subset. After review, `just release-publish "$TAG"`
publishes the immutable GitHub Release. The separate `just release-upload "$TAG"`
dispatches Cargo publication with the reviewed acceptance before deployment approval.
Actions handoffs last 30 days; release assets are durable. Preserve the external
owner and its independent backup for recovery.

Use `just performance-github-assets CURRENT BASELINE` for historical two-file
`sample.json`/`provenance.json` assets. It downloads with authenticated GitHub CLI,
checks provider hashes when available, and bounds extraction. Verify a complete
native two-file pair before claiming live adoption of that inspector. Review host
and workload compatibility independently before comparing its results.

For a complete candidate archive, use the policy and acceptance from the
independently reviewed selection to inspect its format, release, schema, and run
ID. Set `ARCHIVE`, `FINAL_POLICY`, and `ACCEPTANCE` to those three files.

```bash
uv run --locked --group dev research-repo-tools release evidence archive inspect \
  --archive "$ARCHIVE" --policy "$FINAL_POLICY" --acceptance "$ACCEPTANCE"
```

Use the retained complete run and report for performance comparisons. Keep both
trials and their provenance together.
[RRT candidate archive interfaces](https://github.com/acgetchell/research-repo-tools/blob/v0.3.2/docs/candidate-reuse-api.md#public-interfaces)

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
diagnostics. The experiment covers correctness, equivalent ACF timings, differing time-estimation workflows, and dependency costs. See the
[decision report](https://github.com/acgetchell/markov-chain-monte-carlo/blob/v0.5.2/docs/performance/v1/experiments/diagnostic-backends.md).

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
