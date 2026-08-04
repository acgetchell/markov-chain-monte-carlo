# Benchmarking

The Criterion suite protects stable, public MCMC workflows and separates three jobs: quick local regression checks, release-to-release evidence, and broader
profiling. Benchmark results are empirical and machine-dependent; they are not correctness tests or universal performance guarantees.

## Command Guide

| Command | Purpose | Output |
|:--------|:--------|:-------|
| `just bench-latest` | Run the fixed-seed release-signal suite on the current tree | `target/criterion/` |
| `just bench-latest-vs-last` | Measure the current tree and compare it with the saved `last` baseline | `target/bench-reports/performance.md` |
| `just bench-compare [baseline]` | Render existing measurements against `last` or an explicit saved baseline | `target/bench-reports/performance.md` |
| `just bench-save-baseline <tag>` | Measure and save a named local Criterion baseline | `target/criterion/**/<tag>/` |
| `just bench-save-last` | Save the conventional local `last` baseline | `target/criterion/**/last/` |
| `just performance-local` | Compare the current working tree with the latest stable published release in isolated worktrees | `target/bench-reports/performance.{csv,provenance.json,md}` |
| `just performance-github-assets` | Compare the two latest durable GitHub Release benchmark assets without running Cargo | `target/bench-reports/github-assets-performance.{csv,provenance.json,md}` |
| `just performance-release` | Persist, validate, and promote the release comparison; archive the previous curated report | `target/bench-reports/release-performance.{csv,provenance.json}` and `docs/PERFORMANCE.md` |
| `just performance-rerender` | Rebuild and promote the curated Markdown solely from saved release measurements | `docs/PERFORMANCE.md` and `docs/archive/performance/` |
| `just bench` | Run the full current Criterion harness for profiling or exploration | `target/criterion/` |

`just bench-compare <baseline>`, `just performance-github-assets <current-tag> <baseline-tag>`, and
`just performance-release <current-tag> <baseline-tag>` accept explicit baselines or release pairs. Both tags must be supplied together.

## Release-Signal Workloads

`benches/stepping.rs` uses fixed seeds and public APIs. The release-signal set covers:

- by-value stepping and in-place acceptance/rollback paths;
- delayed-proposal accepted, rejected, and no-plan paths;
- fixed 100-step `Sampler` runs for by-value, in-place, and delayed proposals;
- buffered observation, manual online accumulation, `OnlineStats`, and `BinningAnalysis`.

The harness measures transition and observation overhead, not distribution convergence. It has two deliberate fixture-lifecycle contracts:

- Chain-step, 100-step sampler, and buffered-observation benchmarks construct the chain, sampler, proposal, and seeded RNG once outside `b.iter`. Warmup and
  measured iterations advance that same state and RNG. These are steady-state latency or throughput measurements that exclude fixture construction.
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

## Local Saved Baselines

Use a saved baseline for iteration on one machine:

```bash
just bench-save-last

# Make a change, then rerun the release signal and render the comparison.
just bench-latest-vs-last
```

For a named checkpoint:

```bash
just bench-save-baseline before-observer-change
just bench-latest
just bench-compare before-observer-change
```

Criterion baselines below `target/` are local scratch data. Do not commit them or treat measurements from different machines as a controlled comparison.

## Current Tree Versus a Published Release

```bash
just performance-local
```

This command resolves the latest stable published GitHub Release, creates detached temporary worktrees for that tag and the current `HEAD`, applies the current
tracked diff and untracked files to the current worktree, and runs `stepping` in both. Each revision uses its own checked-in benchmark harness; the report
compares benchmark names present in both revisions and lists current-only or baseline-only rows as coverage notes. Temporary worktrees are removed after the
CSV and JSON provenance are saved, reloaded, validated, and rendered to `target/bench-reports/performance.md`.

Before 1.0, public APIs and benchmark harnesses may change in any release. A stable benchmark name means the workload remains intentionally comparable;
renamed, added, or removed workflows appear as coverage notes rather than forced comparisons. Coverage gaps are expected evidence of surface evolution, not
performance regressions. If a workload's meaning changes materially, give it a new benchmark name instead of comparing unlike operations.

This is the preferred pre-PR regression check because both measurements run on the same host while keeping build products and source trees isolated.

## GitHub Release Assets

The `Release Benchmarks` workflow runs when a GitHub Release is published. It saves the release tag as a Criterion baseline and attaches
`markov-chain-monte-carlo-<tag>-criterion-baseline.tar.gz` to the release. The workflow also uploads a 30-day Actions artifact for diagnostics; only the
GitHub Release attachment is the durable historical baseline. Each archive records the release tag and commit, runner operating system and architecture,
rustc version, and Criterion version beside the measurement data.

Compare the latest two assets without local measurements:

```bash
just performance-github-assets
```

Repair or regenerate a particular historical comparison with:

```bash
just performance-github-assets v0.6.0 v0.5.0
```

The command writes an analysis-friendly CSV and structured provenance beside the Markdown report. The native Criterion archives remain the durable release
assets because they retain the full measurement data needed for future tools; the CSV is the stable tabular interchange layer for analysis and report
rendering.

Releases published before the `Release Benchmarks` workflow was introduced are not backfilled. Asset-to-asset comparisons therefore become available after
two releases have been published with the workflow. The asset comparison fails clearly when either archive or requested Criterion sample is absent. Archives
are checked for traversal and link entries before extraction.

This rollout is prospective. The first post-rollout release creates the initial durable asset; the second creates the first complete historical pair and is
the point at which the end-to-end asset workflow can be considered adopted. For example, if the first assets are `v0.4.1` and `v0.4.2`, the first durable
comparison is `v0.4.2` against `v0.4.1`. Pre-1.0 API changes remain allowed: changed workloads should receive new benchmark names and appear as coverage
changes rather than being forced into invalid comparisons.

GitHub-hosted runner hardware can vary between releases. Historical asset reports are useful release records, but a same-host local comparison is stronger
evidence for attributing a change to the code.

## Curated Release Report

During release preparation, after the package version is final:

```bash
just performance-release
```

The command reads the current tag from `Cargo.toml`. If that version is not published, it compares the patched working tree against the latest stable release;
if it is already published, the repair path measures that exact release tag against the preceding stable release. It first writes
`target/bench-reports/release-performance.csv` and `target/bench-reports/release-performance.provenance.json`, reloads and validates both files, and only then
renders and promotes `docs/PERFORMANCE.md`. It archives the previous curated pair under
`docs/archive/performance/<current>-vs-<baseline>.md` and refreshes the archive index. Existing archive files are never overwritten.

For an explicit repair path:

```bash
just performance-release v0.4.1 v0.4.0
```

After a successful measurement, rerender the report without GitHub access, Git worktrees, or Cargo:

```bash
just performance-rerender
```

Rerendering reads only the release CSV and its adjacent provenance file. It rejects reordered or malformed rows, mismatched release metadata, changed report
settings, and any CSV whose SHA-256 digest no longer matches the sidecar.

Review the report's release pair, comparable-row coverage, environment, harness hashes, and lifecycle contracts before committing it. The Markdown report
records Criterion medians and marginal confidence intervals plus the source commits, toolchains, Criterion versions, and host platform and architecture
available for the selected mode. The displayed marginal intervals are not a paired statistical-significance claim.

## Saved Comparison Schema

The versioned CSV stores one deterministic row per benchmark name. Each row is classified as `comparable`, `current_only`, or `baseline_only` and retains the
point estimate and available confidence bounds for both samples. Missing-sample fields are blank by construction. This compact dataset is directly usable by
standard CSV tooling; Parquet would add conversion and dependency overhead without helping these small release-signal tables.

The adjacent JSON records the release pair, report settings, exact benchmark commands, commits, host and Rust/Criterion versions, and SHA-256 hashes for
`Cargo.lock` and `benches/stepping.rs`. A combined source digest covers `Cargo.toml`, `Cargo.lock`, `rust-toolchain.toml`, `benches/stepping.rs`, and every Rust
source file. The sidecar also binds itself to the CSV by digest. GitHub-asset comparisons record the metadata available inside the native Criterion archives
and explicitly leave local-command and source-hash fields absent.

Files below `target/bench-reports/` are ignored local release work products. Keep the CSV and JSON with any external research record that cites the local
same-host comparison. After publication, the native Criterion tarballs attached to GitHub Releases are the durable repository-owned measurements from which
asset-to-asset CSV and Markdown reports can be regenerated.

These timings are evidence about implementation overhead on the recorded hosts, not proof of sampler convergence, mixing quality, or scientific efficiency.
For Praxis comparisons, validate invariant distributions or equilibrium observables first, then report observable-specific effective samples per second,
integrated autocorrelation time, and uncertainty alongside these engineering timings. Apply the same diagnostic and scientific measurements to conventional
and learned proposals.

The repository intentionally starts the curated history prospectively. Earlier releases remain usable as same-host local baselines, but are not represented
as durable assets or retrospective committed reports. A release-preparation report may therefore exist before two durable assets do; that local report does
not substitute for verifying the first asset-to-asset pair after the second post-rollout release.

## Broader Profiling

`just bench` currently runs the same fixed-seed harness without selecting a release baseline. Filter Criterion benchmarks when investigating one path:

```bash
cargo bench --locked --bench stepping chain/step_mut
cargo bench --locked --bench stepping observing/
```

Use a profiler when the question is where time or allocations are spent. Release comparisons answer whether a stable workload changed; they do not identify
the cause.
