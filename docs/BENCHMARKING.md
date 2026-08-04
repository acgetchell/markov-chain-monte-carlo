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
| `just performance-local` | Compare the current working tree with the latest stable published release in isolated worktrees | `target/bench-reports/performance.md` |
| `just performance-github-assets` | Compare the two latest durable GitHub Release benchmark assets without running Cargo | `target/bench-reports/github-assets-performance.md` |
| `just performance-release` | Generate and promote the release comparison, archiving the previous curated report | `docs/PERFORMANCE.md` and `docs/archive/performance/` |
| `just bench` | Run the full current Criterion harness for profiling or exploration | `target/criterion/` |

`just bench-compare <baseline>`, `just performance-github-assets <current-tag> <baseline-tag>`, and
`just performance-release <current-tag> <baseline-tag>` accept explicit baselines or release pairs. Both tags must be supplied together.

## Release-Signal Workloads

`benches/stepping.rs` uses fixed seeds and public APIs. The release-signal set covers:

- by-value stepping and in-place acceptance/rollback paths;
- delayed-proposal accepted, rejected, and no-plan paths;
- fixed 100-step `Sampler` runs for by-value, in-place, and delayed proposals;
- buffered observation, manual online accumulation, `OnlineStats`, and `BinningAnalysis`.

The harness measures transition and observation overhead, not distribution convergence. Fixture setup that establishes valid chain state occurs outside the
timed operation where Criterion's batching API permits it. Every measured iteration starts from the same chain state and `StdRng` seed, and the harness
checks that benchmarks named for accepted, rejected, rollback, or no-plan paths actually enter that path before timing. Timing samples remain empirical and
will vary with the host and surrounding load.

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
report is copied to `target/bench-reports/performance.md`.

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
if it is already published, the repair path measures that exact release tag against the preceding stable release. It writes `docs/PERFORMANCE.md`, archives
the previous curated pair under `docs/archive/performance/<current>-vs-<baseline>.md`, and refreshes the archive index. Existing archive files are never
overwritten.

For an explicit repair path:

```bash
just performance-release v0.6.0 v0.5.0
```

Review the report's release pair, comparable-row coverage, and environment before committing it. The Markdown report records Criterion medians and marginal
confidence intervals plus the source commits, toolchains, Criterion versions, and host identity available for the selected mode. Interval separation is
described as a relation, not as a paired statistical-significance claim.

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
