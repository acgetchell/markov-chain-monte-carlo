# Code Organization Guide

Detailed file and module ownership guide for the `markov-chain-monte-carlo` crate: which file owns what, and where new code usually belongs.

This document complements two related files:

- [`CONTRIBUTING.md`](../CONTRIBUTING.md) — human contributor workflow (setup, tooling, testing, PR process, release).
- [`AGENTS.md`](../AGENTS.md) — canonical rules for AI assistants (git/edit/validation policy, documentation-generation rules).

For contributor setup, test commands, and external tooling, see `CONTRIBUTING.md`. For agent-specific rules, see `AGENTS.md`. This file is the detailed
code/file map: keep ownership and placement guidance here, and keep contributor workflow details elsewhere.

## Full checkout tree

This tree reflects the tracked files in a fresh GitHub checkout. Update it whenever adding, removing, renaming, or moving tracked files or directories.

```text
.
├── .codecov.yml
├── .coderabbit.yml
├── .config/
│   └── nextest.toml
├── .gitattributes
├── .github/
│   ├── CODEOWNERS
│   ├── actions/
│   │   └── setup-toolchain/
│   │       └── action.yml
│   ├── dependabot.yml
│   └── workflows/
│       ├── audit.yml
│       ├── ci.yml
│       ├── codecov.yml
│       ├── codeql.yml
│       ├── dependabot-auto-merge.yml
│       ├── rust-clippy.yml
│       ├── release-benchmarks.yml
│       ├── semgrep-sarif.yml
│       └── zizmor.yml
├── .gitignore
├── .python-version
├── .taplo.toml
├── AGENTS.md
├── CHANGELOG.md
├── CITATION.cff
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── Cargo.lock
├── Cargo.toml
├── LICENSE
├── README.md
├── REFERENCES.md
├── SECURITY.md
├── benches/
│   └── stepping.rs
├── clippy.toml
├── docs/
│   ├── BENCHMARKING.md
│   ├── PERFORMANCE.md
│   ├── RELEASING.md
│   ├── archive/
│   │   └── performance/
│   │       ├── README.md
│   │       ├── v0.4.1-vs-v0.4.0.md
│   │       ├── v0.4.2-vs-v0.4.1.csv
│   │       ├── v0.4.2-vs-v0.4.1.provenance.json
│   │       └── v0.4.2-vs-v0.4.1.svg
│   ├── archives/
│   │   └── changelog/
│   │       ├── 0.1.md
│   │       ├── 0.2.md
│   │       └── 0.3.md
│   ├── assets/
│   │   └── ising_energy_trace.png
│   ├── code_organization.md
│   ├── dev/
│   │   ├── rust.md
│   │   ├── shared-changelog-pilot.md
│   │   └── shared-maintenance-migration.md
│   ├── proposal_validation.md
│   ├── roadmap.md
│   ├── reviewer_guide.md
│   └── scientific_basis.md
├── dprint.json
├── examples/
│   ├── additive_target_bias.rs
│   ├── delayed_chunked_telemetry.rs
│   ├── detailed_balance.rs
│   ├── ising_1d.rs
│   ├── iterator_sampling.rs
│   └── normal_1d.rs
├── justfile
├── notebooks/
│   └── ising_trace_analysis.ipynb
├── pyproject.toml
├── rumdl.toml
├── rust-toolchain.toml
├── rustfmt.toml
├── scripts/
│   ├── README.md
│   ├── archive_performance.py
│   ├── bench_compare.py
│   ├── publish_performance_readme.py
│   ├── update_release_version.py
│   └── tests/
│       ├── __init__.py
│       ├── test_archive_performance.py
│       ├── test_bench_compare.py
│       ├── test_justfile_discoverability.py
│       ├── test_notebooks.py
│       ├── test_publish_performance_readme.py
│       ├── test_release_benchmarks.py
│       ├── test_release_check.py
│       ├── test_tooling_package.py
│       └── test_update_release_version.py
├── semgrep.yaml
├── src/
│   ├── chain.rs
│   ├── diagnostics.rs
│   ├── error.rs
│   ├── lib.rs
│   ├── observable.rs
│   ├── sampler.rs
│   ├── statistics.rs
│   ├── testing.rs
│   └── traits.rs
├── tests/
│   ├── public_api.rs
│   ├── proptest_chain.rs
│   ├── proptest_validators.rs
│   └── semgrep/
│       ├── benches/
│       │   ├── erased_error.rs
│       │   ├── typed_error.rs
│       │   └── unwrap_expect.rs
│       ├── examples/
│       │   ├── deep_import.rs
│       │   ├── erased_error.rs
│       │   ├── typed_error.rs
│       │   └── unwrap_expect.rs
│       ├── github-actions/
│       │   └── workflow_actions.yml
│       ├── docs/
│       │   └── check_fix_order.md
│       ├── scripts/
│       │   ├── python_portability.py
│       │   └── tests/
│       │       └── python_exceptions.py
│       └── src/
│           ├── doctests/
│           │   ├── erased_error.rs
│           │   ├── typed_error.rs
│           │   └── unwrap_expect.rs
│           └── project_rules/
│               ├── algebraic_float.rs
│               └── rust_style.rs
├── ty.toml
├── typos.toml
└── uv.lock
```

## Repository areas

- `src/` — core library modules and crate-level documentation. The detailed source file map is below.
- `examples/` — complete runnable workflows that demonstrate public APIs.
- `notebooks/` — notebook consumers for example-generated artifacts such as exported diagnostic traces.
- `tests/` — integration tests, property-based tests named `tests/proptest_*.rs`, and project-rule tests including Semgrep fixtures under `tests/semgrep/`.
- `benches/` — Criterion benchmarks for stepping, sampler loops, and observing overhead.
- `docs/` — topic guides, release benchmark methodology and archives, and release procedures that support the public API documentation without duplicating
  README or crate-level contract material. `docs/PERFORMANCE.md` is the generated curated release report, while `docs/archive/performance/` owns its tracked
  CSV/JSON evidence and older reports; update them together through `just performance-release` or `just performance-doc`, not by hand. The archived
  pre-evidence v0.4.1 report retains its legacy warning; the current curated report is backed by tracked evidence. Neither report should be hand-edited.
  `just performance-readme` owns the marked README performance section and pair-specific SVGs beside the retained evidence; it never measures benchmarks.
- `docs/archives/changelog/` — completed minor-series release history generated by the shared changelog workflow; never hand-edit.
- `docs/assets/` — tracked images and other documentation media referenced from README or topic guides.
- `scripts/` — Python helpers for benchmark comparison and report promotion, shared-tool integration, consumer release policy, and retained-data README
  publication. `update_release_version.py` composes shared metadata preparation with MCMC policies in one validated transaction; `publish_performance_readme.py`
  retains the historical renderers and MCMC eligibility policy while using shared publication plans. `archive_performance.py` adapts retained CSV/provenance
  schemas, workload selection, and legacy source digests to shared comparison, extraction, and transaction APIs. Common dependency updates, tagging, reviews,
  notebook infrastructure, and fixture validation use the pinned shared CLI directly from `justfile`.
- `.gitattributes` — prevents checkout text conversion of byte-sensitive retained reports and evidence.
- Root configuration files (`Cargo.toml`, `rust-toolchain.toml`, `.python-version`, `pyproject.toml`, `uv.lock`, `justfile`, `semgrep.yaml`, `dprint.json`,
  `rumdl.toml`, `typos.toml`, `.config/nextest.toml`) — build, validation, formatting, release, and project-rule configuration.

## Library module file map

### `src/lib.rs`

The crate root wires the public module layout together. It re-exports the core modules and defines the `prelude` for ergonomic user imports. It includes
`README.md` at the top of rustdoc builds with `include_str!`, then appends crate-level `//!` programming-contract documentation for docs.rs (see
[`AGENTS.md` § Documentation generation](../AGENTS.md#documentation-generation)).

Keep `src/lib.rs` small. Plain project orientation belongs in the README or `docs/`; `src/lib.rs` should stay focused on API semantics, numerical behavior,
and programming contracts. New public surface should usually live in a focused module first, then be re-exported from the crate root or prelude only when it is
part of the intended user API.

### `src/error.rs`

Defines `McmcError`, the crate error type for invalid log-probabilities and proposal ratios.

Add new variants conservatively. `McmcError` is `#[non_exhaustive]`, but error changes still affect user matching and documentation.

### `src/diagnostics.rs`

Defines trace-recording APIs for reusable MCMC diagnostics:

- `ChainId` for stable multi-chain identifiers
- `TraceStepOutcome` for accepted, rejected-proposal, and no-proposal outcomes
- `TraceRecord` for one post-step row
- `TraceRecorder` for recording one chain into a shared-column trace
- `Trace` for multi-chain numeric observable rows and CSV export

Keep diagnostics independent from plotting and notebook rendering. Domain observables should enter this module as numeric columns; visualization and
post-processing belong in notebooks or downstream tools.

### `src/observable.rs`

Defines measurement APIs and collection helpers:

- `Observable<S>` for infallible measurements
- `TryObservable<S>` for fallible measurements
- `ObservedStepError<StepError, ObservationError>` to keep sampling failures and measurement failures orthogonal
- `SampleBuffer<T>` for simple in-memory observation collection

Observables are shared across proposal workflows, so the core observable traits, buffer, and ordinary streaming result aliases belong in the shared prelude.
Highly specialized workflow result aliases should stay at the crate root unless a prelude needs them for ordinary examples or doctests.

### `src/traits.rs`

Defines user extension points:

- `Target<S>` for target distributions through `log_prob(&S) -> f64`
- `Proposal<S>` for by-value proposals
- `ProposalMut<S>` for in-place proposals with rollback through an undo token
- `DelayedProposal<S>` for accept-before-mutation workflows whose plans describe concrete transitions and whose commits must be failure-atomic on error

This is the right place for small, fundamental traits that users implement for their own state spaces. Prefer borrowed parameters by default.

### `src/chain.rs`

Contains `Chain<S>`, the core Metropolis-Hastings state machine.

This module owns:

- current state and current log-probability
- accepted/rejected counters
- invariant-preserving delayed-step telemetry and its read-only accessors
- by-value `step`
- in-place `step_mut`
- state accessors and replacement helpers
- acceptance-rate and counter utilities

Algorithmic correctness belongs here. Higher-level convenience APIs should only move into `Chain` when they are fundamental to a single chain's state.

### `src/sampler.rs`

Contains `Sampler<S, T, P, R>`, an ergonomic wrapper that bundles a chain with its target distribution, proposal, and RNG.

This module owns:

- single-step forwarding methods
- bulk `run` and `run_mut` loops
- `ThinningInterval` parsing and shared thinned-run loops
- observing variants that measure derived quantities after sampling steps
- by-value `Iterator` support
- access to the bundled `Chain`

Use `Sampler` for workflow ergonomics; use `Chain` for the core transition logic.

### `src/statistics.rs`

Defines streaming statistics helpers for post-processing observed samples:

- `OnlineStats` for one-pass means and variances
- `BinningAnalysis` and `BinningEstimate` for correlated-sample uncertainty estimates
- `StatisticsError` for invalid inputs and insufficient data

Statistics helpers are ordinary public API, but they should stay independent of chain mutation and proposal mechanics.

### `src/testing.rs`

Contains test-facing validation utilities for proposal development.

Detailed-balance helpers empirically check discrete by-value, in-place, and delayed proposal transitions by sampling forward/reverse moves and comparing
estimated Metropolis-Hastings transition flows. Keep these helpers explicit at the crate root because they are test-facing diagnostics rather than everyday
sampling imports.

## Examples

New examples go in `examples/`. Each is a complete, runnable workflow:

- `examples/additive_target_bias.rs` — additive model and bias log-weight composition with `AdditiveTarget`.
- `examples/detailed_balance.rs` — by-value, in-place, delayed, and batch detailed-balance checks.
- `examples/normal_1d.rs` — simple by-value random-walk sampler.
- `examples/ising_1d.rs` — in-place mutation with rollback plus energy/magnetization trace CSV export.
- `examples/iterator_sampling.rs` — by-value `Sampler` iterator API.
- `examples/delayed_chunked_telemetry.rs` — delayed-step telemetry and post-step state recorded across resumable chunks.

Keep examples deterministic when possible. The `validate-examples` recipe checks for expected output markers, so example output should remain stable enough for
CI validation.

## Notebooks

Notebook files live in `notebooks/` and should consume generated artifacts rather than owning sampler logic:

- `notebooks/ising_trace_analysis.ipynb` — reads the Ising example CSV trace, plots energy and magnetization traces, and summarizes acceptance statistics.

## Benchmarks

Benchmarks live in `benches/` and use Criterion. Keep their inputs reproducible with fixed seeds and preserve each named workload's setup and state/RNG
lifecycle contract; do not assume a universal per-iteration reset policy. The authoritative contracts live in [`docs/BENCHMARKING.md`](BENCHMARKING.md).
The stepping suite covers by-value, in-place rollback, delayed accepted/rejected/no plan, sampler bulk loops, and observing overhead rather than distribution
convergence.

## See also

- [`CONTRIBUTING.md`](../CONTRIBUTING.md) — contributor setup, external tools, test categories, code style, PR checklist, release process.
- [`AGENTS.md`](../AGENTS.md) — git/edit/validation rules, documentation-generation rules.
- [`docs/proposal_validation.md`](proposal_validation.md) — proposal-author testing patterns.
- [`docs/reviewer_guide.md`](reviewer_guide.md) — short reading path for scientific and engineering reviewers.
- [`docs/scientific_basis.md`](scientific_basis.md) — Metropolis–Hastings contract and scope.
