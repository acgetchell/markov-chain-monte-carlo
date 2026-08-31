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
├── .github/
│   ├── CODEOWNERS
│   ├── actions/
│   │   └── setup-just/
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
├── cliff.toml
├── clippy.toml
├── docs/
│   ├── BENCHMARKING.md
│   ├── PERFORMANCE.md
│   ├── RELEASING.md
│   ├── archive/
│   │   └── performance/
│   │       └── README.md
│   ├── assets/
│   │   └── ising_energy_trace.png
│   ├── code_organization.md
│   ├── dev/
│   │   └── rust.md
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
│   ├── check_notebooks.py
│   ├── postprocess_changelog.py
│   ├── publish_performance_readme.py
│   ├── release_check.py
│   ├── subprocess_utils.py
│   ├── tag_release.py
│   ├── update_cargo_tool_pins.py
│   ├── update_python_dev_pins.py
│   ├── update_release_version.py
│   └── tests/
│       ├── __init__.py
│       ├── test_archive_performance.py
│       ├── test_bench_compare.py
│       ├── test_check_notebooks.py
│       ├── test_justfile_discoverability.py
│       ├── test_postprocess_changelog.py
│       ├── test_publish_performance_readme.py
│       ├── test_release_check.py
│       ├── test_subprocess_utils.py
│       ├── test_tag_release.py
│       ├── test_update_cargo_tool_pins.py
│       ├── test_update_python_dev_pins.py
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
  CSV/JSON evidence and older reports; update them together through `just performance-release` or `just performance-doc`, not by hand. The warning on
  the pre-evidence v0.4.1 report is a one-time migration status marker, not permission to edit generated measurements; the next evidence-backed promotion
  replaces the complete file.
  `just performance-readme` owns the marked README performance section and pair-specific SVGs beside the retained evidence; it never measures benchmarks.
- `docs/assets/` — tracked images and other documentation media referenced from README or topic guides.
- `scripts/` — Python helpers for benchmark comparison and report promotion, notebook checks, changelog post-processing, dependency and tool-pin updates,
  release metadata preparation/validation, retained-data README publication, and release tagging. `update_release_version.py` owns coordinated version/date
  changes; `publish_performance_readme.py` renders README outputs using the evidence schema owned by `archive_performance.py`.
- Root configuration files (`Cargo.toml`, `rust-toolchain.toml`, `justfile`, `semgrep.yaml`, `dprint.json`, `rumdl.toml`, `cliff.toml`, `typos.toml`,
  `.config/nextest.toml`) — build, validation, formatting, release, and project-rule configuration.

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
