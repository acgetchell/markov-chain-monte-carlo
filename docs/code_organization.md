# Code Organization Guide

Detailed file and module ownership guide for the `markov-chain-monte-carlo` crate: which file owns what, and where new code usually belongs.

This document complements two related files:

- [`CONTRIBUTING.md`](../CONTRIBUTING.md) — human contributor workflow (setup, tooling, testing, PR process, release).
- [`AGENTS.md`](../AGENTS.md) — canonical rules for AI assistants (git/edit/validation policy, documentation-generation rules).

For contributor setup, test commands, and external tooling, see `CONTRIBUTING.md`. For agent-specific rules, see `AGENTS.md`. This file is the detailed
code/file map: keep ownership and placement guidance here, and keep contributor workflow details elsewhere.

## Documentation ownership and names

`README.md` introduces the crate, gives an early use-case checklist and quick start, and maps capabilities to their API documentation.
`REFERENCES.md` owns bibliographic records, stable citation identifiers, and the method-to-source index. `docs/scientific_basis.md` owns scientific scope,
shared target/arithmetic conventions, assumptions, methods, and evidence boundaries. Put API selection and scope before detailed methods; order independent
methods lexicographically within coherent groups, retaining prerequisite order. Keep Contents navigation and existing anchors, and link between owners.
Programming contracts and worked API examples belong in rustdoc; proposal testing and analysis recipes belong in their task guides.

Name active documents by primary purpose:

- Use uppercase verbs or gerunds for execution guides: `ANALYZING_CHAINS.md`, `BENCHMARKING.md`, `RELEASING.md`, `VALIDATING_PROPOSALS.md`, and
  `dev/DEVELOPING.md`.
- Use lowercase descriptive names for architecture, scientific discussion, policy, reference material, analysis, and results. A few commands do not turn
  `benchmark_distributions.md` or `reviewer_guide.md` into a task guide. Preserve the existing underscore/hyphen style within each area.
- Keep directory `README.md` indexes, standard root filenames, and retained historical paths such as `archives/changelog/`. The maintainer retired the
  old release-performance series during #154. New reports and indexes under `performance/v1/` belong to `just performance-doc` and are checked with
  `just performance-check`; its first-report state is explicit until two new release baselines exist.

The #154 task-guide renames are `proposal_validation.md` → `VALIDATING_PROPOSALS.md`, `diagnostics.md` → `ANALYZING_CHAINS.md`, and
`dev/rust.md` → `dev/DEVELOPING.md`. Update references and this tree whenever paths change. Rename reports through their configured shared generator,
preserving measurements and provenance; path changes alone do not require benchmark runs.

README is also embedded in rustdoc, so repository destinations use explicit GitHub URLs on `main` and API destinations use `latest` on docs.rs.
Active documentation links stay independent of the library version; release updates preserve these destinations. Other repository guides use relative
links where they resolve within the checkout. Image commit pins and historical release or benchmark evidence links retain their provenance.
Maintain compatibility anchors for renamed README/scientific headings; verify Contents in GitHub-style Markdown and generated rustdoc.

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
│       ├── gitleaks.yml
│       ├── osv.yml
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
│   ├── autocorrelation.rs
│   ├── diagnostic_backends/
│   │   ├── .gitignore
│   │   ├── Cargo.lock
│   │   ├── Cargo.toml
│   │   ├── README.md
│   │   ├── benches/
│   │   │   └── comparison.rs
│   │   ├── src/
│   │   │   └── lib.rs
│   │   └── tests/
│   │       ├── correctness.rs
│   │       └── proptest_extreme_oracle.rs
│   └── stepping.rs
├── clippy.toml
├── docs/
│   ├── ANALYZING_CHAINS.md
│   ├── BENCHMARKING.md
│   ├── RELEASING.md
│   ├── VALIDATING_PROPOSALS.md
│   ├── archives/
│   │   └── changelog/
│   │       ├── 0.1.md
│   │       ├── 0.2.md
│   │       ├── 0.3.md
│   │       └── 0.4.md
│   ├── assets/
│   │   ├── diagnostic_ess_efficiency.png
│   │   ├── diagnostic_rank_overlays.png
│   │   └── ising_energy_trace.png
│   ├── benchmark_distributions.md
│   ├── code_organization.md
│   ├── dev/
│   │   ├── DEVELOPING.md
│   │   ├── shared-changelog-pilot.md
│   │   └── shared-maintenance-migration.md
│   ├── performance/
│   │   └── v1/
│   │       └── experiments/
│   │           ├── diagnostic-backends.json
│   │           └── diagnostic-backends.md
│   ├── roadmap.md
│   ├── reviewer_guide.md
│   └── scientific_basis.md
├── dprint.json
├── examples/
│   ├── adaptive_normal.rs
│   ├── additive_target_bias.rs
│   ├── benchmark_distributions.rs
│   ├── delayed_chunked_telemetry.rs
│   ├── detailed_balance.rs
│   ├── diagnostics.rs
│   ├── diagnostics/
│   │   └── plot_data.rs
│   ├── ising_1d.rs
│   ├── iterator_sampling.rs
│   └── normal_1d.rs
├── justfile
├── notebooks/
│   ├── diagnostic_plots.ipynb
│   └── ising_trace_analysis.ipynb
├── pyproject.toml
├── rumdl.toml
├── rust-toolchain.toml
├── rustfmt.toml
├── semgrep.yaml
├── src/
│   ├── adaptive.rs
│   ├── autocorrelation.rs
│   ├── benchmarks.rs
│   ├── chain.rs
│   ├── continuous_testing.rs
│   ├── convergence.rs
│   ├── diagnostics.rs
│   ├── error.rs
│   ├── ess.rs
│   ├── lib.rs
│   ├── numerics.rs
│   ├── observable.rs
│   ├── ranks.rs
│   ├── sampler.rs
│   ├── statistics.rs
│   ├── testing.rs
│   └── traits.rs
├── tests/
│   ├── adaptive.rs
│   ├── autocorrelation.rs
│   ├── benchmark_distributions.rs
│   ├── combined_rhat.rs
│   ├── continuous_testing.rs
│   ├── convergence.rs
│   ├── ess.rs
│   ├── public_api.rs
│   ├── pooled_ranks.rs
│   ├── rank_normalized_rhat.rs
│   ├── fixtures/
│   │   ├── combined_rhat.json
│   │   ├── diagnostic_plots.json
│   │   ├── ess.json
│   │   ├── generate_combined_rhat.py
│   │   ├── generate_diagnostic_plots.py
│   │   ├── generate_ess.py
│   │   └── rank_normalized_rhat.json
│   ├── proptest_autocorrelation.rs
│   ├── proptest_chain.rs
│   ├── proptest_convergence.rs
│   ├── proptest_ess.rs
│   ├── proptest_validators.rs
│   ├── tracing.rs
│   ├── tooling/
│   │   ├── __init__.py
│   │   ├── test_benchmark_contracts.py
│   │   ├── test_commands.py
│   │   ├── test_ess_fixture.py
│   │   ├── test_notebooks.py
│   │   ├── test_performance_evidence.py
│   │   └── test_release_policy.py
│   └── semgrep/
│       ├── benches/
│       │   ├── erased_error.rs
│       │   ├── typed_error.rs
│       │   └── unwrap_expect.rs
│       ├── examples/
│       │   ├── erased_error.rs
│       │   ├── typed_error.rs
│       │   └── unwrap_expect.rs
│       ├── github-actions/
│       │   └── workflow_actions.yml
│       ├── docs/
│       │   └── check_fix_order.md
│       ├── tests/
│       │   └── tooling/
│       │       └── subprocess_mocks.py
│       └── src/
│           ├── doctests/
│           │   ├── erased_error.rs
│           │   ├── typed_error.rs
│           │   └── unwrap_expect.rs
│           └── project_rules/
│               ├── algebraic_float.rs
│               └── rust_style.rs
├── tooling/
│   ├── benchmark.toml
│   ├── examples.toml
│   ├── performance-interpretation.md
│   └── performance-report.toml
├── ty.toml
├── typos.toml
└── uv.lock
```

## Repository areas

- `src/` — core library modules and crate-level documentation. The detailed source file map is below.
- `examples/` — complete runnable workflows that demonstrate public APIs.
- `notebooks/` — notebook consumers for example-generated artifacts such as exported diagnostic traces.
- `tests/` — integration tests, property-based tests named `tests/proptest_*.rs`, and project-rule tests including Semgrep fixtures under `tests/semgrep/`.
- `benches/` — Criterion benchmarks for stepping, sampler loops, observing overhead, and scalar autocorrelation diagnostics.
- `docs/` — topic guides, release benchmark methodology, and release procedures supporting public API documentation without duplicating README or rustdoc.
  Future shared reports, comparison/evidence JSON, CSV exports and the generated archive index live in `docs/performance/v1/`; `performance.md` is the
  configured current report. The directory currently holds the diagnostic-backend experiment; the release comparison series starts with the next baseline.
  Generate them through `just performance-release` or `just performance-doc`; never hand-edit.
  `just performance-readme` owns future README sections and SVGs using an explicitly reviewed publication configuration.
- `docs/archives/changelog/` — completed minor-series release history generated by the shared changelog workflow; never hand-edit.
- `docs/assets/` — tracked images and other documentation media referenced from README or topic guides.
  Regenerate the analysis guide's rank and ESS figures with `just diagnostic-plots-figures`, and the Ising figure with `just notebook-ising-figure`.
- `tooling/` — declarative stepping inventories, compatibility policy, example output contracts, shared report paths, and scientific prose. Add the reviewed
  `performance-readme.toml` publication selection only when a new measured pair exists; the old legacy adapter and selection were retired.
- `tests/tooling/` — focused consumer configuration, command wiring, workload lifecycle, release policy, evidence transition and notebook checks.
  Reusable implementation and generic regressions live in the pinned shared package. No local Python package or support module remains.
- `.gitattributes` — prevents checkout text conversion of byte-sensitive retained reports and scientific JSON fixtures used in provenance hashes.
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

### `src/adaptive.rs`

Defines `AdaptiveScale`, its validated bounds and typed configuration errors, and the `TunableProposal` capability. The `Sampler::warm_up*` methods tune a
single scale during explicit warmup for all three proposal workflows. Ordinary sampling freezes the final scale. Integration checks in `tests/adaptive.rs`
cover update arithmetic, failure and rollback behavior, chunk continuation, float boundaries, and analytical normal moments after tuning.

### `src/error.rs`

Defines `McmcError`, the crate error type for invalid log-probabilities and proposal ratios.

Add new variants conservatively. `McmcError` is `#[non_exhaustive]`, but error changes still affect user matching and documentation.

### `src/diagnostics.rs`

Defines trace-recording APIs for reusable MCMC diagnostics:

- `ChainId` for stable multi-chain identifiers
- `TraceStepOutcome` for accepted, rejected-proposal, and no-proposal outcomes
- `TraceRecord` for one post-step row
- `TraceRecorder` for recording one chain into a shared-column trace
- `Trace` for multi-chain numeric observable rows, borrowed named-column selection, and CSV export

`Trace::observable_values` owns name lookup and chain selection; estimators consume the selected values without depending on trace storage or column layout.
Local tests cover recorder publication after rejected or interrupted collection; `tests/autocorrelation.rs` exercises named-column alignment across
rejected writes, merges, and retries.
Keep diagnostics independent from plotting and notebook rendering. Domain observables should enter this module as numeric columns; visualization and
post-processing belong in notebooks or downstream tools.

### `src/autocorrelation.rs`

Defines `Autocorrelation`, `IntegratedAutocorrelationTime`, `AutocorrelationError`, and `EssRateError` for scalar ACF estimation, Geyer's initial monotone
sequence integrated-time estimator, single-chain mean ESS, and ESS per measured second. The slice boundary accepts recorded or imported observables without
depending on trace storage or plotting. Callers own burn-in,
chain selection, and regular sample spacing. Independent arithmetic, numerical-boundary, and seeded AR(1) checks live in `tests/autocorrelation.rs`.
`tests/proptest_autocorrelation.rs` checks exact integer covariance ratios, affine invariance, time reversal, and bounded lag prefixes on generated traces.

### `src/convergence.rs`

Defines `SplitRhat`, `RankNormalizedSplitRhat`, `FoldedRankNormalizedSplitRhat`, `CombinedRhat`, and their shared `SplitRhatError` for borrowed scalar chains.
This module owns input checks, splitting, rank-normal scores, folding, within/between variance estimation, and explicit degeneracy errors.
It uses `ranks.rs` for shared average-rank tie grouping and the rank precision bound.
It does not own warmup selection, chain execution, or plotting.
`tests/combined_rhat.rs` checks pinned posterior fixtures in `tests/fixtures/combined_rhat.json`, reproduced in Python by `generate_combined_rhat.py`
in that directory, including scale-only sensitivity, odd lengths, folded degeneracy, extreme finite arithmetic, and preserved component failures.
`tests/convergence.rs` checks exact R-hat and ESS values, numerical boundaries, separated/drifting chains, and seeded independent and AR(1) traces.
`tests/proptest_convergence.rs` checks classical R-hat against exact integer moments across chain counts and lengths, affine transforms, chain/time reversal,
and omitted middle draws. Rank-normalized properties use an exact integer oracle for binary chains and check nonlinear monotone transforms, ties,
chain/time reordering, and omitted middle draws.
`tests/rank_normalized_rhat.rs` checks the rank-normalized component against pinned independent ArviZ fixtures in `tests/fixtures/rank_normalized_rhat.json`,
plus monotone transforms, discrete ties, heavy-tail location shifts, and numerical/input boundaries. Private tests verify pooled ranks, chain reconstruction,
and rank-count rejection before allocation.
`tests/public_api.rs` checks the estimators through crate-root and prelude paths, including use after borrowed trace columns are dropped.

### `src/ess.rs`

Defines `EssEstimate`, `EssEstimator`, `TailEss`, `MeanMcse`, `QuantileMcse`, `MonteCarloError`, `DiagnosticTiming`, and `DiagnosticTimingError`. Owns split
multi-chain autocovariances, raw/ranked/indicator ESS, original-unit mean/quantile uncertainty, retained-count ratios, and checked workload rates. Reuses
`convergence.rs`'s private accepted-input and rank-normal-score infrastructure, backed by `ranks.rs`. Constant individual halves are allowed for ESS;
R-hat policy is unchanged. Quantile
MCSE uses statrs' regularized beta CDF with bounded, checked inversion; default statrs features are disabled. `tests/ess.rs` checks pinned ArviZ fixtures in
`tests/fixtures/ess.json`, reproduced by the isolated `generate_ess.py`, plus numerical ranges and typed failures. `tests/proptest_ess.rs` checks affine unit
changes, chain/time reordering, ties, and nonlinear monotone rank transforms. `tests/public_api.rs` checks crate-root/prelude imports and owned results after
borrowed storage drops. `tests/tooling/test_ess_fixture.py` protects fixture provenance, tolerances, case/metric inventories, and drift rejection.
Plotting and scientific stopping policies belong to callers.

### `src/ranks.rs`

Defines `PooledRanks` and `PooledRankError` for original-chain plotting data. Owns shared average-tie grouping, order reconstruction, and the rank precision
bound reused by `convergence.rs`; normal-score transformation remains with convergence diagnostics. Plot data accept nonempty unequal-length or constant
chains and preserve every original draw, unlike split estimators. `tests/pooled_ranks.rs` covers public contracts and pinned prefix evidence in
`tests/fixtures/diagnostic_plots.json`. Its isolated `generate_diagnostic_plots.py` reuses exact ESS fixture inputs with SciPy ranks and ArviZ estimators;
it can also independently verify example-exported prefixes. `tests/tooling/test_ess_fixture.py` protects retained prefix evidence against drift;
`tests/tooling/test_notebooks.py` checks notebook identity/count/timing rejection, explicit missing values, and byte-exact evidence retention.
No rendering dependencies enter the library.

### `src/numerics.rs`

Owns crate-private compensated summation and count-to-float conversion shared by autocorrelation, convergence, ESS/MCSE, continuous-proposal checks, and
streaming statistics, plus the AS 241 inverse-normal approximation for rank normalization. These arithmetic primitives introduce no public API or statistical
policy. Each calling module owns validation and bounds; the ACF retains its specialized multi-lag traversal while sharing the same ordered accumulator.
Numerical helper tests live with this module.

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
- feature-gated `tracing` events after completed transitions, including no-proposal self-loops

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
- feature-gated DEBUG spans around sampling loops; `src/adaptive.rs` owns warmup spans

Use `Sampler` for workflow ergonomics; use `Chain` for the core transition logic.

`tests/tracing.rs` checks subscriber-visible metrics, loop scopes, failed transitions, counter resets, and preservation of seeded results and RNG state.

### `src/statistics.rs`

Defines streaming statistics helpers for post-processing observed samples:

- `OnlineStats` for one-pass means and variances
- `BinningAnalysis` and `BinningEstimate` for correlated-sample uncertainty estimates
- `StatisticsError` for invalid samples, exhausted counts, and non-finite accumulator updates

Estimate accessors return `None` until their sample requirements are met.

Statistics helpers are ordinary public API, but they should stay independent of chain mutation and proposal mechanics.

### `src/testing.rs`

Contains test-facing validation utilities for proposal development.

Detailed-balance helpers empirically check discrete by-value, in-place, and delayed proposal transitions by sampling forward/reverse moves and comparing
estimated Metropolis-Hastings transition flows. Keep these helpers explicit at the crate root because they are test-facing diagnostics rather than everyday
sampling imports.

### `src/continuous_testing.rs`

Owns `verify_proposal_density` and `verify_proposal_bins`, their reports, and typed errors. These test-facing checks compare independently supplied proposal
densities and sampled bin masses without exact endpoint equality. Callers retain ownership of endpoint evaluation, histogram collection, and rollback;
no proposal trait or sampler behavior changes. Analytical density, exact binomial, numerical-boundary, and seeded generator checks live in
`tests/continuous_testing.rs`. The crate root and scoped testing prelude expose these diagnostics. Each report retains its original tolerance so downstream
collectors can interpret saved successes and violations without storing a separate threshold.

### `src/benchmarks.rs`

Defines the optional `BenchmarkTarget` catalog behind the `benchmarks` feature. It owns fixed two-dimensional reference densities and their analytical
population moments, independently of proposal kernels and diagnostics. The canonical import is at the crate root; the module stays private.
`tests/benchmark_distributions.rs` checks density values, numerical extremes, normalization, and moments through deterministic quadrature with independent
changes of variables. Definitions and moment derivations live in [`docs/benchmark_distributions.md`](benchmark_distributions.md).

## Examples

New examples go in `examples/`. Each is a complete, runnable workflow:

- `examples/additive_target_bias.rs` — additive model and bias log-weight composition with `AdditiveTarget`.
- `examples/benchmark_distributions.rs` — feature-gated reference targets, moment errors, and scalar mean ESS per measured production second.
- `examples/detailed_balance.rs` — by-value, in-place, delayed, and batch detailed-balance checks.
- `examples/diagnostics.rs` — four sequential scalar chains and five illustrative regimes with ACF, single/multi-chain ESS, original-unit mean/quantile
  MCSE, measured production rates, and classical/ranked/folded/combined R-hat. `examples/diagnostics/plot_data.rs` assembles original-chain pooled ranks,
  independently recomputed prefixes, blocked errors, and self-contained CSV/JSON reports through public APIs; assumptions in `docs/ANALYZING_CHAINS.md`.
- `examples/normal_1d.rs` — simple by-value random-walk sampler.
- `examples/adaptive_normal.rs` — bounded proposal-width tuning during warmup, then fixed-width production sampling.
- `examples/ising_1d.rs` — four sequential chains using in-place mutation with rollback; trace/ACF/time CSVs and ESS, timing, and classical split R-hat JSON.
- `examples/iterator_sampling.rs` — by-value `Sampler` iterator API.
- `examples/delayed_chunked_telemetry.rs` — delayed-step telemetry and post-step state recorded across resumable chunks.

Keep examples deterministic when possible. The `validate-examples` recipe checks for expected output markers, so example output should remain stable enough for
CI validation.

## Notebooks

Notebook files live in `notebooks/` and should consume generated artifacts rather than owning sampler logic:

- `notebooks/diagnostic_plots.ipynb` — consumes Rust-exported ranks and diagnostics for rank overlays, ESS/relative-ESS prefix curves, trace/ACF plots,
  scalar and blocked-error tables; retains matching input evidence and a hash manifest without implementing a second ESS/MCSE estimator.
- `notebooks/ising_trace_analysis.ipynb` — reads the Ising CSV and optional timing JSON, plots traces and ACFs, and reports acceptance statistics, integrated
  times, ESS, measured ESS rates, and classical split R-hat for energy and magnetization. Analysis tables are exported under the selected output directory.

## Benchmarks

Benchmarks live in `benches/` and use Criterion. Keep their inputs reproducible with fixed seeds and preserve each named workload's setup and state/RNG
lifecycle contract; do not assume a universal per-iteration reset policy. The authoritative contracts live in [`docs/BENCHMARKING.md`](BENCHMARKING.md).
The stepping suite covers by-value, in-place rollback, delayed accepted/rejected/no plan, sampler bulk loops, and observing overhead rather than distribution
convergence.

`benches/autocorrelation.rs` isolates scalar ACF estimation across sample counts and lag budgets, plus integrated-time estimation from an existing ACF.

`benches/diagnostic_backends/` is an independent, unpublished comparison workspace. It pins candidate dependencies, tests them against independent
oracles, and measures native ACF/IMS against arima ACF and ferromorphic IPS. It does not add dependencies to the library or run under the main CI gate.
Its reproduction commands live in its README; retained evidence and the dependency decision live in `docs/performance/v1/experiments/diagnostic-backends.*`.
These focused diagnostic workloads are separate from the stepping release-signal suite.

## See also

- [`CONTRIBUTING.md`](../CONTRIBUTING.md) — contributor setup, external tools, test categories, code style, PR checklist, release process.
- [`AGENTS.md`](../AGENTS.md) — git/edit/validation rules, documentation-generation rules.
- [`docs/VALIDATING_PROPOSALS.md`](VALIDATING_PROPOSALS.md) — proposal-author testing patterns.
- [`docs/reviewer_guide.md`](reviewer_guide.md) — short reading path for scientific and engineering reviewers.
- [`docs/scientific_basis.md`](scientific_basis.md) — Metropolis–Hastings contract and scope.
