# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- add DelayedProposal for accept-before-mutation workflows
- add Step/DelayedStep telemetry and orthogonal DelayedStepError variants
- implement Chain::step_delayed with no-plan, rejection, acceptance, and commit-failure handling
- extend Sampler with delayed stepping and generic proposal-handle storage
- add focused by-value, in-place, and delayed prelude modules
- update doctests, examples, README snippets, and property-test imports
- add tests for delayed acceptance, rejection, no-plan, invalid numerics, proposal-stage errors, commit atomicity, and run_delayed error stopping
- add Observable and TryObservable traits for infallible and fallible measurements
- add ObservedStepError to keep sampling and observation failures orthogonal
- add SampleBuffer for collected observation outputs
- integrate observing APIs across by-value, in-place, and delayed Sampler paths
- expose minimal workflow preludes for by-value, in-place, and delayed usage
- add doctests, unit tests, and documentation for the new measurement APIs
- simplify bounds, imports, and test names across touched code
- Add OnlineStats and BinningAnalysis for streaming mean, variance, standard error, and blocked autocorrelation-aware estimates
- Add StatisticsError variants for invalid samples and non-finite accumulator state
- Add TryAccumulator and ObservedStreamError for streaming observations into fallible sinks
- Add sampler APIs for streaming by-value, in-place, and delayed observations into accumulators
- Export new statistics and streaming types through minimal workflow preludes
- Document usage in README and doctests
- Ignore local .codex workspace metadata

### Fixed

- keep Metropolis-Hastings acceptance decisions in log space
- explicitly reject arithmetic-created NaN acceptance ratios
- document log_prob semantics and numerical behavior at the crate level
- strengthen ProposalMut::propose_mut(None) contract
- remove unnecessary S: Clone bound from by-value Proposal APIs
- make Sampler chain storage private and expose chain_ref/chain_mut accessors
- update examples, README, and organization docs for by-value proposal wording
- add tests for extreme log-domain acceptance, no-move rollback, state-dependent log_q_ratio, and -inf edge cases

## [0.2.1] - 2026-04-30

### Dependencies

- Bump rand from 0.10.0 to 0.10.1 [#25](https://github.com/acgetchell/markov-chain-monte-carlo/pull/25) [`70cec05`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/70cec057f0070c8a4ebf5084f4f6eb8a2ed4eaa5)

### Documentation

- Add docs/code_organization.md with the crate layout, module responsibilities, testing structure, and development conventions
- Add docs/RELEASING.md with the simplified manual release workflow for this crate
- Link the new developer docs from README.md
- Record the documentation additions in CHANGELOG.md

### Maintenance

- Update Rust toolchain/MSRV to 1.95.0
- Replace tarpaulin coverage with cargo-llvm-cov for local HTML and CI Cobertura reports
- Add CodeRabbit, CodeQL, Codecov, Codacy/OpenGrep, Taplo, rustfmt, clippy, typos, and Semgrep configuration
- Add uv-managed Semgrep tooling with pyproject.toml and uv.lock
- Expand and sort justfile workflows, including lint groups, setup-tools, Semgrep checks, and coverage recipes
- Add citation/reference metadata and README sections for contributing, citation, references, and AI tooling disclosure
- Document Rust/tooling workflow in docs/dev/rust.md and update AGENTS.md guidance

## [0.2.0] - 2026-04-06

markov-chain-monte-carlo v0.2.0

### Added

- Sampler<S, T, P, R> ergonomic wrapper with run/run_mut/Iterator
- Chain fields now private (accessor methods: state(), log_prob(), accepted(), rejected())
- +∞ detection (InfiniteInitialLogProb, InfiniteProposedLogProb, InfiniteLogQRatio)
- McmcError implements Copy, #[non_exhaustive]
- Split crate into modules: chain, error, sampler, traits
- iterator_sampling example, doctests on all public methods
- publish-check justfile recipe

## [0.1.0] - 2026-03-24

v0.1.0: first usable release

### Added

- ProposalMut<S> trait for in-place mutation with rollback
- Chain::step_mut for zero-copy Metropolis-Hastings
- Proposal<S> (clone-based) and Target<S> traits
- NaN detection with automatic state rollback
- Seeded RNG support
- Examples: normal_1d, ising_1d
- Property-based tests for MH invariants

[Unreleased]: https://github.com/acgetchell/markov-chain-monte-carlo/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/acgetchell/markov-chain-monte-carlo/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/acgetchell/markov-chain-monte-carlo/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/acgetchell/markov-chain-monte-carlo/tree/v0.1.0
