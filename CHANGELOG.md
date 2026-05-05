# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-05-05

### Added

- Add delayed-commit proposal API [#36](https://github.com/acgetchell/markov-chain-monte-carlo/pull/36) [`65445f4`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/65445f4e48143a47a217f573bc82c37421092a3d)

  - add DelayedProposal for accept-before-mutation workflows
  - add Step/DelayedStep telemetry and orthogonal DelayedStepError variants
  - implement Chain::step_delayed with no-plan, rejection, acceptance, and commit-failure handling
  - extend Sampler with delayed stepping and generic proposal-handle storage
  - add focused by-value, in-place, and delayed prelude modules
  - update doctests, examples, README snippets, and property-test imports
  - add tests for delayed acceptance, rejection, no-plan, invalid numerics, proposal-stage errors, commit atomicity, and run_delayed error stopping
- Add observable measurement framework [#37](https://github.com/acgetchell/markov-chain-monte-carlo/pull/37) [`82cbaf5`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/82cbaf58a71e1355e370eee43450d22f8c02df91)

  - add Observable and TryObservable traits for infallible and fallible measurements
  - add ObservedStepError to keep sampling and observation failures orthogonal
  - add SampleBuffer for collected observation outputs
  - integrate observing APIs across by-value, in-place, and delayed Sampler paths
  - expose minimal workflow preludes for by-value, in-place, and delayed usage
  - add doctests, unit tests, and documentation for the new measurement APIs
  - simplify bounds, imports, and test names across touched code
- Add streaming statistics and error bars [#38](https://github.com/acgetchell/markov-chain-monte-carlo/pull/38) [`4575047`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/457504708de375dc23ea86de56505c2cdf67419d)

  - Add OnlineStats and BinningAnalysis for streaming mean, variance, standard error, and blocked autocorrelation-aware estimates
  - Add StatisticsError variants for invalid samples and non-finite accumulator state
  - Add TryAccumulator and ObservedStreamError for streaming observations into fallible sinks
  - Add sampler APIs for streaming by-value, in-place, and delayed observations into accumulators
  - Export new statistics and streaming types through minimal workflow preludes
  - Document usage in README and doctests
  - Ignore local .codex workspace metadata
- Add sampler thinning support [#40](https://github.com/acgetchell/markov-chain-monte-carlo/pull/40) [`c03d07e`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/c03d07e8bec3e9b5e67c22273689971f60fad85a)

  - add typed ThinningError and thinned result aliases
  - add state-collecting thinning APIs for by-value, in-place, and delayed samplers
  - add observing, streaming, and fallible thinning variants across sampler workflows
  - re-export thinning types through the public API and appropriate preludes
  - document thinning behavior in README and public doctests
  - cover zero intervals, interval > steps boundaries, and thinned observation behavior
- Add serde checkpointing support [#41](https://github.com/acgetchell/markov-chain-monte-carlo/pull/41) [`d976bc6`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/d976bc67f3373cf2ead697508c127b5ae205d2bc)

  - Add optional `serde` feature and derive serialization for `Chain<S>`
  - Derive `Serialize` for `Sampler` when stored handles support it
  - Document chain checkpointing as the portable resume path
  - Add serde-gated tests for checkpoint roundtrip, resumed sampling, sampler serialization, and non-serializable state construction
  - Mark serde checkpointing as complete in the README
- Add detailed-balance proposal diagnostics [#43](https://github.com/acgetchell/markov-chain-monte-carlo/pull/43) [`cefc035`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/cefc0356ed29f6a273271230f6b59ffbd5f0886c)

  - Add detailed-balance verification APIs for by-value, in-place, delayed, and batch proposal checks.
  - Add typed reports and errors, scoped testing prelude exports, public doctests, and a runnable detailed_balance example.
  - Document proposal validation, scientific basis, roadmap, and refreshed README usage guidance.
  - Add Semgrep guardrails and fixtures to keep examples, benches, and doctests on typed errors.
  - Improve git-cliff and agent commit guidance, then regenerate CHANGELOG.md.
  - Bump serde_json dev-dependency to 1.0.149.
- [**breaking**] Validate sampler construction and checkpoint restores [#46](https://github.com/acgetchell/markov-chain-monte-carlo/pull/46) [`e355a6d`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/e355a6de20234ef5b74a61b8fe2a576793fcda7a)

  - Add ChainCheckpoint as the portable restore format and recompute cached log-probabilities through Chain::from_checkpoint.
  - Make Sampler::new validate the chain against the sampler target and return Result.
  - Add sampler-level reset and replacement helpers so callers do not need mutable access to the underlying Chain.
  - Report checkpoint and current-state cache failures with dedicated McmcError variants.
  - Refresh README organization and run coverage with all crate features enabled in local CI.

### Changed

- Docs/cargo rdme readme refresh [#44](https://github.com/acgetchell/markov-chain-monte-carlo/pull/44) [`0524993`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/052499374aa2e30485c1a429a2d3efd39c31731a)

### Documentation

- Include README in rustdoc [`ac8cd20`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/ac8cd2082d5b450099fc4759fdf3004a65f7061c)

  - Use README.md as the user-facing docs.rs landing page through rustdoc inclusion
  - Remove cargo-rdme generation, CI installation, setup checks, and justfile recipes
  - Keep src/lib.rs focused on semantic and API contract documentation
  - Update contributor, release, agent, and development docs for the new documentation layout

### Fixed

- Harden MCMC acceptance and proposal invariants [#35](https://github.com/acgetchell/markov-chain-monte-carlo/pull/35) [`8ad2703`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/8ad27038dd9c79380ac1d77d35834b481bb51a85)

  - keep Metropolis-Hastings acceptance decisions in log space
  - explicitly reject arithmetic-created NaN acceptance ratios
  - document log_prob semantics and numerical behavior at the crate level
  - strengthen ProposalMut::propose_mut(None) contract
  - remove unnecessary S: Clone bound from by-value Proposal APIs
  - make Sampler chain storage private and expose chain_ref/chain_mut accessors
  - update examples, README, and organization docs for by-value proposal wording
  - add tests for extreme log-domain acceptance, no-move rollback, state-dependent log_q_ratio, and -inf edge cases

### Maintenance

- Add changelog and Python tooling [#39](https://github.com/acgetchell/markov-chain-monte-carlo/pull/39) [`02deb42`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/02deb4291f2bc89311cdb5c254aa284d81ddc4bb)

  - add git-cliff changelog generation with post-processing and release tag helpers
  - add Ruff, Ty, Pytest, and dprint configuration for repository tooling
  - wire Python, Markdown, and Semgrep checks into justfile workflows
  - add Python Semgrep rules and fixtures for script/test hygiene
  - update release, tooling, code organization, README, references, and agent docs
  - regenerate CHANGELOG.md from local git history

### Removed

- Remove pre-release warning from README [`d48bb16`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/d48bb1604f63cad95020a8f027312c48f902cc3f)

## [Unreleased]

### Added

- Add delayed-commit proposal API [#36](https://github.com/acgetchell/markov-chain-monte-carlo/pull/36) [`65445f4`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/65445f4e48143a47a217f573bc82c37421092a3d)

  - add DelayedProposal for accept-before-mutation workflows
  - add Step/DelayedStep telemetry and orthogonal DelayedStepError variants
  - implement Chain::step_delayed with no-plan, rejection, acceptance, and commit-failure handling
  - extend Sampler with delayed stepping and generic proposal-handle storage
  - add focused by-value, in-place, and delayed prelude modules
  - update doctests, examples, README snippets, and property-test imports
  - add tests for delayed acceptance, rejection, no-plan, invalid numerics, proposal-stage errors, commit atomicity, and run_delayed error stopping
- Add observable measurement framework [#37](https://github.com/acgetchell/markov-chain-monte-carlo/pull/37) [`82cbaf5`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/82cbaf58a71e1355e370eee43450d22f8c02df91)

  - add Observable and TryObservable traits for infallible and fallible measurements
  - add ObservedStepError to keep sampling and observation failures orthogonal
  - add SampleBuffer for collected observation outputs
  - integrate observing APIs across by-value, in-place, and delayed Sampler paths
  - expose minimal workflow preludes for by-value, in-place, and delayed usage
  - add doctests, unit tests, and documentation for the new measurement APIs
  - simplify bounds, imports, and test names across touched code
- Add streaming statistics and error bars [#38](https://github.com/acgetchell/markov-chain-monte-carlo/pull/38) [`4575047`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/457504708de375dc23ea86de56505c2cdf67419d)

  - Add OnlineStats and BinningAnalysis for streaming mean, variance, standard error, and blocked autocorrelation-aware estimates
  - Add StatisticsError variants for invalid samples and non-finite accumulator state
  - Add TryAccumulator and ObservedStreamError for streaming observations into fallible sinks
  - Add sampler APIs for streaming by-value, in-place, and delayed observations into accumulators
  - Export new statistics and streaming types through minimal workflow preludes
  - Document usage in README and doctests
  - Ignore local .codex workspace metadata
- Add sampler thinning support [#40](https://github.com/acgetchell/markov-chain-monte-carlo/pull/40) [`c03d07e`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/c03d07e8bec3e9b5e67c22273689971f60fad85a)

  - add typed ThinningError and thinned result aliases
  - add state-collecting thinning APIs for by-value, in-place, and delayed samplers
  - add observing, streaming, and fallible thinning variants across sampler workflows
  - re-export thinning types through the public API and appropriate preludes
  - document thinning behavior in README and public doctests
  - cover zero intervals, interval > steps boundaries, and thinned observation behavior
- Add serde checkpointing support [#41](https://github.com/acgetchell/markov-chain-monte-carlo/pull/41) [`d976bc6`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/d976bc67f3373cf2ead697508c127b5ae205d2bc)

  - Add optional `serde` feature and derive serialization for `Chain<S>`
  - Derive `Serialize` for `Sampler` when stored handles support it
  - Document chain checkpointing as the portable resume path
  - Add serde-gated tests for checkpoint roundtrip, resumed sampling, sampler serialization, and non-serializable state construction
  - Mark serde checkpointing as complete in the README
- Add detailed-balance proposal diagnostics [#43](https://github.com/acgetchell/markov-chain-monte-carlo/pull/43) [`cefc035`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/cefc0356ed29f6a273271230f6b59ffbd5f0886c)

  - Add detailed-balance verification APIs for by-value, in-place, delayed, and batch proposal checks.
  - Add typed reports and errors, scoped testing prelude exports, public doctests, and a runnable detailed_balance example.
  - Document proposal validation, scientific basis, roadmap, and refreshed README usage guidance.
  - Add Semgrep guardrails and fixtures to keep examples, benches, and doctests on typed errors.
  - Improve git-cliff and agent commit guidance, then regenerate CHANGELOG.md.
  - Bump serde_json dev-dependency to 1.0.149.

### Fixed

- Harden MCMC acceptance and proposal invariants [#35](https://github.com/acgetchell/markov-chain-monte-carlo/pull/35) [`8ad2703`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/8ad27038dd9c79380ac1d77d35834b481bb51a85)

  - keep Metropolis-Hastings acceptance decisions in log space
  - explicitly reject arithmetic-created NaN acceptance ratios
  - document log_prob semantics and numerical behavior at the crate level
  - strengthen ProposalMut::propose_mut(None) contract
  - remove unnecessary S: Clone bound from by-value Proposal APIs
  - make Sampler chain storage private and expose chain_ref/chain_mut accessors
  - update examples, README, and organization docs for by-value proposal wording
  - add tests for extreme log-domain acceptance, no-move rollback, state-dependent log_q_ratio, and -inf edge cases

### Maintenance

- Add changelog and Python tooling [#39](https://github.com/acgetchell/markov-chain-monte-carlo/pull/39) [`02deb42`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/02deb4291f2bc89311cdb5c254aa284d81ddc4bb)

  - add git-cliff changelog generation with post-processing and release tag helpers
  - add Ruff, Ty, Pytest, and dprint configuration for repository tooling
  - wire Python, Markdown, and Semgrep checks into justfile workflows
  - add Python Semgrep rules and fixtures for script/test hygiene
  - update release, tooling, code organization, README, references, and agent docs
  - regenerate CHANGELOG.md from local git history

## [0.2.1] - 2026-04-30

### Dependencies

- Bump rand from 0.10.0 to 0.10.1 [#25](https://github.com/acgetchell/markov-chain-monte-carlo/pull/25) [`70cec05`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/70cec057f0070c8a4ebf5084f4f6eb8a2ed4eaa5)

### Documentation

- Add release and code organization guides [`8f0282a`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/8f0282a0b054660856c8f18e9de4768ea7694e1d)

  - Add docs/code_organization.md with the crate layout, module responsibilities, testing structure, and development conventions
  - Add docs/RELEASING.md with the simplified manual release workflow for this crate
  - Link the new developer docs from README.md
  - Record the documentation additions in CHANGELOG.md

### Maintenance

- Upgrade Rust tooling and validation workflows [#32](https://github.com/acgetchell/markov-chain-monte-carlo/pull/32) [`8194a00`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/8194a006e9c81a953549de5caa4194cd34a38dc2)

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
