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

### Added

- Add Sampler API and split into modules [#9](https://github.com/acgetchell/markov-chain-monte-carlo/pull/9) [`d32c9cc`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/d32c9cc1271f1c3d138c6155b6b34315482885f7)

  * feat: add Sampler API and split into modules

  - Add `Sampler<S, T, P, R>` ergonomic wrapper bundling Chain + target +
    proposal + RNG with `step()`/`run()`, `step_mut()`/`run_mut()`, and
    `Iterator` impl for the clone-based path
  - Split crate into modules: `chain`, `error`, `sampler`, `traits`
  - Make Chain bookkeeping fields private; add `log_prob()`, `accepted()`,
    `rejected()`, `total_steps()`, `reset_counters()` accessors
  - Add `McmcError::InfiniteInitialLogProb` and `InfiniteProposedLogProb`
    for +∞ detection with automatic rollback
  - Derive `Copy` on `McmcError`; add `#[must_use]` on `Chain`, `Sampler`,
    and all query methods
  - Add asymmetric proposal tests, -∞/+∞ edge-case tests, error Display
    tests, and doctests for all public Chain/Sampler methods
  - Update examples to use `Sampler` with `reset_counters()` for
    production-only acceptance rates
  - Add `ising_1d` to justfile `examples` and `validate-examples` recipes
  - Update README, CHANGELOG, and crate-level docs

  * Changed: encapsulate Chain state and provide accessor methods

  Make the final public field in Chain private to ensure internal
  consistency between the state and its cached log-probability. Add
  state(), state_mut(), and into_state() accessors. Update examples and
  tests to use the new API. Also mark McmcError as non-exhaustive and
  implement Debug for Sampler.

  * feat: harden API with safe state replacement, +∞ log q-ratio detection, and Debug

  - Replace `state_mut()` with `replace_state()` that recomputes and
    validates `log_prob`, preventing stale-cache bugs
  - Add `into_state()` to consume the chain and recover the state
  - Add `McmcError::InfiniteLogQRatio` for +∞ log q-ratio detection,
    completing the symmetric NaN/+∞ error matrix for all computed values
  - Add `+∞` checks on `log_q_ratio` in both `step` and `step_mut`
    (with rollback in the mut path)
  - Implement `Debug` for `Sampler` (prints chain state)
  - Add tests for all new error paths, accessors, and Debug output

### Dependencies

- Bump proptest in the dependencies group [#6](https://github.com/acgetchell/markov-chain-monte-carlo/pull/6) [`a5c29af`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/a5c29af3588d6f38f18a9f7a9349903147bdcaf4)

## [0.1.0] - 2026-03-24

### Added

- Add GitHub Actions CI/CD workflows and justfile linting recipes [`51b5377`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/51b53775026f736e7359d601558842f70cbc526c)
- Add cargo-tarpaulin coverage recipes and CI linting dependencies [`74c7db5`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/74c7db5e46031cfe142c4b619592f7bb2dfd886d)

  Introduce coverage analysis using cargo-tarpaulin with recipes for local
  HTML reports and CI XML output. Update the CI workflow to install
  required linting tools on Linux and macOS runners and reorganize the
  justfile for better logical grouping.
- Add CI tooling, error handling, prelude, and project infrastruc… [#2](https://github.com/acgetchell/markov-chain-monte-carlo/pull/2) [`6584b92`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/6584b9257b3a0e0cf92f31347ca2536be8fe2444)

  * feat: add CI tooling, error handling, prelude, and project infrastructure

  - Add McmcError enum with NaN detection for log-probabilities and
    proposal ratios; Chain::new and Chain::step now return Result
  - Add prelude module for convenience re-exports
  - Simplify trait bounds: remove unnecessary State bounds from Target,
    Proposal, and Chain struct definition
  - Add rust-toolchain.toml pinned to MSRV 1.94.0
  - Add BSD-3-Clause LICENSE and update Cargo.toml
  - Add README badges (crates.io, docs.rs, CI, codecov, audit, clippy)
  - Add AGENTS.md with project guidance for AI assistants
  - Add doc test with full Metropolis–Hastings example
  - Install yamllint in CI workflow for Linux/macOS; keep actionlint
    as local-only recipe
  - Add coverage and coverage-ci justfile recipes

  * chore: minor project hygiene fixes

  - Add /coverage to .gitignore
  - Consolidate duplicate editing tools policy in AGENTS.md
  - Wire validate-examples into `just ci`
  - Remove redundant "cargo" component from rust-toolchain.toml
  - Link to crates instead of repos in README.md
- Add in-place mutation API (ProposalMut) for non-Clone state spaces [#4](https://github.com/acgetchell/markov-chain-monte-carlo/pull/4) [`ada42eb`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/ada42eb1d5c5719aebd4a8ccbe4bf7e1d907a8e6)

  * feat: add in-place mutation API (ProposalMut) for non-Clone state spaces

  API changes:
  - Remove State marker trait; Chain<S> now works with any S
  - Add ProposalMut<S> trait with associated Undo type for cheap rollback
  - Add Chain::step_mut for zero-copy Metropolis-Hastings
  - Move Clone bound from State to Proposal<S> and Chain::step

  New files:
  - examples/ising_1d.rs: 1-D Ising model demonstrating ProposalMut
  - tests/proptest_chain.rs: property-based tests for MH invariants
    (log_prob consistency, step/step_mut equivalence, counts invariant)

### Changed

- Initial commit: scaffold MCMC crate [`5d7f706`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/5d7f706da2d7c41af619a2f5669cdcd56dae94ba)

[0.3.0]: https://github.com/acgetchell/markov-chain-monte-carlo/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/acgetchell/markov-chain-monte-carlo/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/acgetchell/markov-chain-monte-carlo/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/acgetchell/markov-chain-monte-carlo/tree/v0.1.0
