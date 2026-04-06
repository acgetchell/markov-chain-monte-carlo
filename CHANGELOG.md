# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `Sampler<S, T, P, R>` ergonomic wrapper that bundles a `Chain` with its target, proposal, and RNG
  - `step()` / `run(n)` for clone-based proposals
  - `step_mut()` / `run_mut(n)` for in-place proposals
  - `Iterator` implementation (clone-based path) for composability with `.take(n)` etc.
  - `into_chain()` to recover the inner chain
- `Chain::log_prob()`, `Chain::accepted()`, `Chain::rejected()` accessor methods (fields now private)
- `Chain::total_steps()` convenience method
- `Chain::reset_counters()` for post-burn-in measurement
- `McmcError::InfiniteInitialLogProb` and `McmcError::InfiniteProposedLogProb` error variants for +∞ detection
- `McmcError` now implements `Copy`
- `#[must_use]` on `Chain`, `Sampler`, and all query methods
- `Chain::state()`, `Chain::state_mut()`, `Chain::into_state()` accessor methods (field now private)
- `#[non_exhaustive]` on `McmcError` for forward-compatible error variants
- `Debug` implementation for `Sampler` (prints chain state)
- Doctests for all public `Chain` methods
- Display tests for all `McmcError` variants
- Asymmetric proposal tests (non-zero `log_q_ratio`)
- Edge-case tests for `-∞` and `+∞` log-probabilities

### Changed

- **Breaking:** All `Chain` fields are now private (use accessor methods: `state()`, `log_prob()`, `accepted()`, `rejected()`)
- **Breaking:** `Sampler` added to `prelude`
- Split crate into modules: `chain`, `error`, `sampler`, `traits`
- Examples (`normal_1d`, `ising_1d`) updated to use `Sampler` with `reset_counters()`
- `justfile`: `examples` and `validate-examples` recipes now include `ising_1d`

## [0.1.0] - 2026-03-24

First usable release of the MCMC framework.

### Added

- `Target<S>` trait for target distributions (log-probability)
- `Proposal<S>` trait for clone-based proposal distributions (requires `S: Clone`)
- `ProposalMut<S>` trait for in-place mutation with rollback via associated `Undo` type
- `Chain<S>` with `step` (clone-based) and `step_mut` (in-place) Metropolis–Hastings methods
- `McmcError` with NaN detection for log-probabilities and proposal ratios
- Automatic state rollback on rejection and NaN errors in `step_mut`
- Seeded RNG support for reproducible simulations
- `prelude` module for convenient imports
- `normal_1d` example: sampling from a standard normal distribution
- `ising_1d` example: 1-D Ising model using `ProposalMut` with spin flip undo tokens
- Property-based tests for MH invariants (log_prob consistency, step/step_mut equivalence, counts)
- CI workflows (GitHub Actions), clippy linting, codecov, dependency auditing

## [0.0.1] - 2026-03-22

### Added

- Initial crate scaffold with `State`, `Target`, `Proposal`, `Chain` types
- Basic Metropolis–Hastings `step` method
- `normal_1d` example
- CI/CD infrastructure

[Unreleased]: https://github.com/acgetchell/markov-chain-monte-carlo/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/acgetchell/markov-chain-monte-carlo/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/acgetchell/markov-chain-monte-carlo/releases/tag/v0.0.1
