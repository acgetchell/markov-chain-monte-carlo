# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.0]: https://github.com/acgetchell/markov-chain-monte-carlo/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/acgetchell/markov-chain-monte-carlo/releases/tag/v0.0.1
