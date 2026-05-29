# Roadmap

This roadmap records likely directions for the `markov-chain-monte-carlo` crate. It is not a stability promise; release scope depends on scientific need, API
maturity, and validation quality.

Pre-1.0 releases may still revise public APIs when that makes the crate more correct, but patch releases should stay boring: fixes, documentation, and tooling
hardening that are compatible with the current minor line. New sampler APIs, changed acceptance semantics, and MSRV bumps belong in minor releases so Cargo
users on `0.x` ranges are not surprised by patch updates.

## Completed

### v0.3.0 Scientific Foundations

The v0.3.0 milestone made the crate useful as the sampling layer for downstream research crates:

- [x] Delayed-commit proposals for accept-before-mutation workflows
- [x] Observable measurement framework
- [x] Streaming statistics and binning error estimates
- [x] Sampler thinning
- [x] Optional `serde` checkpointing
- [x] Detailed-balance verification for by-value, in-place, and delayed proposals

## Planned Milestones

### v0.4.0 Resumable CDT Integration

The next feature release should focus on downstream sampler integration rather than a grab bag of diagnostics. This is the right place for the Rust 1.96.0
MSRV/toolchain refresh.

- [#65](https://github.com/acgetchell/markov-chain-monte-carlo/issues/65) - Update Rust toolchain and MSRV to 1.96.0
- [x] [#60](https://github.com/acgetchell/markov-chain-monte-carlo/issues/60) - Expose resumable sampler state for chunked runs
- [#61](https://github.com/acgetchell/markov-chain-monte-carlo/issues/61) - Expose delayed-step hooks or telemetry for domain-specific sampler integration
- [x] [#59](https://github.com/acgetchell/markov-chain-monte-carlo/issues/59) - Support additive target bias terms in Metropolis-Hastings acceptance
- [#48](https://github.com/acgetchell/markov-chain-monte-carlo/issues/48) - Investigate detailed balance for delayed proposals with variable valid-site counts
- [#62](https://github.com/acgetchell/markov-chain-monte-carlo/issues/62) - Clean up doctest/example unwraps and enforce with Semgrep

### v0.5.0 Adaptive Diagnostics

After the CDT-facing continuation and acceptance APIs settle, invest in classical adaptive sampling and diagnostics that help users decide whether a run is
scientifically trustworthy. This should happen before learned-proposal work so there is a baseline to compare against.

- [#10](https://github.com/acgetchell/markov-chain-monte-carlo/issues/10) - Adaptive Metropolis-Hastings
- [#13](https://github.com/acgetchell/markov-chain-monte-carlo/issues/13) - Diagnostics: ESS, autocorrelation, and R-hat
- [#42](https://github.com/acgetchell/markov-chain-monte-carlo/issues/42) - Continuous proposal diagnostics
- [#20](https://github.com/acgetchell/markov-chain-monte-carlo/issues/20) - Benchmark distributions
- [#21](https://github.com/acgetchell/markov-chain-monte-carlo/issues/21) - Tracing integration for long-running simulations

### v0.6.0 Multi-Chain, Tempering, and Learned Proposals

Multi-chain execution and tempering are natural prerequisites for serious learned-proposal experiments: they provide independent-chain comparison, ensemble
diagnostics, and rugged-target baselines. Learned proposals align with near-term AI doctorate work, so they should land while the research context is active
rather than being deferred until the end of the pre-1.0 cycle.

- [#12](https://github.com/acgetchell/markov-chain-monte-carlo/issues/12) - Parallel chains
- [#11](https://github.com/acgetchell/markov-chain-monte-carlo/issues/11) - Simulated annealing / tempering
- [#14](https://github.com/acgetchell/markov-chain-monte-carlo/issues/14) - Learned proposals

### v0.7.0 Portability

Portability work should be done after optional integrations are clearer, so the `std` feature boundary can be designed around real dependencies.

- [#22](https://github.com/acgetchell/markov-chain-monte-carlo/issues/22) - `no_std` support

### v1.0.0 Stabilization

The v1.0.0 release should be about stabilizing the API surface, documentation contract, and compatibility story after the pre-1.0 milestones have shaken out the
sampler, diagnostics, and feature-gating design.

## Maintenance Backlog

These issues can land whenever they are useful and do not need to drive the public release sequence:

- [#53](https://github.com/acgetchell/markov-chain-monte-carlo/issues/53) - Replace Node markdown and YAML tooling with Rust-native tools
- [#56](https://github.com/acgetchell/markov-chain-monte-carlo/issues/56) - Speed up CI without reducing coverage
- [#57](https://github.com/acgetchell/markov-chain-monte-carlo/issues/57) - Evaluate CI shape changes for faster builds

## Non-Goals

The crate should remain a focused MCMC library rather than becoming a full simulation framework. Domain-specific actions, triangulation validity, geometry
kernels, visualization, and physics observables belong in downstream crates.
