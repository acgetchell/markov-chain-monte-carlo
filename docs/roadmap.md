# Roadmap

This roadmap records likely directions for the `markov-chain-monte-carlo` crate. It is not a stability promise; release scope depends on scientific need, API
maturity, and validation quality.

Breaking API changes and MSRV bumps are acceptable in any release up to and including v1.0.0, including patch releases, when they improve correctness,
performance, orthogonality, or invariant safety. Maintainers choose the pre-v1.0 release number based on project scope rather than compatibility impact alone.

## Completed

### v0.3.0 Scientific Foundations

The v0.3.0 milestone made the crate useful as the sampling layer for downstream research crates:

- [x] Delayed-commit proposals for accept-before-mutation workflows
- [x] Observable measurement framework
- [x] Streaming statistics and binning error estimates
- [x] Sampler thinning
- [x] Optional `serde` checkpointing
- [x] Detailed-balance verification for by-value, in-place, and delayed proposals

### v0.4.0 Resumable CDT Integration

The v0.4.0 milestone focused on downstream sampler integration and included the Rust 1.96.0 MSRV/toolchain refresh.

- [x] [#65](https://github.com/acgetchell/markov-chain-monte-carlo/issues/65) - Update Rust toolchain and MSRV to 1.96.0
- [x] [#60](https://github.com/acgetchell/markov-chain-monte-carlo/issues/60) - Expose resumable sampler state for chunked runs
- [x] [#61](https://github.com/acgetchell/markov-chain-monte-carlo/issues/61) - Expose delayed-step hooks or telemetry for domain-specific sampler integration
- [x] [#59](https://github.com/acgetchell/markov-chain-monte-carlo/issues/59) - Support additive target bias terms in Metropolis-Hastings acceptance
- [x] [#48](https://github.com/acgetchell/markov-chain-monte-carlo/issues/48) - Detailed balance for delayed proposals with variable valid-site counts
- [x] [#62](https://github.com/acgetchell/markov-chain-monte-carlo/issues/62) - Clean up doctest/example unwraps and enforce with Semgrep

Resumable chunked runs shipped as `Sampler::run_chunk`, `run_mut_chunk`, and `run_delayed_chunk`, each returning a checkpoint-compatible continuation, plus
`run_delayed_chunk_observing` for per-step delayed telemetry. A combined `(measurements, continuation)` return shape is intentionally deferred: callers keep
measurements in their own buffers and accumulate across chunks today. Revisit the combined shape once `causal-triangulations` integrates against v0.4.0 and
shows whether the callback-plus-continuation composition is sufficient in practice.

### v0.4.1 Invariant and Tooling Hardening

The v0.4.1 milestone hardened validation boundaries, Python tooling, and release evidence while the v0.4 sampler integration API was still fresh. API breaks
were accepted where they made invalid states unrepresentable or preserved validation evidence more directly.

- [x] [#82](https://github.com/acgetchell/markov-chain-monte-carlo/issues/82) - Run general parse-don't-validate audit
- [x] [#84](https://github.com/acgetchell/markov-chain-monte-carlo/issues/84) - Update Python tooling to 3.14 and parse scripts at boundaries
- [x] [#104](https://github.com/acgetchell/markov-chain-monte-carlo/issues/104) - Update Rust toolchain and MSRV to 1.97.1
- [x] [#142](https://github.com/acgetchell/markov-chain-monte-carlo/issues/142) - Update Rust to 1.98.0 and forbid relaxed algebraic `f64` operations

## Planned Milestones

### v0.5.0 Adaptive Diagnostics

After the CDT-facing continuation and acceptance APIs settle, invest in classical adaptive sampling and diagnostics that help users decide whether a run is
scientifically trustworthy. This should happen before learned-proposal work so there is a baseline to compare against.

- [#10](https://github.com/acgetchell/markov-chain-monte-carlo/issues/10) - Adaptive Metropolis-Hastings
- [#13](https://github.com/acgetchell/markov-chain-monte-carlo/issues/13) - Diagnostics: ESS, autocorrelation, and R-hat
- [#42](https://github.com/acgetchell/markov-chain-monte-carlo/issues/42) - Continuous proposal diagnostics
- [#20](https://github.com/acgetchell/markov-chain-monte-carlo/issues/20) - Benchmark distributions
- [#21](https://github.com/acgetchell/markov-chain-monte-carlo/issues/21) - Tracing integration for long-running simulations

### v0.6.0 Multi-Chain, Tempering, and Learned-Proposal Foundations

Multi-chain execution and tempering should come before learned-proposal experiments because they provide independent-chain comparison, ensemble diagnostics,
and rugged-target baselines. Learned proposals remain future work until those baselines make failures visible and comparisons meaningful.

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
