# Roadmap

This roadmap records likely directions for the `markov-chain-monte-carlo` crate. It is not a stability promise; release scope depends on scientific need, API maturity, and validation quality.

## v0.3.0 Scientific Foundations

The v0.3.0 milestone focuses on making the crate useful as the sampling layer for downstream research crates:

- [x] Delayed-commit proposals for accept-before-mutation workflows
- [x] Observable measurement framework
- [x] Streaming statistics and binning error estimates
- [x] Sampler thinning
- [x] Optional `serde` checkpointing
- [x] Detailed-balance verification for by-value, in-place, and delayed proposals

## Near-Term Candidates

Likely follow-up work:

- More convergence diagnostics, such as effective sample size, autocorrelation reports, and R-hat helpers
- Parallel-chain utilities that keep random-stream management explicit
- Additional examples showing multi-chain analysis and checkpoint/resume workflows
- More ergonomic detailed-balance fixtures for common discrete proposal patterns
- Continuous-proposal diagnostics that avoid exact-hit assumptions ([#42](https://github.com/acgetchell/markov-chain-monte-carlo/issues/42))
- Documentation that connects this crate to `causal-triangulations` proposal validation

## Longer-Term Ideas

Exploratory directions:

- Adaptive Metropolis-Hastings with explicit adaptation windows and post-adaptation freezing
- Simulated annealing or tempering workflows
- Learned or data-informed proposal kernels
- More structured diagnostic reports for publication-quality simulation workflows
- Optional integrations with downstream geometry and physics crates

## Non-Goals

The crate should remain a focused MCMC library rather than becoming a full simulation framework. Domain-specific actions, triangulation validity, geometry kernels, visualization, and physics observables belong in downstream crates.
