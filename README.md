# markov-chain-monte-carlo

[![Crates.io](https://img.shields.io/crates/v/markov-chain-monte-carlo.svg)](https://crates.io/crates/markov-chain-monte-carlo) [![Downloads](https://img.shields.io/crates/d/markov-chain-monte-carlo.svg)](https://crates.io/crates/markov-chain-monte-carlo) [![License](https://img.shields.io/crates/l/markov-chain-monte-carlo.svg)](LICENSE) [![Docs.rs](https://docs.rs/markov-chain-monte-carlo/badge.svg)](https://docs.rs/markov-chain-monte-carlo) [![CI](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/ci.yml/badge.svg)](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/ci.yml) [![CodeQL](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/codeql.yml/badge.svg)](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/codeql.yml) [![rust-clippy analyze](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/rust-clippy.yml/badge.svg)](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/rust-clippy.yml) [![Codacy Quality Scan](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/codacy.yml/badge.svg)](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/codacy.yml) [![codecov](https://codecov.io/gh/acgetchell/markov-chain-monte-carlo/graph/badge.svg)](https://codecov.io/gh/acgetchell/markov-chain-monte-carlo) [![Audit dependencies](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/audit.yml/badge.svg)](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/audit.yml)

A composable **Markov Chain Monte Carlo (MCMC)** framework for arbitrary state spaces in Rust.

🚧 **Pre-release (0.x)** — This crate is under active development. APIs may change between minor versions.

See [CHANGELOG.md](CHANGELOG.md) for release history. Citation metadata and background references are available in [CITATION.cff](CITATION.cff) and [REFERENCES.md](REFERENCES.md).

## Introduction

`markov-chain-monte-carlo` provides a small, explicit Metropolis-Hastings toolkit for scientific Rust projects. It is designed for ordinary numeric states, large combinatorial state spaces, and proposal implementations that need rollback-safe mutation or delayed commits.

Use this crate when you want:

- A generic Metropolis-Hastings chain over user-defined state spaces
- By-value, in-place, and delayed-commit proposal APIs
- Log-space acceptance calculations with NaN/+infinity checks
- Observable measurement APIs for collecting derived quantities
- Streaming means, variances, and binning error estimates
- Thinning helpers for long sampler runs
- Optional `serde` checkpointing for chains and sampler handles
- Detailed-balance diagnostics for proposal development

## Features

- [x] `Target<S>` for unnormalized log probabilities or log densities
- [x] `Proposal<S>` for simple by-value proposals
- [x] `ProposalMut<S>` for in-place mutation with rollback tokens
- [x] `DelayedProposal<S>` for accept-before-mutation workflows with concrete plans
- [x] `Chain<S>` with by-value, in-place, and delayed step methods
- [x] `Sampler<S, T, P, R>` for ergonomic bulk runs and iterator-based sampling
- [x] Observables, fallible observations, and sample buffers
- [x] Online statistics and binning analysis for correlated samples
- [x] Thinned run and observation helpers
- [x] Optional `serde` feature for checkpointing
- [x] Detailed-balance checks for by-value, in-place, and delayed proposals

## Scientific Basis and Scope

This crate implements Metropolis-Hastings sampling for user-defined state spaces. The transition rule uses target log-probability differences and proposal probability ratios:

```text
alpha(x, y) = min(1, exp(log pi(y) - log pi(x) + log q(x | y) - log q(y | x)))
```

The library is built around the standard MCMC contract:

- `Target<S>` returns an unnormalized natural log probability, log density, or negative action.
- Proposal implementations must describe the same concrete transition in both the generated move and `log_q_ratio`.
- Detailed balance, or a valid Metropolis-Hastings correction, is a property of the user-provided target+proposal pair.
- Irreducibility, aperiodicity, burn-in, autocorrelation, and convergence are domain-specific analysis questions.

What the crate provides:

- Log-space acceptance calculations to avoid underflow in tail probabilities.
- Explicit rejection of `NaN` and positive-infinite log probabilities or proposal ratios.
- Rollback-safe in-place proposals for large states where cloning is expensive.
- Delayed-commit proposals for workflows that need to score a concrete move before mutating state.
- Empirical detailed-balance checks for representative discrete transitions.
- Streaming statistics and binning analysis for correlated-sample uncertainty estimates.

What the crate does not prove:

- That a proposal is ergodic on a domain-specific state space.
- That a chain has mixed enough for a given scientific observable.
- That a triangulation, graph, or other combinatorial state satisfies external validity constraints.
- That a chosen model is scientifically appropriate for a downstream study.

For a fuller discussion, see [docs/scientific_basis.md](docs/scientific_basis.md).

## Quickstart

Add the crate:

```bash
cargo add markov-chain-monte-carlo
```

Sample a one-dimensional standard normal target with a symmetric random-walk proposal:

```rust
use markov_chain_monte_carlo::prelude::by_value::*;
use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};

#[derive(Clone)]
struct Scalar(f64);

struct StandardNormal;
impl Target<Scalar> for StandardNormal {
    fn log_prob(&self, state: &Scalar) -> f64 {
        -0.5 * state.0 * state.0
    }
}

struct RandomWalk {
    width: f64,
}
impl Proposal<Scalar> for RandomWalk {
    fn propose<R: Rng + ?Sized>(&self, current: &Scalar, rng: &mut R) -> Scalar {
        Scalar(current.0 + rng.random_range(-self.width..self.width))
    }
}

fn main() -> Result<(), McmcError> {
    let target = StandardNormal;
    let proposal = RandomWalk { width: 1.0 };
    let mut rng = StdRng::seed_from_u64(42);

    let chain = Chain::new(Scalar(5.0), &target)?;
    let mut sampler = Sampler::new(chain, &target, &proposal, &mut rng);

    sampler.run(1_000)?;
    sampler.chain_mut().reset_counters();

    let samples = sampler.run(10_000)?;
    assert_eq!(samples.len(), 10_000);
    assert!(sampler.chain_ref().acceptance_rate() > 0.0);
    Ok(())
}
```

## Examples

Complete runnable examples live in [`examples/`](examples/):

- [`examples/normal_1d.rs`](examples/normal_1d.rs) — by-value random-walk sampler for a normal target
- [`examples/ising_1d.rs`](examples/ising_1d.rs) — in-place spin-flip proposals for a non-`Clone` Ising state
- [`examples/iterator_sampling.rs`](examples/iterator_sampling.rs) — `Sampler` as an iterator
- [`examples/detailed_balance.rs`](examples/detailed_balance.rs) — by-value, in-place, delayed, and batch detailed-balance checks

Run them with:

```bash
just examples
```

## Validation and Diagnostics

The crate exposes test-facing and analysis-facing tools for scientific workflows:

- `McmcError` rejects invalid `NaN` and positive-infinite log probabilities or proposal ratios
- `ProposalMut` rollback and `DelayedProposal` commit contracts keep rejected or failed moves from corrupting chain state
- `Observable`, `TryObservable`, and `SampleBuffer` measure derived quantities during sampling
- `OnlineStats` and `BinningAnalysis` support streaming estimates and correlated-sample error bars
- `verify_detailed_balance*` helpers empirically test proposal kernels for representative discrete transitions

These tools are diagnostics, not proofs. Domain-specific state validity, irreducibility, and mixing behavior remain the caller's responsibility.

For proposal-specific testing patterns, see the [proposal validation guide](docs/proposal_validation.md).

## Documentation

- [docs.rs API documentation](https://docs.rs/markov-chain-monte-carlo)
- [Scientific basis and scope](docs/scientific_basis.md)
- [Proposal validation guide](docs/proposal_validation.md)
- [Roadmap](docs/roadmap.md)
- [Code organization guide](docs/code_organization.md)
- [Rust development workflow](docs/dev/rust.md)
- [Release process](docs/RELEASING.md)

## Ecosystem

This crate is part of a broader Rust ecosystem for computational geometry and simulation:

- [`causal-triangulations`](https://crates.io/crates/causal-triangulations) — CDT physics and simulation
- [`delaunay`](https://crates.io/crates/delaunay) — geometric primitives and triangulations
- [`la-stack`](https://crates.io/crates/la-stack) — fixed-size linear algebra

The long-term architecture separates:

- **Geometry**: triangulations and geometric predicates
- **Sampling**: this crate
- **Physics**: CDT actions, observables, and domain-specific dynamics

## Contributing

A short local workflow:

```bash
just setup         # Install/verify dev tools
just check         # Run non-mutating validation
just fix           # Apply formatters/auto-fixes
just ci            # Run the full local CI simulation
just changelog     # Regenerate CHANGELOG.md from local git history
just bench-compile # Compile Criterion benchmarks without measuring
just bench         # Run Criterion benchmarks
```

For the full command list, run `just --list`. AI assistants should follow [`AGENTS.md`](AGENTS.md).

## Citation

If you use this crate in academic work or downstream research software, please cite it using [`CITATION.cff`](CITATION.cff) or GitHub's "Cite this repository" feature.

## References

For canonical background references for Metropolis-Hastings, MCMC, and the example models, see [`REFERENCES.md`](REFERENCES.md).

## AI Agents

This repository contains an `AGENTS.md` file, which defines the canonical rules and invariants for AI coding assistants and autonomous agents working on this codebase.

Portions of this library were developed with the assistance of AI tools including [ChatGPT], [Claude], [Codex], and [CodeRabbit].

All code was written and/or reviewed and validated by the author.

For tool citation metadata, see the [AI-assisted development tools](REFERENCES.md#ai-assisted-development-tools) section of `REFERENCES.md`.

[ChatGPT]: https://openai.com/chatgpt
[Claude]: https://www.anthropic.com/claude
[Codex]: https://openai.com/codex
[CodeRabbit]: https://coderabbit.ai/
