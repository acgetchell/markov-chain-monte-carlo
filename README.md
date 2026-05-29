# markov-chain-monte-carlo

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20033111.svg)](https://doi.org/10.5281/zenodo.20033111)
[![Crates.io](https://img.shields.io/crates/v/markov-chain-monte-carlo.svg)](https://crates.io/crates/markov-chain-monte-carlo)
[![Downloads](https://img.shields.io/crates/d/markov-chain-monte-carlo.svg)](https://crates.io/crates/markov-chain-monte-carlo)
[![License](https://img.shields.io/crates/l/markov-chain-monte-carlo.svg)](https://github.com/acgetchell/markov-chain-monte-carlo/blob/main/LICENSE)
[![Docs.rs](https://docs.rs/markov-chain-monte-carlo/badge.svg)](https://docs.rs/markov-chain-monte-carlo)
[![CI](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/ci.yml/badge.svg)](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/ci.yml)
[![CodeQL](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/codeql.yml/badge.svg)](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/codeql.yml)
[![zizmor](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/zizmor.yml/badge.svg)](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/zizmor.yml)
[![rust-clippy analyze](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/rust-clippy.yml/badge.svg)](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/rust-clippy.yml)
[![codecov](https://codecov.io/gh/acgetchell/markov-chain-monte-carlo/graph/badge.svg)](https://codecov.io/gh/acgetchell/markov-chain-monte-carlo)
[![Audit dependencies](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/audit.yml/badge.svg)](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/audit.yml)

Small, explicit Metropolis-Hastings tools in Rust for ordinary numeric states, large combinatorial state spaces, and proposal implementations that need
rollback-safe mutation or delayed commits.

## 📐 Introduction

This library implements composable Metropolis-Hastings sampling in Rust for workflows where the state space, proposal mechanism, and measurement strategy are
application-specific. The goal is to keep the transition mechanics explicit while supporting cheap cloned states, large rollback-mutable states, delayed-commit
proposals, long streaming runs, and proposal diagnostics.

🚧 **Pre-release (0.x)** — This crate is under active development and not yet ready for production use. APIs may change without notice.

Use this crate when you want:

- a generic Metropolis-Hastings chain over user-defined state spaces
- by-value, in-place, and delayed-commit proposal APIs
- log-space acceptance calculations with NaN/+infinity checks
- observable measurement APIs, streaming statistics, and binning error estimates
- thinning helpers for long sampler runs
- optional `serde` checkpointing with validated resume flows
- detailed-balance diagnostics for proposal development

This crate provides the sampler mechanics; proposal correctness, ergodicity, convergence assessment, and scientific model choice remain domain-specific
responsibilities.

## ✨ Features

- Generic `Chain<S>` over user-defined state spaces with explicit accepted/rejected counters.
- Log-space Metropolis-Hastings acceptance with typed errors for NaN and positive-infinite target or proposal values.
- Three proposal workflows: by-value `Proposal`, rollback-safe in-place `ProposalMut`, and delayed-commit `DelayedProposal`.
- `Sampler` helpers for repeated and chunked runs, iterator-style sampling, thinning, observations, and counter resets after burn-in.
- Streaming `OnlineStats` and `BinningAnalysis` for long correlated runs without retaining every sample.
- `ChainCheckpoint` restore APIs that recompute cached log-probabilities against the resumed target.
- Optional `serde` support for serializing chains, samplers, and portable checkpoints.
- Detailed-balance diagnostics for proposal tests on representative discrete transitions.

## Contents

- [📐 Introduction](#-introduction)
- [✨ Features](#-features)
- [🚀 Quick start](#-quick-start)
- [🧭 Choosing an API](#-choosing-an-api)
- [📦 Cargo features](#-cargo-features)
- [🧪 Examples](#-examples)
- [📖 Documentation](#-documentation)
- [🧩 Ecosystem](#-ecosystem)
- [🤝 Contributing](#-contributing)
- [📚 Citation](#-citation)
- [🔎 References](#-references)
- [🤖 AI Agents](#-ai-agents)
- [📜 License](#-license)

## 🚀 Quick start

Add the library to your crate:

```bash
cargo add markov-chain-monte-carlo
```

Enable checkpoint serialization when needed:

```bash
cargo add markov-chain-monte-carlo --features serde
```

Rust 1.96.0 or newer is required.

Minimal by-value Metropolis-Hastings sampler:

```rust
use markov_chain_monte_carlo::prelude::by_value::*;
use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};

#[derive(Clone)]
struct Scalar(f64);

struct Normal;
impl Target<Scalar> for Normal {
    fn log_prob(&self, state: &Scalar) -> f64 {
        -0.5 * state.0 * state.0
    }
}

struct RandomWalk {
    width: f64,
}
impl Proposal<Scalar> for RandomWalk {
    fn propose<R: Rng + ?Sized>(&self, current: &Scalar, rng: &mut R) -> Scalar {
        let delta = rng.random_range(-self.width..self.width);
        Scalar(current.0 + delta)
    }
}

fn main() -> Result<(), McmcError> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut chain = Chain::new(Scalar(0.0), &Normal)?;
    let proposal = RandomWalk { width: 1.0 };

    for _ in 0..1000 {
        chain.step(&Normal, &proposal, &mut rng)?;
    }

    assert!(chain.acceptance_rate() > 0.0);
    Ok(())
}
```

## 🧭 Choosing an API

- Start with `Proposal` and `Chain::step` when state cloning is cheap.
- Use `ProposalMut` and `Chain::step_mut` when cloning state is expensive and rollback is simple.
- Use `DelayedProposal` and `Chain::step_delayed` when you need to plan and score a concrete move before mutating state.
- Use `DelayedStep` telemetry, `StepOutcome`, and `DelayedProposal::no_plan_info` when delayed proposals need domain-specific per-step records.
- Use `Sampler` when you want ergonomic repeated runs, resumable chunks, iterator-based sampling, or observing helpers.
- Use `Sampler::run_delayed_chunk_observing` to record per-step delayed telemetry and post-step state while resuming chunked runs from a `ChainCheckpoint`.
- Use `verify_detailed_balance*` helpers in proposal tests for representative discrete transitions.
- Use `OnlineStats` and `BinningAnalysis` when long runs should stream statistics instead of retaining every sample.

## 📦 Cargo features

- `serde` — enable `serde::Serialize` for `Chain` and `Sampler`, plus `ChainCheckpoint` serialization/deserialization for validated resume flows.

## 🧪 Examples

Complete runnable examples live in [`examples/`](https://github.com/acgetchell/markov-chain-monte-carlo/tree/main/examples):

- [`examples/normal_1d.rs`](https://github.com/acgetchell/markov-chain-monte-carlo/blob/main/examples/normal_1d.rs) — by-value random-walk sampler for a normal
  target
- [`examples/ising_1d.rs`](https://github.com/acgetchell/markov-chain-monte-carlo/blob/main/examples/ising_1d.rs) — in-place spin-flip proposals for a
  non-`Clone` Ising state
- [`examples/iterator_sampling.rs`](https://github.com/acgetchell/markov-chain-monte-carlo/blob/main/examples/iterator_sampling.rs) — `Sampler` as an iterator
- [`examples/detailed_balance.rs`](https://github.com/acgetchell/markov-chain-monte-carlo/blob/main/examples/detailed_balance.rs) — by-value, in-place, delayed,
  and batch detailed-balance checks
- [`examples/delayed_chunked_telemetry.rs`](https://github.com/acgetchell/markov-chain-monte-carlo/blob/main/examples/delayed_chunked_telemetry.rs) — per-step
  delayed telemetry and post-step state recorded across resumable chunks

Run them with:

```bash
just examples
```

For proposal-specific testing patterns, see the
[proposal validation guide](https://github.com/acgetchell/markov-chain-monte-carlo/blob/main/docs/proposal_validation.md).

## 📖 Documentation

- [docs.rs API documentation](https://docs.rs/markov-chain-monte-carlo)
- [Changelog](https://github.com/acgetchell/markov-chain-monte-carlo/blob/main/CHANGELOG.md)
- [Scientific basis and scope](https://github.com/acgetchell/markov-chain-monte-carlo/blob/main/docs/scientific_basis.md)
- [Proposal validation guide](https://github.com/acgetchell/markov-chain-monte-carlo/blob/main/docs/proposal_validation.md)
- [Roadmap](https://github.com/acgetchell/markov-chain-monte-carlo/blob/main/docs/roadmap.md)
- [Code organization guide](https://github.com/acgetchell/markov-chain-monte-carlo/blob/main/docs/code_organization.md)
- [Rust development workflow](https://github.com/acgetchell/markov-chain-monte-carlo/blob/main/docs/dev/rust.md)
- [Release process](https://github.com/acgetchell/markov-chain-monte-carlo/blob/main/docs/RELEASING.md)
- [Security policy](https://github.com/acgetchell/markov-chain-monte-carlo/blob/main/SECURITY.md)

## 🧩 Ecosystem

This crate is part of a broader Rust ecosystem for computational geometry and simulation:

- [`causal-triangulations`](https://crates.io/crates/causal-triangulations) — CDT physics and simulation
- [`delaunay`](https://crates.io/crates/delaunay) — geometric primitives and triangulations
- [`la-stack`](https://crates.io/crates/la-stack) — fixed-size linear algebra

The long-term architecture separates:

- **Geometry**: triangulations and geometric predicates
- **Sampling**: this crate
- **Physics**: CDT actions, observables, and domain-specific dynamics

## 🤝 Contributing

See [CONTRIBUTING.md](https://github.com/acgetchell/markov-chain-monte-carlo/blob/main/CONTRIBUTING.md) for the full contributor guide (project layout,
development workflow, code style, testing, documentation layout, performance/benchmarking, and the release process). Community expectations live in
[`CODE_OF_CONDUCT.md`](https://github.com/acgetchell/markov-chain-monte-carlo/blob/main/CODE_OF_CONDUCT.md). AI assistants should follow
[`AGENTS.md`](https://github.com/acgetchell/markov-chain-monte-carlo/blob/main/AGENTS.md).

Quick local workflow: run `just setup` once, then run `just check` before opening a pull request. For the full command list, run `just --list`.

## 📚 Citation

If you use this crate in academic work or downstream research software, please cite it using
[`CITATION.cff`](https://github.com/acgetchell/markov-chain-monte-carlo/blob/main/CITATION.cff) or GitHub's "Cite this repository" feature.

## 🔎 References

For canonical background references for Metropolis-Hastings, MCMC, and the example models, see
[`REFERENCES.md`](https://github.com/acgetchell/markov-chain-monte-carlo/blob/main/REFERENCES.md).

## 🤖 AI Agents

This repository contains an `AGENTS.md` file, which defines the canonical rules and invariants for AI coding assistants and autonomous agents working on this
codebase.

Portions of this library were developed with the assistance of AI tools including [ChatGPT], [Claude], [Codex], and [CodeRabbit].

All code was written and/or reviewed and validated by the author.

For tool citation metadata, see the
[AI-assisted development tools](https://github.com/acgetchell/markov-chain-monte-carlo/blob/main/REFERENCES.md#ai-assisted-development-tools) section of
`REFERENCES.md`.

## 📜 License

This project is licensed under the [BSD 3-Clause License](https://github.com/acgetchell/markov-chain-monte-carlo/blob/main/LICENSE).

[ChatGPT]: https://openai.com/chatgpt
[Claude]: https://www.anthropic.com/claude
[Codex]: https://openai.com/codex
[CodeRabbit]: https://coderabbit.ai/
