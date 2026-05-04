# markov-chain-monte-carlo

[![Crates.io](https://img.shields.io/crates/v/markov-chain-monte-carlo.svg)](https://crates.io/crates/markov-chain-monte-carlo) [![Downloads](https://img.shields.io/crates/d/markov-chain-monte-carlo.svg)](https://crates.io/crates/markov-chain-monte-carlo) [![License](https://img.shields.io/crates/l/markov-chain-monte-carlo.svg)](LICENSE) [![Docs.rs](https://docs.rs/markov-chain-monte-carlo/badge.svg)](https://docs.rs/markov-chain-monte-carlo) [![CI](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/ci.yml/badge.svg)](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/ci.yml) [![CodeQL](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/codeql.yml/badge.svg)](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/codeql.yml) [![rust-clippy analyze](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/rust-clippy.yml/badge.svg)](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/rust-clippy.yml) [![Codacy Quality Scan](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/codacy.yml/badge.svg)](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/codacy.yml) [![codecov](https://codecov.io/gh/acgetchell/markov-chain-monte-carlo/graph/badge.svg)](https://codecov.io/gh/acgetchell/markov-chain-monte-carlo) [![Audit dependencies](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/audit.yml/badge.svg)](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/audit.yml)

Small, explicit Metropolis-Hastings tools in Rust for ordinary numeric states, large combinatorial state spaces, and proposal implementations that need rollback-safe mutation or delayed commits.

🚧 **Pre-release (0.x)** — This crate is under active development and not yet ready for production use. APIs may change without notice.

Use this crate when you want:

- a generic Metropolis-Hastings chain over user-defined state spaces
- by-value, in-place, and delayed-commit proposal APIs
- log-space acceptance calculations with NaN/+infinity checks
- observable measurement APIs, streaming statistics, and binning error estimates
- detailed-balance diagnostics for proposal development

This crate provides the sampler mechanics; proposal correctness, ergodicity, convergence assessment, and scientific model choice remain domain-specific responsibilities.

## Contents

- [🚀 Quick start](#-quick-start)
- [🧭 Choosing an API](#-choosing-an-api)
- [📦 Cargo features](#-cargo-features)
- [📚 API guide](#-api-guide)
  - [Features](#features)
  - [Scientific basis and scope](#scientific-basis-and-scope)
  - [Numerical semantics](#numerical-semantics)
  - [Long runs and parallelism](#long-runs-and-parallelism)
  - [Proposal validation](#proposal-validation)
  - [Example](#example)
  - [In-place mutation with rollback](#in-place-mutation-with-rollback)
  - [Delayed commit proposals](#delayed-commit-proposals)
  - [Ergonomic sampling with `Sampler`](#ergonomic-sampling-with-sampler)
  - [Observables and measurements](#observables-and-measurements)
  - [Streaming statistics](#streaming-statistics)
- [🧪 Examples](#-examples)
- [📖 Documentation](#-documentation)
- [🧩 Ecosystem](#-ecosystem)
- [🤝 Contributing](#-contributing)
- [📚 Citation](#-citation)
- [🔎 References](#-references)
- [🤖 AI Agents](#-ai-agents)

## 🚀 Quick start

Add the library to your crate:

```bash
cargo add markov-chain-monte-carlo
```

Enable checkpoint serialization when needed:

```bash
cargo add markov-chain-monte-carlo --features serde
```

Rust 1.95.0 or newer is required.

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
- Use `Sampler` when you want ergonomic repeated runs, iterator-based sampling, or observing helpers.
- Use `verify_detailed_balance*` helpers in proposal tests for representative discrete transitions.
- Use `OnlineStats` and `BinningAnalysis` when long runs should stream statistics instead of retaining every sample.

## 📦 Cargo features

- `serde` — enable `serde::Serialize` / `Deserialize` for `Chain` and `Sampler` checkpointing.

## 📚 API guide

<!-- The block between the cargo-rdme markers below is generated from src/lib.rs //!.
     Do not edit it by hand. To update: edit src/lib.rs and run `just docs-readme`. -->
<!-- dprint-ignore-start -->
<!-- cargo-rdme start -->

Markov Chain Monte Carlo (MCMC) framework.

🚧 **Pre-release (0.x)** — This crate is under active development and
not yet ready for production use. APIs may change without notice.

`markov-chain-monte-carlo` provides a small, explicit Metropolis-Hastings
toolkit for scientific Rust projects. It is designed for ordinary numeric
states, large combinatorial state spaces, and proposal implementations that
need rollback-safe mutation or delayed commits.  The crate aims to provide
a composable, zero-cost abstraction for MCMC methods over arbitrary state
spaces, including discrete and combinatorial systems (e.g., triangulations).

Use this crate when you want:

- A generic Metropolis-Hastings chain over user-defined state spaces
- By-value, in-place, and delayed-commit proposal APIs
- Log-space acceptance calculations with NaN/+infinity checks
- Observable measurement APIs for collecting derived quantities
- Streaming means, variances, and binning error estimates
- Thinning helpers for long sampler runs
- Optional `serde` checkpointing for chains and sampler handles
- Detailed-balance diagnostics for proposal development

[`Target::log_prob`] should return an unnormalized natural log-probability
or log-density.  Additive constants are fine because Metropolis-Hastings
only uses differences, but arbitrary scores or logits will sample a
different distribution.

### Features

- `Target<S>` for unnormalized log probabilities or log densities
- `Proposal<S>` for simple by-value proposals
- `ProposalMut<S>` for in-place mutation with rollback tokens
- `DelayedProposal<S>` for accept-before-mutation workflows with concrete
  plans
- `Chain<S>` with by-value, in-place, and delayed step methods
- `Sampler<S, T, P, R>` for ergonomic bulk runs and iterator-based sampling
- Observables, fallible observations, and sample buffers
- Online statistics and binning analysis for correlated samples
- Thinned run and observation helpers
- Optional `serde` feature for checkpointing
- Detailed-balance checks for by-value, in-place, and delayed proposals

### Scientific basis and scope

This crate implements Metropolis-Hastings sampling for user-defined state
spaces.  The transition rule uses target log-probability differences and
proposal probability ratios:

```text
alpha(x, y) = min(1, exp(log pi(y) - log pi(x) + log q(x | y) - log q(y | x)))
```

The library is built around the standard MCMC contract:

- `Target<S>` returns an unnormalized natural log probability, log
  density, or negative action.
- Proposal implementations must describe the same concrete transition in
  both the generated move and `log_q_ratio`.
- Detailed balance, or a valid Metropolis-Hastings correction, is a
  property of the user-provided target+proposal pair.
- Irreducibility, aperiodicity, burn-in, autocorrelation, and convergence
  are domain-specific analysis questions.

What the crate provides:

- Log-space acceptance calculations to avoid underflow in tail
  probabilities.
- Explicit rejection of `NaN` and positive-infinite log probabilities or
  proposal ratios.
- Rollback-safe in-place proposals for large states where cloning is
  expensive.
- Delayed-commit proposals for workflows that need to score a concrete
  move before mutating state.
- Empirical detailed-balance checks for representative discrete
  transitions.
- Streaming statistics and binning analysis for correlated-sample
  uncertainty estimates.

What the crate does not prove:

- That a proposal is ergodic on a domain-specific state space.
- That a chain has mixed enough for a given scientific observable.
- That a triangulation, graph, or other combinatorial state satisfies
  external validity constraints.
- That a chosen model is scientifically appropriate for a downstream
  study.

For a fuller discussion, see
[docs/scientific_basis.md](https://github.com/acgetchell/markov-chain-monte-carlo/blob/main/docs/scientific_basis.md).

### Numerical semantics

The core Metropolis-Hastings acceptance calculation is performed in log
space using `f64`.  Domain-specific code may use exact arithmetic internally
for predicates or invariant checks, but targets and proposal ratios cross the
crate boundary as log weights:

- finite values represent unnormalized log probability mass/density
- `f64::NEG_INFINITY` represents an impossible or zero-probability state
- `NaN` log-probabilities and log proposal ratios are rejected with
  [`McmcError`]
- `+∞` log-probabilities and log proposal ratios are rejected with
  [`McmcError`]
- acceptance ratios that become `NaN` during arithmetic, such as
  `-∞ - (-∞)`, are treated as rejection

### Long runs and parallelism

`Chain`, `Sampler`, proposal values, and RNGs are ordinary per-instance
values; the crate does not use global mutable state.  Run independent chains
in parallel by giving each worker its own chain, proposal state, and RNG
stream.  This keeps reproducibility and RNG stream splitting under caller
control.

Bulk observing methods return a [`SampleBuffer`], which stores one output
per step.  For production runs with many samples, use compact observables or
single-step observing loops when retaining every measurement is unnecessary.
[`OnlineStats`] and [`BinningAnalysis`] provide constant-memory statistics
for those streaming measurement loops.  Samplers also provide
`*_with_thinning` variants to collect cloned states or measurements only
every k-th completed step while still advancing the chain on every step.

### Proposal validation

The [`verify_detailed_balance`] family of helpers gives proposal authors a
test-facing diagnostic for representative discrete transitions.  Use
[`verify_detailed_balance`] for by-value [`Proposal`] implementations,
[`verify_detailed_balance_mut`] for rollback-based [`ProposalMut`]
implementations, and [`verify_detailed_balance_delayed`] for
[`DelayedProposal`] plans.  The companion batch helpers collect all
per-transition failures in a [`DetailedBalanceBatchReport`], which is useful
when checking a small graph, move table, or list of local states.

These helpers are empirical diagnostics for exact endpoint hits, not a proof
of ergodicity or convergence.  They are intended for tests, examples, and
proposal-development checks over discrete or otherwise exactly comparable
states.

Enable the optional `serde` feature to serialize and deserialize
[`Chain<S>`] when `S` implements serde's traits.  [`Sampler`] also derives
serialization when all stored handles support it, but the portable
checkpoint for resuming a run is the owned [`Chain<S>`]; targets, proposals,
and RNG streams are reconstructed by the caller.

```rust
use markov_chain_monte_carlo::prelude::*;

struct Normal;
impl Target<f64> for Normal {
    fn log_prob(&self, state: &f64) -> f64 { -0.5 * state * state }
}

let Ok(chain) = Chain::new(1.0, &Normal) else {
    unreachable!("normal target returns a finite log probability");
};
let checkpoint = serde_json::to_string(&chain)?;
let restored: Chain<f64> = serde_json::from_str(&checkpoint)?;
assert!((restored.log_prob() - Normal.log_prob(restored.state())).abs() < 1e-12);
```

### Example

Sample from a standard normal distribution using Metropolis–Hastings:

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

struct RandomWalk { width: f64 }
impl Proposal<Scalar> for RandomWalk {
    fn propose<R: Rng + ?Sized>(&self, current: &Scalar, rng: &mut R) -> Scalar {
        let delta: f64 = rng.random_range(-self.width..self.width);
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

    assert!(chain.acceptance_rate() > 0.2);
    Ok(())
}
```

### In-place mutation with rollback

For state spaces where cloning is expensive, use [`ProposalMut`] with
[`Chain::step_mut`].  The proposal mutates the state in place and returns
a small undo token for rollback on rejection:

```rust
use markov_chain_monte_carlo::prelude::in_place::*;
use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};

/// A lattice of spins (not Clone — only mutated in place).
struct SpinChain { spins: Vec<i8> }

/// Energy = −Σ s_i · s_{i+1}  (1-D Ising, no field).
struct Ising;
impl Target<SpinChain> for Ising {
    fn log_prob(&self, state: &SpinChain) -> f64 {
        let s = &state.spins;
        let energy: f64 = s.windows(2)
            .map(|w| -f64::from(w[0]) * f64::from(w[1]))
            .sum();
        -energy  // log_prob = −E  (T = 1)
    }
}

/// Flip one random spin; undo token is the site index.
struct SpinFlip;
impl ProposalMut<SpinChain> for SpinFlip {
    type Undo = usize;
    fn propose_mut<R: Rng + ?Sized>(&self, state: &mut SpinChain, rng: &mut R) -> Option<usize> {
        if state.spins.is_empty() { return None; }
        let idx = rng.random_range(0..state.spins.len());
        state.spins[idx] *= -1;
        Some(idx)
    }
    fn undo(&self, state: &mut SpinChain, idx: usize) {
        state.spins[idx] *= -1;  // flipping twice = identity
    }
}

fn main() -> Result<(), McmcError> {
    let mut rng = StdRng::seed_from_u64(42);
    let state = SpinChain { spins: vec![1; 20] };
    let mut chain = Chain::new(state, &Ising)?;

    for _ in 0..1000 {
        chain.step_mut(&Ising, &SpinFlip, &mut rng)?;
    }

    assert!(chain.acceptance_rate() > 0.0);
    Ok(())
}
```

### Delayed commit proposals

Use [`DelayedProposal`] with [`Chain::step_delayed`] when a proposal can
plan and score a move before mutating the state, then commit only after the
Metropolis-Hastings decision accepts it.

The plan should describe a concrete transition, such as a move kind plus the
local site or handle needed to apply it.  If no valid site can be selected,
return `Ok(None)` from [`DelayedProposal::propose_plan`]; that is an ordinary
rejection, while [`DelayedProposal::commit`] errors are reserved for
exceptional failures applying an already accepted concrete move.

```rust
use core::convert::Infallible;
use markov_chain_monte_carlo::prelude::delayed::*;
use rand::{Rng, SeedableRng, rngs::StdRng};

struct TargetLine;
impl Target<i32> for TargetLine {
    fn log_prob(&self, state: &i32) -> f64 {
        -f64::from(state.abs())
    }
}

struct MoveRight;
impl DelayedProposal<i32> for MoveRight {
    type Plan = i32;
    type Info = i32;
    type Error = Infallible;

    fn propose_plan<R: Rng + ?Sized>(
        &mut self,
        _state: &i32,
        _rng: &mut R,
    ) -> Result<Option<i32>, Self::Error> {
        Ok(Some(1))
    }

    fn proposed_log_prob<T: Target<i32>>(
        &self,
        state: &i32,
        plan: &i32,
        target: &T,
    ) -> Result<f64, Self::Error> {
        Ok(target.log_prob(&(*state + *plan)))
    }

    fn info(&self, plan: &i32) -> i32 {
        *plan
    }

    fn commit<R: Rng + ?Sized>(
        &mut self,
        state: &mut i32,
        plan: i32,
        _rng: &mut R,
    ) -> Result<(), Self::Error> {
        *state += plan;
        Ok(())
    }
}

fn main() -> Result<(), DelayedStepError<Infallible>> {
    let target = TargetLine;
    let mut proposal = MoveRight;
    let mut rng = StdRng::seed_from_u64(42);
    let mut chain = Chain::new(-1, &target).map_err(DelayedStepError::Mcmc)?;

    let step = chain.step_delayed(&target, &mut proposal, &mut rng)?;
    assert!(step.accepted);
    assert_eq!(*chain.state(), 0);
    Ok(())
}
```

### Ergonomic sampling with [`Sampler`]

[`Sampler`] bundles a chain with its target, proposal, and RNG so you
don't have to pass them on every step:

```rust
use markov_chain_monte_carlo::prelude::by_value::*;
use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};

let mut rng = StdRng::seed_from_u64(42);
let chain = Chain::new(Scalar(0.0), &Normal)?;
let mut sampler = Sampler::new(chain, &Normal, &Walk, &mut rng);

// Burn-in
sampler.run(1000)?;
sampler.chain_mut().reset_counters();

// Production
sampler.run(10_000)?;
assert!(sampler.chain_ref().acceptance_rate() > 0.0);
```

### Observables and measurements

Use [`Observable`] or a closure with [`Sampler::run_observing`] to compute
derived quantities during sampling without storing full state histories:

```rust
use markov_chain_monte_carlo::prelude::by_value::*;
use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};

let mut rng = StdRng::seed_from_u64(42);
let chain = Chain::new(Scalar(0.0), &Normal)?;
let mut sampler = Sampler::new(chain, &Normal, &Walk, &mut rng);
let mut energy = |state: &Scalar| 0.5 * state.0 * state.0;

let samples: SampleBuffer<f64> = sampler.run_observing(256, &mut energy)?;
assert_eq!(samples.len(), 256);
```

### Streaming statistics

Use [`OnlineStats`] for Welford mean and variance updates, and
[`BinningAnalysis`] for autocorrelation-aware standard-error estimates:

```rust
use markov_chain_monte_carlo::prelude::*;

let mut energy = OnlineStats::new();
energy.extend([1.0, 2.0, 3.0, 4.0]);

assert_eq!(energy.mean(), Some(2.5));

let mut bins = BinningAnalysis::new();
bins.extend([1.0, 2.0, 3.0, 4.0]);
assert!(bins.standard_error().is_some());
```

`Sampler` can also stream observations directly into these accumulators:

```rust
use core::convert::Infallible;
use markov_chain_monte_carlo::prelude::by_value::*;
use rand::{Rng, SeedableRng, rngs::StdRng};

let mut rng = StdRng::seed_from_u64(42);
let chain = Chain::new(0.0, &T).map_err(ObservedStreamError::Step)?;
let mut sampler = Sampler::new(chain, &T, &P, &mut rng);
let mut coordinate = |state: &f64| *state;
let mut stats = OnlineStats::new();

sampler.run_observing_into(4, &mut coordinate, &mut stats)?;
assert_eq!(stats.count(), 4);
```

<!-- cargo-rdme end -->
<!-- dprint-ignore-end -->

## 🧪 Examples

Complete runnable examples live in [`examples/`](examples/):

- [`examples/normal_1d.rs`](examples/normal_1d.rs) — by-value random-walk sampler for a normal target
- [`examples/ising_1d.rs`](examples/ising_1d.rs) — in-place spin-flip proposals for a non-`Clone` Ising state
- [`examples/iterator_sampling.rs`](examples/iterator_sampling.rs) — `Sampler` as an iterator
- [`examples/detailed_balance.rs`](examples/detailed_balance.rs) — by-value, in-place, delayed, and batch detailed-balance checks

Run them with:

```bash
just examples
```

For proposal-specific testing patterns, see the [proposal validation guide](docs/proposal_validation.md).

## 📖 Documentation

- [docs.rs API documentation](https://docs.rs/markov-chain-monte-carlo)
- [Changelog](CHANGELOG.md)
- [Scientific basis and scope](docs/scientific_basis.md)
- [Proposal validation guide](docs/proposal_validation.md)
- [Roadmap](docs/roadmap.md)
- [Code organization guide](docs/code_organization.md)
- [Rust development workflow](docs/dev/rust.md)
- [Release process](docs/RELEASING.md)

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

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contributor guide (project layout, development workflow, code style, testing, documentation generation via `cargo-rdme`, performance/benchmarking, and the release process). AI assistants should follow [`AGENTS.md`](AGENTS.md).

Quick local workflow: run `just setup` once, then run `just check` before opening a pull request. For the full command list, run `just --list`.

## 📚 Citation

If you use this crate in academic work or downstream research software, please cite it using [`CITATION.cff`](CITATION.cff) or GitHub's "Cite this repository" feature.

## 🔎 References

For canonical background references for Metropolis-Hastings, MCMC, and the example models, see [`REFERENCES.md`](REFERENCES.md).

## 🤖 AI Agents

This repository contains an `AGENTS.md` file, which defines the canonical rules and invariants for AI coding assistants and autonomous agents working on this codebase.

Portions of this library were developed with the assistance of AI tools including [ChatGPT], [Claude], [Codex], and [CodeRabbit].

All code was written and/or reviewed and validated by the author.

For tool citation metadata, see the [AI-assisted development tools](REFERENCES.md#ai-assisted-development-tools) section of `REFERENCES.md`.

[ChatGPT]: https://openai.com/chatgpt
[Claude]: https://www.anthropic.com/claude
[Codex]: https://openai.com/codex
[CodeRabbit]: https://coderabbit.ai/
