# markov-chain-monte-carlo

[![CI](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/ci.yml/badge.svg)](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/ci.yml)
[![rust-clippy analyze](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/rust-clippy.yml/badge.svg)](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/rust-clippy.yml)
[![codecov](https://codecov.io/gh/acgetchell/markov-chain-monte-carlo/graph/badge.svg)](https://codecov.io/gh/acgetchell/markov-chain-monte-carlo)
[![Audit dependencies](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/audit.yml/badge.svg)](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/audit.yml)

A composable **Markov Chain Monte Carlo (MCMC)** framework for arbitrary state spaces in Rust.

🚧 **Pre-release (0.0.x)** — This crate is under active development and **not yet ready for production use**. APIs may change without notice.

---

## Overview

This crate provides:

- A generic Metropolis–Hastings implementation
- Two proposal models:
  - **`Proposal<S>`** — clone-based, for simple/small state spaces
  - **`ProposalMut<S>`** — in-place mutation with rollback, for large combinatorial state spaces (triangulations, graphs) where cloning is expensive
- `Chain<S>` with `step` (clone-based) and `step_mut` (in-place) methods
- NaN detection and automatic state rollback on error
- Seeded RNG support for reproducible simulations

The design emphasizes:

- Zero-cost abstractions
- Log-space numerical stability
- Extensibility for research and experimentation

---

## Quick Start

### Clone-based (simple states)

```rust
use markov_chain_monte_carlo::prelude::*;
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
        Scalar(current.0 + rng.random_range(-self.width..self.width))
    }
}

fn main() -> Result<(), McmcError> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut chain = Chain::new(Scalar(0.0), &Normal)?;

    for _ in 0..1000 {
        chain.step(&Normal, &RandomWalk { width: 1.0 }, &mut rng)?;
    }

    assert!(chain.acceptance_rate() > 0.2);
    Ok(())
}
```

### In-place mutation (combinatorial states)

```rust
use markov_chain_monte_carlo::prelude::*;
use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};

struct SpinChain { spins: Vec<i8> }  // not Clone

struct Ising;
impl Target<SpinChain> for Ising {
    fn log_prob(&self, state: &SpinChain) -> f64 {
        state.spins.windows(2)
            .map(|w| f64::from(w[0]) * f64::from(w[1]))
            .sum()
    }
}

struct SpinFlip;
impl ProposalMut<SpinChain> for SpinFlip {
    type Undo = usize;  // which site was flipped
    fn propose_mut<R: Rng + ?Sized>(&self, state: &mut SpinChain, rng: &mut R) -> Option<usize> {
        let idx = rng.random_range(0..state.spins.len());
        state.spins[idx] *= -1;
        Some(idx)
    }
    fn undo(&self, state: &mut SpinChain, idx: usize) {
        state.spins[idx] *= -1;
    }
}

fn main() -> Result<(), McmcError> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut chain = Chain::new(SpinChain { spins: vec![1; 20] }, &Ising)?;

    for _ in 0..1000 {
        chain.step_mut(&Ising, &SpinFlip, &mut rng)?;
    }

    assert!(chain.acceptance_rate() > 0.0);
    Ok(())
}
```

---

## Relationship to Other Crates

This crate is part of a broader ecosystem:

- [`causal-triangulations`](https://crates.io/crates/causal-triangulations) — CDT physics and simulation
- [`delaunay`](https://crates.io/crates/delaunay) — geometric primitives
- [`la-stack`](https://crates.io/crates/la-stack) — linear algebra

The long-term architecture separates:

- **Geometry** (triangulations)
- **Sampling** (this crate)
- **Physics** (CDT, actions, observables)

---

## Planned Features

- [ ] Adaptive Metropolis–Hastings
- [ ] Simulated annealing / tempering
- [ ] Parallel chains
- [ ] Diagnostics (ESS, autocorrelation)
- [ ] Learned proposals (ML integration)

