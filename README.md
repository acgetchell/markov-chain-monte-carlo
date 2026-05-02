# markov-chain-monte-carlo

[![Crates.io](https://img.shields.io/crates/v/markov-chain-monte-carlo.svg)](https://crates.io/crates/markov-chain-monte-carlo)
[![Downloads](https://img.shields.io/crates/d/markov-chain-monte-carlo.svg)](https://crates.io/crates/markov-chain-monte-carlo)
[![License](https://img.shields.io/crates/l/markov-chain-monte-carlo.svg)](LICENSE)
[![Docs.rs](https://docs.rs/markov-chain-monte-carlo/badge.svg)](https://docs.rs/markov-chain-monte-carlo)
[![CI](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/ci.yml/badge.svg)](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/ci.yml)
[![CodeQL](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/codeql.yml/badge.svg)](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/codeql.yml)
[![rust-clippy analyze](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/rust-clippy.yml/badge.svg)](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/rust-clippy.yml)
[![Codacy Quality Scan](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/codacy.yml/badge.svg)](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/codacy.yml)
[![codecov](https://codecov.io/gh/acgetchell/markov-chain-monte-carlo/graph/badge.svg)](https://codecov.io/gh/acgetchell/markov-chain-monte-carlo)
[![Audit dependencies](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/audit.yml/badge.svg)](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/audit.yml)

A composable **Markov Chain Monte Carlo (MCMC)** framework for arbitrary state spaces in Rust.

🚧 **Pre-release (0.x)** — This crate is under active development. APIs may change between minor versions.

See [CHANGELOG.md](CHANGELOG.md) for release history.
Citation metadata and background references are available in [CITATION.cff](CITATION.cff) and
[REFERENCES.md](REFERENCES.md).

---

## Overview

This crate provides:

- A generic Metropolis–Hastings implementation
- Two proposal models:
  - **`Proposal<S>`** — by-value proposals for simple/small state spaces
  - **`ProposalMut<S>`** — in-place mutation with rollback, for large combinatorial state spaces (triangulations, graphs) where cloning is expensive
- `Chain<S>` with `step` (by-value) and `step_mut` (in-place) methods
- `Sampler<S, T, P, R>` — ergonomic wrapper that bundles a chain with its target, proposal, and RNG; supports `run(n)` / `run_mut(n)` for bulk sampling and implements `Iterator`
- NaN and +∞ detection with automatic state rollback on error
- Chain statistics: `acceptance_rate()`, `total_steps()`, `reset_counters()`
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

### Using `Sampler` (ergonomic wrapper)

```rust
use markov_chain_monte_carlo::prelude::*;
use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};

# #[derive(Clone)] struct Scalar(f64);
# struct Normal;
# impl Target<Scalar> for Normal {
#     fn log_prob(&self, s: &Scalar) -> f64 { -0.5 * s.0 * s.0 }
# }
# struct RandomWalk;
# impl Proposal<Scalar> for RandomWalk {
#     fn propose<R: Rng + ?Sized>(&self, c: &Scalar, r: &mut R) -> Scalar {
#         Scalar(c.0 + r.random_range(-1.0..1.0))
#     }
# }
fn main() -> Result<(), McmcError> {
    let mut rng = StdRng::seed_from_u64(42);
    let chain = Chain::new(Scalar(0.0), &Normal)?;
    let mut sampler = Sampler::new(chain, &Normal, &RandomWalk, &mut rng);

    // Burn-in
    sampler.run(1000)?;
    sampler.chain_mut().reset_counters();

    // Production
    sampler.run(10_000)?;
    assert!(sampler.chain_ref().acceptance_rate() > 0.0);
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
        if state.spins.is_empty() { return None; }
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
- [ ] `serde` feature for chain checkpointing

## Contributing

A short local workflow:

```bash
just setup        # Install/verify dev tools
just check        # Run non-mutating validation
just fix          # Apply formatters/auto-fixes
just ci           # Run the full local CI simulation
```

For the full command list, run `just --list`. Development tooling details live in
[`docs/dev/rust.md`](docs/dev/rust.md), code layout is summarized in
[`docs/code_organization.md`](docs/code_organization.md), release steps live in
[`docs/RELEASING.md`](docs/RELEASING.md), and AI assistants should follow
[`AGENTS.md`](AGENTS.md).

## Citation

If you use this crate in academic work or downstream research software, please cite it using
[`CITATION.cff`](CITATION.cff) or GitHub's "Cite this repository" feature. A Zenodo DOI can be
added after an archived tagged release.

## References

For canonical background references for Metropolis-Hastings, MCMC, and the example models, see
[`REFERENCES.md`](REFERENCES.md).

## AI Agents

This repository contains an `AGENTS.md` file, which defines the canonical rules and invariants
for AI coding assistants and autonomous agents working on this codebase.

AI tools are expected to read and follow `AGENTS.md` when proposing or applying changes.

Portions of this library were developed with the assistance of AI tools including [ChatGPT],
[Claude], [Codex], and [CodeRabbit].

All code was written and/or reviewed and validated by the author.

For tool citation metadata, see the
[AI-assisted development tools](REFERENCES.md#ai-assisted-development-tools) section of
`REFERENCES.md`.

[ChatGPT]: https://openai.com/chatgpt
[Claude]: https://www.anthropic.com/claude
[Codex]: https://openai.com/codex
[CodeRabbit]: https://coderabbit.ai/
