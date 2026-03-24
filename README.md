# markov-chain-monte-carlo

A composable **Markov Chain Monte Carlo (MCMC)** framework for arbitrary state spaces in Rust.

🚧 **Pre-release (0.0.x)** — This crate is under active development and **not yet ready for production use**. APIs may change without notice.

---

## Overview

This crate provides:

- A generic Metropolis–Hastings implementation
- Pluggable proposal distributions
- Support for arbitrary state spaces (including discrete and combinatorial systems)

The design emphasizes:

- Zero-cost abstractions
- Log-space numerical stability
- Extensibility for research and experimentation

---

## Relationship to Other Crates

This crate is part of a broader ecosystem:

- `causal-triangulations` — CDT physics and simulation
- `delaunay` — geometric primitives
- `la-stack` — linear algebra

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

---

## Usage

⚠️ Not yet stabilized. API will change.

