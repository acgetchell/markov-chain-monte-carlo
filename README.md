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
- Pluggable proposal distributions
- Support for arbitrary state spaces (including discrete and combinatorial systems)

The design emphasizes:

- Zero-cost abstractions
- Log-space numerical stability
- Extensibility for research and experimentation

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

---

## Usage

⚠️ Not yet stabilized. API will change.

