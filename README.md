# markov-chain-monte-carlo

[![DOI](https://badgen.net/badge/DOI/10.5281%2Fzenodo.20033111/blue)](https://doi.org/10.5281/zenodo.20033111)
[![Crates.io](https://badgen.net/crates/v/markov-chain-monte-carlo)](https://crates.io/crates/markov-chain-monte-carlo)
[![Downloads](https://badgen.net/crates/d/markov-chain-monte-carlo)](https://crates.io/crates/markov-chain-monte-carlo)
[![License](https://badgen.net/github/license/acgetchell/markov-chain-monte-carlo)](https://github.com/acgetchell/markov-chain-monte-carlo/blob/v0.4.2/LICENSE)
[![Docs.rs](https://docs.rs/markov-chain-monte-carlo/badge.svg)](https://docs.rs/markov-chain-monte-carlo)
[![CI](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/ci.yml/badge.svg)](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/ci.yml)
[![CodeQL](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/codeql.yml/badge.svg)](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/codeql.yml)
[![zizmor](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/zizmor.yml/badge.svg)](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/zizmor.yml)
[![rust-clippy analyze](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/rust-clippy.yml/badge.svg)](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/rust-clippy.yml)
[![codecov](https://codecov.io/gh/acgetchell/markov-chain-monte-carlo/graph/badge.svg)](https://codecov.io/gh/acgetchell/markov-chain-monte-carlo)
[![Audit dependencies](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/audit.yml/badge.svg)](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/audit.yml)

![Ising energy trace](https://raw.githubusercontent.com/acgetchell/markov-chain-monte-carlo/v0.4.2/docs/assets/ising_energy_trace.png)

_Open-boundary 1-D Ising chain with 50 spins, β = 0.5, J = 1, seed 42, 5,000 burn-in steps, and 20,000 recorded steps. `just notebook-check`
regenerates `target/ising_1d_trace.csv`, the executed notebook, and the PNG under `target/notebooks/`; `just notebook-ising-figure` promotes that exact PNG to
the tracked image above._

Research-oriented Metropolis-Hastings tools in Rust for ordinary numeric states, large combinatorial state spaces, and proposal implementations that need
rollback-safe mutation or delayed commits.

## 📐 Introduction

This library implements composable Metropolis-Hastings sampling in Rust for workflows where the state space, proposal mechanism, and measurement strategy are
application-specific. It is designed for research code where proposal kernels, observables, and scientific validity checks live in domain code, while the
sampler owns the transition bookkeeping.

The Metropolis-Hastings contract is explicit: targets return unnormalized natural log weights, proposals describe the same concrete transition they generate,
and proposal asymmetry stays in the Hastings correction. The crate is useful for simple numeric examples, spin systems, triangulation moves, and other large
combinatorial state spaces where cloning, rollback, or delayed commits matter.

🚧 **Pre-release (0.x)** — This is research software under active development. APIs may change before 1.0.

Use this crate when you want:

- a generic Metropolis-Hastings chain over user-defined state spaces
- by-value, in-place, and delayed-commit proposal APIs
- log-space acceptance calculations with NaN/+infinity checks
- additive target composition for bias potentials, energy/action terms, externally supplied learned regularizers, and other log-weight modifiers
- observable measurement APIs, streaming statistics, and binning-based uncertainty estimates for correlated samples
- trace recording and CSV export for downstream MCMC diagnostics
- thinning helpers for long sampler runs
- optional `serde` checkpointing with validated resume flows
- detailed-balance diagnostics for proposal development

This crate provides the sampler mechanics; proposal correctness, ergodicity, convergence assessment, and scientific model choice remain domain-specific
responsibilities.

## 🧪 Scientific basis

The acceptance rule is the standard Metropolis-Hastings correction:

```text
alpha(x, y) = min(1, exp(log pi(y) - log pi(x) + log q(x | y) - log q(y | x)))
```

`Target<S>` supplies `log pi(s)` up to an additive constant. Proposal implementations either use the default symmetric correction or supply the proposal ratio
for the same concrete transition they generate. For asymmetric combinatorial moves, that usually means accounting for move-kind probabilities, valid-site
counts, reverse-site counts, and invalid-move handling.

Physics actions and externally supplied learned regularizer terms fit the same target interface: implement `Target::log_prob` as an unnormalized log weight, or
as `-E(state)` when working in energy/action form. Training learned energies or adaptive proposal policies is outside the current crate scope.

The crate checks local transition mechanics: log-space acceptance, invalid floating-point values, rollback for in-place proposals, delayed commits, counters,
checkpoints, and empirical detailed-balance diagnostics for representative discrete transitions. It does not prove that a proposal is ergodic, that a chain has
mixed, or that a scientific model is appropriate for a downstream study.

For the detailed contract, see the
[scientific basis and scope guide](https://github.com/acgetchell/markov-chain-monte-carlo/blob/v0.4.2/docs/scientific_basis.md).

## ✨ Features

- Generic `Chain<S>` over user-defined state spaces with explicit accepted/rejected counters.
- Log-space Metropolis-Hastings acceptance with typed errors for NaN and positive-infinite target or proposal values.
- `AdditiveTarget` for composing model and bias log-weight terms without mixing them into proposal-ratio corrections.
- Three proposal workflows: by-value `Proposal`, rollback-safe in-place `ProposalMut`, and delayed-commit `DelayedProposal`.
- `Sampler` helpers for repeated and chunked runs, iterator-style sampling, thinning, observations, and counter resets after burn-in.
- Streaming `OnlineStats` and `BinningAnalysis` for long correlated runs without retaining every sample.
- `TraceRecorder` and `Trace` for numeric observable traces with chain IDs, accept/reject metadata, and CSV export.
- `ChainCheckpoint` restore APIs that recompute cached log-probabilities against the resumed target.
- Optional `serde` support for serializing chains and samplers into the same portable checkpoint shape.
- Detailed-balance diagnostics for proposal tests on representative discrete transitions.

## Contents

- [📐 Introduction](#-introduction)
- [🧪 Scientific basis](#-scientific-basis)
- [✨ Features](#-features)
- [🚀 Quick start](#-quick-start)
- [🧭 Choosing an API](#-choosing-an-api)
- [📦 Cargo features](#-cargo-features)
- [🧪 Examples](#-examples)
- [📖 Documentation](#-documentation)
- [Performance](#performance)
- [👀 Reviewer guide](#-reviewer-guide)
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

Rust 1.98.0 or newer is required.

Minimal by-value Metropolis-Hastings sampler. This example demonstrates the transition mechanics; convergence assessment remains a separate analysis step.

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
        let _ = chain.step(&Normal, &proposal, &mut rng)?;
    }

    assert!(chain.acceptance_rate() > 0.0);
    Ok(())
}
```

## 🧭 Choosing an API

- Start with `Proposal` and `Chain::step` when state cloning is cheap.
- Use `ProposalMut` and `Chain::step_mut` when cloning state is expensive and rollback is simple; its `Info` metadata is returned in structured `Step`
  telemetry for accepted and rejected proposals. `StepOutcome::NoProposal` carries metadata only when `ProposalMut::no_proposal_info` provides it.
- Drive `Sampler::step_mut` explicitly when every transition needs metadata. Bulk `Sampler::run_mut*` methods deliberately skip `Info` construction and
  proposal telemetry hooks, so metadata that would be discarded is not constructed.
- Use `DelayedProposal` and `Chain::step_delayed` when you need to plan and score a concrete move before mutating state.
- Use `AdditiveTarget` when the target log weight is the sum of model, bias, energy, action, or externally supplied regularizer terms.
- Use `DelayedStep` telemetry, `StepOutcome`, and `DelayedProposal::no_plan_info` when delayed proposals need domain-specific per-step records.
- Use `Sampler` when you want ergonomic repeated runs, resumable chunks, iterator-based sampling, or observing helpers.
- Parse raw positive thinning counts with `ThinningInterval::new`, then reuse the validated interval across `Sampler::*_with_thinning` calls.
- Use `Sampler::run_delayed_chunk_observing` to record per-step delayed telemetry and post-step state while resuming chunked runs from a `ChainCheckpoint`.
- Use `TraceRecorder` when you need reusable numeric traces with chain IDs, acceptance metadata, target log-probabilities, and CSV export.
- Use `verify_detailed_balance*` helpers in proposal tests for representative discrete transitions.
- Use `OnlineStats` and `BinningAnalysis` when long runs should stream statistics instead of retaining every sample.

When migrating from the previous in-place API, add `ProposalMut::Info`, implement `info` (and optionally `no_proposal_info`), make proposal mutation hooks
accept `&mut self`, and read the `Step` returned by `Chain::step_mut` or `Sampler::step_mut` instead of a boolean. Rejection must restore both the state and
proposal-internal transition state. Keep telemetry hooks observational because bulk execution may skip them. For earlier thinning and
delayed-telemetry APIs, replace raw thinning `usize` arguments with a parsed `ThinningInterval`, handle the underlying sampler or observation error directly
instead of matching `ThinningError::Run`, and replace `Step` field reads with the corresponding `outcome()`, `info()`, `log_prob_before()`,
`log_prob_after()`, and `log_alpha()` accessors.

By-value `Chain::step` and `Sampler::step` now return `Step<()>`, and the by-value `Sampler` iterator yields `Result<Step<()>, McmcError>`; inspect or
explicitly discard that telemetry where older code expected `()`. Construct `DiscreteProposalRatio` with the source-to-destination and
destination-to-source endpoint weights as well as their normalizers. With the `serde` feature, `Sampler` now uses the same canonical checkpoint shape as its
owned `Chain`; deserialize a `ChainCheckpoint`, restore it with `Chain::from_checkpoint(checkpoint, target)`, and pass the resulting `Chain` plus the target,
proposal, and RNG to `Sampler::new`.

## 📦 Cargo features

- `serde` — serialize `Chain` and `Sampler` as canonical checkpoints, plus serialize/deserialize `ChainCheckpoint` for validated resume flows.

## 🧪 Examples

Complete runnable examples live in [`examples/`](https://github.com/acgetchell/markov-chain-monte-carlo/tree/v0.4.2/examples):

- [`examples/normal_1d.rs`](https://github.com/acgetchell/markov-chain-monte-carlo/blob/v0.4.2/examples/normal_1d.rs) — by-value random-walk sampler for a
  normal target
- [`examples/ising_1d.rs`](https://github.com/acgetchell/markov-chain-monte-carlo/blob/v0.4.2/examples/ising_1d.rs) — in-place spin-flip proposals for a
  non-`Clone` open-boundary Ising state, with energy/magnetization trace CSV export
- [`examples/iterator_sampling.rs`](https://github.com/acgetchell/markov-chain-monte-carlo/blob/v0.4.2/examples/iterator_sampling.rs) — `Sampler` as an iterator
- [`examples/detailed_balance.rs`](https://github.com/acgetchell/markov-chain-monte-carlo/blob/v0.4.2/examples/detailed_balance.rs) — by-value, in-place,
  delayed, and batch detailed-balance checks
- [`examples/delayed_chunked_telemetry.rs`](https://github.com/acgetchell/markov-chain-monte-carlo/blob/v0.4.2/examples/delayed_chunked_telemetry.rs) — per-step
  delayed telemetry and post-step state recorded across resumable chunks
- [`examples/additive_target_bias.rs`](https://github.com/acgetchell/markov-chain-monte-carlo/blob/v0.4.2/examples/additive_target_bias.rs) — model and bias
  log-weight terms composed with `AdditiveTarget`

Run them with:

```bash
just examples
```

For proposal-specific testing patterns, see the
[proposal validation guide](https://github.com/acgetchell/markov-chain-monte-carlo/blob/v0.4.2/docs/proposal_validation.md).

The Ising trace notebook lives at
[`notebooks/ising_trace_analysis.ipynb`](https://github.com/acgetchell/markov-chain-monte-carlo/blob/v0.4.2/notebooks/ising_trace_analysis.ipynb). Run
`just notebook-check` to generate `target/ising_1d_trace.csv`, validate the source notebook, and write a headlessly executed copy under `target/notebooks/`.

## 📖 Documentation

- [docs.rs API documentation](https://docs.rs/markov-chain-monte-carlo)
- [Reviewer guide](https://github.com/acgetchell/markov-chain-monte-carlo/blob/v0.4.2/docs/reviewer_guide.md)
- [Changelog](https://github.com/acgetchell/markov-chain-monte-carlo/blob/v0.4.2/CHANGELOG.md)
- [Scientific basis and scope](https://github.com/acgetchell/markov-chain-monte-carlo/blob/v0.4.2/docs/scientific_basis.md)
- [Proposal validation guide](https://github.com/acgetchell/markov-chain-monte-carlo/blob/v0.4.2/docs/proposal_validation.md)
- [Roadmap](https://github.com/acgetchell/markov-chain-monte-carlo/blob/v0.4.2/docs/roadmap.md)
- [Code organization guide](https://github.com/acgetchell/markov-chain-monte-carlo/blob/v0.4.2/docs/code_organization.md)
- [Rust development workflow](https://github.com/acgetchell/markov-chain-monte-carlo/blob/v0.4.2/docs/dev/rust.md)
- [Release process](https://github.com/acgetchell/markov-chain-monte-carlo/blob/v0.4.2/docs/RELEASING.md)
- [Security policy](https://github.com/acgetchell/markov-chain-monte-carlo/blob/v0.4.2/SECURITY.md)

## Performance

<!-- PERFORMANCE:BEGIN -->

**v0.4.2 working tree against v0.4.1**; median elapsed time, with recorded confidence bounds where available.

![Release workload time ratios](https://raw.githubusercontent.com/acgetchell/markov-chain-monte-carlo/v0.4.2/docs/archive/performance/v0.4.2-vs-v0.4.1.svg)

| Workload | Baseline | Current | Relative time |
| --- | --- | --- | --- |
| `chain_step_by_value` | 15.80 ns (15.74 ns - 15.87 ns) | 16.91 ns (16.90 ns - 16.93 ns) | 1.07x slower |
| `chain_step_delayed_no_plan` | 1.03 ns (1.03 ns - 1.03 ns) | 0.73 ns (0.73 ns - 0.73 ns) | 1.42x faster |
| `chain_step_mut_accept` | 12.68 ns (12.66 ns - 12.70 ns) | 13.69 ns (13.65 ns - 13.72 ns) | 1.08x slower |
| `chain_step_mut_reject_rollback` | 102.78 ns (102.21 ns - 103.47 ns) | 198.01 ns (197.44 ns - 198.35 ns) | 1.93x slower |
| `observing_manual_online_sum_100` | 934.48 ns (933.26 ns - 935.27 ns) | 1.28 µs (1.27 µs - 1.28 µs) | 1.36x slower |
| `observing_run_observing_buffer_100` | 1.56 µs (1.56 µs - 1.56 µs) | 1.78 µs (1.77 µs - 1.78 µs) | 1.14x slower |
| `observing_run_observing_into_binning_100` | 1.97 µs (1.96 µs - 1.97 µs) | 2.23 µs (2.23 µs - 2.23 µs) | 1.13x slower |
| `observing_run_observing_into_online_stats_100` | 1.25 µs (1.24 µs - 1.25 µs) | 1.60 µs (1.60 µs - 1.60 µs) | 1.28x slower |
| `sampler_run_by_value_100` | 1.50 µs (1.50 µs - 1.51 µs) | 1.54 µs (1.53 µs - 1.54 µs) | 1.02x slower |
| `sampler_run_mut_100` | 1.04 µs (1.04 µs - 1.04 µs) | 1.28 µs (1.28 µs - 1.28 µs) | 1.23x slower |

Coverage: 10 comparable, 12 current-only, 3 baseline-only workloads.

These workload timings do not measure mixing, convergence, or effective sample size. Ratios are point estimates, not significance tests.

- [Report and measurement context](https://github.com/acgetchell/markov-chain-monte-carlo/blob/v0.4.2/docs/PERFORMANCE.md)
- [CSV measurements](https://github.com/acgetchell/markov-chain-monte-carlo/blob/v0.4.2/docs/archive/performance/v0.4.2-vs-v0.4.1.csv)
- [JSON provenance](https://github.com/acgetchell/markov-chain-monte-carlo/blob/v0.4.2/docs/archive/performance/v0.4.2-vs-v0.4.1.provenance.json)

<!-- PERFORMANCE:END -->

## 👀 Reviewer guide

For a short reading path through the repository's scientific contract, validation strategy, roadmap boundaries, and reproducible local checks, see
[`docs/reviewer_guide.md`](https://github.com/acgetchell/markov-chain-monte-carlo/blob/v0.4.2/docs/reviewer_guide.md).

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

See [CONTRIBUTING.md](https://github.com/acgetchell/markov-chain-monte-carlo/blob/v0.4.2/CONTRIBUTING.md) for the full contributor guide (project layout,
development workflow, code style, testing, documentation layout, performance/benchmarking, and the release process). Community expectations live in
[`CODE_OF_CONDUCT.md`](https://github.com/acgetchell/markov-chain-monte-carlo/blob/v0.4.2/CODE_OF_CONDUCT.md). AI assistants should follow
[`AGENTS.md`](https://github.com/acgetchell/markov-chain-monte-carlo/blob/v0.4.2/AGENTS.md).

Quick local workflow: run `just setup` once, then run `just check` before opening a pull request. For the full command list, run `just --list`.

## 📚 Citation

If you use this crate in academic work or downstream research software, please cite it using
[`CITATION.cff`](https://github.com/acgetchell/markov-chain-monte-carlo/blob/v0.4.2/CITATION.cff) or GitHub's "Cite this repository" feature.

## 🔎 References

For canonical background references for Metropolis-Hastings, MCMC, and the example models, see
[`REFERENCES.md`](https://github.com/acgetchell/markov-chain-monte-carlo/blob/v0.4.2/REFERENCES.md).

## 🤖 AI Agents

AI coding assistants should read [`AGENTS.md`](https://github.com/acgetchell/markov-chain-monte-carlo/blob/v0.4.2/AGENTS.md) before proposing or applying
changes. See [CONTRIBUTING.md](https://github.com/acgetchell/markov-chain-monte-carlo/blob/v0.4.2/CONTRIBUTING.md#ai-assisted-development) for the repository's
AI-assisted development note.

## 📜 License

This project is licensed under the [BSD 3-Clause License](https://github.com/acgetchell/markov-chain-monte-carlo/blob/v0.4.2/LICENSE).
