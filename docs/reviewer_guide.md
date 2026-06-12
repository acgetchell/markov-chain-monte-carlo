# Reviewer Guide

This guide is a short reading path for reviewers who want to evaluate the crate's scientific and engineering claims without reading every module first.

## What to Read First

- [`README.md`](../README.md) — public overview, current scope, quick start, API choices, examples, ecosystem links, citation, and AI-assisted-development
  disclosure.
- [`docs/scientific_basis.md`](scientific_basis.md) — Metropolis-Hastings contract, additive target terms, diagnostics, and user responsibilities.
- [`docs/proposal_validation.md`](proposal_validation.md) — how proposal kernels are tested for rollback behavior, proposal ratios, and representative
  detailed-balance checks.
- [`docs/roadmap.md`](roadmap.md) — planned work and non-goals, including the baseline diagnostics needed before learned-proposal experiments.
- [`REFERENCES.md`](../REFERENCES.md) — canonical MCMC, statistics, statistical-physics, and tooling citations.

## What the README Should Answer

The README should stay compact. It should answer:

- What the crate is: research-oriented Metropolis-Hastings tools in Rust.
- Why it exists: reusable sampler mechanics for downstream scientific crates with domain-specific states, proposals, and observables.
- What is implemented now: by-value, in-place rollback, delayed-commit proposals, additive targets, traces, checkpoints, statistics, and detailed-balance
  diagnostics.
- What is not claimed: proposal ergodicity, convergence, scientific model validity, learned-energy training, or adaptive learned proposal policies.
- How to run it locally: install with Cargo, run examples, and use `just check` for the non-mutating validation gate.

Detailed derivations, proposal-author guidance, and long-form scope discussion belong in `docs/`, not in the README.

## Scientific Contract

The crate implements the standard Metropolis-Hastings transition rule:

```text
alpha(x, y) = min(1, exp(log pi(y) - log pi(x) + log q(x | y) - log q(y | x)))
```

`Target<S>` supplies an unnormalized natural log weight. Proposal implementations must describe the same concrete transition they generate, including
Hastings corrections for asymmetric moves.

The crate checks local transition mechanics: log-space acceptance, invalid floating-point values, rollback for in-place proposals, delayed commits,
checkpoints, counters, and empirical detailed-balance diagnostics. It does not prove that a proposal mixes well, that a chain has converged, or that a
scientific model is appropriate.

## AI and Future Work

AI tools were used as development aids and are cited in [`REFERENCES.md`](../REFERENCES.md). The maintainer reviews, edits, and validates changes before they
land.

Learned proposals are roadmap work. The current crate can compose externally supplied learned regularizer terms as target log weights, but it does not train
energy models or adaptive proposal policies. The roadmap intentionally puts multi-chain execution, tempering, and diagnostics ahead of learned-proposal
experiments so failures have useful baselines.

## Reproducible Local Checks

For a local review run:

```bash
just setup
just check
just examples
cargo test --doc --locked
```

`just check` is the main non-mutating gate. It runs Rust formatting and lint checks, Python support-script checks, notebook linting, workflow/security checks,
Markdown and spelling checks, and project Semgrep rules.
