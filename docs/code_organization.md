# Code Organization Guide

This guide describes how the `markov-chain-monte-carlo` crate is organized and
where new code usually belongs.

## Project Structure

This repository is a single Rust library crate. There is no `src/main.rs`; the
public API starts in `src/lib.rs`.

```text
markov-chain-monte-carlo/
├── .github/
│   ├── CODEOWNERS
│   ├── dependabot.yml
│   └── workflows/
│       ├── audit.yml
│       ├── ci.yml
│       ├── codacy.yml
│       ├── codecov.yml
│       ├── codeql.yml
│       └── rust-clippy.yml
├── docs/
│   ├── dev/
│   │   └── rust.md
│   ├── code_organization.md
│   └── RELEASING.md
├── examples/
│   ├── ising_1d.rs
│   ├── iterator_sampling.rs
│   └── normal_1d.rs
├── src/
│   ├── chain.rs
│   ├── error.rs
│   ├── lib.rs
│   ├── observable.rs
│   ├── sampler.rs
│   └── traits.rs
├── tests/
│   ├── proptest_chain.rs
│   └── semgrep/
├── AGENTS.md
├── CHANGELOG.md
├── Cargo.toml
├── README.md
├── REFERENCES.md
├── justfile
└── semgrep.yaml
```

To refresh the tree, prefer generating it from tracked files:

```bash
git --no-pager ls-files | LC_ALL=C sort
```

## Library Modules

### `src/lib.rs`

The crate root wires the public module layout together. It re-exports the core
modules and defines the `prelude` for ergonomic user imports.

Keep `src/lib.rs` small. New public surface should usually live in a focused
module first, then be re-exported from the crate root or prelude only when it is
part of the intended user API.

### `src/error.rs`

Defines `McmcError`, the crate error type for invalid log-probabilities and
proposal ratios.

Add new variants conservatively. `McmcError` is `#[non_exhaustive]`, but error
changes still affect user matching and documentation.

### `src/observable.rs`

Defines measurement APIs and collection helpers:

- `Observable<S>` for infallible measurements
- `TryObservable<S>` for fallible measurements
- `ObservedStepError<StepError, ObservationError>` to keep sampling failures
  and measurement failures orthogonal
- `SampleBuffer<T>` for simple in-memory observation collection

Observables are shared across proposal workflows, so the core observable traits
and buffer belong in the shared prelude. Workflow-specific result aliases should
stay at the crate root unless a prelude needs them for ordinary examples or
doctests.

### `src/traits.rs`

Defines user extension points:

- `Target<S>` for target distributions through `log_prob(&S) -> f64`
- `Proposal<S>` for by-value proposals
- `ProposalMut<S>` for in-place proposals with rollback through an undo token

This is the right place for small, fundamental traits that users implement for
their own state spaces. Prefer borrowed parameters by default.

### `src/chain.rs`

Contains `Chain<S>`, the core Metropolis-Hastings state machine.

This module owns:

- current state and current log-probability
- accepted/rejected counters
- by-value `step`
- in-place `step_mut`
- state accessors and replacement helpers
- acceptance-rate and counter utilities

Algorithmic correctness belongs here. Higher-level convenience APIs should only
move into `Chain` when they are fundamental to a single chain's state.

### `src/sampler.rs`

Contains `Sampler<S, T, P, R>`, an ergonomic wrapper that bundles a chain with
its target distribution, proposal, and RNG.

This module owns:

- single-step forwarding methods
- bulk `run` and `run_mut` loops
- observing variants that measure derived quantities after sampling steps
- by-value `Iterator` support
- access to the bundled `Chain`

Use `Sampler` for workflow ergonomics; use `Chain` for the core transition
logic.

## Examples

Examples demonstrate complete, runnable sampling workflows:

- `examples/normal_1d.rs` shows a simple by-value random-walk sampler.
- `examples/ising_1d.rs` shows in-place mutation with rollback.
- `examples/iterator_sampling.rs` shows the by-value `Sampler` iterator API.

Keep examples deterministic when possible. The `validate-examples` recipe checks
for expected output markers, so example output should remain stable enough for
CI validation.

## Tests

Tests are split by scope:

- Inline `#[cfg(test)]` modules live beside the code they exercise.
- `tests/proptest_chain.rs` covers broader Metropolis-Hastings invariants.
- `tests/semgrep/` validates repository-owned Semgrep rules.

For narrow behavior changes, prefer focused unit tests near the implementation.
For invariants spanning by-value and in-place paths, prefer integration or
property tests.

## Tooling and Configuration

The main local workflows are defined in `justfile`:

- `just check` runs the non-mutating validation gate.
- `just ci` runs checks, documentation, tests, examples, and example output
  validation.
- `just fix` applies formatting fixes.
- `just publish-check` validates crates.io metadata and runs
  `cargo publish --dry-run`.

Configuration files support the same gate:

- `.github/workflows/` contains CI, coverage, CodeQL, Codacy, Clippy, and audit
  workflows.
- `semgrep.yaml` contains repository-owned Rust style diagnostics.
- `rustfmt.toml`, `clippy.toml`, `.taplo.toml`, `.yamllint`, and `typos.toml`
  keep formatting and linting consistent.

## Organization Conventions

- Keep the crate root focused on module wiring and public re-exports.
- Put core transition semantics in `Chain`; put workflow convenience in
  `Sampler`.
- Keep trait definitions minimal and stable because they shape downstream user
  implementations.
- Prefer borrowed APIs (`&T`, `&mut T`, `&[T]`) unless ownership is necessary.
- Preserve log-space calculations and explicit NaN/+infinity handling.
- Document public API additions with examples or doctests when they clarify
  intended use.
- Add tests near the behavior they protect, then broaden to property tests when
  a change affects Metropolis-Hastings invariants.
- Run `just check` before handing off ordinary changes, and `just ci` before
  release or broad behavior changes.
