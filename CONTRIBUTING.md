# Contributing to markov-chain-monte-carlo

Thank you for your interest in contributing to the [**markov-chain-monte-carlo**][mcmc-lib] crate! This document is a practical guide for contributors. AI
agents and autonomous tooling should follow [`AGENTS.md`](AGENTS.md), which is the canonical rule set; this file mirrors the human-facing parts of those rules.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment Setup](#development-environment-setup)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Code Style and Standards](#code-style-and-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Performance and Benchmarking](#performance-and-benchmarking)
- [Submitting Changes](#submitting-changes)
- [Types of Contributions](#types-of-contributions)
- [Release Process](#release-process)
- [Getting Help](#getting-help)

## Code of Conduct

This project is governed by [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md). The community is built on:

- **Respectful collaboration** in scientific computing and statistics
- **Inclusive participation** regardless of background or experience level
- **Excellence in numerical correctness** and idiomatic Rust
- **Open knowledge sharing** about MCMC, Metropolis–Hastings, and proposal design

## Getting Started

### Prerequisites

Before you begin, ensure you have:

1. **Rust 1.98.0** (pinned via [`rust-toolchain.toml`](rust-toolchain.toml) — automatically handled by rustup)
2. **Git** for version control
3. **Just** (command runner): `cargo install just`
4. **uv** (Python 3.14 tooling): install the repository-pinned version with
   `curl -LsSf "https://astral.sh/uv/$(just --evaluate uv_version)/install.sh" | sh` (see [astral.sh/uv][uv])

### Quick Start

1. **Fork and clone** the repository:

   ```bash
   git clone https://github.com/yourusername/markov-chain-monte-carlo.git
   cd markov-chain-monte-carlo
   ```

2. **Setup the development environment** (installs repository-managed tools and verifies system prerequisites — see
   [Development Environment Setup](#development-environment-setup) for what gets installed or checked):

   ```bash
   just setup
   ```

3. **Run tests**:

   ```bash
   just test            # Focused unit + doc tests (fast)
   just test-all        # Broad release Rust + doc + Python tooling tests
   ```

4. **Try the examples**:

   ```bash
   just examples        # Runs all six examples, including additive-target and delayed-telemetry workflows
   ```

5. **Run benchmarks** (optional):

   ```bash
   just bench-compile   # Compile Criterion benchmarks without measuring
   just bench           # Run Criterion benchmarks
   ```

6. **Code-quality checks**:

   ```bash
   just check           # Run all non-mutating linters / validators
   just ci              # Full local CI simulation (mirrors .github/workflows/ci.yml)
   just fix             # Apply formatters / auto-fixes (mutating)
   ```

## Development Environment Setup

### Automatic Toolchain Management

This project pins its Rust toolchain via [`rust-toolchain.toml`](rust-toolchain.toml). When you enter the project directory, `rustup` will automatically:

- install the correct Rust version (1.98.0) if you don't have it
- switch to the pinned version for this project
- install required components (clippy, rustfmt, rust-docs, rust-std, rust-src, rust-analyzer)

**No manual toolchain setup is needed** — just have `rustup` installed ([rustup.rs][rustup]).

### External Tools

`just setup` installs repository-managed tools and verifies the system prerequisites the project relies on:

- [actionlint](https://github.com/rhysd/actionlint) — GitHub Actions workflow linter, installed through `actionlint-py` in the uv dev environment
- [cargo-edit](https://github.com/killercup/cargo-edit) — Cargo dependency requirement updates
- [cargo-llvm-cov](https://github.com/taiki-e/cargo-llvm-cov) — coverage reports
- [cargo-nextest](https://nexte.st/) — Rust unit and integration test runner
- [cargo-update](https://github.com/nabijaczleweli/cargo-update) — installed Cargo tool updates
- [dprint](https://dprint.dev/) — YAML formatter
- [git-cliff](https://git-cliff.org/) — changelog generation
- [jq](https://jqlang.github.io/jq/download/) — system-provided JSON validator; install it with your package manager or the official instructions
- [rumdl](https://rumdl.dev/) — Markdown formatter / linter
- [taplo](https://taplo.tamasfe.dev/) — TOML formatter / linter
- [typos](https://github.com/crate-ci/typos) — spell checker
- [uv](https://docs.astral.sh/uv/) — Python package and tool runner (used for `semgrep`, `ruff`, `ty`, and the changelog/tagging Python helpers in `scripts/`)
- [zizmor](https://github.com/zizmorcore/zizmor) — GitHub Actions security analyzer

The recipe checks `uv` and `jq` before managed installation work begins. If either system prerequisite is unavailable or the pinned `uv` version does not
match, it exits with installation guidance; Cargo and uv then install or synchronize the repository-managed tools.

## Project Structure

This crate is a single Rust library (no `src/main.rs`). The detailed file/module map lives in [`docs/code_organization.md`](docs/code_organization.md); use it
when asking, "I'm adding a new function/type/trait, which file owns it?"

At a high level:

- `src/` contains the public library modules and the canonical crate-level `//!` documentation.
- `examples/` contains complete runnable workflows.
- `tests/` and `benches/` contain integration validation and Criterion benchmarks.
- `docs/` contains topic guides such as scientific scope, proposal validation, roadmap, release, and Rust tooling notes.
- `scripts/` contains Python helpers for changelog and release workflows.
- Root configuration files (`justfile`, `Cargo.toml`, `rust-toolchain.toml`, `semgrep.yaml`, `dprint.json`, `cliff.toml`, `typos.toml`) define automation, build
  metadata, validation, formatting, and release behavior.

This file (`CONTRIBUTING.md`) covers contributor workflow and tooling. [`docs/code_organization.md`](docs/code_organization.md) covers the narrower
architectural placement question.

## Development Workflow

### Just Command Runner

This project uses [Just] as the primary task automation tool. The justfile defines every dev workflow.
Run bare `just` for the curated workflow guide and `just --list` for the complete grouped recipe reference. Public recipes are documented, grouped, and kept
in lexicographic source order so both views remain easy to scan.

**Essential Just commands:**

```bash
just setup           # Install managed tools / verify system prerequisites
just update          # Update dependencies, managed Cargo tools, and tool pins
just check           # Run linters / validators (non-mutating)
just ci              # Full local CI simulation (mirrors .github/workflows/ci.yml)
just fix             # Apply formatters / auto-fixes (mutating)
just test            # Focused unit + doc tests (fast)
just test-rust       # Broad release Rust tests + doctests
just test-all        # Broad Rust + Python tooling tests
just examples        # Run all examples
just bench           # Run Criterion benchmarks
just bench-compile   # Compile benchmarks without measuring
just changelog      # Regenerate CHANGELOG.md from git history
just clean           # Clean build artifacts
```

`just update` advances Cargo dependency requirements and lockfile entries, resolves the latest compatible versions for exact Python development-tool pins,
upgrades the Cargo-installed CLI tools managed by `just setup`, and reconciles their root justfile pins together with the active uv version. Review the
resulting manifest, lockfile, and tool-pin changes before committing them.

**Workflow help:**

```bash
just --list          # All available commands
just help-workflows  # Detailed workflow guidance
```

### Typical Development Cycle

1. **Start a feature/fix branch.** Prefer `{type}/{issue}-descriptor`, e.g. `fix/307-acceptance-rate`, `feat/315-thinning-helpers`, `doc/329-citation-notes`:

   ```bash
   git checkout -b feat/your-feature
   ```

2. **Iterate with the smallest changed-surface validator:**

   ```bash
   # edit code and docs
   just test-unit       # library unit-test changes
   just test-doc        # doctest-only changes
   just test-integration # one integration-test crate or the integration bucket
   just notebook-lint   # notebook or notebook-checker changes
   just fix             # apply formatters
   ```

3. **Compose each relevant focused bucket once for final non-core changes.** For core Rust changes or a GitHub-equivalent local run, use the full gate:

   ```bash
   just check           # non-mutating linters and validators
   just ci              # flat GitHub-equivalent validation union
   ```

4. **Submit:**

   ```bash
   git commit           # see commit-message rules below
   git push origin feat/your-feature
   # open a pull request
   ```

## Code Style and Standards

### Rust Code Style

- **Edition**: Rust 2024
- **MSRV**: 1.98.0 (pinned in `rust-toolchain.toml`)
- **Formatting**: `cargo fmt --all` (configured in `rustfmt.toml`)
- **Linting**: strict clippy with warnings as errors

### Python Tooling Style

- Keep Python helpers and tests portable across Linux, macOS, and Windows. Use `pathlib`, avoid platform-reserved fixture names, compare paths using native
  `Path` values or normalized relative POSIX text as appropriate, and sort filesystem-derived output with explicit platform-neutral keys.

### Linting Configuration

The fast `just clippy` recipe checks the core library surface used by `just check`:

```bash
CARGO_BUILD_WARNINGS=deny cargo clippy --locked --workspace --all-features --lib -- -W clippy::pedantic -W clippy::nursery -W clippy::cargo -A clippy::multiple_crate_versions
```

Cargo owns the warning-as-error policy so changing the policy does not invalidate compiled artifacts. The explicit `-W` flags remain because they enable
Clippy lint groups; `CARGO_BUILD_WARNINGS=deny` changes the severity of enabled lints but does not select those groups. The crate forbids `unsafe_code` and
warns on `missing_docs`; broken intra-doc links are denied. The full `just ci` gate uses `just clippy-all-targets` to match the GitHub Clippy SARIF workflow.
Focused test, example, and benchmark validators still provide execution or compile-contract evidence because ordinary compilation does not execute Clippy
lints.

### API Style

- **Prefer borrowed APIs by default.** Take references (`&T`, `&mut T`, `&[T]`) and return borrowed views when possible. Take ownership or return `Vec` only
  when required.
- **Log-space numerics.** Targets and proposal ratios cross the API boundary as `f64` log weights. `NaN` and `+∞` are explicit error conditions
  ([`McmcError`](https://docs.rs/markov-chain-monte-carlo/latest/markov_chain_monte_carlo/enum.McmcError.html)); `-∞` is a legal "impossible state" marker.
- **Defined floating-point semantics.** The five `f64::algebraic_{add,sub,mul,div,rem}` methods are forbidden throughout repository-owned Rust because their
  unspecified transformations can change precision, non-finite and signed-zero behavior, acceptance decisions, and reproducibility. Ordinary IEEE-754
  arithmetic and deliberate `f64::mul_add` use remain allowed. Any other relaxed or fast-math facility requires a separate tracked scientific review.
- **Rollback safety.** `ProposalMut::propose_mut` must pair with `undo` so that a rejected mutation leaves state observably unchanged. `DelayedProposal::commit`
  errors are reserved for genuinely exceptional failures applying an already-accepted concrete move.
- **Detailed balance.** New proposal kinds should ship with a `verify_detailed_balance*` test for representative discrete transitions.

### Project-Specific Semgrep Rules

The repo enforces some project conventions via Semgrep (`semgrep.yaml`). They cover things like avoiding `stdio` diagnostics in `src/`, banning `Box<dyn Error>`
in `src/`/examples/benches/doctests, banning `unwrap()`/`expect()` in doctests/examples/benches (prefer `?` and concrete errors, or a benches fixture helper),
requiring `expect()` reasons, forbidding unwrap-default-on-non-finite, and rejecting `f64::algebraic_*` operations while preserving `mul_add`. Run them with
`just semgrep` and `just semgrep-test`.

## Testing

### Test Categories

- **Library tests** — inline `#[cfg(test)] mod tests` in each source file:

  ```bash
  just test-unit
  ```

- **Doctests** — examples in `///` and `//!` doc comments:

  ```bash
  just test-doc
  ```

- **Integration tests** — `tests/` directory:

  ```bash
  just test-integration
  ```

- **Python tooling tests** — `pytest` over the `scripts/` helpers:

  ```bash
  just test-python
  ```

- **Benchmark compilation** (no measurement):

  ```bash
  cargo bench --no-run
  ```

- **Broad Rust CI tests** — all-feature library unit and integration tests share one optimized build, while doctests remain separate:

  ```bash
  just test-rust-ci
  just test-doc
  ```

- **Notebooks** — validate JSON, stable cells, output hygiene, cell compilation, Ruff formatting/linting, and Ty before headless execution:

  ```bash
  just notebook-lint
  just notebook-check
  just notebook-check-slow       # only explicitly configured heavy notebooks
  just notebook-clear-outputs-all # mutating cleanup before committing
  ```

  Executed notebooks and runtime caches are written under `target/notebooks/`; source notebooks are never overwritten by validation.

`just test` runs focused library tests through nextest plus rustdoc doctests through `cargo test --doc`; `just test-all` runs the broad release-profile Rust
test bucket, rustdoc doctests, and Python tooling tests.

### Property-Based Tests

For numerical/statistical invariants, use [proptest](https://docs.rs/proptest/) (already a dev-dependency). Put property-based Rust tests in integration
files named `tests/proptest_*.rs`; keep `src` unit tests focused on deterministic local behavior unless a private helper cannot be reached otherwise. Seed
RNGs explicitly so failing cases reproduce.

### Test Conventions

- Use deterministic seeds (`StdRng::seed_from_u64(...)`) for randomized tests.
- Keep individual unit tests under ~1 second.
- For detailed-balance diagnostics, exercise both the per-transition and batch helpers.
- Doctests are part of the API contract — when you change behavior, update the doctest, don't `,no_run`/`,ignore` it away.

## Documentation

This crate carries a layered documentation set: a README landing page that is included at the top of docs.rs, a long-form `//!` block appended below it on
docs.rs, per-topic docs under `docs/`, and academic references in `REFERENCES.md`.

### Documentation Layout

`src/lib.rs` includes the README during rustdoc builds:

```rust
#![cfg_attr(any(doc, doctest), doc = include_str!("../README.md"))]
```

Rules:

- Edit `README.md` directly for the public landing page: badges, pitch/status, install snippet, MSRV, Cargo features, minimal quick start, API-choice guide,
  examples, docs, contributing, citation, and ecosystem links.
- Keep README code examples valid as doctests. The README is included during `cargo test --doc`.
- Keep `src/lib.rs //!` focused on programming-contract material that should appear below the README on docs.rs: API semantics, numerical behavior, proposal
  responsibilities, checkpoint behavior, detailed-balance diagnostics, and streaming statistics.
- Avoid duplicating long-form content between README and `src/lib.rs //!`. Short orientation overlap is fine; scientific scope belongs in `docs/`, API behavior
  belongs in `src/lib.rs //!`, and landing-page prose belongs in README.

For the full agent-facing rule set, see the `## Documentation generation` section of [`AGENTS.md`](AGENTS.md).

### Other Documentation Standards

- **Public APIs**: every public function, struct, trait, and module needs a `///` (or `//!` for modules) doc comment. The crate has `missing_docs = "warn"`.
- **Worked examples**: include a runnable doctest for non-trivial public items.
- **Mathematical context**: explain the statistical / numerical meaning of values, especially log-space conventions and where `NaN` / `±∞` are meaningful.
- **References**: when adding a new method or paper, add an entry to [`REFERENCES.md`](REFERENCES.md) and cite it from the relevant doc.

### Generating Docs Locally

```bash
just doc                                   # cargo doc --no-deps --document-private-items
RUSTDOCFLAGS="-D warnings" cargo doc       # fail on rustdoc warnings
```

### Per-topic docs

Long-form discussion lives under `docs/`. Update these alongside code changes that affect them.

- [`docs/code_organization.md`](docs/code_organization.md) — per-module "where does new code go?" guidance for `src/*.rs`
- [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md) — local regression checks, release comparisons, durable assets, and report promotion
- [`docs/reviewer_guide.md`](docs/reviewer_guide.md) — short reading path for scientific and engineering reviewers
- [`docs/scientific_basis.md`](docs/scientific_basis.md) — Metropolis–Hastings contract and scope discussion that expands the README scientific-basis summary
- [`docs/proposal_validation.md`](docs/proposal_validation.md) — proposal-author testing patterns and `verify_detailed_balance*` usage
- [`docs/roadmap.md`](docs/roadmap.md) — planned feature work
- [`docs/dev/rust.md`](docs/dev/rust.md) — Rust toolchain notes and tooling deep-dive
- [`docs/RELEASING.md`](docs/RELEASING.md) — release procedure

## Performance and Benchmarking

Benchmarks live in [`benches/`](benches/) and use [Criterion](https://docs.rs/criterion).

```bash
just bench-latest           # run the fixed-seed release-signal set
just bench-latest-vs-last   # rerun and compare with the saved local baseline
just performance-local      # compare the current tree with the latest stable release
just performance-rerender   # rebuild the curated report from saved release measurements
just bench                  # run all benchmarks for broader profiling
just bench-compile          # compile benchmark harness without measuring
cargo bench --bench stepping <filter>   # run a subset
```

Use `just bench-save-last` before the first `bench-latest-vs-last` run. Release maintainers use `just performance-release` to save validated CSV/JSON evidence
and update the curated report, `just performance-rerender` to reproduce that report without remeasuring, and `just performance-github-assets` for comparisons
that consume durable release artifacts without local measurements. See
[`docs/BENCHMARKING.md`](docs/BENCHMARKING.md) for the command contracts and interpretation limits.

Performance guidelines:

- profile before optimizing
- prefer borrowed APIs and constant-memory accumulators (`OnlineStats`, `BinningAnalysis`) for hot loops
- avoid intermediate `Vec` allocations in step / observe paths
- be honest about what hot-path work crosses the API boundary as log weights vs. internal exact arithmetic

Coverage:

`just setup` and the Codecov workflow install `llvm-tools-preview` because `cargo-llvm-cov` needs the LLVM coverage tools. The pinned toolchain keeps
that coverage-only component out of the default `rustup` install path.

```bash
just coverage        # local HTML report (target/llvm-cov/html/index.html)
just coverage-ci     # cobertura.xml for CI
```

## Submitting Changes

### Pull Request Process

1. Fork and create a feature branch (`{type}/{issue}-descriptor`).
2. Make changes following the coding standards above.
3. Add tests for new functionality.
4. Run `just check` (or `just ci` for the full local simulation).
5. Update relevant `docs/*.md` and add to `REFERENCES.md` if you introduced a new method/paper.
6. Open a pull request with a descriptive title and body.

### Commit Messages

Commit subjects and bodies feed `CHANGELOG.md` through `git-cliff`. Use [Conventional Commits](https://www.conventionalcommits.org/):

```text
<type>: <summary>

Optional body explaining the change in plain prose.

- specific change
- another specific change

Refs: #123
```

Valid `<type>` values:

- `feat` — Added (new feature)
- `fix` — Fixed (bug fix)
- `perf` — Performance
- `docs` — Documentation
- `refactor` / `test` / `style` — Changed
- `build` / `chore` / `ci` — Maintenance

Breaking changes use bang notation (`feat!: remove deprecated API`) or a `BREAKING CHANGE:` footer trailer so `git-cliff` detects them.

Avoid Markdown headings (`#` through `###`) in the body — they conflict with changelog section headings. Use plain labels like `Refs:` or `Migration:` instead.

Do not include test commands, validation results, or `Tests:` sections in commit messages unless explicitly requested. Put validation summaries in PR
descriptions or review notes instead.

### Pull Request Checklist

- [ ] Tests pass (`just test-all`)
- [ ] Code is formatted (`cargo fmt`)
- [ ] No clippy warnings (`just clippy`)
- [ ] Doctests pass (`cargo test --doc`)
- [ ] Relevant `docs/*.md` updated
- [ ] No long-form API/contract content duplicated between the README and `src/lib.rs //!`; short landing-summary overlap is fine
- [ ] `just check` passes (`fmt-check`, `clippy`, `python-check`, `notebook-lint`, `validate-json`, `yaml-check`, `action-lint`, `zizmor`,
      `justfile-fmt-check`, `toml-fmt-check`, `toml-lint`, `markdown-check`, `spell-check`, `release-check`, `semgrep`, `semgrep-test`)
- [ ] Commit message follows the Conventional Commits format above

## Types of Contributions

### Bug Reports

- File an issue on GitHub
- Provide a minimal reproduction, ideally as a failing test or doctest
- Include relevant numerical context (state size, seed, dimensions, etc.)

### Feature Requests

- Open a discussion or issue first for non-trivial features
- Describe the use case and the proposed API surface
- Consider whether it belongs in this crate (sampling) or in a sibling crate (`delaunay`, `causal-triangulations`)

### Code Contributions

- Start with a focused PR (one feature or one fix)
- Add tests, including detailed-balance checks for new proposal kinds
- Document the numerical/statistical contract (log-space conventions, when `NaN`/`±∞` is rejected, what rollback guarantees hold)

### Documentation Contributions

- Fix typos and improve clarity
- Add worked examples to `///` or `//!` doc comments
- Improve `docs/*.md` topic guides
- Add references to [`REFERENCES.md`](REFERENCES.md) and cite them from the relevant docs

## Release Process

The full release procedure lives in [`docs/RELEASING.md`](docs/RELEASING.md). Highlights:

1. Update version metadata in `Cargo.toml`, `CITATION.cff`, and `pyproject.toml`; refresh `Cargo.lock` and `uv.lock`.
2. Regenerate `CHANGELOG.md` for the new tag: `just changelog-unreleased v0.X.Y`.
3. Run `just release-check` to confirm the release metadata and active current-version references agree.
4. Run `just performance-release`, review the curated report, and confirm it reproduces with `just performance-rerender`.
5. Run `just fix`, `just ci`, and `just publish-check` for the final local release validation.
6. Commit and push the release PR.
7. After merge, tag the release with `just tag v0.X.Y` and create the GitHub Release.
8. Publish to crates.io.

Doc-only changes still require a version bump on crates.io, so prefer to land documentation updates **before** publishing.

Breaking API changes and MSRV bumps are allowed in any release up to and including v1.0.0, including patch releases. The release number during this
pre-stability period reflects project scope rather than compatibility impact alone; breaking changes still use `feat!` or a `BREAKING CHANGE:` trailer.

After v1.0.0, this project follows [Semantic Versioning](https://semver.org/):

- **MAJOR**: breaking API changes (also flagged by `feat!`/`BREAKING CHANGE:` in commit messages)
- **MINOR**: new features (backwards compatible)
- **PATCH**: bug fixes and improvements

## Getting Help

- **GitHub Issues** — bug reports and feature requests
- **GitHub Discussions** — general questions and design discussion
- **`docs/`** — topic guides
- **`REFERENCES.md`** — academic references and AI-assisted-development tool citations
- **`AGENTS.md`** — canonical rules for AI assistants (mirrors a subset here)
- **`just help-workflows`** — local workflow guidance

For mathematical / statistical questions about the underlying algorithms (Metropolis–Hastings, detailed balance, autocorrelation analysis), see the references
cited from [`docs/scientific_basis.md`](docs/scientific_basis.md) and [`REFERENCES.md`](REFERENCES.md).

---

Thank you for contributing!

[mcmc-lib]: https://github.com/acgetchell/markov-chain-monte-carlo
[rustup]: https://rustup.rs/
[Just]: https://github.com/casey/just
[uv]: https://docs.astral.sh/uv/
