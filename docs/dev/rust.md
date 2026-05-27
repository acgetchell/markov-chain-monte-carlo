# Rust Development

This repository is a single Rust library crate using Rust 1.95.0 and edition 2024.

## Core Commands

```bash
just check            # Non-mutating validation gate
just check-fast       # Fast compile check
just ci               # Full CI simulation
just fix              # Apply formatters/auto-fixes (mutating)
just lint             # All lint groups
just setup            # Install/verify external dev tools
just test             # Lib + doc tests
just test-all         # Lib + doc + integration + Python tooling tests
just bench-compile    # Compile Criterion benchmarks without measuring
just bench            # Criterion benchmarks
just examples         # Run all examples
```

## Validation

`just check` is the primary non-mutating gate. It currently runs:

- `just fmt-check` - Rust formatting check
- `just clippy` - Clippy with `pedantic`, `nursery`, and `cargo` warnings
- `just python-check` - Ruff formatting/linting and Ty type checking for Python tooling
- `just yaml-check` - YAML formatting check through dprint Pretty YAML
- `just action-lint` - GitHub Actions validation through `actionlint`
- `just zizmor` - GitHub Actions security analysis through `zizmor`
- `just toml-fmt-check` - TOML formatting check through Taplo
- `just toml-lint` - TOML validation through Taplo
- `just markdown-check` - Markdown formatting check through dprint
- `just spell-check` - Spellcheck through `typos`
- `just semgrep` - Repository-owned Rust and Python policy rules
- `just semgrep-test` - Tests for the repository-owned Semgrep rules

For cross-repo muscle memory, the same checks are also available through grouped lint aliases:

- `just lint` - all lint groups
- `just lint-code` - Rust formatting, Clippy, Python checks, Semgrep, and Semgrep rule tests
- `just lint-config` - JSON, TOML, YAML, GitHub Actions, and Actions security validation
- `just lint-docs` - Markdown formatting and spellcheck

`just ci` runs `just check` (including `zizmor`), benchmark harness compilation, documentation, tests, and deterministic example-output validation.

## Setup

Run `just setup` or `just setup-tools` to install and verify external tools:

- `actionlint`
- `cargo-llvm-cov`
- `cargo-nextest`
- `dprint`
- `git-cliff`
- `jq`
- `rumdl`
- `taplo`
- `typos`
- `uv`
- `zizmor`

The setup recipe uses Cargo for Rust tools and `uv sync --group dev` for project-managed Python tools. Semgrep, Ruff, Ty, actionlint, and the changelog helper
tests are pinned in `pyproject.toml` and invoked through `uv`.

## Line Length

Non-Rust tooling uses a 160-column policy for Ruff-managed Python support scripts, `rumdl`-managed Markdown, Taplo-managed TOML, and dprint-managed YAML. Rust
remains on the narrower `rustfmt` `max_width = 100` setting because wide Rust signatures, trait bounds, and method chains are harder to scan at 160 columns.

## Testing

- All tests: `just test-all`
- Fast Rust tests (library tests through nextest plus rustdoc doctests): `just test`
- Integration tests through nextest: `just test-integration`
- Python tooling tests: `just test-python`
- Single runnable test by name filter: `cargo nextest run chain_samples_near_mode`
- Examples: `just examples` builds all examples once, then runs the compiled binaries.

## Benchmarks

Benchmarks use Criterion and are intentionally deterministic. Run all benchmarks with:

```bash
just bench
```

The full CI simulation (`just ci`) compiles benchmark harnesses with `just bench-compile`, but it does not run Criterion measurements.

The initial `benches/stepping.rs` suite protects core transition costs:

- by-value `Chain::step`
- in-place `Chain::step_mut` acceptance and rollback paths
- delayed `Chain::step_delayed` accepted, rejected, and no-plan paths
- bulk `Sampler` run loops
- observing with `SampleBuffer` versus manual online accumulation

## Coverage

Coverage uses `cargo llvm-cov` with all crate features enabled.

- Local HTML report: `just coverage`
- CI Cobertura XML: `just coverage-ci`

`just coverage` generates and opens:

```text
target/llvm-cov/html/index.html
```

`just coverage-ci` generates:

```text
coverage/cobertura.xml
```

## Tooling

The lightweight tooling layer mirrors the useful parts of the `delaunay` repo:

- `.coderabbit.yml` configures CodeRabbit to use focused Rust, Actions, secret-scan, and Semgrep checks.
- `.codecov.yml` configures coverage thresholds and ignores examples.
- `.github/workflows/codeql.yml` runs CodeQL for Rust and GitHub Actions.
- `.github/workflows/semgrep-sarif.yml` uploads repository-owned Semgrep rule results to GitHub Code Scanning.
- `.github/workflows/zizmor.yml` runs zizmor for GitHub Actions security analysis.
- `clippy.toml` pins Clippy's MSRV to the crate MSRV.
- `cliff.toml` configures offline `git-cliff` changelog generation from squash commit bodies, annotated tag notes, and filtered dependency commits.
- `dprint.json` configures YAML formatting through dprint Pretty YAML with the repository's 160-column non-Rust line length.
- `pyproject.toml` pins Python-based development tools and configures Ruff's 160-column line length.
- `rumdl.toml` configures Markdown linting and formatting with the repository's 160-column non-Rust line length.
- `scripts/` contains changelog post-processing and release-tag helpers.
- `rustfmt.toml` keeps stable Rust formatting explicit at 100 columns.
- `.taplo.toml` keeps TOML formatting stable and Cargo-like with the repository's 160-column non-Rust line length.
- `typos.toml` configures spellcheck exclusions and project vocabulary.
- `ty.toml` restricts Ty type checking to Python tooling.
- `semgrep.yaml` contains repository-owned Rust and Python policy rules.

Keep these checks focused. Avoid broad community rule packs unless they prove low-noise for this crate.

## Rust Style

- Prefer borrowed APIs by default: take `&T`, `&mut T`, and `&[T]` when possible.
- Return borrowed views (`&T`, `&[T]`) when possible.
- Only take ownership or allocate returned `Vec`s when required.
- Keep production fallible paths typed; prefer `McmcError` or a specific error type over dynamic error erasure.
- Avoid `unwrap`, `expect`, and `panic!` in production `src/` code.
- Use `#[expect(..., reason = "...")]` rather than `#[allow(clippy::...)]`.

## Publishing

Before publishing, prefer updating documentation first. Doc-only changes still require a version bump on crates.io. Release version updates should keep
`Cargo.toml`, `Cargo.lock`, `CITATION.cff`, `pyproject.toml`, and `uv.lock` in sync.
