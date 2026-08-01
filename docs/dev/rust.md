# Rust Development

This repository is a single Rust library crate using Rust 1.97.1 and edition 2024.

## Core Commands

```bash
just check            # Non-mutating validation gate
just check-fast       # Fast compile check
just ci               # Full CI simulation
just ci-rust          # Rust correctness subset
just ci-portability   # Portability subset
just ci-repository-tooling  # Repository tooling subset
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
- `just markdown-check` - Markdown formatting check through rumdl
- `just spell-check` - Spellcheck through `typos`
- `just semgrep` - Repository-owned Rust and Python policy rules
- `just semgrep-test` - Tests for the repository-owned Semgrep rules

For cross-repo muscle memory, the same checks are also available through grouped lint aliases:

- `just lint` - all lint groups
- `just lint-code` - Rust formatting, Clippy, Python checks, Semgrep, and Semgrep rule tests
- `just lint-config` - JSON, TOML, YAML, GitHub Actions, and Actions security validation
- `just lint-docs` - Markdown formatting and spellcheck

`just ci` remains the comprehensive local and GitHub Actions entrypoint. It runs repository tooling checks, Python support-script tests, Rust correctness
checks, documentation, library tests, doctests, integration tests, deterministic example-output validation, and benchmark harness compilation.

The full command is factored into named subsets so CI-shape trade-offs are explicit and easy to time without changing coverage:

- `just ci-rust` - Rust formatting, Clippy, documentation, library tests, doctests, integration tests, and deterministic example-output validation.
- `just ci-portability` - fast compile checking, library tests, doctests, integration tests, and deterministic example-output validation for platform smoke
  checks.
- `just ci-repository-tooling` - Python, YAML, GitHub Actions, TOML, Markdown, spelling, Semgrep, Semgrep rule tests, and Python support-script tests.

The GitHub Actions `CI` workflow intentionally runs `just ci` on Linux, macOS, and Windows so all supported development platforms exercise the same
comprehensive validation gate.

## Rust 1.97.1 Audit

The MSRV and contributor toolchain use Rust 1.97.1 rather than 1.97.0 because the point release fixes an LLVM miscompilation. The audit below follows the
official [Rust 1.97.0 release notes](https://doc.rust-lang.org/stable/releases.html#version-1970-2026-07-09) and
[Rust 1.97.1 announcement](https://blog.rust-lang.org/2026/07/16/Rust-1.97.1/).

| Surface | Decision |
| --- | --- |
| Cargo warning policy | `just clippy` uses `CARGO_BUILD_WARNINGS=deny` instead of `-D warnings`, so changing warning severity does not invalidate the build cache. The explicit Clippy `-W` selectors remain because Cargo changes the severity of enabled lints but does not enable `pedantic`, `nursery`, or `cargo`. The standalone rustdoc command keeps `RUSTDOCFLAGS="-D warnings"` to express rustdoc-specific policy at that command boundary. |
| `Result<T, Infallible>` must-use behavior | Keep the typed infallible results. Library tests, doctests, examples, integration tests, and benchmark compilation exercise these APIs under 1.97.1 without exposing ignored results. |
| Symbol mangling and code generation | Keep the new v0 symbol mangling default. Repository benchmarks, backtraces, and LLVM coverage do not parse legacy symbol names. The 1.97.1 LLVM fix is the reason to require the point release rather than 1.97.0. |
| Linker messages | Keep the lint enabled and add no suppression. Local validation checks the host linker, while the Linux, macOS, and Windows CI matrix checks the supported platform linkers. Any future suppression must be limited to a demonstrated platform-specific false positive. |
| Integer, `NonZero`, and `RepeatN` APIs | No change. Existing `NonZeroUsize` values model validated counts rather than bit masks, and the crate does not construct a reusable `RepeatN`; adopting the new APIs would add churn without clarifying an invariant. |
| `resolver.lockfile-path` | No change. The crate commits `Cargo.lock`, and its build, benchmark, release, and coverage workflows use writable checkouts; there is no read-only source workflow that needs a relocated lockfile. |
| Clippy, rustfmt, and rustdoc | Keep the current configuration. The 1.97 Clippy lint set is enforced by `just clippy`; rustfmt produces no formatting changes; and the new rustdoc `--emit` and path-remapping options do not improve the current local or docs.rs workflow. |

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
- Property-based Rust tests live in integration files named `tests/proptest_*.rs`; keep `src` unit tests deterministic unless a private helper requires a
  local test.

## Benchmarks

Benchmarks use Criterion and are intentionally deterministic. Run all benchmarks with:

```bash
just bench
```

The full CI simulation (`just ci`) compiles benchmark harnesses with `just bench-compile`, but it does not run Criterion measurements. Benchmark harness
compilation uses all crate features so optional feature paths stay covered.

The initial `benches/stepping.rs` suite protects core transition costs:

- by-value `Chain::step`
- in-place `Chain::step_mut` acceptance and rollback paths
- delayed `Chain::step_delayed` accepted, rejected, and no-plan paths
- bulk `Sampler` run loops
- observing with `SampleBuffer` versus manual online accumulation

## Coverage

Coverage uses `cargo llvm-cov` with all crate features enabled.
`just setup` and the Codecov workflow install `llvm-tools-preview`, which provides the LLVM coverage tools used by `cargo-llvm-cov`. It stays out of
`rust-toolchain.toml` because normal build, lint, doc, and test workflows do not need it.

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
- `.github/workflows/ci.yml` runs `just ci` on Linux, macOS, and Windows.
- PR-running workflows cache Cargo-installed CI, coverage, and SARIF helper tools through `taiki-e/cache-cargo-install-action`.
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
- Avoid `unwrap()`/`expect()` in doctests, examples, and benchmarks too; prefer `?` with concrete errors, or an explicit fixture helper in benches.
- Use `#[expect(..., reason = "...")]` rather than `#[allow(clippy::...)]`.

## Publishing

Before publishing, prefer updating documentation first. Doc-only changes still require a version bump on crates.io. Release version updates should keep
`Cargo.toml`, `Cargo.lock`, `CITATION.cff`, `pyproject.toml`, and `uv.lock` in sync.
