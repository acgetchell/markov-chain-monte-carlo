# Rust Development

This repository is a single Rust library crate using Rust 1.98.0 and edition 2024. Auxiliary repository tooling requires Python 3.14 and is managed by uv.

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
just setup            # Install managed tools / verify system prerequisites
just update           # Update dependencies, managed Cargo tools, and tool pins
just test             # Focused unit + doc tests
just test-unit        # Focused library unit tests
just test-integration # Focused integration tests
just test-rust-ci     # All-feature release lib + integration tests in one nextest pass
just test-rust        # Broad Rust CI tests + doctests
just test-all         # Broad Rust + Python tooling tests
just notebook-check   # Notebook lint + fast headless execution
just bench-compile    # Compile Criterion benchmarks without measuring
just bench            # Criterion benchmarks
just examples         # Run all examples
```

## Validation

`just check` is the primary non-mutating gate. It currently runs:

- `just fmt-check` - Rust formatting check
- `just clippy` - Clippy with `pedantic`, `nursery`, and `cargo` warnings
- `just python-check` - Ruff formatting/linting and Ty type checking for Python tooling
- `just notebook-lint` - notebook JSON, output hygiene, cell compilation, Ruff, and Ty checks
- `just validate-json` - JSON syntax validation
- `just yaml-check` - YAML formatting check through dprint Pretty YAML
- `just action-lint` - GitHub Actions validation through `actionlint`
- `just zizmor` - GitHub Actions security analysis through `zizmor`
- `just justfile-fmt-check` - Justfile formatting check
- `just toml-fmt-check` - TOML formatting check through Taplo
- `just toml-lint` - TOML validation through Taplo
- `just markdown-check` - Markdown formatting check through rumdl
- `just spell-check` - Spellcheck through `typos`
- `just release-check` - synchronized release metadata and active current-version reference validation
- `just semgrep` - Repository-owned Rust and Python policy rules
- `just semgrep-test` - Tests for the repository-owned Semgrep rules

For cross-repo muscle memory, the same checks are also available through grouped lint aliases:

- `just lint` - all lint groups
- `just lint-code` - Rust formatting, Clippy, Python checks, Semgrep, and Semgrep rule tests
- `just lint-config` - JSON, TOML, YAML, GitHub Actions, and Actions security validation
- `just lint-docs` - Markdown formatting and spellcheck

`just ci` is the comprehensive local and GitHub Actions entrypoint. Its dependency list is a flat union of focused validators: GitHub Actions, Markdown,
spelling, release metadata, JSON, TOML, YAML/CFF, Python, Python tests, Semgrep, notebooks, Rust formatting and all-target Clippy, documentation, broad Rust
runnable tests, doctests, benchmark-harness compilation, and deterministic example validation. It does not depend on nested `ci-*`, `check`, `lint`, or
`test-all` bundles.

Runnable library unit and integration tests across all public features share one release-profile nextest invocation through `just test-rust-ci`:

```bash
cargo nextest run --locked --release --profile ci --all-features --lib --tests --verbose
```

Doctests remain in `just test-doc` because nextest does not execute rustdoc examples. The fast `just clippy` recipe checks the core library for `just check`,
while `just ci` uses `just clippy-all-targets` to match `.github/workflows/rust-clippy.yml`. Test, example, and benchmark validators still own their execution
or compile-contract evidence because ordinary compilation does not execute Clippy lints.

The named subsets remain available for focused timing or platform work, but `just ci` does not compose through them:

- `just ci-rust` - Rust formatting, core Clippy, documentation, broad release-profile Rust tests, doctests, and deterministic example-output validation.
- `just ci-portability` - fast compile checking, broad release-profile Rust tests, doctests, and deterministic example-output validation for platform smoke
  checks.
- `just ci-repository-tooling` - Python checks and tests, notebook linting, JSON, YAML, GitHub Actions, TOML, Markdown, spelling, Semgrep, and Semgrep rule
  tests.

The GitHub Actions `CI` workflow intentionally runs `just ci` on Linux, macOS, and Windows so all supported development platforms exercise the same
comprehensive validation gate.

## Rust 1.98.0 Audit

The MSRV and contributor toolchain use Rust 1.98.0. The audit below follows the final official
[Rust 1.98.0 release notes](https://doc.rust-lang.org/stable/releases.html#version-1980-2026-08-20) and
[release announcement](https://blog.rust-lang.org/2026/08/20/Rust-1.98.0/).

| Surface | Decision |
| --- | --- |
| Algebraic floating-point methods | Forbid `f64::algebraic_{add,sub,mul,div,rem}` throughout repository-owned Rust. Their unspecified reassociation and precision plus relaxed NaN, infinity, and signed-zero behavior can change log densities, acceptance decisions, diagnostics, and seeded-run reproducibility. The repository Semgrep rule covers receiver, associated, qualified, function-item, alias, and callback forms. Ordinary IEEE-754 operators and deliberate `f64::mul_add` remain allowed. |
| New lints and compatibility changes | Keep the warning policy and add no suppressions. The runtime-symbol and `c_void` lints do not intersect this safe Rust library, and the trait-object lifetime, ambiguous-import, attribute-validation, structural-equality, and temporary-scope changes require no source changes. The full validation gate exercises library, test, doctest, example, and benchmark targets. |
| Stabilized library APIs | No change. The new substring/subslice range recovery, buffered integer formatting, `NonZero::from_str_radix`, explicit-endian UTF-16 decoding, circumfix stripping, mutable atomic-slice, and process-argument APIs do not simplify an existing path or address a demonstrated bottleneck. |
| Platform support | No change. The new and promoted PowerPC64, AArch64 pointer-authentication, and Thumb targets do not alter the declared Linux, macOS, and Windows MSVC matrix. |
| Cargo, Clippy, rustfmt, and rustdoc | Keep the existing command and configuration shape apart from the aligned Clippy MSRV. Cargo 1.98's stable changes are fixes rather than useful new workflow controls here; the pinned Clippy, rustfmt, and rustdoc components are validated through `just ci`. |

## Setup

Run `just setup` or `just setup-tools` to install repository-managed tools and verify system prerequisites:

- `actionlint`
- `cargo-edit`
- `cargo-llvm-cov`
- `cargo-nextest`
- `cargo-update`
- `dprint`
- `git-cliff`
- `jq` (system-provided; use a package manager or the [official installation instructions](https://jqlang.github.io/jq/download/))
- `rumdl`
- `taplo`
- `typos`
- `uv`
- `zizmor`

The setup recipe verifies `uv` and `jq` before managed installation work begins, then uses Cargo for Rust tools and the canonical `uv sync --locked` for the
project-managed Python 3.14 environment. Semgrep, Ruff, Ty, actionlint, and the support-script tests are pinned in `pyproject.toml` and invoked through the
`uv_version` release pinned in the root [`justfile`](../../justfile).

Run `just update` when intentionally refreshing repository dependencies and managed tooling. It updates Cargo requirements and lockfile entries, advances
exact Python development-tool pins without changing ranged requirements, upgrades the Cargo-installed CLI set, and then reconciles justfile pins with the
installed versions and active uv release.

## Line Length

Non-Rust tooling uses a 160-column policy for Ruff-managed Python support scripts, `rumdl`-managed Markdown, Taplo-managed TOML, and dprint-managed YAML. Rust
remains on the narrower `rustfmt` `max_width = 100` setting because wide Rust signatures, trait bounds, and method chains are harder to scan at 160 columns.

## Testing

- All Rust and Python tests: `just test-all`
- Focused library unit tests plus rustdoc doctests: `just test`
- Focused unit tests: `just test-unit`
- Focused integration tests: `just test-integration`
- Broad all-feature release-profile unit and integration tests: `just test-rust-ci`
- Broad Rust runnable tests plus doctests: `just test-rust`
- Python tooling tests: `just test-python`
- Single runnable test by name filter: `cargo nextest run chain_samples_near_mode`
- Examples: `just examples` builds all examples once, then runs the compiled binaries.
- Property-based Rust tests live in integration files named `tests/proptest_*.rs`; keep `src` unit tests deterministic unless a private helper requires a
  local test.

For the fast development cycle, run the smallest changed test, doctest, or integration-test crate first. For final validation of non-core changes, compose
the relevant focused buckets once without replaying broader overlapping suites. Run `just ci` for core Rust changes or whenever GitHub-equivalent evidence is
required.

## Notebooks

`just notebook-lint` validates every source notebook without execution: JSON shape and stable unique cell IDs, empty outputs and execution counts, cell-aware
Python compilation, and extracted-code Ruff format/check plus Ty. `just notebook-check` then generates the Ising example artifact and executes the fast
notebook set headlessly with `MPLBACKEND=Agg`.

`just notebook-ising-figure` runs that validated workflow and promotes `target/notebooks/ising_energy_trace.png` to the tracked README asset. The notebook
keeps plot labels and PNG metadata valid for an explicitly supplied `MCMC_TRACE_PATH`; set `MCMC_NOTEBOOK_OUTPUT_DIR` when external or read-only trace storage
requires a separate writable figure directory. The README caption records the fixed parameters used for the tracked default figure.

Executed notebooks, IPython state, and Matplotlib caches are written below `target/notebooks/`; source notebooks remain unchanged. Heavier notebooks must be
listed explicitly in the `slow_notebooks` justfile variable and run only through `just notebook-check-slow`. Use `just notebook-clear-outputs-all` for
intentional in-place cleanup before committing.

## Benchmarks

Benchmarks use Criterion with fixed seeds and workload-specific fixture lifecycles. Chain-step, 100-step sampler, and buffered-observation workloads create
their state and RNG once outside `b.iter` and measure steady-state execution as those values advance. The manual accumulator, `OnlineStats`, and
`BinningAnalysis` comparisons use `iter_batched` to provide a fresh chain and RNG outside each timed batch. Run all benchmarks with:

```bash
just bench
```

For the release-signal, saved-baseline, isolated-worktree, GitHub Release asset, and curated-report workflows, see
[`docs/BENCHMARKING.md`](../BENCHMARKING.md). The shortest local regression loop is:

```bash
just bench-save-last
just bench-latest-vs-last
```

The full CI simulation (`just ci`) compiles benchmark harnesses with `just bench-compile`, but it does not run Criterion measurements. Benchmark harness
compilation uses all crate features so optional feature paths stay covered.

The initial `benches/stepping.rs` suite protects core transition costs:

- by-value `Chain::step`
- in-place `Chain::step_mut` acceptance and rollback paths
- delayed `Chain::step_delayed` accepted, rejected, and no-plan paths
- bulk `Sampler` run loops
- observing with `SampleBuffer` versus manual online accumulation

The observing group also covers `OnlineStats` and `BinningAnalysis`. Release reports compare only benchmark names present in both revisions and identify
unmatched rows explicitly.

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
`Cargo.toml`, `Cargo.lock`, `CITATION.cff`, `pyproject.toml`, and `uv.lock` in sync through `just update-version "$TAG"`. This requires GitHub CLI for stable
release discovery. Follow [RELEASING.md](../RELEASING.md) for the shared dependency refresh, preparation, retained-evidence publication, and post-merge order.
