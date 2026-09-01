# Justfile for markov-chain-monte-carlo development workflow
# Install just: https://github.com/casey/just
# Usage: just <command> or just --list

# Use bash with strict error handling for all recipes
set shell := ["bash", "-euo", "pipefail", "-c"]

cargo_edit_version := "0.13.13"
cargo_llvm_cov_version := "0.9.0"
cargo_nextest_version := "0.9.143"
cargo_update_version := "22.1.1"
clippy_sarif_version := "0.8.0"
dprint_version := "0.57.0"
git_cliff_version := "2.13.1"
just_version := "1.58.0"
python_version := "3.14"
rumdl_version := "0.2.62"
sarif_fmt_version := "0.8.0"
taplo_version := "0.10.0"
typos_version := "1.50.0"
uv_version := "0.12.7"
zizmor_version := "1.30.0"
example_names := "detailed_balance normal_1d ising_1d iterator_sampling delayed_chunked_telemetry additive_target_bias"
fast_notebooks := "notebooks/ising_trace_analysis.ipynb"
slow_notebooks := ""

# Common cargo-llvm-cov arguments for all coverage runs.
# Excludes examples from reports while allowing tests to exercise library code.
_coverage_base_args := '''--ignore-filename-regex '(^|/)examples/' \
  --workspace --all-features --lib --tests \
  --verbose'''

# Examples
_build-examples:
    cargo build --locked --examples

# Internal helpers: ensure external tooling is installed
_ensure-actionlint: _ensure-uv
    #!/usr/bin/env bash
    set -euo pipefail
    uv run --locked actionlint -version >/dev/null

_ensure-cargo-edit:
    #!/usr/bin/env bash
    set -euo pipefail
    installed_version=""
    if cargo upgrade --version >/dev/null 2>&1; then
        installed_version="$(cargo upgrade --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
    fi
    if [[ "$installed_version" != "{{ cargo_edit_version }}" ]]; then
        echo "❌ 'cargo-edit' {{ cargo_edit_version }} not found. Install with:"
        echo "   cargo install --locked cargo-edit --version {{ cargo_edit_version }}"
        exit 1
    fi

_ensure-cargo-install-update:
    #!/usr/bin/env bash
    set -euo pipefail
    installed_version=""
    if command -v cargo-install-update >/dev/null; then
        installed_version="$(cargo-install-update --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)"
    fi
    if [[ "$installed_version" != "{{ cargo_update_version }}" ]]; then
        echo "❌ 'cargo-update' {{ cargo_update_version }} not found. Run 'just setup-tools' or install it with:"
        echo "   cargo install --locked cargo-update --version {{ cargo_update_version }}"
        exit 1
    fi

_ensure-cargo-llvm-cov:
    #!/usr/bin/env bash
    set -euo pipefail
    installed_version=""
    if command -v cargo-llvm-cov >/dev/null; then
        installed_version="$(cargo llvm-cov --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
    fi
    if [[ "$installed_version" != "{{ cargo_llvm_cov_version }}" ]]; then
        echo "❌ 'cargo-llvm-cov' {{ cargo_llvm_cov_version }} not found. Install with:"
        echo "   cargo install --locked cargo-llvm-cov --version {{ cargo_llvm_cov_version }}"
        exit 1
    fi

_ensure-cargo-nextest:
    #!/usr/bin/env bash
    set -euo pipefail
    installed_version=""
    if cargo nextest --version >/dev/null 2>&1; then
        installed_version="$(cargo nextest --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
    fi
    if [[ "$installed_version" != "{{ cargo_nextest_version }}" ]]; then
        echo "❌ 'cargo-nextest' {{ cargo_nextest_version }} not found. Install with:"
        echo "   cargo install --locked cargo-nextest --version {{ cargo_nextest_version }}"
        exit 1
    fi

_ensure-dprint:
    #!/usr/bin/env bash
    set -euo pipefail
    installed_version=""
    if command -v dprint >/dev/null; then
        installed_version="$(dprint --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
    fi
    if [[ "$installed_version" != "{{ dprint_version }}" ]]; then
        echo "❌ 'dprint' {{ dprint_version }} not found. Install with:"
        echo "   cargo install --locked dprint --version {{ dprint_version }}"
        exit 1
    fi

_ensure-gh:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v gh >/dev/null || {
        echo "❌ 'gh' not found. Install GitHub CLI: https://cli.github.com/" >&2
        exit 1
    }

_ensure-git-cliff:
    #!/usr/bin/env bash
    set -euo pipefail
    installed_version=""
    if command -v git-cliff >/dev/null; then
        installed_version="$(git-cliff --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
    fi
    if [[ "$installed_version" != "{{ git_cliff_version }}" ]]; then
        echo "❌ 'git-cliff' {{ git_cliff_version }} not found. Install with:"
        echo "   cargo install --locked git-cliff --version {{ git_cliff_version }}"
        exit 1
    fi

_ensure-jq:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v jq >/dev/null || {
        echo "❌ 'jq' not found. Install it with your system package manager or follow:"
        echo "   https://jqlang.github.io/jq/download/"
        exit 1
    }

_ensure-rumdl:
    #!/usr/bin/env bash
    set -euo pipefail
    installed_version=""
    if command -v rumdl >/dev/null; then
        installed_version="$(rumdl --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
    fi
    if [[ "$installed_version" != "{{ rumdl_version }}" ]]; then
        echo "❌ 'rumdl' {{ rumdl_version }} not found. Install with:"
        echo "   cargo install --locked rumdl --version {{ rumdl_version }}"
        exit 1
    fi

_ensure-taplo:
    #!/usr/bin/env bash
    set -euo pipefail
    installed_version=""
    if command -v taplo >/dev/null; then
        installed_version="$(taplo --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
    fi
    if [[ "$installed_version" != "{{ taplo_version }}" ]]; then
        echo "❌ 'taplo' {{ taplo_version }} not found. Install with:"
        echo "   cargo install --locked taplo-cli --version {{ taplo_version }}"
        exit 1
    fi

_ensure-typos:
    #!/usr/bin/env bash
    set -euo pipefail
    installed_version=""
    if command -v typos >/dev/null; then
        installed_version="$(typos --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
    fi
    if [[ "$installed_version" != "{{ typos_version }}" ]]; then
        echo "❌ 'typos' {{ typos_version }} not found. Install with:"
        echo "   cargo install --locked typos-cli --version {{ typos_version }}"
        exit 1
    fi

_ensure-uv:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v uv >/dev/null || { echo "❌ 'uv' not found. Install with the official installer: https://docs.astral.sh/uv/getting-started/installation/"; exit 1; }
    installed_version="$(uv --version 2>/dev/null || true)"
    if [[ "$installed_version" =~ ^uv[[:space:]]+([0-9]+\.[0-9]+\.[0-9]+) ]]; then
        installed_version="${BASH_REMATCH[1]}"
    fi
    if [[ "$installed_version" != "{{ uv_version }}" ]]; then
        echo "❌ 'uv' version mismatch: expected {{ uv_version }}, found ${installed_version:-unknown}."
        echo "   Install it with: curl -LsSf https://astral.sh/uv/{{ uv_version }}/install.sh | sh"
        exit 1
    fi

_ensure-uv-available:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v uv >/dev/null || {
        echo "❌ 'uv' not found. Install it from https://github.com/astral-sh/uv" >&2
        exit 1
    }
    uv --version >/dev/null

_ensure-zizmor:
    #!/usr/bin/env bash
    set -euo pipefail
    installed_version=""
    if command -v zizmor >/dev/null; then
        installed_version="$(zizmor --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
    fi
    if [[ "$installed_version" != "{{ zizmor_version }}" ]]; then
        echo "❌ 'zizmor' {{ zizmor_version }} not found. Install with:"
        echo "   cargo install --locked zizmor --version {{ zizmor_version }}"
        exit 1
    fi

# GitHub Actions workflow validation
[group('validation')]
action-lint: _ensure-actionlint
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(git ls-files -co --exclude-standard -z -- '.github/workflows/*.yml' '.github/workflows/*.yaml')
    if [ "${#files[@]}" -gt 0 ]; then
        # actionlint 1.7.12 predates $/ syntax; ignore only this valid self-repository reference.
        # Remove when https://github.com/rhysd/actionlint/issues/711 is released.
        printf '%s\0' "${files[@]}" | xargs -0 uv run --locked actionlint \
            -ignore '^specifying action "\$/\.github/actions/setup-just" in invalid format because ref is missing\.'
    else
        echo "No workflow files found to lint."
    fi

# Run the Criterion benchmark suite.
[group('benchmarks and performance')]
bench:
    cargo bench --locked --bench stepping

# Render existing Criterion measurements against an explicit saved baseline.
[group('benchmarks and performance')]
bench-compare baseline="last": python-sync
    uv run --locked bench-compare {{ quote(baseline) }}

# Compile benchmark harnesses without running Criterion measurements.
[group('benchmarks and performance')]
bench-compile:
    cargo bench --locked --all-features --no-run

# Run the fixed-seed MCMC release-signal benchmark set.
[group('benchmarks and performance')]
bench-latest: bench

# Run latest measurements and compare them with a saved Criterion baseline.
[group('benchmarks and performance')]
bench-latest-vs-last baseline="last": bench-latest python-sync
    uv run --locked bench-compare {{ quote(baseline) }}

# Save the complete MCMC release-signal set under a Criterion baseline name.
[group('benchmarks and performance')]
bench-save-baseline tag:
    cargo bench --locked --bench stepping -- --save-baseline {{ quote(tag) }}

# Save the current release signal under the conventional local `last` name.
[group('benchmarks and performance')]
bench-save-last:
    just bench-save-baseline last

# Build the library.
[group('build and setup')]
build:
    cargo build --locked

# Changelog generation (git-cliff + post-processing)
[group('release')]
changelog: _ensure-git-cliff _ensure-rumdl python-sync
    #!/usr/bin/env bash
    set -euo pipefail
    changelog_path="CHANGELOG.md"
    staging_file="$(mktemp "${changelog_path}.tmp.XXXXXX")"
    trap 'rm -f "$staging_file"' EXIT
    cp -p "$changelog_path" "$staging_file"
    GIT_CLIFF_OFFLINE=true git-cliff -o "$staging_file"
    uv run --locked postprocess-changelog "$staging_file"
    mv -f "$staging_file" "$changelog_path"
    trap - EXIT

# Regenerate CHANGELOG.md for a release tag before the tag exists
[group('release')]
changelog-unreleased version: _ensure-git-cliff _ensure-rumdl python-sync
    #!/usr/bin/env bash
    set -euo pipefail
    changelog_path="CHANGELOG.md"
    staging_file="$(mktemp "${changelog_path}.tmp.XXXXXX")"
    trap 'rm -f "$staging_file"' EXIT
    cp -p "$changelog_path" "$staging_file"
    GIT_CLIFF_OFFLINE=true git-cliff --tag {{ quote(version) }} -o "$staging_file"
    uv run --locked postprocess-changelog "$staging_file"
    mv -f "$staging_file" "$changelog_path"
    trap - EXIT
    uv run --locked update-release-version {{ quote(version) }} --sync-changelog-date

# Non-mutating validation gate
[group('workflows')]
check: check-rust check-repository-tooling
    @echo "✅ Checks complete!"

# Fast compile check (no binary produced)
[group('build and setup')]
check-fast:
    cargo check --locked

# Repository tooling that does not need to be repeated across operating systems.
[group('validation')]
check-repository-tooling: python-check notebook-lint validate-json yaml-check action-lint zizmor justfile-fmt-check toml-fmt-check toml-lint markdown-check spell-check release-check semgrep-test semgrep
    @echo "✅ Repository tooling checks complete!"

# Rust validation that is meaningful for source portability and user-facing API correctness.
[group('validation')]
check-rust: fmt-check clippy
    @echo "✅ Rust checks complete!"

# Runnable Rust unit and integration tests share one release-profile nextest pass;
# rustdoc doctests remain separate because nextest does not execute them.
# Run the flat union of GitHub-equivalent validators and tests, including the
# same all-target Clippy scope uploaded by the SARIF workflow.
[group('workflows')]
ci: action-lint zizmor justfile-fmt-check markdown-check spell-check release-check validate-json toml-fmt-check toml-lint yaml-check python-check semgrep-test semgrep test-python notebook-check fmt-check clippy-all-targets doc test-rust-ci test-doc bench-compile validate-examples
    @echo "🎯 CI checks complete!"

# CI subset for macOS and Windows portability confidence.
[group('workflows')]
ci-portability: check-fast test-rust-ci test-doc validate-examples
    @echo "✅ Portability CI checks complete!"

# CI subset for repository tooling and support-script tests.
[group('workflows')]
ci-repository-tooling: check-repository-tooling test-python
    @echo "✅ Repository tooling CI checks complete!"

# CI subset for Rust correctness.
[group('workflows')]
ci-rust: check-rust doc test-rust-ci test-doc validate-examples
    @echo "✅ Rust CI checks complete!"

# Clean build artifacts
[group('build and setup')]
clean:
    cargo clean
    rm -rf target/llvm-cov
    rm -rf coverage

# Fast core-library Clippy linting used by `just check`.
[group('validation')]
clippy:
    CARGO_BUILD_WARNINGS=deny cargo clippy --locked --workspace --all-features --lib -- -W clippy::pedantic -W clippy::nursery -W clippy::cargo -A clippy::multiple_crate_versions

# Full Cargo-target Clippy sweep used by `just ci` and the GitHub SARIF workflow.
[group('validation')]
clippy-all-targets:
    CARGO_BUILD_WARNINGS=deny cargo clippy --locked --workspace --all-features --all-targets -- -W clippy::pedantic -W clippy::nursery -W clippy::cargo -A clippy::multiple_crate_versions

# Coverage analysis for local development (HTML output)
[group('tests and coverage')]
coverage: _ensure-cargo-llvm-cov
    #!/usr/bin/env bash
    set -euo pipefail

    mkdir -p target/llvm-cov
    cargo llvm-cov {{ _coverage_base_args }} --open --output-dir target/llvm-cov
    echo "Coverage report generated: target/llvm-cov/html/index.html"

# Coverage analysis for CI (XML output for codecov)
[group('tests and coverage')]
coverage-ci: _ensure-cargo-llvm-cov
    #!/usr/bin/env bash
    set -euo pipefail

    mkdir -p coverage
    cargo llvm-cov {{ _coverage_base_args }} --cobertura --output-path coverage/cobertura.xml

# Show curated workflows when Just is invoked without a recipe.
[default]
[private]
default: help-workflows

# Build rustdoc for the library.
[group('validation')]
doc:
    cargo doc --locked --no-deps --document-private-items

# Run one example by name, e.g. `just example ising_1d`.
[group('tests and coverage')]
example name:
    cargo run --locked --example "{{ name }}"

# Build and run every Rust example.
[group('tests and coverage')]
examples: _build-examples
    #!/usr/bin/env bash
    set -euo pipefail
    suffix=""
    if [[ "${OS:-}" == "Windows_NT" ]]; then
        suffix=".exe"
    fi
    for example in {{ example_names }}; do
        "target/debug/examples/${example}${suffix}"
    done

# Fix (mutating): apply formatters
[group('workflows')]
fix: fmt justfile-fmt markdown-fix yaml-fix python-fix toml-fix
    @echo "✅ Fixes applied!"

# Format Rust source files.
[group('validation')]
fmt:
    cargo fmt --all

# Check Rust source formatting without modifying files.
[group('validation')]
fmt-check:
    cargo fmt --all -- --check

# Show the curated entry points for common repository workflows.
[group('workflows')]
help-workflows:
    @echo "Common Just workflows:"
    @echo "  just changelog      # Regenerate CHANGELOG.md from local git history"
    @echo "  just changelog-unreleased <tag> # Generate notes using the prepared release date"
    @echo "  just check          # Run lint/validators (non-mutating)"
    @echo "  just check-fast     # Fast compile check (cargo check)"
    @echo "  just ci             # Full CI simulation, including zizmor and benchmark compile"
    @echo "  just ci-portability # Portability subset for CI-shape timing"
    @echo "  just ci-repository-tooling # Repository tooling subset for CI-shape timing"
    @echo "  just ci-rust        # Rust correctness subset for CI-shape timing"
    @echo "  just fix            # Apply formatters/auto-fixes (mutating)"
    @echo "  just release-check  # Validate synchronized release metadata and references"
    @echo "  just setup          # Install managed tools and verify system prerequisites"
    @echo "  just tag <ver>      # Create annotated release tag from CHANGELOG.md"
    @echo "  just update         # Update dependencies, managed Cargo tools, and tool pins"
    @echo "  just update-version <tag> # Prepare release metadata from one stable tag"
    @echo ""
    @echo "Quality groups:"
    @echo "  just justfile-fmt-check # Validate canonical Justfile formatting"
    @echo "  just lint           # All linting (code + docs + config)"
    @echo "  just lint-code      # Rust + Python + Semgrep checks"
    @echo "  just lint-config    # JSON, TOML, YAML, GitHub Actions, and Actions security checks"
    @echo "  just lint-docs      # Markdown and spelling checks"
    @echo "  just notebook-check # Lint all notebooks and execute the fast notebook set"
    @echo "  just notebook-check-slow # Include explicitly configured heavy notebooks"
    @echo "  just notebook-ising-figure # Regenerate the tracked Ising trace figure"
    @echo "  just notebook-lint  # Validate JSON, output hygiene, and extracted Python"
    @echo "  just python-check   # Ruff + Ty checks for Python tooling"
    @echo "  just zizmor         # GitHub Actions security analysis"
    @echo ""
    @echo "Testing:"
    @echo "  just bench          # Run Criterion benchmarks"
    @echo "  just bench-compare [baseline] # Render existing measurements against a baseline"
    @echo "  just bench-compile  # Compile benchmarks without measuring"
    @echo "  just bench-latest   # Run the fixed-seed release-signal set"
    @echo "  just bench-latest-vs-last # Measure and compare against the saved 'last' baseline"
    @echo "  just bench-save-baseline <tag> # Save a named local Criterion baseline"
    @echo "  just bench-save-last # Save the conventional local 'last' baseline"
    @echo "  just coverage       # Generate and open HTML coverage report"
    @echo "  just coverage-ci    # Generate Cobertura XML coverage report"
    @echo "  just example <name> # Run one example, e.g. just example ising_1d"
    @echo "  just examples       # Run all examples"
    @echo "  just performance-doc # Rebuild the curated report from retained measurements"
    @echo "  just performance-github-assets # Compare durable GitHub Release assets"
    @echo "  just performance-local # Compare the current tree with the latest stable release"
    @echo "  just performance-readme # Publish the README table and SVG from retained evidence"
    @echo "  just performance-release # Promote/archive the release-to-release report"
    @echo "  just test           # Focused unit + doctest buckets"
    @echo "  just test-all       # Broad Rust + Python tooling tests"
    @echo "  just test-rust      # Broad release Rust tests + doctests"
    @echo ""
    @echo "Use 'just --list' for the complete grouped recipe reference."

# Format the Just command layer canonically.
[group('validation')]
justfile-fmt:
    just --fmt

# Check Justfile formatting without modifying it.
[group('validation')]
justfile-fmt-check:
    just --fmt --check

# All linting: code + documentation + configuration
[group('validation')]
lint: lint-code lint-docs lint-config

# Check Rust, Python, and repository-owned Semgrep rules.
[group('validation')]
lint-code: fmt-check clippy python-check semgrep-test semgrep

# Check JSON, TOML, YAML, GitHub Actions, and Just configuration.
[group('validation')]
lint-config: validate-json toml-fmt-check toml-lint yaml-check action-lint zizmor justfile-fmt-check

# Check Markdown and spelling.
[group('validation')]
lint-docs: markdown-check spell-check

# Check Markdown formatting and lint rules.
[group('validation')]
markdown-check: _ensure-rumdl
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        case "$file" in
            CHANGELOG.md) continue ;;
        esac
        files+=("$file")
    done < <(git ls-files -co --exclude-standard -z -- '*.md')
    if [ "${#files[@]}" -gt 0 ]; then
        printf '%s\0' "${files[@]}" | xargs -0 -n100 rumdl check
    else
        echo "No Markdown files found to check."
    fi

# Apply Markdown formatting and lint fixes.
[group('validation')]
markdown-fix: _ensure-rumdl
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        case "$file" in
            CHANGELOG.md) continue ;;
        esac
        files+=("$file")
    done < <(git ls-files -co --exclude-standard -z -- '*.md')
    if [ "${#files[@]}" -gt 0 ]; then
        printf '%s\0' "${files[@]}" | xargs -0 -n100 rumdl check --fix
    else
        echo "No Markdown files found to format."
    fi

# Alias for the canonical Markdown check.
[group('validation')]
markdown-lint: markdown-check

# Lint and execute the configured fast notebook set.
[group('notebooks')]
notebook-check: notebook-lint notebook-execute-fast
    @echo "📓 Fast notebook checks complete!"

# Lint and execute the configured fast and slow notebook sets.
[group('notebooks')]
notebook-check-slow: notebook-check notebook-execute-slow
    @echo "📓 Slow notebook checks complete!"

# Clear outputs from every source notebook explicitly.
[group('notebooks')]
notebook-clear-outputs-all: notebook-sync
    uv run --locked --group dev --group notebook check-notebooks clear --repo-root .

# Execute the configured fast notebook set headlessly.
[group('notebooks')]
notebook-execute-fast: notebook-sync validate-ising-example
    #!/usr/bin/env bash
    set -euo pipefail
    notebooks=( {{ fast_notebooks }} )
    MPLBACKEND=Agg uv run --locked --group dev --group notebook check-notebooks execute "${notebooks[@]}" --repo-root . --output-dir target/notebooks

# Execute the explicitly configured slow notebook set headlessly.
[group('notebooks')]
notebook-execute-slow: _ensure-uv
    #!/usr/bin/env bash
    set -euo pipefail
    notebooks=( {{ slow_notebooks }} )
    if [ "${#notebooks[@]}" -eq 0 ]; then
        echo "No slow notebooks configured."
        exit 0
    fi
    MPLBACKEND=Agg uv run --locked --group dev --group notebook check-notebooks execute "${notebooks[@]}" --repo-root . --output-dir target/notebooks --timeout 1800

# Regenerate the tracked Ising figure from the example trace and notebook.
[group('notebooks')]
notebook-ising-figure: notebook-check
    cp target/notebooks/ising_energy_trace.png docs/assets/ising_energy_trace.png

# Validate source notebook structure and extracted Python without execution.
[group('notebooks')]
notebook-lint: _ensure-uv
    uv run --locked --group dev --group notebook check-notebooks lint --repo-root .

# Compatibility entry point for the canonical Python environment sync.
[group('notebooks')]
notebook-sync: python-sync

# Rebuild and promote the curated report from tracked or explicitly saved release measurements.
[group('benchmarks and performance')]
performance-doc measurements_path="": python-sync
    #!/usr/bin/env bash
    set -euo pipefail
    measurements_path={{ quote(measurements_path) }}
    if [[ -n "$measurements_path" ]]; then
        uv run --locked archive-performance --rerender "$measurements_path" --promote
    else
        uv run --locked archive-performance --rerender --promote
    fi

# Compare stored GitHub Release benchmark assets without local benchmark runs.
[group('benchmarks and performance')]
performance-github-assets current_tag="" baseline_tag="": python-sync
    #!/usr/bin/env bash
    set -euo pipefail
    current_tag={{ quote(current_tag) }}
    baseline_tag={{ quote(baseline_tag) }}
    if [[ -n "$current_tag" || -n "$baseline_tag" ]]; then
        if [[ -z "$current_tag" || -z "$baseline_tag" ]]; then
            echo "current_tag and baseline_tag must be provided together" >&2
            exit 2
        fi
        uv run --locked archive-performance "$current_tag" "$baseline_tag" --github-assets --measurements-output target/bench-reports/github-assets-performance.csv --output target/bench-reports/github-assets-performance.md
    else
        uv run --locked archive-performance --published-latest --github-assets --measurements-output target/bench-reports/github-assets-performance.csv --output target/bench-reports/github-assets-performance.md
    fi

# Compare the current tree with the latest stable published release locally.
[group('benchmarks and performance')]
performance-local: python-sync
    uv run --locked archive-performance --current-vs-latest --measurements-output target/bench-reports/performance.csv --output target/bench-reports/performance.md

# Publish the README table, SVG, and pinned links from validated retained release evidence.
[group('benchmarks and performance')]
performance-readme: python-sync
    uv run --locked publish-performance-readme

# Generate a release-to-release report, promote it, and archive the previous report.
[group('benchmarks and performance')]
performance-release current_tag="" baseline_tag="": python-sync
    #!/usr/bin/env bash
    set -euo pipefail
    current_tag={{ quote(current_tag) }}
    baseline_tag={{ quote(baseline_tag) }}
    if [[ -n "$current_tag" || -n "$baseline_tag" ]]; then
        if [[ -z "$current_tag" || -z "$baseline_tag" ]]; then
            echo "current_tag and baseline_tag must be provided together" >&2
            exit 2
        fi
        uv run --locked archive-performance "$current_tag" "$baseline_tag" --measurements-output target/bench-reports/release-performance.csv --promote
    else
        uv run --locked archive-performance --infer-release --measurements-output target/bench-reports/release-performance.csv --promote
    fi

# Pre-publish validation: checks crates.io metadata rules that cargo publish --dry-run does NOT catch
[group('release')]
publish-check: _ensure-jq
    #!/usr/bin/env bash
    set -euo pipefail
    echo "🔍 Validating crates.io metadata..."
    errors=0

    # Keywords: max 5, each ≤20 chars, ASCII alphanumeric/hyphen only
    keywords=$(cargo metadata --no-deps --format-version=1 2>/dev/null \
        | jq -r '.packages[0].keywords[]')
    count=0
    while IFS= read -r kw; do
        [[ -z "$kw" ]] && continue
        count=$((count + 1))
        if (( ${#kw} > 20 )); then
            echo "  ❌ keyword '${kw}' exceeds 20-char limit (${#kw} chars)"
            errors=1
        fi
        if ! [[ "$kw" =~ ^[a-zA-Z0-9_-]+$ ]]; then
            echo "  ❌ keyword '${kw}' contains invalid characters"
            errors=1
        fi
    done <<< "$keywords"
    if (( count > 5 )); then
        echo "  ❌ too many keywords ($count > 5)"
        errors=1
    fi
    echo "  ✓ keywords ($count): $keywords"

    # Categories: max 5
    cat_count=$(cargo metadata --no-deps --format-version=1 2>/dev/null \
        | jq '.packages[0].categories | length')
    if (( cat_count > 5 )); then
        echo "  ❌ too many categories ($cat_count > 5)"
        errors=1
    fi
    echo "  ✓ categories ($cat_count)"

    # Description: required, ≤1000 chars
    desc=$(cargo metadata --no-deps --format-version=1 2>/dev/null \
        | jq -r '.packages[0].description // ""')
    if [[ -z "$desc" ]]; then
        echo "  ❌ description is missing"
        errors=1
    elif (( ${#desc} > 1000 )); then
        echo "  ❌ description exceeds 1000-char limit (${#desc} chars)"
        errors=1
    fi
    echo "  ✓ description (${#desc} chars)"

    if (( errors )); then
        echo ""
        echo "❌ Metadata validation failed. Fix Cargo.toml before publishing."
        exit 1
    fi

    echo ""
    echo "📦 Running cargo publish --dry-run..."
    cargo publish --locked --allow-dirty --dry-run
    echo ""
    echo "✅ Publish check passed!"

# Check Python support scripts with Ruff and Ty.
[group('validation')]
python-check: python-typecheck
    uv run --locked ruff format --check scripts/
    uv run --locked ruff check scripts/

# Apply Ruff fixes and formatting to Python support scripts.
[group('validation')]
python-fix: python-sync
    uv run --locked ruff check scripts/ --fix
    uv run --locked ruff format scripts/

# Alias for the canonical Python check.
[group('validation')]
python-lint: python-check

# Synchronize the complete default Python environment once.
[group('build and setup')]
python-sync: _ensure-uv
    uv sync --locked

# Type-check Python support scripts with Ty.
[group('validation')]
python-typecheck: python-sync
    uv run --locked ty check scripts/

# Validate synchronized release metadata and active version references.
[group('release')]
release-check: python-sync
    uv run --locked release-check

# Repository-owned Semgrep rules for project-specific diagnostics.
[group('validation')]
semgrep: _ensure-uv
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        case "$file" in
            tests/semgrep/*) continue ;;
        esac
        files+=("$file")
    done < <(git ls-files -co --exclude-standard -z --)
    if [ "${#files[@]}" -gt 0 ]; then
        uv run --locked semgrep --metrics off --error --strict --timeout 30 --config semgrep.yaml "${files[@]}"
    else
        echo "No tracked or untracked repository files found to scan."
    fi

# Validate repository-owned Semgrep rules against annotated fixtures.
[group('validation')]
semgrep-test: _ensure-uv
    #!/usr/bin/env bash
    set -euo pipefail
    cd tests/semgrep

    uv run --locked semgrep scan --metrics off --test --strict --config ../../semgrep.yaml scripts/python_portability.py

    expect_semgrep_count() {
        local expected="$1"
        local rule="$2"
        local target="$3"
        local json
        local count

        json="$(uv run --locked semgrep scan --metrics off --json --quiet --strict --config ../../semgrep.yaml "$target")"
        count="$(printf '%s\n' "$json" | { grep -o "\"check_id\":\"$rule\"" || true; } | wc -l | tr -d '[:space:]')"

        if [[ "$count" != "$expected" ]]; then
            echo "expected $expected findings for $rule in $target, got $count" >&2
            exit 1
        fi
    }

    expect_semgrep_count 2 mcmc.rust.no-stdio-diagnostics-in-src src/project_rules/rust_style.rs
    expect_semgrep_count 1 mcmc.rust.no-nonfinite-unwrap-defaults src/project_rules/rust_style.rs
    expect_semgrep_count 3 mcmc.rust.no-production-unwrap-panic src/project_rules/rust_style.rs
    expect_semgrep_count 2 mcmc.rust.thinning-interval-parameters-use-refined-type src/project_rules/rust_style.rs
    expect_semgrep_count 5 mcmc.rust.step-telemetry-fields-private src/project_rules/rust_style.rs
    expect_semgrep_count 1 mcmc.rust.no-box-dyn-error-in-src src/project_rules/rust_style.rs
    expect_semgrep_count 2 mcmc.rust.public-error-enums-non-exhaustive src/project_rules/rust_style.rs
    expect_semgrep_count 1 mcmc.rust.no-clippy-allow-lints src/project_rules/rust_style.rs
    expect_semgrep_count 1 mcmc.rust.expect-requires-reason src/project_rules/rust_style.rs
    uv run --locked semgrep scan --metrics off --test --strict --config ../../semgrep.yaml src/project_rules/algebraic_float.rs
    expect_semgrep_count 60 mcmc.rust.no-algebraic-f64-operations src/project_rules/algebraic_float.rs

    uv run --locked semgrep scan --metrics off --test --strict --config ../../semgrep.yaml examples/deep_import.rs
    expect_semgrep_count 3 mcmc.rust.no-box-dyn-error-in-examples-benches examples/erased_error.rs
    expect_semgrep_count 0 mcmc.rust.no-box-dyn-error-in-examples-benches examples/typed_error.rs
    expect_semgrep_count 3 mcmc.rust.no-box-dyn-error-in-examples-benches benches/erased_error.rs
    expect_semgrep_count 0 mcmc.rust.no-box-dyn-error-in-examples-benches benches/typed_error.rs
    expect_semgrep_count 3 mcmc.rust.no-box-dyn-error-in-doctests src/doctests/erased_error.rs
    expect_semgrep_count 0 mcmc.rust.no-box-dyn-error-in-doctests src/doctests/typed_error.rs
    expect_semgrep_count 7 mcmc.rust.no-unwrap-expect-in-doctests src/doctests/unwrap_expect.rs
    expect_semgrep_count 2 mcmc.rust.no-unwrap-expect-in-benches-examples examples/unwrap_expect.rs
    expect_semgrep_count 2 mcmc.rust.no-unwrap-expect-in-benches-examples benches/unwrap_expect.rs
    expect_semgrep_count 1 mcmc.github-actions.external-action-sha-pinned github-actions/workflow_actions.yml
    expect_semgrep_count 1 mcmc.github-actions.external-action-approved-allowlist github-actions/workflow_actions.yml
    expect_semgrep_count 1 mcmc.github-actions.external-action-version-comment github-actions/workflow_actions.yml
    expect_semgrep_count 2 mcmc.docs.check-before-fix-command-order docs/check_fix_order.md
    uv run --locked semgrep scan --metrics off --test --strict --config ../../semgrep.yaml scripts/tests/python_exceptions.py

# Install managed tools and verify system prerequisites.
[group('build and setup')]
setup: setup-tools

# Install repository-managed tools and verify system prerequisites.
[group('build and setup')]
setup-tools: _ensure-uv _ensure-jq
    #!/usr/bin/env bash
    set -euo pipefail

    have() { command -v "$1" >/dev/null 2>&1; }

    echo "Ensuring Rust components..."
    rustup component add clippy rustfmt rust-docs rust-src llvm-tools-preview
    echo ""

    echo "Ensuring cargo tools..."
    cargo_update_version="{{ cargo_update_version }}"
    if ! have cargo-install-update || [[ "$(cargo-install-update --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)" != "$cargo_update_version" ]]; then
        cargo install --locked cargo-update --version "$cargo_update_version"
    fi
    cargo_edit_version="{{ cargo_edit_version }}"
    if ! cargo upgrade --version >/dev/null 2>&1 || [[ "$(cargo upgrade --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)" != "$cargo_edit_version" ]]; then
        cargo install --locked cargo-edit --version "$cargo_edit_version"
    fi
    cargo_llvm_cov_version="{{ cargo_llvm_cov_version }}"
    if ! have cargo-llvm-cov || [[ "$(cargo llvm-cov --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)" != "$cargo_llvm_cov_version" ]]; then
        cargo install --locked cargo-llvm-cov --version "$cargo_llvm_cov_version"
    fi
    cargo_nextest_version="{{ cargo_nextest_version }}"
    if ! cargo nextest --version >/dev/null 2>&1 || [[ "$(cargo nextest --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)" != "$cargo_nextest_version" ]]; then
        cargo install --locked cargo-nextest --version "$cargo_nextest_version"
    fi
    dprint_version="{{ dprint_version }}"
    if ! have dprint || [[ "$(dprint --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)" != "$dprint_version" ]]; then
        cargo install --locked dprint --version "$dprint_version"
    fi
    git_cliff_version="{{ git_cliff_version }}"
    if ! have git-cliff || [[ "$(git-cliff --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)" != "$git_cliff_version" ]]; then
        cargo install --locked git-cliff --version "$git_cliff_version"
    fi
    just_version="{{ just_version }}"
    if ! have just || [[ "$(just --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)" != "$just_version" ]]; then
        cargo install --locked just --version "$just_version"
    fi
    taplo_version="{{ taplo_version }}"
    if ! have taplo || [[ "$(taplo --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)" != "$taplo_version" ]]; then
        cargo install --locked taplo-cli --version "$taplo_version"
    fi
    rumdl_version="{{ rumdl_version }}"
    if ! have rumdl || [[ "$(rumdl --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)" != "$rumdl_version" ]]; then
        cargo install --locked rumdl --version "$rumdl_version"
    fi
    typos_version="{{ typos_version }}"
    if ! have typos || [[ "$(typos --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)" != "$typos_version" ]]; then
        cargo install --locked typos-cli --version "$typos_version"
    fi
    zizmor_version="{{ zizmor_version }}"
    if ! have zizmor || [[ "$(zizmor --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)" != "$zizmor_version" ]]; then
        cargo install --locked zizmor --version "$zizmor_version"
    fi
    echo ""

    echo "Ensuring uv-managed Python tools..."
    uv sync --locked
    echo ""

    echo "Verifying required commands..."
    missing=0
    for cmd in cargo-install-update cargo-llvm-cov cargo-upgrade dprint git-cliff jq just rumdl taplo typos uv zizmor; do
        if have "$cmd"; then
            echo "  ✓ $cmd"
        else
            echo "  ✗ $cmd"
            missing=1
        fi
    done
    if [ "$missing" -ne 0 ]; then
        echo ""
        echo "❌ Some required tools are still missing."
        echo "Fix the installs above and re-run: just setup-tools"
        exit 1
    fi
    if cargo nextest --version >/dev/null 2>&1; then
        echo "  ✓ cargo nextest"
    else
        echo "  ✗ cargo nextest"
        exit 1
    fi

    uv run --locked actionlint -version >/dev/null
    echo "  ✓ actionlint (uv)"
    uv run --locked semgrep --version >/dev/null
    echo "  ✓ semgrep (uv)"
    uv run --locked ruff --version >/dev/null
    echo "  ✓ ruff (uv)"
    uv run --locked ty --version >/dev/null
    echo "  ✓ ty (uv)"

    echo ""
    echo "✅ Tooling setup complete."

# Check repository spelling.
[group('validation')]
spell-check: _ensure-typos
    typos --config typos.toml --force-exclude .

# Create an annotated git tag from the CHANGELOG.md section for the given version
[group('release')]
tag version: python-sync
    uv run --locked tag-release {{ version }}

# Recreate an existing tag from the CHANGELOG.md section for the given version
[group('release')]
tag-force version: python-sync
    uv run --locked tag-release {{ version }} --force

# Focused local Rust buckets: unit tests plus rustdoc doctests.
[group('tests and coverage')]
test: test-unit test-doc

# Broad Rust correctness plus Python tooling tests.
[group('tests and coverage')]
test-all: test-rust test-python
    @echo "✅ All tests passed"

# Run rustdoc doctests.
[group('tests and coverage')]
test-doc:
    cargo test --locked --doc --verbose

# Integration tests
[group('tests and coverage')]
test-integration: _ensure-cargo-nextest
    cargo nextest run --locked --test '*' --verbose

# Backward-compatible alias for the former recipe name.
[group('tests and coverage')]
test-lib: test-unit

# Run Python support-script tests.
[group('tests and coverage')]
test-python: python-sync
    uv run --locked pytest -q

# Broad Rust test workflow; doctests remain a separate cargo-test bucket.
[group('tests and coverage')]
test-rust: test-rust-ci test-doc
    @echo "✅ Rust tests passed"

# Broad release-profile Rust CI bucket: lib unit and integration tests together.
[group('tests and coverage')]
test-rust-ci: _ensure-cargo-nextest
    cargo nextest run --locked --release --profile ci --all-features --lib --tests --verbose

# Focused library unit tests for changed-surface validation.
[group('tests and coverage')]
test-unit: _ensure-cargo-nextest
    cargo nextest run --locked --lib --verbose

# Apply canonical TOML formatting.
[group('validation')]
toml-fix: toml-fmt

# Format tracked TOML files.
[group('validation')]
toml-fmt: _ensure-taplo
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(git ls-files -co --exclude-standard -z -- '*.toml')
    if [ "${#files[@]}" -gt 0 ]; then
        taplo fmt "${files[@]}"
    else
        echo "No TOML files found to format."
    fi

# Check tracked TOML formatting without modifying files.
[group('validation')]
toml-fmt-check: _ensure-taplo
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(git ls-files -co --exclude-standard -z -- '*.toml')
    if [ "${#files[@]}" -gt 0 ]; then
        taplo fmt --check "${files[@]}"
    else
        echo "No TOML files found to check."
    fi

# Lint tracked TOML files.
[group('validation')]
toml-lint: _ensure-taplo
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(git ls-files -co --exclude-standard -z -- '*.toml')
    if [ "${#files[@]}" -gt 0 ]; then
        taplo lint "${files[@]}"
    else
        echo "No TOML files found to lint."
    fi

# Update dependency requirements, locks, managed Cargo tools, and the active uv pin.
[group('build and setup')]
update: _ensure-cargo-install-update update-dependencies update-cargo-tools
    @echo "✅ Repository dependencies and tools updated."

# Advance Cargo dependency declarations and lockfile entries.
[doc('Update repository Cargo dependency requirements and lockfiles.')]
[group('build and setup')]
update-cargo-dependencies: _ensure-cargo-edit
    cargo upgrade --incompatible allow
    cargo update

# Update locally installed Cargo CLI tools and reconcile their pins plus the active uv version.
[doc('Update managed Cargo CLI tools and reconcile all root justfile tool pins.')]
[group('build and setup')]
update-cargo-tools: _ensure-cargo-install-update _ensure-uv-available
    #!/usr/bin/env bash
    set -euo pipefail

    packages=(
        cargo-edit
        cargo-llvm-cov
        cargo-nextest
        cargo-update
        dprint
        git-cliff
        just
        rumdl
        taplo-cli
        typos-cli
        zizmor
    )
    cargo install-update --locked "${packages[@]}"
    uv run --locked update-tool-pins

# Advance Cargo and exact Python development requirements plus their lockfiles.
[doc('Update Cargo and Python development requirements plus all Cargo/uv locked dependencies.')]
[group('build and setup')]
update-dependencies: _ensure-cargo-edit _ensure-uv-available update-cargo-dependencies update-python-dependencies

# Resolve latest exact Python development tools, retain ranged requirements, and sync.
[doc('Update exact dependency-groups.dev pins and uv.lock through uv.')]
[group('build and setup')]
update-python-dependencies: _ensure-uv-available
    uv run --locked update-python-dev-pins
    uv lock --upgrade
    uv sync --locked --group dev

# Prepare versions, dates, and active references from a stable tag without upgrading dependencies.
[group('release')]
update-version tag: _ensure-gh python-sync
    uv run --locked update-release-version {{ quote(tag) }}

# Validate the Ising example once while generating the notebook input trace.
# Validate example output (seeded, deterministic)
[group('tests and coverage')]
validate-examples: _build-examples validate-ising-example
    #!/usr/bin/env bash
    set -euo pipefail

    example_binary() {
        local example="$1"
        local suffix=""
        if [[ "${OS:-}" == "Windows_NT" ]]; then
            suffix=".exe"
        fi
        printf 'target/debug/examples/%s%s' "$example" "$suffix"
    }

    validate_example() {
        local example="$1"
        shift
        local output
        output=$("$(example_binary "$example")")
        echo "$output"
        for marker in "$@"; do
            echo "$output" | grep -q "$marker" || { echo "❌ ${example}: Missing marker '${marker}'"; exit 1; }
        done
        echo "✅ ${example} validated"
    }

    validate_example detailed_balance "Detailed balance checks passed" "by-value residual"
    validate_example normal_1d "Sample mean" "Acceptance rate"
    validate_example iterator_sampling "Sample mean" "Acceptance rate"
    validate_example delayed_chunked_telemetry "Per-step telemetry" "Delayed chunked telemetry complete"
    validate_example additive_target_bias "AdditiveTarget bias example" "observed P(true)"

# Validate the Ising example output and produce its trace for notebook checks.
[group('tests and coverage')]
validate-ising-example: _build-examples
    #!/usr/bin/env bash
    set -euo pipefail
    suffix=""
    if [[ "$(uname -s)" == *MINGW* || "$(uname -s)" == *MSYS* || "$(uname -s)" == *CYGWIN* ]]; then
        suffix=".exe"
    fi
    binary="target/debug/examples/ising_1d${suffix}"
    output=$("$binary")
    printf '%s\n' "$output"
    for marker in "<m>" "acceptance rate"; do
        if ! grep -Fq "$marker" <<< "$output"; then
            echo "Example ising_1d missing expected marker: $marker" >&2
            exit 1
        fi
    done
    echo "✅ ising_1d validated"

# Validate tracked JSON files.
[group('validation')]
validate-json: _ensure-jq
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(git ls-files -co --exclude-standard -z -- '*.json')
    if [ "${#files[@]}" -gt 0 ]; then
        printf '%s\0' "${files[@]}" | xargs -0 -n1 jq empty
    else
        echo "No JSON files found to validate."
    fi

# YAML formatting check
[group('validation')]
yaml-check: _ensure-dprint
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(git ls-files -co --exclude-standard -z -- '*.yml' '*.yaml')
    if [ "${#files[@]}" -gt 0 ]; then
        printf '%s\0' "${files[@]}" | xargs -0 dprint check
    else
        echo "No YAML files found to check."
    fi

# YAML formatting
[group('validation')]
yaml-fix: _ensure-dprint
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(git ls-files -co --exclude-standard -z -- '*.yml' '*.yaml')
    if [ "${#files[@]}" -gt 0 ]; then
        printf '%s\0' "${files[@]}" | xargs -0 dprint fmt
    else
        echo "No YAML files found to format."
    fi

# Alias for the canonical YAML check.
[group('validation')]
yaml-lint: yaml-check

# GitHub Actions security analysis
[group('validation')]
zizmor: _ensure-zizmor
    zizmor .github
