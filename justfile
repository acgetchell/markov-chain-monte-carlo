# Justfile for markov-chain-monte-carlo development workflow
# Install just: https://github.com/casey/just
# Usage: just <command> or just --list

# Use bash with strict error handling for all recipes
set shell := ["bash", "-euo", "pipefail", "-c"]

cargo_nextest_version := "0.9.137"
cargo_llvm_cov_version := "0.8.7"
dprint_version := "0.54.0"
git_cliff_version := "2.13.1"
rumdl_version := "0.2.3"
taplo_version := "0.10.0"
typos_version := "1.46.3"
zizmor_version := "1.25.2"
example_names := "detailed_balance normal_1d ising_1d iterator_sampling"

# Common cargo-llvm-cov arguments for all coverage runs.
# Excludes examples from reports while allowing tests to exercise library code.
_coverage_base_args := '''--ignore-filename-regex '(^|/)examples/' \
  --workspace --all-features --lib --tests \
  --verbose'''

# Internal helpers: ensure external tooling is installed
_ensure-actionlint:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v uv >/dev/null || { echo "❌ 'uv' not found. Install with the official installer: https://docs.astral.sh/uv/getting-started/installation/"; exit 1; }
    uv run actionlint -version >/dev/null

_ensure-cargo-llvm-cov:
    #!/usr/bin/env bash
    set -euo pipefail
    installed_version=""
    if command -v cargo-llvm-cov >/dev/null; then
        installed_version="$(cargo llvm-cov --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
    fi
    if [[ "$installed_version" != "{{cargo_llvm_cov_version}}" ]]; then
        echo "❌ 'cargo-llvm-cov' {{cargo_llvm_cov_version}} not found. Install with:"
        echo "   cargo install --locked cargo-llvm-cov --version {{cargo_llvm_cov_version}}"
        exit 1
    fi

_ensure-cargo-nextest:
    #!/usr/bin/env bash
    set -euo pipefail
    installed_version=""
    if cargo nextest --version >/dev/null 2>&1; then
        installed_version="$(cargo nextest --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
    fi
    if [[ "$installed_version" != "{{cargo_nextest_version}}" ]]; then
        echo "❌ 'cargo-nextest' {{cargo_nextest_version}} not found. Install with:"
        echo "   cargo install --locked cargo-nextest --version {{cargo_nextest_version}}"
        exit 1
    fi

_ensure-dprint:
    #!/usr/bin/env bash
    set -euo pipefail
    installed_version=""
    if command -v dprint >/dev/null; then
        installed_version="$(dprint --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
    fi
    if [[ "$installed_version" != "{{dprint_version}}" ]]; then
        echo "❌ 'dprint' {{dprint_version}} not found. Install with:"
        echo "   cargo install --locked dprint --version {{dprint_version}}"
        exit 1
    fi

_ensure-git-cliff:
    #!/usr/bin/env bash
    set -euo pipefail
    installed_version=""
    if command -v git-cliff >/dev/null; then
        installed_version="$(git-cliff --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
    fi
    if [[ "$installed_version" != "{{git_cliff_version}}" ]]; then
        echo "❌ 'git-cliff' {{git_cliff_version}} not found. Install with:"
        echo "   cargo install --locked git-cliff --version {{git_cliff_version}}"
        exit 1
    fi

_ensure-jq:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v jq >/dev/null || { echo "❌ 'jq' not found. Install with your system package manager."; exit 1; }

_ensure-rumdl:
    #!/usr/bin/env bash
    set -euo pipefail
    installed_version=""
    if command -v rumdl >/dev/null; then
        installed_version="$(rumdl --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
    fi
    if [[ "$installed_version" != "{{rumdl_version}}" ]]; then
        echo "❌ 'rumdl' {{rumdl_version}} not found. Install with:"
        echo "   cargo install --locked rumdl --version {{rumdl_version}}"
        exit 1
    fi

_ensure-taplo:
    #!/usr/bin/env bash
    set -euo pipefail
    installed_version=""
    if command -v taplo >/dev/null; then
        installed_version="$(taplo --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
    fi
    if [[ "$installed_version" != "{{taplo_version}}" ]]; then
        echo "❌ 'taplo' {{taplo_version}} not found. Install with:"
        echo "   cargo install --locked taplo-cli --version {{taplo_version}}"
        exit 1
    fi

_ensure-typos:
    #!/usr/bin/env bash
    set -euo pipefail
    installed_version=""
    if command -v typos >/dev/null; then
        installed_version="$(typos --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
    fi
    if [[ "$installed_version" != "{{typos_version}}" ]]; then
        echo "❌ 'typos' {{typos_version}} not found. Install with:"
        echo "   cargo install --locked typos-cli --version {{typos_version}}"
        exit 1
    fi

_ensure-uv:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v uv >/dev/null || { echo "❌ 'uv' not found. Install with the official installer: https://docs.astral.sh/uv/getting-started/installation/"; exit 1; }

_ensure-zizmor:
    #!/usr/bin/env bash
    set -euo pipefail
    installed_version=""
    if command -v zizmor >/dev/null; then
        installed_version="$(zizmor --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
    fi
    if [[ "$installed_version" != "{{zizmor_version}}" ]]; then
        echo "❌ 'zizmor' {{zizmor_version}} not found. Install with:"
        echo "   cargo install --locked zizmor --version {{zizmor_version}}"
        exit 1
    fi

# GitHub Actions workflow validation
action-lint: _ensure-actionlint
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(git ls-files -z '.github/workflows/*.yml' '.github/workflows/*.yaml')
    if [ "${#files[@]}" -gt 0 ]; then
        printf '%s\0' "${files[@]}" | xargs -0 uv run actionlint
    else
        echo "No workflow files found to lint."
    fi

# Benchmarks
bench:
    cargo bench --locked --bench stepping

# Compile benchmark harnesses without running Criterion measurements.
bench-compile:
    cargo bench --locked --all-features --no-run

# Build
build:
    cargo build --locked

# Changelog generation (git-cliff + post-processing)
changelog: _ensure-git-cliff python-sync
    #!/usr/bin/env bash
    set -euo pipefail
    GIT_CLIFF_OFFLINE=true git-cliff -o CHANGELOG.md
    uv run postprocess-changelog

# Regenerate CHANGELOG.md for a release tag before the tag exists
changelog-unreleased version: _ensure-git-cliff python-sync
    #!/usr/bin/env bash
    set -euo pipefail
    GIT_CLIFF_OFFLINE=true git-cliff --tag {{version}} -o CHANGELOG.md
    uv run postprocess-changelog

# Rust validation that is meaningful for source portability and user-facing API correctness.
check-rust: fmt-check clippy
    @echo "✅ Rust checks complete!"

# Repository tooling that does not need to be repeated across operating systems.
check-repository-tooling: python-check yaml-check action-lint zizmor toml-fmt-check toml-lint markdown-check spell-check semgrep semgrep-test
    @echo "✅ Repository tooling checks complete!"

# Non-mutating validation gate
check: check-rust check-repository-tooling
    @echo "✅ Checks complete!"

# Fast compile check (no binary produced)
check-fast:
    cargo check --locked

# CI subset for Rust correctness.
ci-rust: check-rust doc test test-integration validate-examples
    @echo "✅ Rust CI checks complete!"

# CI subset for macOS and Windows portability confidence.
ci-portability: check-fast test test-integration validate-examples
    @echo "✅ Portability CI checks complete!"

# CI subset for repository tooling and support-script tests.
ci-repository-tooling: check-repository-tooling test-python
    @echo "✅ Repository tooling CI checks complete!"

# CI simulation: comprehensive validation.
# Depends on repository tooling, including `zizmor` GitHub Actions security analysis.
ci: ci-repository-tooling ci-rust bench-compile
    @echo "🎯 CI checks complete!"

# Clean build artifacts
clean:
    cargo clean
    rm -rf target/llvm-cov
    rm -rf coverage

# Clippy linting
clippy:
    cargo clippy --locked --workspace --all-targets -- -D warnings -W clippy::pedantic -W clippy::nursery -W clippy::cargo -A clippy::multiple_crate_versions

# Coverage analysis for local development (HTML output)
coverage: _ensure-cargo-llvm-cov
    #!/usr/bin/env bash
    set -euo pipefail

    mkdir -p target/llvm-cov
    cargo llvm-cov {{_coverage_base_args}} --open --output-dir target/llvm-cov
    echo "Coverage report generated: target/llvm-cov/html/index.html"

# Coverage analysis for CI (XML output for codecov)
coverage-ci: _ensure-cargo-llvm-cov
    #!/usr/bin/env bash
    set -euo pipefail

    mkdir -p coverage
    cargo llvm-cov {{_coverage_base_args}} --cobertura --output-path coverage/cobertura.xml

# Default recipe shows available commands
default:
    @just --list

# Documentation
doc:
    cargo doc --locked --no-deps --document-private-items

# Examples
_build-examples:
    cargo build --locked --examples

examples: _build-examples
    #!/usr/bin/env bash
    set -euo pipefail
    suffix=""
    if [[ "${OS:-}" == "Windows_NT" ]]; then
        suffix=".exe"
    fi
    for example in {{example_names}}; do
        "target/debug/examples/${example}${suffix}"
    done

# Fix (mutating): apply formatters
fix: fmt markdown-fix yaml-fix python-fix toml-fix
    @echo "✅ Fixes applied!"

# Rust formatting
fmt:
    cargo fmt --all

# Rust format check
fmt-check:
    cargo fmt --all -- --check

help-workflows:
    @echo "Common Just workflows:"
    @echo "  just check          # Run lint/validators (non-mutating)"
    @echo "  just check-fast     # Fast compile check (cargo check)"
    @echo "  just ci-rust        # Rust correctness subset for CI-shape timing"
    @echo "  just ci-portability # Portability subset for CI-shape timing"
    @echo "  just ci-repository-tooling # Repository tooling subset for CI-shape timing"
    @echo "  just ci             # Full CI simulation, including zizmor and benchmark compile"
    @echo "  just fix            # Apply formatters/auto-fixes (mutating)"
    @echo "  just setup          # Install/verify external dev tools"
    @echo "  just changelog      # Regenerate CHANGELOG.md from local git history"
    @echo "  just changelog-unreleased <ver>  # Regenerate CHANGELOG.md for a release tag"
    @echo "  just tag <ver>      # Create annotated release tag from CHANGELOG.md"
    @echo ""
    @echo "Quality groups:"
    @echo "  just lint           # All linting (code + docs + config)"
    @echo "  just lint-code      # Rust + Python + Semgrep checks"
    @echo "  just lint-config    # JSON, TOML, YAML, GitHub Actions, and Actions security checks"
    @echo "  just lint-docs      # Markdown and spelling checks"
    @echo "  just python-check   # Ruff + Ty checks for Python tooling"
    @echo "  just zizmor         # GitHub Actions security analysis"
    @echo ""
    @echo "Testing:"
    @echo "  just test           # Lib + doc tests"
    @echo "  just test-all       # Lib + doc + integration + Python tooling tests"
    @echo "  just bench          # Run Criterion benchmarks"
    @echo "  just bench-compile  # Compile benchmarks without measuring"
    @echo "  just coverage       # Generate and open HTML coverage report"
    @echo "  just coverage-ci    # Generate Cobertura XML coverage report"
    @echo "  just examples       # Run all examples"

# All linting: code + documentation + configuration
lint: lint-code lint-docs lint-config

lint-code: fmt-check clippy python-check semgrep semgrep-test

lint-config: validate-json toml-fmt-check toml-lint yaml-check action-lint zizmor

lint-docs: markdown-check spell-check

markdown-check: _ensure-rumdl
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        case "$file" in
            CHANGELOG.md) continue ;;
        esac
        files+=("$file")
    done < <(git ls-files -z '*.md')
    if [ "${#files[@]}" -gt 0 ]; then
        printf '%s\0' "${files[@]}" | xargs -0 -n100 rumdl check
    else
        echo "No Markdown files found to check."
    fi

markdown-fix: _ensure-rumdl
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        case "$file" in
            CHANGELOG.md) continue ;;
        esac
        files+=("$file")
    done < <(git ls-files -z '*.md')
    if [ "${#files[@]}" -gt 0 ]; then
        printf '%s\0' "${files[@]}" | xargs -0 -n100 rumdl check --fix
    else
        echo "No Markdown files found to format."
    fi

markdown-lint: markdown-check

# Pre-publish validation: checks crates.io metadata rules that cargo publish --dry-run does NOT catch
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

python-check: python-typecheck
    uv run ruff format --check scripts/
    uv run ruff check scripts/

python-fix: python-sync
    uv run ruff check scripts/ --fix
    uv run ruff format scripts/

python-lint: python-check

python-sync: _ensure-uv
    uv sync --group dev

python-typecheck: python-sync
    uv run ty check scripts/

# Repository-owned Semgrep rules for project-specific diagnostics.
semgrep: _ensure-uv
    uv run semgrep --metrics off --error --strict --timeout 30 --config semgrep.yaml .

semgrep-test: _ensure-uv
    #!/usr/bin/env bash
    set -euo pipefail
    cd tests/semgrep

    expect_semgrep_count() {
        local expected="$1"
        local rule="$2"
        local target="$3"
        local json
        local count

        json="$(uv run semgrep scan --metrics off --json --quiet --strict --config ../../semgrep.yaml "$target")"
        count="$(printf '%s\n' "$json" | { grep -o "\"check_id\":\"$rule\"" || true; } | wc -l | tr -d '[:space:]')"

        if [[ "$count" != "$expected" ]]; then
            echo "expected $expected findings for $rule in $target, got $count" >&2
            exit 1
        fi
    }

    expect_semgrep_count 2 mcmc.rust.no-stdio-diagnostics-in-src src/project_rules/rust_style.rs
    expect_semgrep_count 1 mcmc.rust.no-nonfinite-unwrap-defaults src/project_rules/rust_style.rs
    expect_semgrep_count 3 mcmc.rust.no-production-unwrap-panic src/project_rules/rust_style.rs
    expect_semgrep_count 1 mcmc.rust.no-box-dyn-error-in-src src/project_rules/rust_style.rs
    expect_semgrep_count 2 mcmc.rust.public-error-enums-non-exhaustive src/project_rules/rust_style.rs
    expect_semgrep_count 1 mcmc.rust.no-clippy-allow-lints src/project_rules/rust_style.rs
    expect_semgrep_count 1 mcmc.rust.expect-requires-reason src/project_rules/rust_style.rs

    uv run semgrep scan --metrics off --test --strict --config ../../semgrep.yaml examples/deep_import.rs
    expect_semgrep_count 3 mcmc.rust.no-box-dyn-error-in-examples-benches examples/erased_error.rs
    expect_semgrep_count 0 mcmc.rust.no-box-dyn-error-in-examples-benches examples/typed_error.rs
    expect_semgrep_count 3 mcmc.rust.no-box-dyn-error-in-examples-benches benches/erased_error.rs
    expect_semgrep_count 0 mcmc.rust.no-box-dyn-error-in-examples-benches benches/typed_error.rs
    expect_semgrep_count 3 mcmc.rust.no-box-dyn-error-in-doctests src/doctests/erased_error.rs
    expect_semgrep_count 0 mcmc.rust.no-box-dyn-error-in-doctests src/doctests/typed_error.rs
    expect_semgrep_count 1 mcmc.github-actions.external-action-sha-pinned github-actions/workflow_actions.yml
    expect_semgrep_count 1 mcmc.github-actions.external-action-approved-allowlist github-actions/workflow_actions.yml
    expect_semgrep_count 1 mcmc.github-actions.external-action-version-comment github-actions/workflow_actions.yml
    expect_semgrep_count 2 mcmc.docs.check-before-fix-command-order docs/check_fix_order.md
    uv run semgrep scan --metrics off --test --strict --config ../../semgrep.yaml scripts/tests/python_exceptions.py

setup: setup-tools

setup-tools:
    #!/usr/bin/env bash
    set -euo pipefail

    have() { command -v "$1" >/dev/null 2>&1; }

    echo "Ensuring Rust components..."
    rustup component add clippy rustfmt rust-docs rust-src llvm-tools-preview
    echo ""

    echo "Ensuring cargo tools..."
    cargo_llvm_cov_version="{{cargo_llvm_cov_version}}"
    if ! have cargo-llvm-cov || [[ "$(cargo llvm-cov --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)" != "$cargo_llvm_cov_version" ]]; then
        cargo install --locked cargo-llvm-cov --version "$cargo_llvm_cov_version"
    fi
    cargo_nextest_version="{{cargo_nextest_version}}"
    if ! cargo nextest --version >/dev/null 2>&1 || [[ "$(cargo nextest --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)" != "$cargo_nextest_version" ]]; then
        cargo install --locked cargo-nextest --version "$cargo_nextest_version"
    fi
    dprint_version="{{dprint_version}}"
    if ! have dprint || [[ "$(dprint --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)" != "$dprint_version" ]]; then
        cargo install --locked dprint --version "$dprint_version"
    fi
    git_cliff_version="{{git_cliff_version}}"
    if ! have git-cliff || [[ "$(git-cliff --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)" != "$git_cliff_version" ]]; then
        cargo install --locked git-cliff --version "$git_cliff_version"
    fi
    taplo_version="{{taplo_version}}"
    if ! have taplo || [[ "$(taplo --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)" != "$taplo_version" ]]; then
        cargo install --locked taplo-cli --version "$taplo_version"
    fi
    rumdl_version="{{rumdl_version}}"
    if ! have rumdl || [[ "$(rumdl --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)" != "$rumdl_version" ]]; then
        cargo install --locked rumdl --version "$rumdl_version"
    fi
    typos_version="{{typos_version}}"
    if ! have typos || [[ "$(typos --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)" != "$typos_version" ]]; then
        cargo install --locked typos-cli --version "$typos_version"
    fi
    zizmor_version="{{zizmor_version}}"
    if ! have zizmor || [[ "$(zizmor --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)" != "$zizmor_version" ]]; then
        cargo install --locked zizmor --version "$zizmor_version"
    fi
    echo ""

    if have uv; then
        echo "Ensuring uv-managed Python tools..."
        uv sync --group dev
        echo ""
    else
        echo "❌ uv missing; cannot install project-managed Python tools."
        echo "Install uv and re-run: just setup-tools"
        exit 1
    fi

    echo "Verifying required commands..."
    missing=0
    for cmd in cargo-llvm-cov dprint git-cliff jq rumdl taplo typos uv zizmor; do
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

    uv run actionlint -version >/dev/null
    echo "  ✓ actionlint (uv)"
    uv run semgrep --version >/dev/null
    echo "  ✓ semgrep (uv)"
    uv run ruff --version >/dev/null
    echo "  ✓ ruff (uv)"
    uv run ty --version >/dev/null
    echo "  ✓ ty (uv)"

    echo ""
    echo "✅ Tooling setup complete."

spell-check: _ensure-typos
    typos --config typos.toml --force-exclude .

# Create an annotated git tag from the CHANGELOG.md section for the given version
tag version: python-sync
    uv run tag-release {{version}}

# Recreate an existing tag from the CHANGELOG.md section for the given version
tag-force version: python-sync
    uv run tag-release {{version}} --force

# Testing: runnable Rust tests use nextest; rustdoc doctests remain on cargo test.
test: test-lib test-doc

test-lib: _ensure-cargo-nextest
    cargo nextest run --locked --lib --verbose

test-doc:
    cargo test --locked --doc --verbose

# All tests (lib + doc + integration + Python tooling)
test-all: test test-integration test-python
    @echo "✅ All tests passed"

# Integration tests
test-integration: _ensure-cargo-nextest
    cargo nextest run --locked --test '*' --verbose

test-python: python-sync
    uv run pytest -q

toml-fmt: _ensure-taplo
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(git ls-files -z '*.toml')
    if [ "${#files[@]}" -gt 0 ]; then
        taplo fmt "${files[@]}"
    else
        echo "No TOML files found to format."
    fi

toml-fmt-check: _ensure-taplo
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(git ls-files -z '*.toml')
    if [ "${#files[@]}" -gt 0 ]; then
        taplo fmt --check "${files[@]}"
    else
        echo "No TOML files found to check."
    fi

toml-lint: _ensure-taplo
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(git ls-files -z '*.toml')
    if [ "${#files[@]}" -gt 0 ]; then
        taplo lint "${files[@]}"
    else
        echo "No TOML files found to lint."
    fi

toml-fix: toml-fmt

# Validate example output (seeded, deterministic)
validate-examples: _build-examples
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
    validate_example ising_1d "<m>" "acceptance rate"
    validate_example iterator_sampling "Sample mean" "Acceptance rate"

validate-json: _ensure-jq
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(git ls-files -z '*.json')
    if [ "${#files[@]}" -gt 0 ]; then
        printf '%s\0' "${files[@]}" | xargs -0 -n1 jq empty
    else
        echo "No JSON files found to validate."
    fi

# YAML formatting check
yaml-check: _ensure-dprint
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(git ls-files -z '*.yml' '*.yaml')
    if [ "${#files[@]}" -gt 0 ]; then
        printf '%s\0' "${files[@]}" | xargs -0 dprint check
    else
        echo "No YAML files found to check."
    fi

# YAML formatting
yaml-fix: _ensure-dprint
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(git ls-files -z '*.yml' '*.yaml')
    if [ "${#files[@]}" -gt 0 ]; then
        printf '%s\0' "${files[@]}" | xargs -0 dprint fmt
    else
        echo "No YAML files found to format."
    fi

yaml-lint: yaml-check

# GitHub Actions security analysis
zizmor: _ensure-zizmor
    zizmor .github
