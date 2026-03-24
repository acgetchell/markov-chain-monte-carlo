# Justfile for markov-chain-monte-carlo development workflow
# Install just: https://github.com/casey/just
# Usage: just <command> or just --list

# Use bash with strict error handling for all recipes
set shell := ["bash", "-euo", "pipefail", "-c"]

# Coverage (cargo-tarpaulin)
#
# Common tarpaulin arguments for all coverage runs
# Note: -t 300 sets per-test timeout to 5 minutes (needed for slow CI environments)
_coverage_base_args := '''--exclude-files 'examples/*' \
  --workspace --lib --tests \
  -t 300 --verbose --implicit-test-threads'''

# Internal helpers: ensure external tooling is installed
_ensure-actionlint:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v actionlint >/dev/null || { echo "❌ 'actionlint' not found. Install: brew install actionlint"; exit 1; }

_ensure-yamllint:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v yamllint >/dev/null || { echo "❌ 'yamllint' not found. Install: brew install yamllint"; exit 1; }

# GitHub Actions workflow validation
action-lint: _ensure-actionlint
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(git ls-files -z '.github/workflows/*.yml' '.github/workflows/*.yaml')
    if [ "${#files[@]}" -gt 0 ]; then
        printf '%s\0' "${files[@]}" | xargs -0 actionlint
    else
        echo "No workflow files found to lint."
    fi

# Build
build:
    cargo build

# Fast compile check (no binary produced)
check: fmt-check clippy yaml-lint action-lint
    @echo "✅ Checks complete!"

# CI simulation: comprehensive validation
ci: check doc test examples
    @echo "🎯 CI checks complete!"

# Clean build artifacts
clean:
    cargo clean
    rm -rf target/tarpaulin
    rm -rf coverage

# Clippy linting
clippy:
    cargo clippy --workspace --all-targets -- -D warnings -W clippy::pedantic -W clippy::nursery -W clippy::cargo -A clippy::multiple_crate_versions

# Coverage analysis for local development (HTML output)
coverage:
    #!/usr/bin/env bash
    set -euo pipefail

    if ! command -v cargo-tarpaulin >/dev/null 2>&1; then
        echo "cargo-tarpaulin not found. Install with: cargo install cargo-tarpaulin"
        exit 1
    fi

    mkdir -p target/tarpaulin
    cargo tarpaulin {{_coverage_base_args}} --out Html --output-dir target/tarpaulin
    echo "Coverage report generated: target/tarpaulin/tarpaulin-report.html"

# Coverage analysis for CI (XML output for codecov)
coverage-ci:
    #!/usr/bin/env bash
    set -euo pipefail

    if ! command -v cargo-tarpaulin >/dev/null 2>&1; then
        echo "cargo-tarpaulin not found. Install with: cargo install cargo-tarpaulin"
        exit 1
    fi

    mkdir -p coverage
    cargo tarpaulin {{_coverage_base_args}} --out Xml --output-dir coverage

# Default recipe shows available commands
default:
    @just --list

# Documentation
doc:
    cargo doc --no-deps --document-private-items

# Examples
examples:
    cargo run --quiet --example normal_1d

# Fix (mutating): apply formatters
fix: fmt
    @echo "✅ Fixes applied!"

# Rust formatting
fmt:
    cargo fmt --all

fmt-check:
    cargo fmt --all -- --check

# Testing
test:
    cargo test --lib --verbose
    cargo test --doc --verbose

test-all: test test-integration
    @echo "✅ All tests passed"

test-integration:
    cargo test --tests --verbose

# Validate example output (seeded, deterministic)
validate-examples:
    #!/usr/bin/env bash
    set -euo pipefail
    output=$(cargo run --quiet --example normal_1d)
    echo "$output"
    echo "$output" | grep -q "Sample mean" || { echo "❌ Missing sample mean"; exit 1; }
    echo "$output" | grep -q "Acceptance rate" || { echo "❌ Missing acceptance rate"; exit 1; }
    echo "✅ Example output validated"

# YAML lint
yaml-lint: _ensure-yamllint
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(git ls-files -z '*.yml' '*.yaml')
    if [ "${#files[@]}" -gt 0 ]; then
        echo "🔍 yamllint (${#files[@]} files)"
        yamllint --strict -c .yamllint "${files[@]}"
    else
        echo "No YAML files found to lint."
    fi
