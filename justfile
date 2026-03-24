# Justfile for markov-chain-monte-carlo development workflow
# Install just: https://github.com/casey/just
# Usage: just <command> or just --list

# Default recipe shows available commands
default:
    @just --list

# Build
build:
    cargo build

# Fast compile check (no binary produced)
check: fmt-check clippy
    @echo "✅ Checks complete!"

# CI simulation: comprehensive validation
ci: check doc test examples
    @echo "🎯 CI checks complete!"

# Clean build artifacts
clean:
    cargo clean

# Clippy linting
clippy:
    cargo clippy --workspace --all-targets -- -D warnings -W clippy::pedantic -W clippy::nursery -W clippy::cargo -A clippy::multiple_crate_versions

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

# Validate example output (seeded, deterministic)
validate-examples:
    #!/usr/bin/env bash
    set -euo pipefail
    output=$(cargo run --quiet --example normal_1d)
    echo "$output"
    echo "$output" | grep -q "Sample mean" || { echo "❌ Missing sample mean"; exit 1; }
    echo "$output" | grep -q "Acceptance rate" || { echo "❌ Missing acceptance rate"; exit 1; }
    echo "✅ Example output validated"

# Testing
test:
    cargo test --lib --verbose
    cargo test --doc --verbose

test-all: test test-integration
    @echo "✅ All tests passed"

test-integration:
    cargo test --tests --verbose
