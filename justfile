# Justfile for markov-chain-monte-carlo development workflow
# Install just: https://github.com/casey/just
# Usage: just <command> or just --list

# Use bash with strict error handling for all recipes
set shell := ["bash", "-euo", "pipefail", "-c"]

cargo_llvm_cov_version := "0.8.5"

# Common cargo-llvm-cov arguments for all coverage runs.
# Excludes examples from reports while allowing tests to exercise library code.
_coverage_base_args := '''--ignore-filename-regex '(^|/)examples/' \
  --workspace --lib --tests \
  --verbose'''

# Internal helpers: ensure external tooling is installed
_ensure-actionlint:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v actionlint >/dev/null || { echo "❌ 'actionlint' not found. Install: brew install actionlint"; exit 1; }

_ensure-cargo-llvm-cov:
    #!/usr/bin/env bash
    set -euo pipefail
    if ! command -v cargo-llvm-cov >/dev/null; then
        echo "❌ 'cargo-llvm-cov' not found. Install with:"
        echo "   cargo install --locked cargo-llvm-cov --version {{cargo_llvm_cov_version}}"
        exit 1
    fi

_ensure-cargo-rdme:
    #!/usr/bin/env bash
    set -euo pipefail
    if ! cargo rdme --version >/dev/null 2>&1; then
        echo "❌ 'cargo-rdme' not found. Install with:"
        echo "   cargo install --locked cargo-rdme"
        exit 1
    fi

_ensure-dprint:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v dprint >/dev/null || { echo "❌ 'dprint' not found. Install: cargo install dprint"; exit 1; }

_ensure-git-cliff:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v git-cliff >/dev/null || { echo "❌ 'git-cliff' not found. Install: cargo install git-cliff"; exit 1; }

_ensure-jq:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v jq >/dev/null || { echo "❌ 'jq' not found. Install: brew install jq"; exit 1; }

_ensure-taplo:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v taplo >/dev/null || { echo "❌ 'taplo' not found. Install: brew install taplo or cargo install taplo-cli"; exit 1; }

_ensure-typos:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v typos >/dev/null || { echo "❌ 'typos' not found. Install: cargo install typos-cli"; exit 1; }

_ensure-uv:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v uv >/dev/null || { echo "❌ 'uv' not found. Install: brew install uv"; exit 1; }

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

# Benchmarks
bench:
    cargo bench --bench stepping

# Compile benchmark harnesses without running Criterion measurements.
bench-compile:
    cargo bench --no-run

# Build
build:
    cargo build

# Changelog generation (git-cliff + post-processing)
changelog: _ensure-git-cliff python-sync
    #!/usr/bin/env bash
    set -euo pipefail
    GIT_CLIFF_OFFLINE=true git-cliff -o CHANGELOG.md
    uv run postprocess-changelog

# Prepend unreleased changes to CHANGELOG.md for the given version
changelog-unreleased version: _ensure-git-cliff python-sync
    #!/usr/bin/env bash
    set -euo pipefail
    GIT_CLIFF_OFFLINE=true git-cliff --unreleased --tag {{version}} --prepend CHANGELOG.md
    uv run postprocess-changelog

# Non-mutating validation gate
check: fmt-check clippy python-check yaml-lint action-lint toml-fmt-check toml-lint markdown-check spell-check semgrep semgrep-test docs-readme-check
    @echo "✅ Checks complete!"

# Fast compile check (no binary produced)
check-fast:
    cargo check

# CI simulation: comprehensive validation
ci: check bench-compile doc test-all examples validate-examples
    @echo "🎯 CI checks complete!"

# Clean build artifacts
clean:
    cargo clean
    rm -rf target/llvm-cov
    rm -rf coverage

# Clippy linting
clippy:
    cargo clippy --workspace --all-targets -- -D warnings -W clippy::pedantic -W clippy::nursery -W clippy::cargo -A clippy::multiple_crate_versions

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
    cargo doc --no-deps --document-private-items

# Regenerate README.md API section from src/lib.rs //! (mutating).
docs-readme: _ensure-cargo-rdme
    cargo rdme --force

# Verify README.md API section matches src/lib.rs //! (non-mutating).
docs-readme-check: _ensure-cargo-rdme
    cargo rdme --check

# Examples
examples:
    cargo run --quiet --example detailed_balance
    cargo run --quiet --example normal_1d
    cargo run --quiet --example ising_1d
    cargo run --quiet --example iterator_sampling

# Fix (mutating): apply formatters
fix: fmt markdown-fix python-fix toml-fmt docs-readme
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
    @echo "  just ci             # Full CI simulation, including benchmark compile"
    @echo "  just fix            # Apply formatters/auto-fixes (mutating)"
    @echo "  just setup          # Install/verify external dev tools"
    @echo "  just changelog      # Regenerate CHANGELOG.md from local git history"
    @echo "  just changelog-unreleased <ver>  # Prepend unreleased changes for a version"
    @echo "  just tag <ver>      # Create annotated release tag from CHANGELOG.md"
    @echo ""
    @echo "Quality groups:"
    @echo "  just lint           # All linting (code + docs + config)"
    @echo "  just lint-code      # Rust + Python + Semgrep checks"
    @echo "  just lint-config    # JSON, TOML, YAML, GitHub Actions"
    @echo "  just lint-docs      # Markdown format check + spell check"
    @echo "  just python-check   # Ruff + Ty checks for Python tooling"
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

lint-config: validate-json toml-lint toml-fmt-check yaml-lint action-lint

lint-docs: markdown-check spell-check

markdown-check: _ensure-dprint
    dprint check

markdown-fix: _ensure-dprint
    dprint fmt

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
    expect_semgrep_count 1 mcmc.rust.no-clippy-allow-lints src/project_rules/rust_style.rs
    expect_semgrep_count 1 mcmc.rust.expect-requires-reason src/project_rules/rust_style.rs

    uv run semgrep scan --metrics off --test --strict --config ../../semgrep.yaml examples/deep_import.rs
    expect_semgrep_count 3 mcmc.rust.no-box-dyn-error-in-examples-benches examples/erased_error.rs
    expect_semgrep_count 0 mcmc.rust.no-box-dyn-error-in-examples-benches examples/typed_error.rs
    expect_semgrep_count 3 mcmc.rust.no-box-dyn-error-in-examples-benches benches/erased_error.rs
    expect_semgrep_count 0 mcmc.rust.no-box-dyn-error-in-examples-benches benches/typed_error.rs
    expect_semgrep_count 3 mcmc.rust.no-box-dyn-error-in-doctests src/doctests/erased_error.rs
    expect_semgrep_count 0 mcmc.rust.no-box-dyn-error-in-doctests src/doctests/typed_error.rs
    uv run semgrep scan --metrics off --test --strict --config ../../semgrep.yaml scripts/tests/python_exceptions.py

setup: setup-tools

setup-tools:
    #!/usr/bin/env bash
    set -euo pipefail

    have() { command -v "$1" >/dev/null 2>&1; }

    echo "Ensuring Rust components..."
    rustup component add clippy rustfmt rust-docs rust-src llvm-tools-preview
    echo ""

    if have brew; then
        echo "Ensuring Homebrew tools..."
        brew install actionlint dprint git-cliff jq taplo uv yamllint || true
        echo ""
    else
        echo "Homebrew not found; skipping brew-managed tools."
        echo "Install manually if missing: actionlint dprint jq taplo uv yamllint"
        echo ""
    fi

    echo "Ensuring cargo tools..."
    if ! have cargo-llvm-cov; then
        cargo install --locked cargo-llvm-cov --version {{cargo_llvm_cov_version}}
    fi
    if ! cargo rdme --version >/dev/null 2>&1; then
        cargo install --locked cargo-rdme
    fi
    if ! have dprint; then
        cargo install --locked dprint
    fi
    if ! have git-cliff; then
        cargo install --locked git-cliff
    fi
    if ! have typos; then
        cargo install --locked typos-cli
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
    for cmd in actionlint cargo-llvm-cov dprint git-cliff jq taplo typos uv yamllint; do
        if have "$cmd"; then
            echo "  ✓ $cmd"
        else
            echo "  ✗ $cmd"
            missing=1
        fi
    done
    if cargo rdme --version >/dev/null 2>&1; then
        echo "  ✓ cargo-rdme"
    else
        echo "  ✗ cargo-rdme"
        missing=1
    fi

    if [ "$missing" -ne 0 ]; then
        echo ""
        echo "❌ Some required tools are still missing."
        echo "Fix the installs above and re-run: just setup-tools"
        exit 1
    fi

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

# Testing
test:
    cargo test --lib --verbose
    cargo test --doc --verbose

# All tests (lib + doc + integration + Python tooling)
test-all: test test-integration test-python
    @echo "✅ All tests passed"

# Integration tests
test-integration:
    cargo test --tests --verbose

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

# Validate example output (seeded, deterministic)
validate-examples:
    #!/usr/bin/env bash
    set -euo pipefail
    output=$(cargo run --quiet --example detailed_balance)
    echo "$output"
    echo "$output" | grep -q "Detailed balance checks passed" || { echo "❌ detailed_balance: Missing success marker"; exit 1; }
    echo "$output" | grep -q "by-value residual" || { echo "❌ detailed_balance: Missing by-value residual"; exit 1; }
    echo "✅ detailed_balance validated"
    output=$(cargo run --quiet --example normal_1d)
    echo "$output"
    echo "$output" | grep -q "Sample mean" || { echo "❌ normal_1d: Missing sample mean"; exit 1; }
    echo "$output" | grep -q "Acceptance rate" || { echo "❌ normal_1d: Missing acceptance rate"; exit 1; }
    echo "✅ normal_1d validated"
    output=$(cargo run --quiet --example ising_1d)
    echo "$output"
    echo "$output" | grep -q "<m>" || { echo "❌ ising_1d: Missing magnetization"; exit 1; }
    echo "$output" | grep -q "acceptance rate" || { echo "❌ ising_1d: Missing acceptance rate"; exit 1; }
    echo "✅ ising_1d validated"
    output=$(cargo run --quiet --example iterator_sampling)
    echo "$output"
    echo "$output" | grep -q "Sample mean" || { echo "❌ iterator_sampling: Missing sample mean"; exit 1; }
    echo "$output" | grep -q "Acceptance rate" || { echo "❌ iterator_sampling: Missing acceptance rate"; exit 1; }
    echo "✅ iterator_sampling validated"

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
