# Justfile for markov-chain-monte-carlo development workflow
# Install just: https://github.com/casey/just
# Usage: just <command> or just --list

# Use bash with strict error handling for all recipes
set shell := ["bash", "-euo", "pipefail", "-c"]

_run := "uv run --locked --group dev research-repo-tools toolchain run --"

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
    {{ _run }} cargo build --locked --examples

# System prerequisites remain outside the shared managed toolchain.
_ensure-gh:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v gh >/dev/null || {
        echo "❌ 'gh' not found. Install GitHub CLI: https://cli.github.com/" >&2
        exit 1
    }

_ensure-jq:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v jq >/dev/null || {
        echo "❌ 'jq' not found. Install it with your system package manager or follow:"
        echo "   https://jqlang.github.io/jq/download/"
        exit 1
    }

_ensure-uv-stable:
    uv run --no-config --no-sync --no-python-downloads research-repo-tools deps check-uv

_notebook-all mode:
    #!/usr/bin/env bash
    set -euo pipefail
    notebooks=()
    while IFS= read -r -d '' file; do
        [[ -f "$file" ]] || continue
        case "$file" in
            */.ipynb_checkpoints/*) continue ;;
        esac
        notebooks+=("$file")
    done < <(git ls-files -co --exclude-standard -z -- 'notebooks/*.ipynb')
    if [ "${#notebooks[@]}" -gt 0 ]; then
        uv run --locked --group dev --group notebook research-repo-tools notebooks {{ quote(mode) }} "${notebooks[@]}"
    else
        echo "No notebooks found."
    fi

# GitHub Actions workflow validation
[group('validation')]
action-lint:
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(git ls-files -co --exclude-standard -z -- '.github/workflows/*.yml' '.github/workflows/*.yaml')
    if [ "${#files[@]}" -gt 0 ]; then
        # actionlint 1.7.12 predates $/ syntax; ignore only this valid self-repository reference.
        # Remove when https://github.com/rhysd/actionlint/issues/711 is released.
        printf '%s\0' "${files[@]}" | xargs -0 uv run --locked --group dev actionlint \
            -ignore '^specifying action "\$/\.github/actions/setup-toolchain" in invalid format because ref is missing\.'
    else
        echo "No workflow files found to lint."
    fi

# Run the Criterion benchmark suite.
[group('benchmarks and performance')]
bench:
    {{ _run }} cargo bench --locked --bench stepping

# Render existing Criterion measurements against an explicit saved baseline.
[group('benchmarks and performance')]
bench-compare baseline="last": python-sync
    uv run --locked --group dev bench-compare {{ quote(baseline) }}

# Compile benchmark harnesses without running Criterion measurements.
[group('benchmarks and performance')]
bench-compile:
    {{ _run }} cargo bench --locked --all-features --no-run

# Run the fixed-seed MCMC release-signal benchmark set.
[group('benchmarks and performance')]
bench-latest: bench

# Run latest measurements and compare them with a saved Criterion baseline.
[group('benchmarks and performance')]
bench-latest-vs-last baseline="last": bench-latest python-sync
    uv run --locked --group dev bench-compare {{ quote(baseline) }}

# Save the complete MCMC release-signal set under a Criterion baseline name.
[group('benchmarks and performance')]
bench-save-baseline tag:
    {{ _run }} cargo bench --locked --bench stepping -- --save-baseline {{ quote(tag) }}

# Save the current release signal under the conventional local `last` name.
[group('benchmarks and performance')]
bench-save-last:
    just bench-save-baseline last

# Build the library.
[group('build and setup')]
build:
    {{ _run }} cargo build --locked

# Changelog generation (git-cliff + post-processing)
[group('release')]
changelog: python-sync
    {{ _run }} research-repo-tools changelog generate

# Rotate completed minor series without regenerating history
[group('release')]
changelog-archive: python-sync
    uv run --locked --group dev research-repo-tools changelog archive

# Strictly validate the root changelog and every archive
[group('release')]
changelog-check: python-sync
    uv run --locked --group dev research-repo-tools changelog check

# Preview generated history without publishing root or archive files
[group('release')]
[positional-arguments]
changelog-preview *args: python-sync
    {{ _run }} research-repo-tools changelog generate --dry-run "$@"

# Generate a prospective release with an explicit ISO date
[group('release')]
changelog-release tag date: python-sync
    {{ _run }} research-repo-tools changelog generate --tag {{ quote(tag) }} --date {{ quote(date) }}

alias changelog-unreleased := changelog-release

# Non-mutating validation gate
[group('workflows')]
check: check-rust check-repository-tooling
    @echo "✅ Checks complete!"

# Fast compile check (no binary produced)
[group('build and setup')]
check-fast:
    {{ _run }} cargo check --locked

# Repository tooling that does not need to be repeated across operating systems.
[group('validation')]
check-repository-tooling: changelog-check python-check notebook-lint validate-json yaml-check action-lint zizmor justfile-fmt-check toml-fmt-check toml-lint markdown-check spell-check release-check semgrep-test semgrep
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
ci: changelog-check action-lint zizmor justfile-fmt-check markdown-check spell-check release-check validate-json toml-fmt-check toml-lint yaml-check python-check semgrep-test semgrep test-python notebook-check fmt-check clippy-all-targets doc test-rust-ci test-doc bench-compile validate-examples
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
    {{ _run }} cargo clean
    rm -rf target/llvm-cov
    rm -rf coverage

# Fast core-library Clippy linting used by `just check`.
[group('validation')]
clippy:
    CARGO_BUILD_WARNINGS=deny {{ _run }} cargo clippy --locked --workspace --all-features --lib -- -W clippy::pedantic -W clippy::nursery -W clippy::cargo -A clippy::multiple_crate_versions

# Full Cargo-target Clippy sweep used by `just ci` and the GitHub SARIF workflow.
[group('validation')]
clippy-all-targets:
    CARGO_BUILD_WARNINGS=deny {{ _run }} cargo clippy --locked --workspace --all-features --all-targets -- -W clippy::pedantic -W clippy::nursery -W clippy::cargo -A clippy::multiple_crate_versions

# Coverage analysis for local development (HTML output)
[group('tests and coverage')]
coverage:
    #!/usr/bin/env bash
    set -euo pipefail

    mkdir -p target/llvm-cov
    {{ _run }} cargo llvm-cov {{ _coverage_base_args }} --open --output-dir target/llvm-cov
    echo "Coverage report generated: target/llvm-cov/html/index.html"

# Coverage analysis for CI (XML output for codecov)
[group('tests and coverage')]
coverage-ci:
    #!/usr/bin/env bash
    set -euo pipefail

    mkdir -p coverage
    {{ _run }} cargo llvm-cov {{ _coverage_base_args }} --cobertura --output-path coverage/cobertura.xml

# Show curated workflows when Just is invoked without a recipe.
[default]
[private]
default: help-workflows

# Build rustdoc for the library.
[group('validation')]
doc:
    {{ _run }} cargo doc --locked --no-deps --document-private-items

# Run one example by name, e.g. `just example ising_1d`.
[group('tests and coverage')]
example name:
    {{ _run }} cargo run --locked --example "{{ name }}"

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
    {{ _run }} cargo fmt --all

# Check Rust source formatting without modifying files.
[group('validation')]
fmt-check:
    {{ _run }} cargo fmt --all -- --check

# Show the curated entry points for common repository workflows.
[group('workflows')]
help-workflows:
    @echo "Common Just workflows:"
    @echo "  just changelog      # Regenerate CHANGELOG.md from local git history"
    @echo "  just changelog-unreleased <tag> <date> # Generate notes with an explicit ISO date"
    @echo "  just check          # Run lint/validators (non-mutating)"
    @echo "  just check-fast     # Fast compile check (cargo check)"
    @echo "  just ci             # Full CI simulation, including zizmor and benchmark compile"
    @echo "  just ci-portability # Portability subset for CI-shape timing"
    @echo "  just ci-repository-tooling # Repository tooling subset for CI-shape timing"
    @echo "  just ci-rust        # Rust correctness subset for CI-shape timing"
    @echo "  just fix            # Apply formatters/auto-fixes (mutating)"
    @echo "  just release-check  # Validate synchronized release metadata and references"
    @echo "  just setup          # Install managed tools and verify system prerequisites"
    @echo "  just tools-check    # Inspect the declared toolchain without installing anything"
    @echo "  just tag <ver>      # Create annotated release tag from CHANGELOG.md"
    @echo "  just update         # Update dependencies, managed Cargo tools, and tool pins"
    @echo "  just update-version <tag> # Prepare release metadata from one stable tag"
    @echo ""
    @echo "Local CodeRabbit review:"
    @echo "  just review [base]      # Review the branch and local edits; base defaults to origin/main"
    @echo "  just review-uncommitted # Review only local edits, including new files"
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
    @echo "  just notebook-lint  # Validate structure, output hygiene, and native notebook Python"
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
    {{ _run }} just --fmt

# Check Justfile formatting without modifying it.
[group('validation')]
justfile-fmt-check:
    {{ _run }} just --fmt --check

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
markdown-check:
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
        printf '%s\0' "${files[@]}" | xargs -0 -n100 {{ _run }} rumdl check
    else
        echo "No Markdown files found to check."
    fi

# Apply Markdown formatting and lint fixes.
[group('validation')]
markdown-fix:
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
        printf '%s\0' "${files[@]}" | xargs -0 -n100 {{ _run }} rumdl check --fix
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
notebook-clear-outputs-all: (_notebook-all 'clear')

# Execute the configured fast notebook set headlessly.
[group('notebooks')]
notebook-execute-fast: notebook-sync validate-ising-example
    #!/usr/bin/env bash
    set -euo pipefail
    notebooks=( {{ fast_notebooks }} )
    uv run --locked --group dev --group notebook research-repo-tools notebooks execute "${notebooks[@]}"

# Execute the explicitly configured slow notebook set headlessly.
[group('notebooks')]
notebook-execute-slow:
    #!/usr/bin/env bash
    set -euo pipefail
    notebooks=( {{ slow_notebooks }} )
    if [ "${#notebooks[@]}" -eq 0 ]; then
        echo "No slow notebooks configured."
        exit 0
    fi
    just notebook-sync
    uv run --locked --group dev --group notebook research-repo-tools notebooks execute "${notebooks[@]}" --timeout 1800

# Regenerate the tracked Ising figure from the example trace and notebook.
[group('notebooks')]
notebook-ising-figure: notebook-check
    cp target/notebooks/ising_energy_trace.png docs/assets/ising_energy_trace.png

# Validate source notebook structure and native notebook Python without execution.
[group('notebooks')]
notebook-lint: (_notebook-all 'lint')

# Synchronize notebook dependencies and register the project-local kernel.
[group('notebooks')]
notebook-sync:
    uv run --locked --managed-python --only-group tooling research-repo-tools notebooks sync

# Rebuild and promote the curated report from tracked or explicitly saved release measurements.
[group('benchmarks and performance')]
performance-doc measurements_path="": python-sync
    #!/usr/bin/env bash
    set -euo pipefail
    measurements_path={{ quote(measurements_path) }}
    if [[ -n "$measurements_path" ]]; then
        {{ _run }} archive-performance --rerender "$measurements_path" --promote
    else
        {{ _run }} archive-performance --rerender --promote
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
        {{ _run }} archive-performance "$current_tag" "$baseline_tag" --github-assets --measurements-output target/bench-reports/github-assets-performance.csv --output target/bench-reports/github-assets-performance.md
    else
        {{ _run }} archive-performance --published-latest --github-assets --measurements-output target/bench-reports/github-assets-performance.csv --output target/bench-reports/github-assets-performance.md
    fi

# Compare the current tree with the latest stable published release locally.
[group('benchmarks and performance')]
performance-local: python-sync
    {{ _run }} archive-performance --current-vs-latest --measurements-output target/bench-reports/performance.csv --output target/bench-reports/performance.md

# Publish the README table, SVG, and pinned links from validated retained release evidence.
[group('benchmarks and performance')]
performance-readme: python-sync
    uv run --locked --group dev publish-performance-readme

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
        {{ _run }} archive-performance "$current_tag" "$baseline_tag" --measurements-output target/bench-reports/release-performance.csv --promote
    else
        {{ _run }} archive-performance --infer-release --measurements-output target/bench-reports/release-performance.csv --promote
    fi

# Pre-publish validation: checks crates.io metadata rules that cargo publish --dry-run does NOT catch
[group('release')]
publish-check: _ensure-jq
    #!/usr/bin/env bash
    set -euo pipefail
    echo "🔍 Validating crates.io metadata..."
    errors=0

    # Keywords: max 5, each ≤20 chars, ASCII alphanumeric/hyphen only
    keywords=$({{ _run }} cargo metadata --no-deps --format-version=1 2>/dev/null \
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
    cat_count=$({{ _run }} cargo metadata --no-deps --format-version=1 2>/dev/null \
        | jq '.packages[0].categories | length')
    if (( cat_count > 5 )); then
        echo "  ❌ too many categories ($cat_count > 5)"
        errors=1
    fi
    echo "  ✓ categories ($cat_count)"

    # Description: required, ≤1000 chars
    desc=$({{ _run }} cargo metadata --no-deps --format-version=1 2>/dev/null \
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
    {{ _run }} cargo publish --locked --allow-dirty --dry-run
    echo ""
    echo "✅ Publish check passed!"

# Check Python support scripts with Ruff and Ty.
[group('validation')]
python-check: python-typecheck
    uv run --locked --group dev ruff format --check scripts/
    uv run --locked --group dev ruff check scripts/

# Apply Ruff fixes and formatting to Python support scripts.
[group('validation')]
python-fix: python-sync
    uv run --locked --group dev ruff check scripts/ --fix
    uv run --locked --group dev ruff format scripts/

# Alias for the canonical Python check.
[group('validation')]
python-lint: python-check

# Synchronize the locked development environment with the declared toolchain.
[group('build and setup')]
python-sync:
    {{ _run }} uv sync --locked --managed-python --group dev

# Type-check Python support scripts with Ty.
[group('validation')]
python-typecheck: python-sync
    uv run --locked --group dev ty check scripts/

# Validate synchronized release metadata and active version references.
[group('release')]
release-check: python-sync
    uv run --locked --group dev research-repo-tools release check --final-release

# Extract release notes from the root changelog or its archives
[group('release')]
release-notes tag: python-sync
    uv run --locked --group dev research-repo-tools changelog notes {{ quote(tag) }}

# Review committed and local changes against the PR base with CodeRabbit.
[group('review')]
review base="origin/main":
    uv run --locked --group dev research-repo-tools review branch --base={{ quote(base) }}

# Review staged, unstaged, and new files without committed branch changes.
[group('review')]
review-uncommitted:
    uv run --locked --group dev research-repo-tools review uncommitted

# Repository-owned Semgrep rules for project-specific diagnostics.
[group('validation')]
semgrep:
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        [[ -f "$file" ]] || continue
        case "$file" in
            tests/semgrep/*) continue ;;
        esac
        files+=("$file")
    done < <(git ls-files -co --exclude-standard -z --)
    if [ "${#files[@]}" -gt 0 ]; then
        uv run --locked --group dev semgrep --metrics off --error --strict --timeout 30 --config semgrep.yaml "${files[@]}"
    else
        echo "No tracked or untracked repository files found to scan."
    fi

# Validate repository-owned Semgrep rules against annotated fixtures.
[group('validation')]
semgrep-test:
    uv run --locked --group dev research-repo-tools semgrep check-fixtures

# Install managed tools and verify system prerequisites.
[group('build and setup')]
setup: _ensure-jq
    uv run --locked --managed-python --only-group tooling research-repo-tools setup

# Compatibility entry point for shared setup.
[group('build and setup')]
setup-tools: setup

# Check repository spelling.
[group('validation')]
spell-check:
    {{ _run }} typos --config typos.toml --force-exclude .

# Create an annotated git tag from the CHANGELOG.md section for the given version
[group('release')]
tag version: python-sync
    uv run --locked --group dev research-repo-tools changelog tag {{ quote(version) }}

# Recreate an existing tag from the CHANGELOG.md section for the given version
[group('release')]
tag-force version: python-sync
    uv run --locked --group dev research-repo-tools changelog tag {{ quote(version) }} --force

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
    {{ _run }} cargo test --locked --doc --verbose

# Integration tests
[group('tests and coverage')]
test-integration:
    {{ _run }} cargo nextest run --locked --test '*' --verbose

# Backward-compatible alias for the former recipe name.
[group('tests and coverage')]
test-lib: test-unit

# Run Python support-script tests.
[group('tests and coverage')]
test-python: python-sync
    uv run --locked --group dev pytest -q

# Broad Rust test workflow; doctests remain a separate cargo-test bucket.
[group('tests and coverage')]
test-rust: test-rust-ci test-doc
    @echo "✅ Rust tests passed"

# Broad release-profile Rust CI bucket: lib unit and integration tests together.
[group('tests and coverage')]
test-rust-ci:
    {{ _run }} cargo nextest run --locked --release --profile ci --all-features --lib --tests --verbose

# Focused library unit tests for changed-surface validation.
[group('tests and coverage')]
test-unit:
    {{ _run }} cargo nextest run --locked --lib --verbose

# Apply canonical TOML formatting.
[group('validation')]
toml-fix: toml-fmt

# Format tracked TOML files.
[group('validation')]
toml-fmt:
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(git ls-files -co --exclude-standard -z -- '*.toml')
    if [ "${#files[@]}" -gt 0 ]; then
        {{ _run }} taplo fmt "${files[@]}"
    else
        echo "No TOML files found to format."
    fi

# Check tracked TOML formatting without modifying files.
[group('validation')]
toml-fmt-check:
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(git ls-files -co --exclude-standard -z -- '*.toml')
    if [ "${#files[@]}" -gt 0 ]; then
        {{ _run }} taplo fmt --check "${files[@]}"
    else
        echo "No TOML files found to check."
    fi

# Lint tracked TOML files.
[group('validation')]
toml-lint:
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(git ls-files -co --exclude-standard -z -- '*.toml')
    if [ "${#files[@]}" -gt 0 ]; then
        {{ _run }} taplo lint "${files[@]}"
    else
        echo "No TOML files found to lint."
    fi

# Inspect declared tools without installing or synchronizing anything.
[group('build and setup')]
tools-check: _ensure-jq
    uv run --locked --no-sync --no-python-downloads research-repo-tools toolchain check

# Upgrade tools, then Cargo and Python dependencies.
[group('build and setup')]
update: update-tools update-dependencies
    @echo "✅ Repository dependencies and tools updated."

# Advance Cargo dependency requirements and lockfile entries.
[group('build and setup')]
update-cargo-dependencies:
    uv run --locked --only-group tooling --inexact research-repo-tools toolchain run -- cargo upgrade --incompatible allow
    uv run --locked --only-group tooling --inexact research-repo-tools toolchain run -- cargo update

# Upgrade declared Cargo tools and publish their verified TOML pins.
[group('build and setup')]
update-cargo-tools:
    uv run --locked --only-group tooling --inexact research-repo-tools toolchain upgrade

# Advance Cargo and Python requirements and their lockfiles.
[group('build and setup')]
update-dependencies: update-cargo-dependencies update-python-dependencies

# Resolve direct dev pins, upgrade the Python lock, and synchronize dev.
[group('build and setup')]
update-python-dependencies: _ensure-uv-stable
    uv run --locked --only-group tooling --inexact research-repo-tools deps update-python
    uv lock --upgrade
    uv run --locked --no-sync --no-python-downloads research-repo-tools toolchain run -- uv sync --locked --managed-python --group dev

alias update-python-deps := update-python-dependencies

# Upgrade uv and declared Cargo tools, then synchronize the declared environment.
[group('build and setup')]
update-tools: update-uv update-cargo-tools setup

# Upgrade uv through its installation owner and reconcile the exact project pin.
[group('build and setup')]
update-uv:
    uv run --no-config --no-sync --no-python-downloads research-repo-tools deps update-uv

# Prepare versions, dates, and active references from a stable tag without upgrading dependencies.
[group('release')]
update-version tag: _ensure-gh python-sync
    uv run --locked --group dev update-release-version {{ quote(tag) }}

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
yaml-check:
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(git ls-files -co --exclude-standard -z -- '*.yml' '*.yaml')
    if [ "${#files[@]}" -gt 0 ]; then
        printf '%s\0' "${files[@]}" | xargs -0 {{ _run }} dprint check
    else
        echo "No YAML files found to check."
    fi

# YAML formatting
[group('validation')]
yaml-fix:
    #!/usr/bin/env bash
    set -euo pipefail
    files=()
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(git ls-files -co --exclude-standard -z -- '*.yml' '*.yaml')
    if [ "${#files[@]}" -gt 0 ]; then
        printf '%s\0' "${files[@]}" | xargs -0 {{ _run }} dprint fmt
    else
        echo "No YAML files found to format."
    fi

# Alias for the canonical YAML check.
[group('validation')]
yaml-lint: yaml-check

# GitHub Actions security analysis
[group('validation')]
zizmor:
    {{ _run }} zizmor .github
