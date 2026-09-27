# Justfile for markov-chain-monte-carlo development workflow
# Install just: https://github.com/casey/just
# Usage: just <command> or just --list

# Use bash with strict error handling for all recipes
set shell := ["bash", "-euo", "pipefail", "-c"]

_run := "uv run --locked --group dev research-repo-tools toolchain run --"

fast_notebooks := "notebooks/ising_trace_analysis.ipynb"
slow_notebooks := ""

# Common cargo-llvm-cov arguments for all coverage runs.
# Excludes examples from reports while allowing tests to exercise library code.
_coverage_base_args := '''--ignore-filename-regex '(^|/)examples/' \
  --workspace --all-features --lib --tests \
  --verbose'''

# Examples
_build-examples:
    {{ _run }} cargo build --locked --all-features --examples

# System prerequisites remain outside the shared managed toolchain.
_ensure-jq:
    uv run --locked --only-group tooling research-repo-tools validation require jq

_ensure-uv-stable:
    uv run --no-config --no-sync --no-python-downloads research-repo-tools deps check-uv

_notebook-all mode:
    uv run --locked --group dev --group notebook research-repo-tools files run --include '*.ipynb' --exclude '*/.ipynb_checkpoints/*' -- research-repo-tools notebooks {{ quote(mode) }}

# GitHub Actions workflow validation
[group('validation')]
action-lint:
    # actionlint 1.7.12 predates $/ syntax; remove this exception when issue #711 is released.
    uv run --locked --group dev research-repo-tools files run --include '.github/workflows/*.yml' --include '.github/workflows/*.yaml' -- actionlint -ignore '^specifying action "\$/\.github/actions/setup-toolchain" in invalid format because ref is missing\.'

# Audit every maintained Python and Rust lockfile without executing dependency code.
[group('security')]
audit:
    uv run --locked --group dev research-repo-tools security osv uv.lock Cargo.lock benches/diagnostic_backends/Cargo.lock

# Run the fixed-seed stepping benchmark suite.
[group('benchmarks and performance')]
bench:
    {{ _run }} cargo bench --locked --bench stepping

# Render existing Criterion measurements against an explicit saved baseline.
[group('benchmarks and performance')]
bench-compare baseline="last": sync
    uv run --locked --group dev research-repo-tools performance compare target/criterion target/criterion --baseline-sample {{ quote(baseline) }} --format markdown --output target/bench-reports/performance.md

# Compile benchmark harnesses without running Criterion measurements.
[group('benchmarks and performance')]
bench-compile:
    {{ _run }} cargo bench --locked --all-features --no-run

# Run latest measurements and compare them with a saved Criterion baseline.
[group('benchmarks and performance')]
bench-latest-vs-last baseline="last": bench (bench-compare baseline)

# Save the complete MCMC release-signal set under a Criterion baseline name.
[group('benchmarks and performance')]
bench-save-baseline tag:
    {{ _run }} cargo bench --locked --bench stepping -- --save-baseline {{ quote(tag) }}

# Build the library.
[group('build and setup')]
build:
    {{ _run }} cargo build --locked

# Changelog generation (git-cliff + post-processing)
[group('release')]
changelog: sync
    {{ _run }} research-repo-tools changelog generate

# Rotate completed minor series without regenerating history
[group('release')]
changelog-archive: sync
    uv run --locked --group dev research-repo-tools changelog archive

# Strictly validate the root changelog and every archive
[group('release')]
changelog-check: sync
    uv run --locked --group dev research-repo-tools changelog check

# Preview generated history without publishing root or archive files
[group('release')]
[positional-arguments]
changelog-preview *args: sync
    {{ _run }} research-repo-tools changelog generate --dry-run "$@"

# Generate a prospective release with an explicit ISO date
[group('release')]
changelog-release tag date: sync
    {{ _run }} research-repo-tools changelog generate --tag {{ quote(tag) }} --date {{ quote(date) }}

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
check-repository-tooling: changelog-check python-check notebook-lint validate-json yaml-check action-lint zizmor justfile-fmt-check toml-fmt-check toml-lint markdown-check spell-check release-check performance-check semgrep-test semgrep
    @echo "✅ Repository tooling checks complete!"

# Rust validation that is meaningful for source portability and user-facing API correctness.
[group('validation')]
check-rust: fmt-check clippy
    @echo "✅ Rust checks complete!"

# Runnable Rust unit and integration tests share one release-profile nextest pass;
# rustdoc doctests remain separate because nextest does not execute them.
# Run repository validators, all-target Clippy, tests, docs, examples, and benchmark compilation.
[group('workflows')]
ci: changelog-check action-lint zizmor justfile-fmt-check markdown-check spell-check release-check performance-check validate-json toml-fmt-check toml-lint yaml-check python-check semgrep-test semgrep test-python notebook-check fmt-check clippy-all-targets doc test-rust-ci test-doc bench-compile validate-examples
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
    just coverage-report

# Summarize existing Cobertura data without rerunning instrumented tests.
[group('tests and coverage')]
[positional-arguments]
coverage-report *args:
    uv run --locked --group dev research-repo-tools coverage report "$@"

# List all public recipes when Just is invoked without a recipe.
[default]
[private]
default:
    @just --justfile {{ quote(justfile()) }} --list

# Build rustdoc for the library.
[group('validation')]
doc:
    {{ _run }} cargo doc --locked --all-features --no-deps --document-private-items

# Run one example by name, e.g. `just example ising_1d`.
[group('tests and coverage')]
example name:
    {{ _run }} cargo run --locked --all-features --example "{{ name }}"

# Build and run every Rust example.
[group('tests and coverage')]
examples: _build-examples
    {{ _run }} research-repo-tools validation run tooling/examples.toml

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
    {{ _run }} research-repo-tools files run --include '*.md' --exclude CHANGELOG.md --exclude 'docs/performance/**' -- rumdl check

# Apply Markdown formatting and lint fixes.
[group('validation')]
markdown-fix:
    {{ _run }} research-repo-tools files run --include '*.md' --exclude CHANGELOG.md --exclude 'docs/performance/**' -- rumdl check --fix

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

# Forward explicit shared performance operations.
[group('benchmarks and performance')]
[positional-arguments]
performance +args:
    {{ _run }} research-repo-tools performance "$@"

# Measure a clean tagged checkout and package its shared release baseline.
[group('benchmarks and performance')]
performance-baseline tag:
    {{ _run }} research-repo-tools performance baseline tooling/benchmark.toml {{ quote(tag) }} {{ quote("markov-chain-monte-carlo-" + tag + "-criterion-baseline.tar.gz") }}

# Check retained reports, allowing an empty inventory before the first comparison.
[group('validation')]
performance-check:
    #!/usr/bin/env bash
    set -euo pipefail
    shopt -s nullglob
    artifacts=(
        docs/performance/v1/*.comparison.json
        docs/performance/v1/*.evidence.json
        docs/performance/v1/*.csv
        docs/performance/v1/*.svg
        docs/performance/v1/v*-vs-v*.md
    )
    if [[ -e docs/performance/v1/performance.md || -L docs/performance/v1/performance.md ]]; then
        uv run --locked --group dev research-repo-tools performance promote tooling/performance-report.toml --check
    elif (( ${#artifacts[@]} )) || [[ -e tooling/performance-readme.toml || -L tooling/performance-readme.toml ]]; then
        echo "Release evidence exists without its current report; restore or promote the report." >&2
        exit 1
    else
        echo "No release comparison yet; the next tagged release establishes the baseline."
    fi

# Rebuild and promote the shared report from retained or explicitly saved evidence.
[group('benchmarks and performance')]
[positional-arguments]
performance-doc *args: sync
    uv run --locked --group dev research-repo-tools performance promote tooling/performance-report.toml "$@"

# Compare stored GitHub Release benchmark assets without local benchmark runs.
[group('benchmarks and performance')]
[positional-arguments]
performance-github-assets *tags: sync
    uv run --locked --group dev research-repo-tools performance assets "$@" --order published --repository acgetchell/markov-chain-monte-carlo --asset-template 'markov-chain-monte-carlo-{tag}-criterion-baseline.tar.gz' --payload target/bench-reports/github-assets.comparison.json --manifest target/bench-reports/github-assets.evidence.json --report target/bench-reports/github-assets.md

# Compare the current tree with the latest stable published release locally.
[group('benchmarks and performance')]
performance-local: sync
    {{ _run }} research-repo-tools performance measure tooling/benchmark.toml --mode current-vs-latest --order published --allow-git-mutations --payload target/bench-reports/local.comparison.json --manifest target/bench-reports/local.evidence.json --report target/bench-reports/local.md

# Publish the README table, SVG, and pinned links from validated retained release evidence.
[group('benchmarks and performance')]
[positional-arguments]
performance-readme *args: sync
    uv run --locked --group dev research-repo-tools performance publish tooling/performance-readme.toml "$@"

# Generate a release-to-release report, promote it, and archive the previous report.
[group('benchmarks and performance')]
[positional-arguments]
performance-release *tags: sync
    {{ _run }} research-repo-tools performance measure tooling/benchmark.toml "$@" --order published --allow-git-mutations --payload target/bench-reports/release.comparison.json --manifest target/bench-reports/release.evidence.json
    uv run --locked --group dev research-repo-tools performance promote tooling/performance-report.toml --payload target/bench-reports/release.comparison.json --manifest target/bench-reports/release.evidence.json

# Publish the reviewed crate to crates.io using the managed Cargo toolchain.
[group('release')]
publish:
    {{ _run }} cargo publish --locked

# Pre-publish validation: checks crates.io metadata rules that cargo publish --dry-run does NOT catch
[group('release')]
publish-check:
    {{ _run }} research-repo-tools validation cargo-metadata
    {{ _run }} cargo publish --locked --allow-dirty --dry-run

# Check all Python files and fixtures with the complete configured Ruff and Ty policy.
[group('validation')]
python-check: python-typecheck
    uv run --locked --group dev research-repo-tools files run --include '*.py' --include '*.pyi' -- ruff format --check --no-force-exclude
    uv run --locked --group dev research-repo-tools files run --include '*.py' --include '*.pyi' -- ruff check --no-fix --no-force-exclude

# Apply configured Ruff fixes and formatting to all Python files and fixtures.
[group('validation')]
python-fix: sync
    uv run --locked --group dev research-repo-tools files run --include '*.py' --include '*.pyi' -- ruff check --fix --no-force-exclude
    uv run --locked --group dev research-repo-tools files run --include '*.py' --include '*.pyi' -- ruff format --no-force-exclude

# Type-check every discovered Python file and fixture with Ty.
[group('validation')]
python-typecheck: sync
    uv run --locked --no-sync --no-python-downloads research-repo-tools toolchain python-check
    uv run --locked --group dev research-repo-tools files run --include '*.py' --include '*.pyi' -- ty check --no-force-exclude

# Validate a stable release tag, package metadata, and generated notes without publishing.
[group('release')]
release-check tag="": sync
    #!/usr/bin/env bash
    set -euo pipefail
    package_version="$({{ _run }} cargo read-manifest | jq -er 'if .name == "markov-chain-monte-carlo" then .version else error("expected package.name = markov-chain-monte-carlo") end')"
    release_tag={{ quote(tag) }}
    release_tag="${release_tag:-v$package_version}"
    if [[ ! "$release_tag" =~ ^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$ ]]; then
        printf 'Release preflight failed: expected a stable vX.Y.Z tag, got %s\n' "$release_tag" >&2
        exit 1
    fi
    if [[ "$release_tag" != "v$package_version" ]]; then
        printf 'Release preflight failed: tag %s does not match package version %s\n' "$release_tag" "$package_version" >&2
        exit 1
    fi
    uv run --locked --group dev research-repo-tools release check --final-release
    uv run --locked --group dev research-repo-tools changelog notes "$release_tag" > /dev/null
    printf 'Release preflight passed for %s.\n' "$release_tag"

# Extract release notes from the root changelog or its archives
[group('release')]
release-notes tag: sync
    uv run --locked --group dev research-repo-tools changelog notes {{ quote(tag) }}

# Update release metadata using an explicit version, previous tag, and date.
[group('release')]
[positional-arguments]
release-update version previous date *args: sync
    uv run --locked --group dev research-repo-tools release update "v${1#v}" --previous-release "$2" --date "$3" "${@:4}"

# Inspect the GitHub Release and the exact published crates.io version.
[group('release')]
[positional-arguments]
release-verify tag:
    gh release view "$1" --json tagName,isDraft,isPrerelease,body,assets
    {{ _run }} cargo info "markov-chain-monte-carlo@${1#v}" --registry crates-io

# Review committed and local changes against the PR base with CodeRabbit.
[group('review')]
review base="origin/main":
    uv run --locked --group dev research-repo-tools review branch --base={{ quote(base) }}

# Review staged, unstaged, and new files without committed branch changes.
[group('review')]
review-uncommitted:
    uv run --locked --group dev research-repo-tools review uncommitted

# Run the network-dependent vulnerability and full-history secret gates.
[group('security')]
security: audit security-secrets

# Scan all reachable Git history and current tracked/nonignored files with redacted reports.
[group('security')]
security-secrets:
    uv run --locked --group dev research-repo-tools security secrets

# Repository-owned Semgrep rules for project-specific diagnostics.
[group('validation')]
semgrep:
    uv run --locked --group dev research-repo-tools files run --exclude 'tests/semgrep/**' --exclude 'docs/performance/**' --timeout 600 -- semgrep --metrics off --error --strict --timeout 30 --config semgrep.yaml

# Validate repository-owned Semgrep rules against annotated fixtures.
[group('validation')]
semgrep-test:
    uv run --locked --group dev research-repo-tools semgrep check-fixtures

# Install managed tools and verify system prerequisites.
[group('build and setup')]
setup: _ensure-jq
    uv run --locked --managed-python --only-group tooling research-repo-tools setup

# Preview a shared-package/Python migration outside the current environment.
[group('build and setup')]
[positional-arguments]
shared-python-plan version:
    uvx --no-config --isolated --managed-python --from "research-repo-tools==$1" research-repo-tools toolchain adopt --dry-run

# Adopt the selected shared package and its Python baseline, environment, and kernel.
[group('build and setup')]
[positional-arguments]
shared-python-update version:
    uvx --no-config --isolated --managed-python --from "research-repo-tools==$1" research-repo-tools toolchain adopt --apply

# Check repository spelling.
[group('validation')]
spell-check:
    {{ _run }} typos --config typos.toml --force-exclude .

# Synchronize the locked development environment with the declared toolchain.
[group('build and setup')]
sync:
    {{ _run }} uv sync --locked --managed-python --group dev

# Create an annotated git tag from the CHANGELOG.md section for the given version
[group('release')]
tag version: sync
    uv run --locked --group dev research-repo-tools changelog tag {{ quote(version) }}

# Recreate an existing tag from the CHANGELOG.md section for the given version
[group('release')]
tag-force version: sync
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
    {{ _run }} cargo test --locked --all-features --doc --verbose

# Run integration tests across all public features.
[group('tests and coverage')]
test-integration:
    {{ _run }} cargo nextest run --locked --all-features --test '*' --verbose

# Run Python consumer integration tests.
[group('tests and coverage')]
test-python: sync
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
toml-fix:
    {{ _run }} research-repo-tools files run --include '*.toml' -- taplo fmt

# Check tracked TOML formatting without modifying files.
[group('validation')]
toml-fmt-check:
    {{ _run }} research-repo-tools files run --include '*.toml' -- taplo fmt --check

# Lint tracked TOML files.
[group('validation')]
toml-lint:
    {{ _run }} research-repo-tools files run --include '*.toml' -- taplo lint

# Inspect declared tools without installing or synchronizing anything.
[group('build and setup')]
tools-check: _ensure-jq
    uv run --locked --no-sync --no-python-downloads research-repo-tools toolchain check

# Preview obsolete shared tool installs; --apply removes them, --keep-root retains other consumers' pins.
[group('build and setup')]
[positional-arguments]
tools-clean *args:
    uv run --locked --no-sync --no-python-downloads research-repo-tools toolchain clean "$@"

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

# Upgrade uv and declared Cargo tools, then synchronize the declared environment.
[group('build and setup')]
update-tools: update-uv update-cargo-tools setup

# Upgrade uv through its installation owner and reconcile the exact project pin.
[group('build and setup')]
update-uv:
    uv run --no-config --no-sync --no-python-downloads research-repo-tools deps update-uv

# Validate the Ising example once while generating the notebook input trace.
# Validate example output (seeded, deterministic)
[group('tests and coverage')]
validate-examples: _build-examples validate-ising-example
    {{ _run }} research-repo-tools validation run tooling/examples.toml detailed_balance normal_1d iterator_sampling delayed_chunked_telemetry additive_target_bias benchmark_distributions adaptive_normal diagnostics

# Validate the Ising example output and produce its trace for notebook checks.
[group('tests and coverage')]
validate-ising-example: _build-examples
    {{ _run }} research-repo-tools validation run tooling/examples.toml ising_1d

# Validate tracked JSON files.
[group('validation')]
validate-json: _ensure-jq
    uv run --locked --group dev research-repo-tools files run --include '*.json' --batch-size 1 -- jq empty

# YAML formatting check
[group('validation')]
yaml-check:
    {{ _run }} research-repo-tools files run --include '*.yml' --include '*.yaml' -- dprint check

# YAML formatting
[group('validation')]
yaml-fix:
    {{ _run }} research-repo-tools files run --include '*.yml' --include '*.yaml' -- dprint fmt

# GitHub Actions security analysis
[group('validation')]
[positional-arguments]
zizmor *args:
    uv run --locked --group dev research-repo-tools zizmor check "$@" .github
