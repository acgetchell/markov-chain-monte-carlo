# AGENTS.md

Essential guidance for AI assistants working in this repository.

## Priorities

When making changes in this repo, prioritize (in order):

- Correctness
- Speed
- Coverage (but keep the code idiomatic Rust)

## Core Rules

### Git Operations

- **NEVER** run `git commit`, `git push`, `git tag`, or any git commands that modify version control state
- **ALLOWED**: Run read-only git commands (e.g. `git --no-pager status`, `git --no-pager diff`,
  `git --no-pager log`, `git --no-pager show`, `git --no-pager blame`) to inspect changes/history
- **ALWAYS** use `git --no-pager` when reading git output
- Suggest git commands that modify version control state for the user to run manually

### Commit Messages

When user requests commit message generation:

1. Run `git --no-pager diff --cached --stat`
2. Generate conventional commit format: `<type>: <brief summary>`
3. Types: `feat`, `fix`, `refactor`, `perf`, `docs`, `test`, `chore`, `style`, `ci`, `build`
4. Include body with organized bullet points and test results
5. Present in code block (no language) - user will commit manually

### Code Quality

- **ALLOWED**: Run formatters/linters: `cargo fmt`, `cargo clippy`, `cargo doc`
- **NEVER**: Use `sed`, `awk`, `perl` for code edits
- **ALWAYS**: Use `edit_files` tool for edits (and `create_file` for new files)
- **EXCEPTION**: Shell text tools OK for read-only analysis only

### Validation

- **GitHub Actions**: Validate workflows with `just action-lint` (uses `actionlint`)
- **YAML**: Use `just yaml-lint`

### Rust

- Prefer borrowed APIs by default:
  take references (`&T`, `&mut T`, `&[T]`) as arguments and return borrowed views (`&T`, `&[T]`) when possible.
  Only take ownership or return `Vec`/allocated data when required.

## Common Commands

```bash
just fix              # Apply formatters/auto-fixes (mutating)
just check            # Lint/validators (non-mutating)
just ci               # Full CI simulation (checks + tests + examples)
just test             # Lib + doc tests (fast)
just test-all         # All tests (lib + doc + integration)
just examples         # Run all examples
```

### Detailed Command Reference

- All tests: `just test-all`
- Build (debug): `cargo build` (or `just build`)
- Coverage (CI XML): `just coverage-ci`
- Coverage (HTML): `just coverage`
- Fast Rust tests (lib + doc): `just test`
- Format: `cargo fmt` (or `just fmt`)
- Integration tests: `just test-integration`
- Lint (Clippy): `just clippy`
- Lint/validate: `just check`
- Pre-commit validation / CI simulation: `just ci`
- Run a single test (by name filter): `cargo test chain_samples_near_mode`
- Run examples: `just examples` (or `cargo run --example normal_1d`)

### GitHub Issues

Use the `gh` CLI to read, create, and edit issues:

- **Read**: `gh issue view <number>` (or `--json title,body,labels,milestone` for structured data)
- **List**: `gh issue list` (add `--label enhancement`, etc. to filter)
- **Create**: `gh issue create --title "..." --body "..." --label enhancement --label rust`
- **Edit**: `gh issue edit <number> --add-label "..."`, `--milestone "..."`, `--title "..."`
- **Comment**: `gh issue comment <number> --body "..."`
- **Close**: `gh issue close <number>` (with optional `--reason completed` or `--reason "not planned"`)

When creating or updating issues:

- **Labels**: Use appropriate labels: `enhancement`, `bug`, `performance`, `documentation`, `rust`, etc.
- **Dependencies**: Document relationships in issue body and comments:
  - "Depends on: #XXX" - this issue cannot start until #XXX is complete
  - "Blocks: #YYY" - #YYY cannot start until this issue is complete
  - "Related: #ZZZ" - related work but not blocking
- **Issue body format**: Include clear sections: Summary, Current State, Proposed Changes, Benefits, Implementation Notes
- **Cross-referencing**: Always reference related issues/PRs using #XXX notation for automatic linking

## Code structure (big picture)

- This is a single Rust *library crate* (no `src/main.rs`). The crate root is `src/lib.rs`.
- The MCMC framework is implemented in `src/lib.rs`:
  - `McmcError`: error type for NaN detection in log-probabilities and proposal ratios
  - `State`: marker trait for MCMC state types (requires `Clone`)
  - `Target<S>`: trait for target distributions (log-probability)
  - `Proposal<S>`: trait for proposal distributions (propose + log q-ratio)
  - `Chain<S>`: Metropolis–Hastings chain with acceptance tracking
  - `prelude`: convenience re-exports for common usage
- Rust tests are inline `#[cfg(test)]` modules in `src/lib.rs`.
- The `justfile` defines all dev workflows (see `just --list`).
- Examples live in `examples/` (e.g. `normal_1d.rs`).

## Publishing note

- If you publish this crate to crates.io, prefer updating documentation *before* publishing a new version (doc-only changes still require a version bump on crates.io).

## Editing tools policy

- Never use `sed`, `awk`, `python`, or `perl` to edit code or write file changes.
- These tools may be used for read-only inspection, parsing, or analysis, but never for writing.
