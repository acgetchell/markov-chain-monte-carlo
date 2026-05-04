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
- **ALLOWED**: Run read-only git commands (e.g. `git --no-pager status`, `git --no-pager diff`, `git --no-pager log`, `git --no-pager show`, `git --no-pager blame`) to inspect changes/history
- **ALWAYS** use `git --no-pager` when reading git output
- Suggest git commands that modify version control state for the user to run manually
- When suggesting branch names, prefer `{type}/{issue}-descriptor-or-two`, e.g. `fix/307-acceptance-rate`, `ci/312-rust-tooling`, or `doc/329-citation-notes`. If an environment requires an owner/tool prefix, keep this structure after the prefix, e.g. `codex/ci/312-rust-tooling`.

### Commit Message Generation

When generating commit messages:

1. Run `git --no-pager diff --cached --stat`
2. Use conventional commits: `<type>: <summary>`
3. Valid types: `feat`, `fix`, `refactor`, `perf`, `docs`, `test`, `chore`, `style`, `ci`, `build`
4. Include bullet-point body describing key changes
5. Include test results
6. Present inside a code block so the user can commit manually

#### Changelog-Aware Body Text

Commit subjects and bodies feed `CHANGELOG.md` through `git-cliff`. Write them as clean, readable release-note prose:

- Keep the subject line concise; it becomes the primary changelog bullet.
- The type determines the changelog section (`feat` -> Added, `fix` -> Fixed, `refactor`/`test`/`style` -> Changed, `perf` -> Performance, `docs` -> Documentation, `build`/`chore`/`ci` -> Maintenance).
- Include PR references as `(#N)` in the subject when known; `git-cliff` auto-links them.
- Body text appears as indented supporting detail under the changelog bullet.
- Avoid Markdown headings `#` through `###` in the body because they conflict with changelog release and section headings. Use plain labels such as `Tests:` instead.
- Keep body text as plain prose or simple bullet lists. Avoid deep nesting.
- Include only release-note-worthy implementation details; avoid dumping internal command output.

#### Breaking Changes

Breaking changes must use one of these conventional commit markers so `git-cliff` can detect them:

- Bang notation: `feat!: remove deprecated API`
- Footer trailer: `BREAKING CHANGE: <description>`

Examples of breaking changes include removing or renaming public API items, changing default behavior, bumping MSRV, altering serialized checkpoint formats, changing numerical semantics, or changing acceptance/error behavior.

### Code Quality

- **ALLOWED**: Run formatters/linters: `cargo fmt`, `cargo clippy`, `cargo doc`, `just check`
- **NEVER**: Use `sed`, `awk`, `python`, or `perl` to edit code or write file changes
- **ALWAYS**: Use `edit_files` tool for edits (and `create_file` for new files)
- **EXCEPTION**: Shell text tools OK for read-only analysis only

### Validation

- **Primary gate**: Run `just check` for non-mutating local validation (Rust format check, Clippy, Python checks, YAML, GitHub Actions, TOML, Markdown, spell check, Semgrep, Semgrep rule tests)
- **Full CI simulation**: Run `just ci` before handing off broad tooling or behavior changes
- **GitHub Actions**: Validate workflows with `just action-lint` (uses `actionlint`)
- **YAML**: Use `just yaml-lint`
- **TOML**: Use `just toml-fmt-check` and `just toml-lint` (uses Taplo)
- **Spell check**: Use `just spell-check` (uses `typos`)
- **Project rules**: Use `just semgrep` and `just semgrep-test` (Semgrep is pinned in `pyproject.toml` and run through `uv`)

### Rust

- Prefer borrowed APIs by default: take references (`&T`, `&mut T`, `&[T]`) as arguments and return borrowed views (`&T`, `&[T]`) when possible. Only take ownership or return `Vec`/allocated data when required.
- Rust/tooling details live in [`docs/dev/rust.md`](docs/dev/rust.md).

## Common Commands

```bash
just fix              # Apply formatters/auto-fixes (mutating)
just check            # Lint/validators (non-mutating)
just ci               # Full CI simulation (checks + tests + examples)
just lint             # Grouped lint aliases (code + docs + config)
just setup            # Install/verify external dev tools
just test             # Lib + doc tests (fast)
just test-all         # All tests (lib + doc + integration + Python tooling)
just examples         # Run all examples
```

For detailed command references, coverage, and tooling notes, see [`docs/dev/rust.md`](docs/dev/rust.md).

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
- The MCMC framework is split across focused modules, all re-exported from `src/lib.rs`:
  - `src/error.rs` — `McmcError`: error type for NaN/+∞ detection in log-probabilities and proposal ratios
  - `src/observable.rs` — observation traits and buffers for measuring derived quantities while sampling
  - `src/traits.rs` — `Target<S>`, `Proposal<S>` (clone-based), `ProposalMut<S>` (in-place with rollback)
  - `src/chain.rs` — `Chain<S>`: Metropolis–Hastings chain with `step` (clone-based) and `step_mut` (in-place)
  - `src/sampler.rs` — `Sampler<S,T,P,R>`: ergonomic wrapper bundling a chain with its target, proposal, and RNG; provides `run`/`run_mut` for bulk sampling and implements `Iterator` for the clone-based path
  - `src/statistics.rs` — streaming summary statistics and binning analysis helpers
  - `src/testing.rs` — test-facing detailed-balance verification helpers
  - `prelude` module in `src/lib.rs`: convenience re-exports (`Chain`, `McmcError`, `Proposal`, `ProposalMut`, `Sampler`, `Target`)
- Rust tests are inline `#[cfg(test)]` modules in each source file.
- The `justfile` defines all dev workflows (see `just --list`).
- Examples live in `examples/` (e.g. `normal_1d.rs`, `ising_1d.rs`, `detailed_balance.rs`).

## Publishing note

- If you publish this crate to crates.io, prefer updating documentation *before* publishing a new version (doc-only changes still require a version bump on crates.io).

## Editing tools policy

See [Code Quality](#code-quality) above for the canonical editing tools policy.
