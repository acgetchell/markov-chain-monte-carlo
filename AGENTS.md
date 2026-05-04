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

This is a single Rust *library crate* (no `src/main.rs`). The crate root is `src/lib.rs`.

Use [`docs/code_organization.md`](docs/code_organization.md) as the detailed checkout tree and file/module map when deciding where new functions, traits, examples, tests, or docs belong. Keep this section high-level so `AGENTS.md` remains focused on operating rules rather than duplicating the architecture guide.

When adding, removing, renaming, or moving tracked files or directories, update `docs/code_organization.md` in the same change so its full checkout tree and ownership guidance stay current.

At a glance:

- `src/` — public library modules and crate-level `//!` documentation.
- `examples/` — complete runnable workflows.
- `tests/` and `benches/` — integration validation and Criterion benchmarks.
- `docs/` — topic guides.
- `scripts/` — changelog and release helpers.

## Documentation map

The repository's docs have overlapping topics but distinct roles. When in doubt, consult the file whose role best matches the task:

- **`AGENTS.md`** (this file) — canonical rules for AI assistants and autonomous tooling: git/edit/validation policy, commit-message format, code-quality rules, documentation-generation rules. Authoritative when rules conflict.
- **`CONTRIBUTING.md`** — human contributor workflow: prerequisites, `just setup` external tools, high-level repository layout, test categories, code style, performance/benchmarking, PR checklist, release process. Mirrors the human-facing parts of this file.
- **`docs/code_organization.md`** — full tracked checkout tree, detailed file/module map, and "where does new code go?" guidance for `src/*.rs`, examples, tests, benches, docs, and scripts. Consult when adding a new function/type/trait and unsure which file owns it, and update it whenever tracked files move, appear, or disappear. Does **not** repeat contributor workflow or tooling procedures.
- **`README.md`** — public GitHub landing page. It should lead with short hand-written orientation (badges, pitch, concise table of contents for long READMEs, install, quick start, API-choice guide, Cargo features), then include the cargo-rdme generated API guide from `src/lib.rs` `//!` (see [Documentation generation](#documentation-generation)), then short links to examples, docs, ecosystem, contributing, citation, references, and agent guidance.
- **`docs/scientific_basis.md`** — Metropolis–Hastings contract and scope discussion (extends the `# Scientific basis and scope` section in `lib.rs //!`).
- **`docs/proposal_validation.md`** — proposal-author testing patterns and `verify_detailed_balance*` usage.
- **`docs/roadmap.md`** — planned feature work.
- **`docs/dev/rust.md`** — Rust toolchain notes and tooling deep-dive.
- **`docs/RELEASING.md`** — release procedure (also referenced from `## Publishing note` below).
- **`REFERENCES.md`** — academic references and AI-assisted-development tool citations.
- **`CHANGELOG.md`** — generated by `git-cliff` from commit history; **never hand-edit**. To change changelog content, fix the upstream commit message (see [Commit Message Generation](#commit-message-generation)) or `cliff.toml`.

## Documentation generation

The `README.md` API section is **generated by `cargo-rdme`** from the crate-level `//!` doc comment in `src/lib.rs`. The generated region is bracketed by:

```text
<!-- dprint-ignore-start -->
<!-- cargo-rdme start -->
...generated content (do not hand-edit)...
<!-- cargo-rdme end -->
<!-- dprint-ignore-end -->
```

Rules:

- **NEVER** hand-edit content between `<!-- cargo-rdme start -->` and `<!-- cargo-rdme end -->`. Edits there will be overwritten the next time `cargo rdme` runs.
- To change the generated region, edit `src/lib.rs` `//!` and then regenerate with `just docs-readme` (or `cargo rdme`).
- `just check` and CI run `cargo rdme --check`, which fails if the marker region drifts from `//!`.
- The `<!-- dprint-ignore-* -->` markers exist because dprint's markdown formatter (`textWrap = "never"`) would otherwise rewrap the cargo-rdme region and re-introduce drift.
- The README should remain a useful GitHub landing page, not just a docs.rs mirror. Keep hand-written, short, scannable sections **before** the cargo-rdme block for the pitch, pre-release/status note, concise table of contents when the README is long, installation, MSRV, Cargo features, a minimal quick start, and a "which API should I choose?" guide. Treat the marker region as the detailed API guide.
- For long READMEs, maintain a hand-written `## Contents` section before `## 🚀 Quick start`. Include the major landing-page sections and the main generated API-guide subsections, but do not enumerate every generated paragraph. When changing README headings or `src/lib.rs //!` headings that render into the API guide, update this table of contents in the same change.
- Content **outside** the markers (landing sections before the generated block; Examples directory, Documentation, Ecosystem, Contributing, Citation, References, AI Agents, License after it) is hand-written and may be edited normally. Keep it concise and avoid duplicating long-form details that already live in `//!`.

The `lib.rs //!` block is the canonical long-form crate documentation. It should still lead with short, scannable sections (pitch → use-cases → `# Features`) before deeper contract content (`# Scientific basis and scope`, `# Numerical semantics`, validation hierarchy, worked examples).

## Publishing note

- If you publish this crate to crates.io, prefer updating documentation *before* publishing a new version (doc-only changes still require a version bump on crates.io).
- Before tagging, run `just docs-readme` so the README's marker region is in sync with the latest `lib.rs` `//!`, then re-run `just check` to confirm `cargo rdme --check` passes.

## Editing tools policy

See [Code Quality](#code-quality) above for the canonical editing tools policy.
