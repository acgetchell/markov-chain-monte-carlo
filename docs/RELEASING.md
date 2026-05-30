# Releasing markov-chain-monte-carlo

This guide documents the release flow for this crate. Changelog generation and annotated release tags are automated locally with `git-cliff` plus the Python
helpers in `scripts/`.

Applies to versions `vX.Y.Z`. Prefer updating documentation before publishing to crates.io.

## Conventions

Use a tag with a leading `v` and a Cargo version without it:

```bash
TAG=vX.Y.Z
VERSION=${TAG#v}
```

Start from an up-to-date `main`:

```bash
git checkout main
git pull --ff-only
git --no-pager status --short
```

## Release PR

The release PR should primarily contain version, changelog, and documentation updates. Major behavior or API changes should already be merged before release
prep begins.

Small release-critical fixes are acceptable if they are discovered during validation, but keep the PR focused.

### 1. Create the release branch

```bash
git checkout -b release/$TAG
```

### 2. Bump package and citation versions

Preferred, if `cargo-edit` is installed:

```bash
cargo set-version "$VERSION"
```

Alternatively, edit `Cargo.toml` manually and update the package `version`.

Also update release metadata that duplicates the package version:

- `CITATION.cff` `version`
- `pyproject.toml` project `version`

Then refresh lockfiles/build metadata:

```bash
cargo check
uv sync
```

`uv sync` installs the default `dev` and `notebook` dependency groups (configured in `[tool.uv] default-groups`), refreshing `uv.lock` so the notebook
execution dependencies exercised by `just ci` are provisioned during release validation.

### 3. Generate the changelog

Regenerate `CHANGELOG.md` with the release section for commits since the previous tag:

```bash
just changelog-unreleased "$TAG"
```

Review `CHANGELOG.md` for accuracy. Do not hand-edit generated changelog content; if a release note is wrong, fix the source commit message, `cliff.toml`, or
the changelog post-processing helper, then regenerate.

The generator is intentionally local/offline. It uses squash commit bodies for unreleased entries and annotated tag notes for older tagged releases. Put
release-note-worthy bullets in the squash commit body before merging feature PRs; details that live only in GitHub PR descriptions or old hand edits are not
recoverable from local git history.

Pass the tag form, including the leading `v`, so compare links point at the eventual release tag.

For a patch release, keep the notes focused on fixes, dependency updates, tooling, and documentation. Do not pull planned feature-roadmap issues into a patch
release unless they are small and explicitly intended for that patch.

Review version references:

```bash
git grep -nE '\bv?[0-9]+\.[0-9]+\.[0-9]+\b' -- README.md docs/ Cargo.toml Cargo.lock CITATION.cff pyproject.toml uv.lock CHANGELOG.md
```

### 4. Validate locally

Run the normal release validation gates:

```bash
just fix
just ci
just publish-check
```

`just fix` reruns formatters and auto-fixes. `just ci` covers formatting, Clippy, Python tooling checks, benchmark harness compilation, docs, tests, example
output validation, YAML, TOML, Markdown, spelling, GitHub Actions, and Semgrep checks. `just publish-check` validates crates.io metadata and runs
`cargo publish --locked --allow-dirty --dry-run`.

### 5. Commit, push, and open the PR

Review the diff carefully:

```bash
git --no-pager diff
git --no-pager status --short
```

Suggested PR metadata:

- Title: `chore(release): release $TAG`
- Summary: version bump, changelog update, and release documentation updates
- Validation: include the `just ci` and `just publish-check` results

## After the PR Merges

Sync `main` to the merge commit:

```bash
git checkout main
git pull --ff-only
```

Create and push an annotated tag from the release changelog section:

```bash
just tag "$TAG"
git push origin "$TAG"
```

Create the GitHub release:

```bash
gh release create "$TAG" --title "$TAG" --notes-from-tag
```

Publish to crates.io:

```bash
cargo publish --locked
```

## Notes

- Do not tag or publish until the release PR has merged.
- Keep release PRs small: version, changelog, docs, and release-critical fixes.
- `just changelog` regenerates the full changelog from local git history; do not hand-edit generated changelog content.
- `just changelog-unreleased <tag>` regenerates `CHANGELOG.md` as if `<tag>` were the release tag, without creating the tag.
- `cliff.toml` skips release-prep commits, filters CI/action dependency churn, and keeps Rust-library dependency bumps concise.
- `just tag <tag>` creates the annotated release tag from the matching `CHANGELOG.md` section.
- Run `just ci` before handing off a release PR.
- Run `just publish-check` before publishing.
