# Releasing markov-chain-monte-carlo

This guide documents the manual release flow for this crate. It is intentionally
lighter than the Delaunay release process: this repository does not currently
generate changelogs, archive changelog sections, run benchmarks, or produce
performance baselines as part of release automation.

Applies to versions `vX.Y.Z`. Prefer updating documentation before publishing to
crates.io.

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

The release PR should primarily contain version, changelog, and documentation
updates. Major behavior or API changes should already be merged before release
prep begins.

Small release-critical fixes are acceptable if they are discovered during
validation, but keep the PR focused.

### 1. Create the release branch

```bash
git checkout -b release/$TAG
```

### 2. Bump the Cargo version

Preferred, if `cargo-edit` is installed:

```bash
cargo set-version "$VERSION"
```

Alternatively, edit `Cargo.toml` manually and update the package `version`.

Then refresh the lockfile/build metadata:

```bash
cargo check
```

### 3. Update documentation and changelog

Update `CHANGELOG.md` by hand:

- move relevant `[Unreleased]` entries under `## [X.Y.Z] - YYYY-MM-DD`
- add a fresh empty `## [Unreleased]` section
- update comparison links at the bottom

For a patch release, keep the notes focused on fixes, dependency updates,
tooling, and documentation. Do not pull planned feature-roadmap issues into a
patch release unless they are small and explicitly intended for that patch.

Review version references:

```bash
rg -n '\bv?[0-9]+\.[0-9]+\.[0-9]+\b' README.md docs/ Cargo.toml CHANGELOG.md
```

### 4. Validate locally

Run the normal release validation gates:

```bash
just fix
just ci
just publish-check
```

`just ci` covers formatting, Clippy, benchmark harness compilation, docs,
tests, examples, example output validation, YAML, TOML, spelling, GitHub
Actions, and Semgrep checks.
`just publish-check` validates crates.io metadata and runs
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

Create and push an annotated tag:

```bash
git tag -a "$TAG" -m "markov-chain-monte-carlo $TAG"
git push origin "$TAG"
```

Create the GitHub release:

```bash
gh release create "$TAG" --title "$TAG" --notes-file CHANGELOG.md
```

For a small release, it is also fine to paste the specific changelog section into
the GitHub release notes instead of using the full changelog file.

Publish to crates.io:

```bash
cargo publish --locked
```

## Notes

- Do not tag or publish until the release PR has merged.
- Keep release PRs small: version, changelog, docs, and release-critical fixes.
- `CHANGELOG.md` is currently maintained manually.
- If releases become frequent enough to make manual changelog updates noisy,
  add changelog automation as a separate tooling issue.
- Run `just ci` before handing off a release PR.
- Run `just publish-check` before publishing.
