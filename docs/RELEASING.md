# Releasing markov-chain-monte-carlo

Use this procedure for each stable release. Run the command blocks in order from
the repository root, keeping the release variables in the same shell. Stop and
resolve any failed command before continuing. Git mutations and publication are
maintainer actions; assistants follow the restrictions in [AGENTS.md](../AGENTS.md).

## Release workflow

1. Prepare a release PR containing synchronized metadata, generated notes, and any
   reviewed performance publication files. Validate it and merge it into `main`.
2. Tag the reviewed merge and create a draft GitHub Release from the tag annotation.
3. Publish the crate to crates.io.
4. Dispatch **Release Benchmarks**. It attaches the benchmark baseline and publishes
   the draft only after the attachment succeeds.
5. Verify the published crate, GitHub Release, and retained baseline, then remove
   the merged release branch.

## Prepare the environment

Complete [contributor setup](../CONTRIBUTING.md#development-environment-setup).
GitHub CLI must be installed and authenticated, and your Cargo credentials must
permit publishing this crate to crates.io.

Set the release inputs once. This example prepares `0.5.0` after `v0.4.2`:

```bash
VERSION=0.5.0
TAG="v$VERSION"
PREVIOUS_TAG=v0.4.2
RELEASE_DATE="$(date -u +%F)"
```

For later releases, replace `VERSION` and `PREVIOUS_TAG` with the target version
and actual previous published stable tag. The `date` command fills `RELEASE_DATE`
with today's UTC date; you can instead assign an intended ISO date explicitly.
Both metadata preparation and changelog generation receive this same value.

Commit and merge substantive implementation changes before starting release
preparation. Start with a clean working tree, then verify authentication and the
remote, synchronize `main`, and sync the locked development environment:

```bash
gh auth status
git --no-pager remote -v
git --no-pager status --short
git switch main
git pull --ff-only
just sync
```

If dependency or tool upgrades are needed, run `just update`, review and validate
its changes, and land them separately before creating the release branch.
Release metadata preparation does not upgrade dependencies.

## Prepare the release PR

### Create the release branch

```bash
git switch -c "release/$TAG"
```

Keep this PR focused on release metadata, generated notes, release documentation,
and applicable retained performance evidence.

### Update metadata and generate the changelog

After substantive changes are committed, run:

```bash
just release-update "$VERSION" "$PREVIOUS_TAG" "$RELEASE_DATE"
just changelog-preview --tag "$TAG" --date "$RELEASE_DATE"
just changelog-release "$TAG" "$RELEASE_DATE"
```

`release-update` synchronizes Cargo versions and lock metadata, citation
version/date, installation examples, and configured active links. Supplying the
previous tag avoids GitHub release discovery. The dependency-only Python
environment keeps its independent placeholder version. The stable Zenodo concept
DOI remains `10.5281/zenodo.20033111`; the updater validates its references.

The recipe accepts a version such as `0.5.0` or a tag such as `v0.5.0` and passes
a canonical stable tag to the shared updater. Additional shared CLI options follow
the three release inputs; for example, preview metadata changes without writing:

```bash
just release-update "$VERSION" "$PREVIOUS_TAG" "$RELEASE_DATE" --dry-run
```

`changelog-preview` prints prospective notes; `changelog-release` writes
`CHANGELOG.md` and archives completed minor series under `docs/archives/changelog/`.
Neither command creates a tag.

Generated notes use committed history. Staged and unstaged implementation changes
are absent from the notes. If a substantive fix is needed during preparation,
commit it, regenerate the notes, and repeat the affected validation before merge.
Never hand-edit generated changelog content; correct source commit messages or
the shared generator instead.

The repository uses a declared release date. Keep `RELEASE_DATE` unchanged when
resuming preparation. To change it intentionally, assign the new date and rerun
all three commands above before review. Updating only the changelog date would
conflict with the prepared metadata.

### Retained performance evidence and publication

For **v0.5.0**, skip the commands in this subsection and continue to validation.
This release establishes the first baseline after the performance reset; the
tagged workflow records it after crate publication. A comparison needs two
releases from the new series. Do not backfill retired releases.

For subsequent releases, once the latest published release belongs to the new
series, measure the prepared working tree against that baseline:

```bash
just performance-release
```

Review the retained evidence and report under `docs/performance/v1/`. Create or
update `tooling/performance-readme.toml` from that real pair using the
[publication instructions](BENCHMARKING.md#release-preparation-and-readme-publication),
including the `prepare` tag policy and source/harness fingerprints. Then publish
the reviewed selection and check that the report reproduces:

```bash
just performance-readme --preview
just performance-readme
just performance-doc --check
```

Include the report, evidence, archives, publication configuration, SVG, and README
together in the release PR. Measurements create new observations on each run;
report reproduction does not measure again. Other comparison modes and evidence
contracts belong in [BENCHMARKING.md](BENCHMARKING.md).

### Validate the release artifacts

```bash
just release-check "$TAG"
just ci
just publish-check
just security
```

`just release-check "$TAG"` requires a canonical stable tag matching the Cargo
package version, synchronized release metadata, and dated, nonempty generated
notes with valid references. It creates no tag and publishes nothing. The
no-argument form infers the tag from Cargo and remains part of CI.

`just ci` includes changelog/history checks, release metadata checks, Rust and
Python tests, doctests, notebook execution, example validation, docs, and benchmark
compilation. `just publish-check` validates crates.io metadata and runs Cargo's
packaging dry run without publishing.

`just security` runs `just audit` for vulnerabilities in `uv.lock` and both Cargo
lockfiles, then `just security-secrets` for secrets in Git history and current
files. It requires network access and a complete Git checkout, and remains
separate from `just ci`. Findings or scanner errors fail the check; inspect the
reports under `target/security/`. There is no need to run `just audit` separately
when running `just security`.

If formatting changes are needed, run `just fix`, inspect its changes, and rerun
the affected validation. Reuse successful validation while its inputs remain
unchanged.

### Review, commit, and submit the release PR

Inspect the changed metadata, notes, archives, documentation, and any performance
publication files before staging:

```bash
git --no-pager status --short
git --no-pager diff
```

Stage only reviewed paths with `git add`, including any new generated archive.
Inspect the staged diff, then commit and submit the release PR:

```bash
git --no-pager diff --cached
git commit -m "chore(release): release $TAG"
git push -u origin "release/$TAG"
gh pr create --base main --head "release/$TAG" --title "chore(release): release $TAG"
```

Include validation results in the PR description. Resolve review findings and
require passing CI before merging. Do not tag or publish before the PR merges.

## Publish after the PR merges

### Synchronize the reviewed release

Use a clean checkout and keep the release variables from preparation. In a new
shell, restore the same version, previous tag, and declared release date.

```bash
git switch main
git pull --ff-only
git --no-pager status --short
just sync
just check
just release-check "$TAG"
just release-notes "$TAG"
```

Confirm the checkout contains the release PR and the extracted notes are the
reviewed notes. The configured `declared` date policy preserves the prepared date;
crossing midnight does not require changing it.

### Create and push the annotated tag

```bash
just tag "$TAG"
git --no-pager show --no-patch "$TAG"
test "$(git --no-pager rev-parse "$TAG^{commit}")" = "$(git --no-pager rev-parse HEAD)"
git push origin "$TAG"
```

Verify the tag identifies the reviewed release commit. If it already exists or
the push is rejected, stop and inspect the local tag, remote tag, GitHub Release,
and crates.io state. Do not force an existing published tag to a new commit.

### Create the draft GitHub Release

```bash
gh release create "$TAG" --title "$TAG" --notes-from-tag --draft --verify-tag
gh release view "$TAG" --json tagName,isDraft,isPrerelease,body,assets
```

Confirm the tag and notes match the reviewed release, `isDraft` is true, and
`isPrerelease` is false. Keep this release as a draft until the benchmark workflow
publishes it.

### Publish the crate

```bash
uv run --locked --group dev research-repo-tools toolchain run -- cargo publish --locked
```

Wait for successful publication before proceeding. If it fails, leave the GitHub
Release as a draft and follow the recovery instructions below.

### Attach the baseline and publish the GitHub Release

```bash
gh workflow run release-benchmarks.yml --ref main -f release_tag="$TAG"
gh run list --workflow release-benchmarks.yml --limit 5
gh run watch --exit-status
```

Select the run titled `Release benchmark $TAG` when `gh run watch` prompts.
Wait for it to succeed. The workflow validates the draft, measures the tagged
`stepping` suite, uploads `markov-chain-monte-carlo-$TAG-criterion-baseline.tar.gz`,
and publishes the draft. Do not publish the draft manually while this runs.

### Verify publication

```bash
gh release view "$TAG" --json tagName,isDraft,isPrerelease,body,assets
uv run --locked --group dev research-repo-tools toolchain run -- cargo info "markov-chain-monte-carlo@$VERSION" --registry crates-io
```

Confirm the GitHub Release has the expected tag and notes, is no longer a draft,
and contains the named baseline archive. Confirm crates.io reports the exact
published version. A published crate alone does not establish that the GitHub
Release or baseline upload succeeded.

The release attachment is the durable baseline; the 30-day Actions artifact is
temporary. It contains selected Criterion estimates and provenance, not the raw
Criterion time series. Preserve raw samples separately when needed for reanalysis.

## Recover a failed attempt

- If preparation or validation fails, fix the source, regenerate notes after any
  substantive implementation commit, and repeat the affected checks before merge.
- If crate publication fails, inspect crates.io and the command output before
  retrying. If the version was accepted, continue with benchmark publication.
  Published crate contents cannot be replaced; a content fix needs a new version.
- If benchmark measurement fails, fix the cause before retrying. Keep the GitHub
  Release in draft until a complete baseline can be attached.
- If attachment succeeds but GitHub publication fails, rerun the failed
  publication job to reuse the saved Actions artifact. An existing attachment
  must match byte for byte; different bytes stop publication and are never
  overwritten. See [release asset contracts](BENCHMARKING.md#github-release-assets).

## Finish the release

Record the published version, successful benchmark run, and verification results
in the release PR or tracking issue. After confirming both publications and the
durable baseline, remove the merged release branch:

```bash
git branch -d "release/$TAG"
git push origin --delete "release/$TAG"
```
