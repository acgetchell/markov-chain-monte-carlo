# Releasing markov-chain-monte-carlo

Release preparation follows the same command sequence as la-stack. Choose one stable tag, `vX.Y.Z`; commands infer the package version, prior published stable
GitHub release, and current UTC release date. Dependency and tool upgrades should land separately before the release PR. Feature and API work should already
be merged; keep the release PR focused on metadata, generated notes, retained benchmark evidence, and publication assets.

## Prerequisites

Start from an up-to-date `main` and follow [contributor setup](../CONTRIBUTING.md#development-environment-setup). Shared setup installs the declared managed
tools and locked environment; Git, Bash/sh, a native build toolchain, uv, and jq remain system prerequisites. Install and authenticate GitHub CLI
(`gh auth login`) for stable release discovery and publication. Network access is required for dependency refreshes, GitHub discovery, and uncached benchmark
builds.

### Dependency and tool refresh

Run the dependency and tool refresh as a separate maintenance change before release preparation:

```bash
just update
```

`just update` upgrades uv through its installation owner, reconciles its pin, upgrades declared Cargo tools, and runs setup. It then updates Cargo requirements
and lockfiles, resolves exact `dependency-groups.dev` pins as one compatible set, refreshes the full Python lock, and syncs dev. The Python resolver retains
project and other development constraints, including constraints on the same distribution as an exact pin. Ranged, compound, wildcard, marked, runtime, and
build requirements are not rewritten. Symlinked `uv.lock` is rejected before resolving or mutating pins. Review and validate all dependency, lockfile, and
tool-pin changes, then merge them into `main` before creating `release/$TAG`. Do not carry unreviewed dependency changes into the release branch.

## Preparation sequence

After the prerequisite changes are merged, start with a clean working tree. The commands below update `main` to the reviewed changes and create the release
branch before preparing metadata. Supply `TAG`, then copy the prepared citation date into `DATE`:

```bash
TAG=vX.Y.Z
git checkout main
git pull --ff-only
git checkout -b "release/$TAG"
just update-version "$TAG"
# Set DATE to the date-released value now recorded in CITATION.cff.
DATE=YYYY-MM-DD
just changelog-unreleased "$TAG" "$DATE"
just performance-release
# Review the shared evidence, then update tooling/performance-readme.toml selections and provenance pins.
just performance-readme --preview
just performance-readme
just ci
uv run --locked --group dev research-repo-tools toolchain run -- cargo publish --locked --allow-dirty --dry-run
```

### Release metadata

`just update-version "$TAG"` requires canonical stable `vX.Y.Z` syntax and an available `gh`. It excludes drafts and prereleases when inferring the prior
published stable release. It prepares Cargo versions and lock metadata, citation version/date, active installation examples, and non-artifact README links.
The dependency-only Python environment is not a releasable package; its placeholder version stays independent of Cargo.
All policy is declarative in `pyproject.toml`, including fixed DOI assertions, historical exclusions,
and canonical stable tags. Version-independent benchmark examples need no callback.

For an offline preview, supply the previous published tag and date explicitly:

```bash
just update-version "$TAG" --previous-release "$PREVIOUS_TAG" --date "$DATE" --dry-run
```

See the [migration record](dev/shared-maintenance-migration.md) for the v0.1.5 ownership map.

The stable Zenodo concept DOI stays `10.5281/zenodo.20033111` across releases. Keep it in `CITATION.cff`, the README badge target, and `REFERENCES.md`; do not
substitute a version DOI or add version-specific citation identifiers. The updater validates these existing DOI references rather than changing them.

The release date is the current UTC day. Rerunning with the same tag on the same UTC day leaves contents unchanged. Rerunning on another UTC day deliberately
updates the citation and any existing target changelog heading together. The updater does not upgrade dependencies, generate changelog content, measure
benchmarks, or redirect existing performance artifact links to a tag whose artifacts have not been published yet.

### Generated changelog

Set `DATE` explicitly to the ISO date in the prepared `CITATION.cff`.
`just changelog-unreleased "$TAG" "$DATE"` generates prospective notes without
creating a tag or changing release metadata. The shared package normalizes and
archives completed minor series and validates Markdown before publishing.
To intentionally move the release date, rerun `just update-version "$TAG"` and
update `DATE` to match.

Review `CHANGELOG.md` and `docs/archives/changelog/`; never hand-edit generated
content. Fix source commit messages or the shared package and regenerate.
Run `just changelog-check` to validate the complete root/archive history and
`just release-check` to verify metadata consistency. Both are included in the
normal validation gates. Release-note extraction validates the requested release
and its required links; it is not a substitute for whole-history validation.
The [pilot comparison](dev/shared-changelog-pilot.md) records the behavior verified
with the pinned shared package, including preserved dates and repeatable archives.

### Retained performance evidence and publication

`just performance-release` infers the current tag from `Cargo.toml` and measures against the appropriate prior stable release in isolated worktrees. An
unpublished current version uses the patched working tree; an already published version uses that tag against its predecessor. Explicit
`just performance-release <current-tag> <baseline-tag>` pairs are for repairs. Measurement reruns produce new observations and are not idempotent.

The command retains shared comparison/evidence JSON and CSV under `docs/performance/v1/`,
updates its `current.md`, and archives the previous shared report. Historical files under
`docs/PERFORMANCE.md` and `docs/archive/performance/` remain unchanged.
Review coverage, host/toolchain context, and workload contracts in [BENCHMARKING.md](BENCHMARKING.md).

Before publishing the README, update `tooling/performance-readme.toml` with the reviewed
pair paths, independently verified revisions/releases, SVG links, and selected workload rows.
Future-release preparation uses `tag-policy = "prepare"` and requires current shared source
and harness fingerprints. Tagged evidence uses `existing`; any existing tag must contain
the exact linked and generated bytes. The publisher checks the current Cargo version and both
report identities and rejects stale input before updating the README section and SVG together.

The checked-in historical selection cannot publish converted legacy data as fresh measurement.
Keep the old README bytes until the next measured release is prepared. Do not bypass tag checks
to replace artifacts already published under an existing tag.

Reproduce retained reports without measurement or network access:

```bash
just performance-doc --check
just performance-doc
```

An explicit shared pair can be promoted with `--payload` and `--manifest` arguments.
These transformations are content-idempotent. Commit the report, evidence, archive index,
reviewed publication configuration, SVG, and README together before creating the new tag.

For development use `just performance-local`; same-version working-tree comparisons remain
supported but cannot be promoted. `just performance-github-assets` compares authenticated
release assets without local measurements.

### Validation and release PR

`just ci` covers repository checks, Rust and Python tests, doctests, notebook execution, example validation, docs, and benchmark compilation. If formatting is
needed, run `just fix`, review the resulting diff, and rerun CI. `just release-check` is included in CI and verifies synchronized package versions, active
references, concept DOI, and citation/changelog dates. Historical documentation and performance artifact links keep their own release identity.

The explicit Cargo dry run verifies packaging without publishing. `just publish-check` additionally validates crates.io metadata and runs the same dry run.
Review the full diff, commit and push the release branch, and open a PR. Include the CI and packaging validation results. Do not tag or publish before the
release PR merges.

## After the PR merges

Sync `main`, create an annotated tag from the generated release notes, and verify that it targets the release merge before pushing:

```bash
git checkout main
git pull --ff-only
just check
just tag "$TAG"
git --no-pager show --no-patch "$TAG"
test "$(git rev-parse "$TAG^{commit}")" = "$(git rev-parse HEAD)"
git push origin "$TAG"
gh release create "$TAG" --title "$TAG" --notes-from-tag --draft --verify-tag
cargo publish --locked
gh workflow run release-benchmarks.yml -f release_tag="$TAG"
```

Use a normal tag push even after `just tag-force`; Git rejects a conflicting remote tag. If the push is rejected, stop. Intentional retagging requires a
separately verified recovery procedure covering the existing remote tag object and target commit, the intended replacement, and the GitHub Release and
crates.io publication state.

Create the draft GitHub Release before publishing to crates.io. Keep the draft unpublished if crate publication fails. After successful crate publication,
manually dispatch `Release Benchmarks`. The workflow verifies that the tag names a mutable draft, measures the `stepping` suite, attaches
`markov-chain-monte-carlo-$TAG-criterion-baseline.tar.gz`, and publishes the draft after the upload succeeds. Do not publish the draft separately while the
workflow is running. Verify the workflow succeeds and the immutable release contains the durable attachment:

```bash
gh release view "$TAG" --json assets --jq '.assets[].name'
```

If upload succeeds but publication fails, rerun the failed publication job to reuse the saved Actions artifact. The workflow downloads any existing asset
with the same name and requires an exact byte match before publishing. A different asset stops publication for investigation; it is never overwritten.

The 30-day Actions artifact is diagnostic only. The shared release archive contains selected Criterion
estimates and provenance, not the original raw Criterion time series. Preserve raw samples separately
when needed for reanalysis. Historical releases are not backfilled. The first release containing this
workflow creates the first shared baseline; its successor enables the first complete asset pair.
Verify that live pair before treating release-asset adoption as complete.

After confirming publication and the durable baseline, delete the merged release branch locally and remotely:

```bash
git branch -d "release/$TAG"
git push origin --delete "release/$TAG"
```
