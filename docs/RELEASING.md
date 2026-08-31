# Releasing markov-chain-monte-carlo

Release preparation follows the same command sequence as la-stack. Choose one stable tag, `vX.Y.Z`; commands infer the package version, prior published stable
GitHub release, and current UTC release date. Dependency and tool upgrades should land separately before the release PR. Feature and API work should already
be merged; keep the release PR focused on metadata, generated notes, retained benchmark evidence, and publication assets.

## Prerequisites

Start from an up-to-date `main` and install the prerequisites described in [CONTRIBUTING.md](../CONTRIBUTING.md): Git, Bash, `rustup`/Cargo and a native build
toolchain, the pinned `just` and `uv`, and `jq`. Run `just setup` to install managed tools, Rust components, and the locked Python environment. Setup checks
`uv` and `jq` before managed installations; it does not install these system prerequisites. Install and authenticate GitHub CLI (`gh auth login`) for stable
release discovery and publication. Network access is required for dependency refreshes, GitHub discovery, and uncached benchmark builds.

## Preparation sequence

The commands below are the canonical order. Run `just update` first and review, validate, and merge its dependency/tool changes separately. Then choose the
release tag once and create the release branch before preparing metadata. The only manually supplied release value is `TAG`:

```bash
just update
TAG=vX.Y.Z
git checkout main
git pull --ff-only
git checkout -b "release/$TAG"
just update-version "$TAG"
just changelog-unreleased "$TAG"
just performance-release
just performance-readme
just ci
cargo publish --locked --allow-dirty --dry-run
```

### Dependency and tool refresh

`just update` updates Cargo requirements and lockfiles, resolves exact `dependency-groups.dev` pins as one compatible set, upgrades the managed Cargo tools,
reconciles their pins and the active uv pin, and syncs Python. The Python resolver retains project and other development constraints, including constraints
on the same distribution as an exact pin. Ranged, compound, wildcard, marked, runtime, and build requirements are not rewritten. Symlinked `uv.lock` is
rejected before resolving or mutating pins. Review the dependency diff and validate it before opening the release PR.

### Release metadata

`just update-version "$TAG"` requires canonical stable `vX.Y.Z` syntax and an available `gh`. It excludes drafts and prereleases when inferring the prior
published stable release. It prepares the root versions in `Cargo.toml`, `Cargo.lock`, `pyproject.toml`, and `uv.lock`, citation version/date, active
installation and release-command examples, and non-artifact README links. Dependency lock entries are preserved. It validates the complete proposed metadata
before replacing files and restores earlier contents if publication fails.

The stable Zenodo concept DOI stays `10.5281/zenodo.20033111` across releases. Keep it in `CITATION.cff`, the README badge target, and `REFERENCES.md`; do not
substitute a version DOI or add version-specific citation identifiers. The updater validates these existing DOI references rather than changing them.

The release date is the current UTC day. Rerunning with the same tag on the same UTC day leaves contents unchanged. Rerunning on another UTC day deliberately
updates the citation and any existing target changelog heading together. The updater does not upgrade dependencies, generate changelog content, measure
benchmarks, or redirect existing performance artifact links to a tag whose artifacts have not been published yet.

### Generated changelog

`just changelog-unreleased "$TAG"` generates notes from local Git history without creating a tag, applies Markdown hygiene, and synchronizes the new heading
with the prepared citation date. This date synchronization is offline, so crossing UTC midnight during generation does not split the citation and changelog
dates. To intentionally move the release date, rerun `just update-version "$TAG"`.

Review `CHANGELOG.md`; never hand-edit generated content. Fix source commit messages, `cliff.toml`, or the post-processing helper and regenerate. Squash commit
bodies supply unreleased details; annotated tag notes supply older release details. Put release-note-worthy bullets in squash commit bodies because GitHub
PR descriptions and old manual edits are not recoverable from local history. Keep patch-release notes focused on the changes intended for that patch.

### Retained performance evidence and publication

`just performance-release` infers the current tag from `Cargo.toml` and measures against the appropriate prior stable release in isolated worktrees. An
unpublished current version uses the patched working tree; an already published version uses that tag against its predecessor. Explicit
`just performance-release <current-tag> <baseline-tag>` pairs are for repairs. Measurement reruns produce new observations and are not idempotent.

The command saves and validates `target/bench-reports/release-performance.{csv,provenance.json}`, promotes `docs/PERFORMANCE.md`, and retains the compact pair
under `docs/archive/performance/`. Review comparable-row coverage, host/toolchain details, and harness hashes. If harness hashes differ, verify the shared
workloads against the lifecycle contract in [BENCHMARKING.md](BENCHMARKING.md). Rename changed workloads instead of comparing unlike operations.

`just performance-readme` consumes this validated retained pair without GitHub discovery or new measurements. It replaces only the marked README performance
section and the pair's SVG under `docs/archive/performance/`, with tag-pinned report, CSV, JSON, and image links. Tracked paths are checked for containment
individually, independent of link labels. Missing evidence names `just performance-release` as the recovery command before changing the README; a digest
mismatch remains a separate integrity failure. Local release tags must be current: read-only Git checks require an existing current tag to contain the exact
report, CSV, JSON, and rendered SVG before publication. Repaired or same-version evidence that differs cannot be published under that tag; keep it local or
prepare a new release. New-release working-tree comparisons may target the future release tag; commit all linked artifacts before creating it. Identical
tagged artifacts can be republished without changes, and same-version comparisons retain their source label.

To reproduce the curated report without measuring again:

```bash
just performance-doc
just performance-readme
```

`performance-doc` reads the current report's retained CSV/JSON pair by default. An explicit CSV path can repair another saved pair. It replaces the former
`performance-rerender` command; there is no compatibility alias. These retained-data transformations are content-idempotent with unchanged evidence and the
locked rendering environment. Native Criterion archives attached to GitHub Releases remain richer raw evidence for independent reanalysis.

Commit the report, compact evidence, archive index, SVG, and README together. For development without modifying committed publication artifacts, use
`just performance-local`; same-version current-tree-versus-published-tag comparisons remain supported. `just performance-github-assets` compares durable
release assets without local Cargo measurements. See [BENCHMARKING.md](BENCHMARKING.md) for measurement limits.

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
just tag "$TAG"
git --no-pager show --no-patch "$TAG"
test "$(git rev-parse "$TAG^{commit}")" = "$(git rev-parse HEAD)"
git push origin "$TAG"
cargo publish --locked
gh release create "$TAG" --title "$TAG" --notes-from-tag
```

Publishing to crates.io precedes creating the GitHub release. The GitHub release triggers `Release Benchmarks`, which measures the `stepping` suite and
attaches `markov-chain-monte-carlo-$TAG-criterion-baseline.tar.gz`. Verify the workflow succeeds and the durable release attachment exists:

```bash
gh release view "$TAG" --json assets --jq '.assets[].name'
```

The 30-day Actions artifact is diagnostic only. Historical releases are not backfilled: `v0.4.1` and earlier releases have no Criterion baseline attachment.
`v0.4.2` establishes the initial asset. After publishing `v0.4.3`, run `just performance-github-assets` and verify the `v0.4.3`-against-`v0.4.2` pair before
treating release-benchmark adoption as complete.

After confirming publication and the durable baseline, delete the merged release branch locally and remotely:

```bash
git branch -d "release/$TAG"
git push origin --delete "release/$TAG"
```
