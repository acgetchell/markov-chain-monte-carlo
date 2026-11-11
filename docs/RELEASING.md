# Releasing markov-chain-monte-carlo

Prepare each `vX.Y.Z` release in a dedicated PR. After that PR is merged,
create and push the annotated tag. `Prepare GitHub release` verifies the source
and automatically creates its draft. Start `Release Benchmarks`
to attach the durable baseline, then review and publish the draft with
`just release-publish`. The published release starts `Publish to crates.io`;
approve its `crates-io` deployment to permit the Cargo upload.

The local benchmarks used to prepare the release PR do not require a draft.
The draft is needed later as the upload destination for the benchmark archive
produced on GitHub Actions.

The changelog is generated for the target tag before the tag exists, so the
release process does not require a temporary local tag.

Release recipes are content-idempotent where their inputs permit it. Repeating
`update-version` on the same UTC date or repeating `performance-readme` for the
same retained bundle produces no file changes. Rerunning `update-version` on a
later UTC date intentionally advances the declared release date. Crossing
midnight alone requires no update: the `declared` date policy allows preparation
and tagging on different UTC days when citation and changelog dates agree.
Repeated `performance-release` runs can produce different measurements.
With unchanged history and configuration, repeating `changelog-release`
reuses the declared citation date and preserves the generated contents,
including when a rerun happens after UTC midnight.

The command names, arguments and operational order follow Delaunay's release
workflow. Published `research-repo-tools==0.2.8` implements common metadata,
changelog, tag, release-event, registry and asset operations. MCMC supplies its
own scientific validation, feature selections and benchmark evidence policy.
Main CI retains signed source-validation evidence. Byte-identical Cargo archive promotion remains deferred; see the
[artifact promotion assessment](dev/shared-maintenance-migration.md#artifact-promotion-assessment-202).

## Contents

- [Reusable CI evidence](#reusable-ci-evidence)
- [Prepare the environment](#prepare-the-environment)
- [Step 1: Prepare the release PR](#step-1-prepare-the-release-pr)
- [Step 2: Publish after the PR is merged](#step-2-publish-after-the-pr-is-merged)
  - [Verify the release commit before tagging](#2-verify-the-release-commit)
  - [Inspect the prepared draft GitHub release](#5-inspect-the-prepared-draft-github-release)
- [Configure or verify Trusted Publishing](#configure-trusted-publishing-before-the-next-release)

## Reusable CI evidence

The installed shared package and `tooling/release-evidence.toml` bind the exact
main-push source to successful repository tooling, all three native CI jobs,
clean default/all-feature Cargo package verification and a separate signing
job. `just release-evidence` verifies the original run, attempt, jobs, artifact
digest and signature before a tag exists. PRs, manual dispatches and partial
reruns cannot substitute for that main-push attempt.

The publication workflow freezes the selected identity before environment
approval and rechecks it afterward. Release metadata, exact-commit SARIF gates,
benchmark assets, registry absence and maintainer approval remain required.
This proves signed source validation; Cargo packages are verified again from
that source rather than promoted as byte-identical producer archives.

The release commands share Delaunay's operator-facing roles:

| Recipe | Purpose |
| --- | --- |
| `changelog-release TAG [DATE]` | Generate prospective notes using the prepared citation date |
| `changelog-preview [--tag TAG] [--date DATE]` | Preview the same generation without publishing files |
| `performance-prepare` | Infer release identities and capture the measurement plan without timing |
| `performance-release` | Measure, retain and promote the release comparison |
| `performance-doc` | Reproduce the retained report without measuring |
| `performance-readme [--preview]` | Publish or preview the reviewed README selection |
| `release-package` | Verify clean default/all-feature Cargo packages without uploading |
| `release-benchmarks TAG` | Dispatch the benchmark workflow at that exact tag and return |
| `release-evidence` | Verify retained signed-source evidence without building or publishing |
| `release-publish TAG` | Check notes and evidence, then publish the reviewed GitHub draft |
| `release-verify TAG` | Verify GitHub assets and the exact crates.io version |

The target supplied to `update-version` becomes the package's prospective
release identity. Performance preparation infers its published predecessor;
reporting and README publication reuse the retained identities. The tag push
provides the same identity to automatic draft preparation. No manual baseline
labels, commit hashes, notes files or draft creation are needed in this flow.

## Prepare the environment

Set the target tag once:

```bash
TAG=vX.Y.Z
```

Verify GitHub authentication and the repository remotes, then synchronize
`main`:

```bash
gh auth status
git --no-pager remote -v
git switch main
git pull --ff-only
```

Install or verify the declared development tools before running maintenance
recipes. Shared setup installs managed tools and explicitly synchronizes `dev`;
see [contributor setup](../CONTRIBUTING.md#development-environment-setup):

```bash
just setup
```

Refresh Cargo dependency requirements, Python development-tool pins, lockfiles
and repository-owned tool pins before creating the release branch:

```bash
just update
```

Review any tracked changes and land them separately before continuing with the
release PR, then synchronize `main` again. Dependency and tool upgrades remain
independently reviewable; `just update-version` deliberately does not run
`just update`. Assistants follow the Git and publication approval rules in
[AGENTS.md](../AGENTS.md).

`update-version` and release performance recipes use published, non-draft,
non-prerelease GitHub Releases as authoritative stable release history.

## Step 1: Prepare the release PR

Keep the release PR focused on version metadata, the generated changelog,
benchmark artifacts, and release documentation. Major code changes should
already be on `main`.

### 1. Create the release branch

```bash
git switch -c "release/$TAG"
```

### 2. Update release metadata

```bash
just update-version "$TAG"
```

The shared updater requires a stable target no older than the latest published
stable release and infers its predecessor. It synchronizes the root Cargo
manifest and both Cargo lockfiles, citation version/date, and installation
metadata. The dependency-only Python environment keeps its independent version.
Active guide links stay on `main`, API links on docs.rs `latest`, and image and
historical evidence links retain their provenance. The Zenodo concept DOI remains
`10.5281/zenodo.20033111`.

The recipe accepts optional shared CLI arguments, including `--previous-release`,
`--date`, `--offline` and `--dry-run`. It does not upgrade dependencies, generate
the changelog or run benchmarks. The shared updater validates prepared metadata;
final changelog validation runs after generation in step 6. Review its diff before continuing.

### 3. Generate the release changelog

```bash
just changelog-release "$TAG"
```

This generates `CHANGELOG.md` as though the target tag already existed, archives
completed minor series under `docs/archives/changelog/`, and defaults to the
`date-released` already recorded in `CITATION.cff` by `update-version`.
No `RELEASE_DATE` shell variable is needed, and generation does not read today's
date or update package or citation metadata. The citation version must match
the target tag, and its date must be a valid ISO `YYYY-MM-DD` date; missing or
invalid metadata stops generation with preparation guidance. Reruns reuse the
recorded date even after UTC midnight and preserve existing release dates.

The optional explicit date remains supported. For troubleshooting, populate
`RELEASE_DATE` manually and pass it as the second argument:

```bash
RELEASE_DATE="$(date -u +%F)"
just changelog-release "$TAG" "$RELEASE_DATE"
```

This override explicitly chooses today's UTC date; keep it consistent with
`CITATION.cff`. To intentionally change the declared release date, update the
metadata first with `just update-version "$TAG" --date "$RELEASE_DATE"`.
You can also set `RELEASE_DATE` to a reviewed `YYYY-MM-DD` value instead of today.

Use `just changelog-preview --tag "$TAG"` for a read-only preview, and
`just changelog-check` to validate the root and every archive. Tagged previews
use the same citation-date default; `--date YYYY-MM-DD` remains available as an
explicit override. Review generated notes and archives; never hand-edit them.
Correct source commit messages or shared generator behavior and regenerate instead.

### 4. Generate release performance measurements

Run this after the package version has been updated:

```bash
just release-dependency-builds
just performance-prepare --release-intent
just performance-release --release-intent --checkpoint target/bench-reports/release-checkpoint
```

Preparation verifies the published dependency inputs and captures independent
source, harness and release assertions without timing. The release recipe measures
the prospective package version against the latest stable published release in two
paired trials with opposite phase orders. It promotes the report to
`docs/performance/current.md` and retains the complete run and provenance under
`docs/performance/runs/`. Existing v1 evidence stays frozen under
`docs/performance/v1/`; the new declaration requires at least v0.5.2 and does not
backfill retired releases. Review scientific workload validity, source
and harness identities, measurement scope and evidence before using timings.
An invariant-violating run is a failed release check.

If measurement succeeds but report promotion needs retrying, use
`just performance-doc` with the saved payload/evidence paths as described in
[benchmark preparation](BENCHMARKING.md#release-preparation-and-readme-publication).
It does not measure again. Repeated measurements create new observations.

### 5. Publish the README performance snapshot

Review both retained trials and the independently declared 13-workload selection
from trial 1 in `tooling/performance-publication.toml`. Then run:

```bash
just performance-readme --preview
just performance-readme
```

The shared publisher consumes retained evidence and updates the marked README
table, CSV and SVG without invoking Cargo or Criterion. It reads the frozen
preparation identities rather than requiring manual provenance edits. Include the report, evidence,
publication configuration and README assets together in the release PR. Check
reproduction with `just performance-doc --check`.

### 6. Validate the release branch

```bash
just ci
just release-version-check
just publish-check
just security
```

`just ci` retains MCMC's deterministic and property tests, all-feature nextest
tests, rustdoc doctests, examples, notebooks, documentation and benchmark
compilation. `release-version-check` delegates final package, citation and
generated-note checks to the shared CLI. `publish-check` validates crates.io
metadata and dry-runs Cargo packaging from the intentionally dirty release tree.
`security` adds network vulnerability checks for all maintained locks and a
full-history secret scan; it stays separate from platform CI.

Require native Linux, macOS and Windows CI for the final release commit. A local
macOS run, including platform emulation tests, establishes only macOS validation.
Reuse successful checks while their relevant inputs remain unchanged.

### 7. Review, stage, and commit the release artifacts

```bash
git --no-pager status --short
git --no-pager diff
```

Review metadata, generated notes/archives and retained performance files. Stage
only reviewed paths, then inspect and commit the staged release delta:

```bash
git --no-pager diff --cached
git commit -m "chore(release): release $TAG

- Prepare reviewed release metadata, generated notes and retained performance evidence"
```

### 8. Push the branch and open the PR

```bash
git push -u origin "release/$TAG"
gh pr create --base main --head "release/$TAG" --title "chore(release): release $TAG"
```

Describe validation in the PR and require review and passing CI before merge.
If a substantive fix is needed during preparation, commit it, regenerate notes,
and repeat affected validation. If measured code, dependencies, toolchain or
harness inputs changed, repeat measurement and README publication too. Keep
earlier evidence and its provenance intact.

### Handling fixes found during preparation

Commit critical fixes before regenerating the affected release outputs. Changes
to measured code, harnesses, dependencies, toolchain or measurement policy require
fresh `performance-release` evidence and reviewed `performance-readme` output.
For unchanged retained measurements, use `performance-doc` to reproduce the report
without acquiring new observations. Regenerate notes with
`just changelog-release "$TAG"` and repeat the affected validation.

File noncritical fixes for a later release. Preserve the prepared citation date
unless deliberately advancing it through `update-version`; never hand-edit the
generated changelog to record a known issue.

### Resuming an unpublished release after another PR

Prepare and validate corrective artifacts on the follow-up PR. Do not repeat
version or dependency updates solely because another PR is required. After it
merges, resume Step 2 from the new clean `main` commit and require that exact
commit's complete signed evidence and SARIF gates.

If an unpublished tag or draft already exists, stop active benchmark/upload
runs and review the remote tag, draft assets and registry state before recovery.
MCMC does not expose Delaunay's `tag-force` recipe; do not substitute an
unreviewed force-push. Published tags and releases require a new version.

## Step 2: Publish after the PR is merged

### 1. Synchronize `main`

```bash
git switch main
git pull --ff-only
test -z "$(git --no-pager status --porcelain --untracked-files=all)"
RELEASE_COMMIT="$(git --no-pager rev-parse HEAD)"
git --no-pager show --no-patch "$RELEASE_COMMIT"
```

Require an empty status and verify the full SHA against the merged PR and its
successful CI. If further commits landed, review and validate the exact release
commit before continuing. Stop on stale notes, metadata or pending/failed checks.

### 2. Verify the release commit

```bash
just release-evidence
```

Record the full reviewed SHA and require its complete trusted main-push CI run.
Use `gh run list --workflow ci.yml --event push --branch main --limit 5` to
inspect a missing or pending producer; verification itself does not wait.
`release-evidence` independently reads the clean checkout's SHA and Cargo version,
then verifies the exact published package's signed-source policy without building,
measuring, tagging, publishing or waiting. Pending, failed, missing, ambiguous,
expired or wrong-source evidence stops preparation. A PR or manual dispatch is
not a substitute for the trusted main-push producer.

The policy requires repository-tooling, native Linux/macOS/Windows validation,
clean default/all-feature Cargo packaging and the separate signing job in one
run attempt. It checks live artifact identity/digest, signer/workflow/source and
the original validation steps. Artifacts are retained for 30 days. A partial
rerun cannot assemble evidence from different attempts; a full rerun requires a
fresh selection and verification. If a bot merge did not produce the required
main-push evidence, stop and resolve that producer failure before tagging.

Native `build` checks, `repository-tooling`, `Clippy SARIF Analysis` and
`Repository Rule SARIF Analysis` also remain independently required on this
exact commit, bound to GitHub Actions app ID `15368`. Signed-source validation
does not promote previously built Cargo bytes: the publication workflow still
verifies and repackages the same clean source without registry credentials.

### 3. Create and verify the annotated tag

```bash
test "$(git --no-pager rev-parse HEAD)" = "$RELEASE_COMMIT"
just tag "$TAG"
test "$(git --no-pager rev-parse "$TAG^{commit}")" = "$RELEASE_COMMIT"
git --no-pager tag -l --format='%(contents)' "$TAG"
git --no-pager show --no-patch "$TAG"
```

`tag` first runs `release-version-check`, then uses shared changelog notes for
the annotation. Confirm the exact reviewed SHA and annotation. If the tag already
exists, inspect remote and registry state before recovery; never move a published
tag.

### 4. Push the tag

```bash
git push origin "$TAG"
```

Stop on rejection and inspect the existing remote tag without force-pushing.

### 5. Inspect the prepared draft GitHub release

```bash
gh run list --workflow prepare-release.yml --limit 5
gh release view "$TAG"
```

The tag push starts `Prepare GitHub release`. Before repository Python runs,
the workflow checks the canonical tag, peeled commit and protected-main
ancestry. Isolated published tooling then verifies metadata, exact required
checks and signed-source evidence, and creates a stable draft with the tag as
title and generated notes as its body. It rechecks the frozen source and
evidence immediately before creating the draft.

Wait for success before dispatching benchmarks. A retry accepts only the same
mutable stable draft with identical title, tag, target and notes, preserving
any assets; it does not rewrite or publish an existing release. A conflicting
draft, changed identity or missing evidence stops the workflow. GitHub release
immutability is enabled for this repository.

### 6. Run benchmarks and review the draft

```bash
just release-benchmarks "$TAG"
```

The recipe dispatches the workflow at the exact tag with its required `tag` input and returns without waiting or measuring locally.
Monitor separately with `gh run list --workflow release-benchmarks.yml --limit 5`
and `gh run watch --exit-status` if desired.
Select the run titled `Release benchmark $TAG`. Trusted preflight requires the workflow ref and source SHA to match that tag,
resolves its full commit and uses shared `release-draft --commit` to capture the
repository, tag, release ID, commit, and title assertion. Trusted job outputs carry that target independently of benchmark files. The benchmark job checks
out the captured commit, verifies HEAD and the local tag before setup, measures the clean checkout and attaches
`markov-chain-monte-carlo-$TAG-criterion-baseline.tar.gz`. The benchmark job has
read-only permissions and restores no dependency caches. Privileged validation
and attachment jobs install only the exact published shared package, without
checking out benchmark code. The writer restores the original target outside the checkout and supplies `release-upload --target`; a missing, malformed,
or changed identity fails instead of selecting a fresh release. The workflow leaves the draft unpublished for review.
Wait for success; do not move the tag or edit/publish the draft while it runs.

### 7. Publish the reviewed draft and approve Cargo publication

`release-publish` rechecks signed evidence before publishing the draft. The
credential-free publication job freezes the verified selection as a job output;
after environment approval, the upload job must revalidate that exact selection
before registry-absence checks and short-lived authentication. A changed run,
attempt or artifact requires renewed verification and approval; it cannot be
silently substituted during the approved job.

From the clean tagged checkout using your human GitHub CLI session:

```bash
just release-publish "$TAG"
gh run list --workflow publish-crates.yml --limit 5
```

The shared gate checks version/metadata, clean checkout, remote tag commit,
protected default-branch ancestry, successful exact-commit checks and the
required baseline. The thin consumer check also binds the release title/body
to the tag and generated notes. Human authentication is essential: publication
with a workflow's `GITHUB_TOKEN` suppresses the downstream release event.

The publication workflow checks out the event SHA, requires the exact repository,
release ID, stable tag/ref/commit and immutable release, then verifies clean
default and all-feature Cargo packages without upload credentials. Approve its
`crates-io` environment deployment in GitHub. The upload job rechecks the evidence
and proven registry absence before requesting a short-lived OIDC token. Only this
job has `id-token: write`; it has no repository write permission or dependency
caches. All uploads for this crate are serialized without cancellation.

Cargo repackages the same clean, revalidated source using `--locked --no-verify`
so build scripts do not run with the crates.io token. Native verification already
succeeded in the separate job. The dry-run archive is not transferred between
jobs, and byte-identical archives are not asserted. There is no second manual
Cargo upload.

### 8. Verify registry availability and downstream resolution

```bash
just release-verify "$TAG"
```

This requires the published stable GitHub release, nonempty durable baseline
and exact non-yanked registry version with a checksum. It retries proven absence
seven times at ten-second intervals, never an upload. A GitHub release alone is
insufficient. Check a fresh downstream consumer in a disposable shell:

```bash
CONSUMER="$(mktemp -d)"
RELEASE_CARGO_HOME="$(mktemp -d)"
RELEASE_CRATE="markov-chain-monte-carlo@=${TAG#v}"
CARGO_HOME="$RELEASE_CARGO_HOME" cargo init --vcs none --name mcmc_release_check "$CONSUMER"
CARGO_HOME="$RELEASE_CARGO_HOME" cargo add --manifest-path "$CONSUMER/Cargo.toml" "$RELEASE_CRATE"
CARGO_HOME="$RELEASE_CARGO_HOME" cargo check --locked --manifest-path "$CONSUMER/Cargo.toml"
CARGO_HOME="$RELEASE_CARGO_HOME" cargo add --manifest-path "$CONSUMER/Cargo.toml" "$RELEASE_CRATE" --all-features
CARGO_HOME="$RELEASE_CARGO_HOME" cargo check --locked --manifest-path "$CONSUMER/Cargo.toml"
```

Record the tag, full SHA, benchmark and publication runs, approved deployment,
registry checksum and downstream results in #176. Keep that issue open until
real publication and account configuration are verified. Preserve the durable
GitHub baseline; the temporary Actions artifact expires after 30 days.

### Recovering a failed run

Missing or failed signed evidence requires a complete trusted main-push attempt
for the exact release source. A partial job rerun cannot combine attempts.
Reverify any new selection before tagging or publishing, and renew environment
approval when the frozen run, attempt or artifact changes.

Benchmark failures leave the release in draft. If only attachment failed, reuse
the retained Actions artifact; an existing attachment must match byte for byte.
Different bytes stop the run without overwrite. An expired artifact requires a
new measurement, with any conflicting draft attachment resolved by the maintainer.
Published immutable releases cannot be repaired by the draft workflow.

If Cargo upload fails or times out, inspect the original run and exact registry
state before retrying. Visibility verification still runs after upload failure
and preserves the failed job result. An existing version blocks another upload;
use `release-verify` and the downstream check to finish verification. Registry
transport/authentication errors never prove absence. Investigate an uncertain
accepted-upload outcome before authorizing a rerun. Corrections to published
contents require a new version. Never move a published tag.

### 9. Verify the Zenodo release metadata

After Zenodo archives the GitHub release, inspect the new version record reached
from concept DOI `10.5281/zenodo.20033111`. Verify its software version and date
against the release tag and `CITATION.cff`. The concept landing page alone does
not establish that the new release was archived.

### 10. Remove the merged release branch

After publication, baseline and Zenodo verification succeed:

```bash
git branch -d "release/$TAG"
git push origin --delete "release/$TAG"
```

## Configure Trusted Publishing before the next release

Follow [deployment setup and evidence](dev/crates-io-publishing.md) and the
current [crates.io setup requirements](https://crates.io/docs/trusted-publishing).
Use owner `acgetchell`, repository `markov-chain-monte-carlo`, workflow
`publish-crates.yml`, and environment `crates-io`. Configure required maintainer
review, disable administrator bypass, permit only `v*` deployment tags, and add
only the pinned authentication action to the live allowlist while preserving
existing entries. Draft preparation and publishing workflow changes require maintainer review rather
than Dependabot automatic approval. No long-lived registry token is needed.

Source configuration does not establish that account settings are applied.
Verify the actual trusted publisher, environment, permissions and action policy
before release and record the evidence in #176.
