# Releasing markov-chain-monte-carlo

Prepare each stable release in a dedicated PR. Acquire the comparison once
locally during Step 1, keeping original assertions outside the checkout.
After merge and native CI, verify that acquisition against the final clean
commit before tagging. The tag push prepares a draft automatically. Stage the
accepted records and dispatch `Release Benchmarks` to verify and attach the
original archive. Publish the reviewed draft, explicitly dispatch
`Publish to crates.io` with the independent publication selection, then approve
its `crates-io` deployment.

Benchmark attachment and Cargo publication require the maintainer's authenticated
dispatch against the exact tag. Step 2 reuses evidence without measuring.
The changelog is generated before the tag exists; no temporary local tag is needed.

With unchanged history and configuration, changelog generation reuses the prepared
citation date, and report/README retries reuse retained measurements. Fresh
acquisition requires a new checkpoint. Preserve original measurements and source
provenance through capture, staging, attachment and publication recovery.

## Contents

- [Reusable CI evidence](#reusable-ci-evidence)
- [Release command alignment](#release-command-alignment)
- [Prepare the environment](#prepare-the-environment)
- [Step 1: Prepare the release PR](#step-1-prepare-the-release-pr)
- [Step 2: Publish after the PR is merged](#step-2-publish-after-the-pr-is-merged)
  - [Verify the final release commit before tagging](#2-verify-the-final-release-commit)
  - [Wait for the draft GitHub release](#5-wait-for-the-draft-github-release)
- [Configure or verify Trusted Publishing](#configure-trusted-publishing-before-the-next-release)

## Reusable CI evidence

MCMC adapted the released [la-stack v0.5.4 process](https://github.com/acgetchell/la-stack/blob/v0.5.4/docs/RELEASING.md)
at commit `560b8dd736cdc83441e888e36c25dc5ddc069fea`. The current RRT pin and its
adoption history are recorded in the [shared maintenance record](dev/shared-maintenance-migration.md).
`tooling/release_candidate.py` retains the same operator selection and transfer
protocol; RRT owns verification, signatures, archive handling and provider effects.

Protected main pushes retain signed evidence after repository tooling, native
Linux/macOS/Windows validation and clean default/all-feature package verification.
`tooling/release-evidence.toml` binds those actual jobs/steps, hosted OS labels,
signer, package, exact commit and original run/attempt. The guarantee remains
`signed-source`; Cargo's later nonbuilding repackaging is not byte-identical
promotion of a retained producer archive. Separate required SARIF checks remain
publishing gates. PR/manual-dispatch/partial-rerun results cannot replace the
original main-push evidence; expired or changed artifacts stop verification.

Step 1 registers `owner.json` under
`~/.local/state/markov-chain-monte-carlo/releases/<checkout-sha256>/<tag>/`.
The acquisition itself lives under
`~/.local/state/markov-chain-monte-carlo/acquisitions/<tag>/<checkpoint-path-sha256>/`.
Keep both directories and an independent backup under exclusive maintainer
control. Original raw evidence, preparation, policy, publisher receipt and their
digests are immutable. An ignored checkout directory is not an owner boundary.

`just release-evidence` recovers that original owner and verifies the final clean
commit against its acquisition and signed native evidence. It writes acceptance
outside the checkout at `<owner>/final/<commit>/<selection>/`; the default named
selection is `reviewed`. Inspection and retries do not build, test, sample, tag
or publish. A newer complete native attempt requires an explicitly reviewed new
selection; existing records are never replaced.

The same four staged inert assets and `transfer.json` feed benchmark attachment.
After review, `publication-transfer.json` also freezes the canonical archive ID
and digest. Cargo dispatch requires that independent transfer and the full
reviewed acceptance SHA-256. Hosted verification freezes its exact acceptance
in an immutable artifact, then checks its ID/digest and current run/attempt
again after protected-environment approval and before OIDC credentials.

## Release command alignment

| Command | Shared behavior and MCMC declaration |
| --- | --- |
| `update-version TAG [ARGS...]` | Prepare Cargo/citation metadata without dependency updates |
| `changelog-release TAG [DATE]` | Generate notes using the prepared citation date |
| `release-version-check` | Require synchronized final metadata and generated notes |
| `publish-check` | Verify clean default/all-feature Cargo packages |
| `release-evidence [OWNER] [SELECTION]` | Verify original acquisition and exact native/package evidence without rebuilding |
| `tag TAG [OWNER] [SELECTION] [--force]` | Require an existing reviewed selection and create or explicitly replace the local annotated tag |
| `release-notes TAG` | Print generated notes used by automatic draft preparation |
| `release-stage [OWNER] [TARGET] [SELECTION]` | Stage original inert records against the authenticated draft target |
| `release-benchmarks TAG` | Dispatch retained archive attachment at the explicit tag |
| `release-publish TAG [OWNER] [SELECTION]` | Revalidate acceptance and asset identities, then publish the captured draft |
| `release-upload TAG [OWNER] [SELECTION]` | Dispatch Cargo publication with the independent existing acceptance |
| `release-verify TAG [TARGET]` | Verify generated notes, stable GitHub assets, and the exact registry version, optionally bound to a captured release target |
| `release-visibility TAG [TARGET]` | Check immutable GitHub assets and the exact registry version, optionally bound to a captured release target |

MCMC-specific declarations retain all-feature checks, its scientific workload
and paired-trial policy, the independent diagnostic workspace, Zenodo identity
and the frozen v1 evidence boundary. Common command names, arguments, receipt
protocol, immutable publication and recovery follow the released reference.

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

### 4. Generate the release performance comparison

After updating the package version and preparing dependency provenance, run:

```bash
just release-dependency-builds
just performance-release
```

The recipe fetches tags, prepares independent source assertions and acquires the
complete comparison once under the original external owner. It retains the raw
evidence and exact publisher receipt before promoting the report. Routine commands
recover that owner through the same immutable registration used by la-stack v0.5.4;
no source-selection shell variables are needed.

MCMC retains its own measurement policy: two paired trials in opposite phase
orders, all stepping and thinning cases, 100 Criterion samples per case, and
independent case inventory checks. The report goes to `docs/performance/current.md`
with complete run/index records under `docs/performance/runs/`. Baselines require
v0.5.2 or newer. Frozen `docs/performance/v1/` bytes and links stay unchanged;
Issues #209 and #221 govern their later guard retirement.

The default checkpoint is `target/bench-reports/release-checkpoint`. Fresh
acquisition refuses an existing checkpoint. Resume eligible missing work with:

```bash
just performance-resume
```

Complete suite pairs retain original bytes and sessions; an interrupted pair is
recollected under the shared recovery contract.

If measured inputs change before tagging, retain the earlier checkpoint, raw
evidence, external owner directory, and `owner.json`. Start from the reviewed
revised release branch in a fresh clone at a new path. After setup, run
`just release-dependency-builds` and `just performance-release` in that clone.
The new checkout and checkpoint paths select a fresh owner and registration.
Use that clone for README publication and the post-merge Step 2 commands so
they recover the newly selected acquisition.

Preparation alone remains available through
`just performance-prepare`, without timing; fetching requires its explicit
`--fetch-tags --allow-git-mutations` flags.

After acquisition, retry receipt capture independently with
`just performance-capture "$OWNER"`. Use `--retained-scratch` only while controlling
the original acquisition if its callback failed before retaining raw bytes.
Retry report reproduction/promotion with `just performance-doc` and the original
payload/evidence paths. None of these recovery commands starts acquisition.
See [benchmark preparation](BENCHMARKING.md#release-preparation-and-readme-publication).

### 5. Publish the README performance snapshot

Review both retained trials and MCMC's independent 13-workload trial-1 selection
in `tooling/performance-publication.toml`, then run:

```bash
just performance-readme --preview
just performance-readme
```

The shared publisher reads the original preparation, retained evidence and
declared output contract. It updates the README table and CSV/SVG without Cargo
or Criterion. Include the complete report, run/index records, publication
declaration and README assets together. Reproduce the report with
`just performance-doc --check`. Preserve all source, harness, host/session/time,
commands and raw samples; output-only changes do not relabel measurements.

### 6. Validate the release branch

```bash
just ci
just release-version-check
just security
```

`just ci` retains MCMC's deterministic and property tests, all-feature nextest
tests, rustdoc doctests, examples, notebooks, documentation and benchmark
compilation. `release-version-check` delegates final package, citation and
generated-note checks to the shared CLI. After committing the reviewed release artifacts, `just publish-check` verifies
clean default and all-feature Cargo packages. Main CI verifies the final merged
commit; Step 2 reuses that signed source evidence without package-verification builds.
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
just publish-check
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

### 1. Synchronize `main` and wait for CI

```bash
git switch main
git pull --ff-only
test -z "$(git --no-pager status --porcelain)"
```

Use a clean checkout of the reviewed release commit. If `main` advanced, review
the intervening changes and refresh generated notes through a PR as needed.
Wait for that commit's complete main-push CI run. The next step derives the commit
and checks eligible evidence automatically; pending results stop it with a retry
diagnostic. Use GitHub's Actions page or the
[manual inspection commands](#manual-context-and-inspection) when troubleshooting.

Require the complete run: `repository-tooling`, all three native `build (OS)`
jobs, default/all-feature Cargo verification in `package`, and signed
`release-evidence`. The evidence policy checks the actual jobs, successful steps
and hosted runner labels. A development
supplier pin skips signing and cannot provide eligible release evidence.
Inspect the independent SARIF checks for this exact commit too:

```bash
gh run list --workflow rust-clippy.yml --event push --branch main --commit "$(git --no-pager rev-parse HEAD)" --limit 20
gh run list --workflow semgrep-sarif.yml --event push --branch main --commit "$(git --no-pager rev-parse HEAD)" --limit 20
```

Wait for successful `Clippy SARIF Analysis` and `Repository Rule SARIF Analysis`
from GitHub Actions (app ID `15368`). They are separate publication gates;
`release-evidence` verifies only the jobs and signer declared in its evidence
policy, not every independent workflow. PR review remains a protected-branch
requirement and cannot replace merged main-push evidence.

### 2. Verify the final release commit

The normal commands need no `OWNER` or `RELEASE_COMMIT` shell variables:

```bash
just release-evidence
just release-archive
```

Require native Linux, macOS and Windows CI, default/all-feature Cargo packaging and
signed main-push evidence, plus the independently checked SARIF results. The candidate verifier
also checks that the final tree differs from the measured candidate only through
the accepted publication contract. It preserves the original measured revision
separately from the final commit. Do not tag until verification succeeds.

If main-push evidence needs renewal, rerun every job in the original run and
explicitly review a new selection. Manual CI dispatch cannot supply release
evidence. Pending, failed, cancelled or expired evidence blocks publication.

### 3. Create the annotated tag

```bash
just tag "$TAG"
```

`just tag` first runs `release-version-check` and revalidates the existing reviewed
acceptance for clean HEAD. A different commit or missing receipt requires an
explicit `release-evidence` review before tagging. The recipe builds the annotation
from the matching active or archived changelog section. It links that section
when the annotation exceeds 125,000 bytes.

Use `just tag "$TAG" --force` only for an explicitly reviewed local tag replacement.
The same metadata and evidence checks run first. For a tag already pushed to
GitHub, follow [draft preparation recovery](#draft-preparation-recovery).

### 4. Push the tag

After `just tag` succeeds, push the tag.
This starts `Prepare GitHub release`; pushing does not wait for or verify CI.

```bash
git push origin "$TAG"
```

### 5. Wait for the draft GitHub release

The tag push starts `prepare-release.yml`. Let its original attempt finish;
the next `release-stage` command automatically retrieves and verifies that exact
preparation receipt. Missing or pending preparation stops staging; wait for the
original run and retry. The manual inspection commands below can show that run.
Let preparation create the draft; manual creation is unnecessary.
The read-only job checks the exact tagged source, main ancestry, final metadata
and retained signed native evidence, then freezes that selection and generated
notes. An isolated pinned writer with no checkout authenticates the handoff by
immutable artifact ID, digest, run and attempt, revalidates the same native
selection and creates the unpublished draft with `release assets prepare-authenticated`.

Retain the handoff and separate draft-receipt artifact IDs, ZIP digests and
producer run/attempts in independent release review notes. Preserve the captured
release ID in `target.json`. An existing exact matching draft is unchanged;
conflicting notes, title, lifecycle, moved tags or missing evidence stop work.
Retain the separate `draft-state-RUN_ID-ATTEMPT` artifact as well. It contains
the original `producer.json`, `handoff.json`, `draft-receipt.json` and captured
`target.json`; recovery must restore that complete state before retrying.
Preparation builds the declared native Python dependencies for provenance;
it runs no tests, Rust package builds, release-evidence signing or benchmark
acquisition.

Keep the release title identical to the tag, including its leading `v`.
Leave it as a stable draft: publishing now would make the release immutable
before the benchmark archive can be attached. GitHub recommends this
[draft-first sequence for immutable releases](https://docs.github.com/en/code-security/concepts/supply-chain-security/immutable-releases).

### 6. Attach accepted benchmarks and review the draft

Stage the four inert records:

```bash
just release-stage
```

Staging uses the installed `release assets retrieve` operation and
`tooling/draft-receipt.toml`. It authenticates the exact tag-push workflow source,
original successful attempt, required jobs/steps, immutable artifact ID and ZIP
digest before retaining `selection.json`, `target.json` and `draft-receipt.json`
under the owner's final record. Retries revalidate that frozen provider selection
instead of searching again. A replacement draft is rejected before any upload.

This revalidates acceptance, freezes the prepared draft ID before any upload, and prints
the independent `transfer.json` path. A retry uses that same draft and identical
bytes. Dispatch from the exact tag using the printed path:

```bash
just release-benchmarks "$TAG"
```

The recipe resolves the existing transfer from the original owner record and
clean HEAD, validates the explicitly supplied tag, and returns after dispatch.
It never substitutes another owner or tag, polls, creates a draft,
acquires measurements or runs `performance-readme`. Inspect the matching
`Release Benchmarks` run and require success before publishing.

The no-checkout retrieval job checks original release/asset identities and
downloads the selected bytes. A separate read-only checkout job verifies
acquisition, final binding and full frozen native acceptance. The no-checkout
attachment job receives an immutable artifact by exact ID, digest and current
attempt, rechecks the draft and original assertions, then attaches
`markov-chain-monte-carlo-$TAG-criterion-baseline.tar.gz`. The verification job prepares native
Python dependency provenance; no job measures benchmarks or rebuilds the Rust
package.

The archive retains original evidence, current Criterion files and a separate
final-release binding. Local ownership, measured revisions, hosts, sessions and
timestamps remain unchanged; acceptance does not relabel their source.

Review the draft title, generated notes and archive. The workflow leaves it
unpublished. Do not edit, replace or publish it while a job is running. Runs for
the same tag serialize without cancellation. Privileged jobs use isolated
pinned tooling and execute no checkout code.

### 7. Publish the reviewed draft and approve Cargo publication

From the clean tagged checkout and your human GitHub CLI session:

```bash
just release-publish "$TAG"
```

This revalidates the reviewed acceptance and the identities of the staged assets
and canonical archive. It freezes those assertions in `publication-transfer.json`
and uses the shared gate to check the remote notes. The publish command checks
the sealed IDs and available digests immediately before publishing the captured
draft by ID, then confirms those assertions on the immutable release.

Keep other draft writers idle from sealing through verification. GitHub's
release update API has no conditional asset precondition. If the command
cannot confirm the immutable result, inspect the release before retrying or
dispatching Cargo publication. See
[GitHub's conditional request rules](https://docs.github.com/en/rest/using-the-rest-api/best-practices-for-using-the-rest-api#use-conditional-requests).

Dispatch Cargo publication from that independently retained selection:

```bash
just release-upload "$TAG"
gh run list --workflow publish-crates.yml --limit 5
```

`release-upload` revalidates the already existing acceptance and original
publication transfer, checks the immutable release, and sends a snapshot of the
same assertions and acceptance digest. Missing review or changed evidence stops
dispatch; this command never creates a review selection or chooses a newer run.
It returns after dispatch and does not approve the deployment.

The first hosted job checks immutable release state, canonical and transport asset
IDs/hashes, original archive acceptance, final clean commit and generated notes.
It verifies current native evidence against the separately reviewed full
acceptance digest and freezes the exact receipt in an immutable Actions artifact.
After it succeeds, approve the `crates-io` deployment in GitHub. The upload job
retrieves that artifact by exact ID, digest and current run/attempt, repeats
verification against its frozen receipt, proves the registry version
absent, then requests an OIDC token. Only that job has `id-token: write`; it
has no repository write permission or dependency caches. Uploads serialize
without cancellation.

Original main CI built default and all-feature packages without upload credentials.
Cargo repackages the clean commit with `--locked --no-verify`, so build scripts
do not run with the crates.io token. This is signed source validation, not
byte-identical Cargo archive promotion. No benchmark workspace member is published.

### 8. Verify registry availability and downstream resolution

After the publishing workflow completes:

```bash
just release-verify "$TAG"
```

This checks the generated notes against the checkout and requires the published
stable GitHub release, a nonempty durable baseline asset, and the exact non-yanked
crates.io version with a checksum. The workflow passes its captured `target.json`
to also check the reviewed release ID and tag commit. A confirmed registry 404
allows another poll, and individual lookups allow bounded transient HTTP retries.

For a remote asset and registry check from the repository, use
`just release-visibility "$TAG"`. Both recipes accept a captured target file as
their second argument. From outside a checkout, use the installed RRT CLI.
[RRT publication visibility](https://github.com/acgetchell/research-repo-tools/blob/v0.3.0/README.md#releases)

Check a clean consumer with a fresh Cargo home:

```bash
(
  set -eu
  CONSUMER="$(mktemp -d)"
  CARGO_HOME="$(mktemp -d)"
  export CARGO_HOME
  RELEASE_CRATE="markov-chain-monte-carlo@=${TAG#v}"
  cargo init --vcs none --name mcmc_release_check "$CONSUMER"
  printf '%s\n' 'use markov_chain_monte_carlo as _;' 'fn main() {}' > "$CONSUMER/src/main.rs"
  cargo add --manifest-path "$CONSUMER/Cargo.toml" "$RELEASE_CRATE"
  cargo check --locked --manifest-path "$CONSUMER/Cargo.toml"
  cargo add --manifest-path "$CONSUMER/Cargo.toml" "$RELEASE_CRATE" --features benchmarks,serde,tracing
  cargo check --locked --manifest-path "$CONSUMER/Cargo.toml"
)
```

Preserve the output and record the tag, commit, benchmark run, publishing run,
registry URL/checksum, and consumer results in the current release tracker.
Link the merged release PR and any follow-up issue for an explicitly accepted
verification gap. Close the tracker when its agreed validation boundary is
satisfied. See [release tracking](dev/release-flow-integration.md#release-tracking).

Hosted jobs restore no dependency caches. The checkout verifier has read-only
repository permissions. Draft retrieval and attachment require write access
but execute no checkout code. See
[release assets](BENCHMARKING.md#github-release-assets) and
[workflow validation](dev/release-flow-integration.md) for the
exact handoff, local regressions and separate hosted evidence requirements.

### Manual context and inspection

The normal commands recover context after restarting the shell. Explicit paths,
selections and GitHub commands remain available for older acquisitions, a moved
checkout or deliberate recovery. They do not replace or rewrite the normal
external owner selection. Do not infer missing assertions from received assets.

```bash
RELEASE_COMMIT=$(git --no-pager rev-parse HEAD)
OWNER="/path/to/original/independent/acquisition"
just release-evidence "$OWNER"
just release-archive "$OWNER"
```

The same explicit owner and optional selection arguments are supported by
`tag`, `release-publish` and `release-upload` during recovery.

Use the recorded workflow/run or inspect the exact current commit:

```bash
gh run list --workflow ci.yml --event push --branch main --commit "$RELEASE_COMMIT" --limit 20
gh run list --workflow prepare-release.yml --event push --branch "$TAG" --commit "$RELEASE_COMMIT" --limit 20
gh run watch ORIGINAL_RUN_ID --exit-status
```

For manually reviewed preparation retrieval, the shared CLI preserves the same
source, original-attempt, job and artifact checks:

```bash
just release-assets retrieve tooling/draft-receipt.toml --tag "$TAG" --commit "$RELEASE_COMMIT" \
  --run-id ORIGINAL_PREPARATION_RUN_ID --output /path/to/independent/preparation
just release-stage "$OWNER" /path/to/independent/preparation/target.json
export CANDIDATE_TRANSFER_FILE="$OWNER/final/$RELEASE_COMMIT/reviewed/transfer.json"
just release-benchmarks "$TAG"
just release-publish "$TAG" "$OWNER"
just release-upload "$TAG" "$OWNER"
```

For direct dispatch troubleshooting, `publish-crates.yml` still accepts
`--ref "$TAG" -f tag="$TAG" -F candidate=@/original/publication-transfer.json`
and `-f acceptance=REVIEWED_FULL_ACCEPTANCE_SHA256`. The explicit selection must
be the same original owner record; do not copy a digest from incoming assets.

### Recovering a failed run

A failed draft, attachment, or upload step does not invalidate completed
benchmark measurements. Preserve the external owner record, preparation,
checkpoint, archive, transfer files, and workflow receipts. Do not run
`just clean` or restart acquisition to troubleshoot those later phases.

Inspect the failed stage and retained records before selecting a recovery command.
An unconfirmed write outcome requires provider inspection before retrying.
Cold hosted setup and composed-job timing for this adoption remain required;
see the [integration record](dev/release-flow-integration.md).

If HEAD changes, return to Step 2's synchronization and exact-commit verification.
If a native run/attempt changes, explicitly review a new named selection as
described below. Missing, expired, PR, dispatch or partial-rerun evidence never
authorizes fallback to a replacement run or a local rebuild.

#### Draft preparation recovery

Automatic preparation accepts only its original attempt and rejects prior runs
for the same tag. Do not rerun a failed preparation or recreate its tag to obtain
a fresh receipt. Preserve the original authenticated handoff and journal on
success and failure. A pending journal means creation may have happened; a
captured journal binds the existing release ID. A missing or edited draft is not
permission to create a replacement.

Recover the original artifacts by their independently recorded immutable IDs and
verify their ZIP digests and producer run/attempts. Restore the writer state as a
unit (`producer.json`, `handoff.json`, `draft-receipt.json`, and `target.json` if
captured). Run an explicitly reviewed recovery in a GitHub Actions job outside
the checkout. Pass the restored state, original producer and artifact assertions,
and current executing run and attempt to `release assets prepare-authenticated`.
For a pending journal, independent provider review can authorize
`--recover-release-id REVIEWED_ID` to record the existing draft without creating
another release. See the
[RRT recovery procedure](https://github.com/acgetchell/research-repo-tools/blob/v0.3.0/docs/PREPARING_DRAFTS.md#retry-and-recovery).

The workflow retains records for 30 days; hard runner termination or upload
failure may prevent journal retention. GitHub history guards are not a permanent
intent ledger. Keep workflow identity/history intact for retriggerable tags. If
history or the journal is unavailable, stop automation and reconcile intent and
provider state manually before authorizing any further creation. Recovery writers
must use the same repository/tag serialization group as preparation and attachment.

#### Benchmark and publication recovery

Resume local acquisition with `just performance-resume`; keep the original
checkpoint, preparation and owner record. Retry capture, report promotion and
README publication separately after acquisition completes. Source/harness drift
requires a new reviewed acquisition, not repaired receipts.

For native evidence, rerun **all** jobs in the original main-push run. Partial
attempts and manual CI dispatch remain ineligible. Accept a new complete native
attempt through an explicit new review selection:

```bash
just release-evidence "$OWNER" renewed-ci
```

Before GitHub publication, use the new selection for archive, staging and
attachment. After GitHub publication, keep the original immutable archive,
transport assets and `publication-transfer.json`. Review the new acceptance and
dispatch `publish-crates.yml` again with its full SHA-256 as `acceptance` and
the original publication selection as `candidate`. This renews CI evidence
without reacquiring timings or replacing release assets. A changed native
attempt after deployment approval requires a fresh reviewed dispatch.

For hosted recovery, rerun all jobs or dispatch again with the same independent
selection. Inert artifacts are selected by exact ID and attempt and retained for
30 days. Failed-jobs-only attachment reruns cannot reuse another attempt's
verification. These retries perform no timings.

Upload retries reuse the original draft ID and identical bytes. Conflicting
or incomplete assets stop without overwriting or deletion. Inspect failures
before manual correction. Keep published tags and independently captured
digests fixed throughout recovery.

A rerun uses the tagged workflow. A later workflow fix requires a reviewed new
candidate. If Cargo reports an ambiguous timeout, inspect exact registry
visibility before retrying; a present version must not be uploaded again.
See [registry and downstream verification](#8-verify-registry-availability-and-downstream-resolution).

### 9. Verify the Zenodo release metadata

After Zenodo archives the GitHub release, inspect the new version record reached
from concept DOI `10.5281/zenodo.20033111`. Verify its software version and date
against the release tag and `CITATION.cff`. The concept landing page alone does
not establish that the new release was archived.

### 10. Remove the merged release branch

After publication and baseline verification succeed:

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
Verify the trusted publisher, environment, permissions and action policy before
release, then record the evidence in the current release tracker. See
[release tracking](dev/release-flow-integration.md#release-tracking).
