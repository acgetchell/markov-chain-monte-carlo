# Releasing markov-chain-monte-carlo

1. Prepare a release PR with synchronized metadata, generated notes, and one
   retained benchmark comparison. Review the scientific report and pass the
   local release checks.
2. After merge, wait for the exact main-push CI run and the separate SARIF
   checks. Run `just release-evidence`, create the annotated tag with
   `just tag "$TAG"`, then push it explicitly.
3. Let GitHub prepare the draft. Stage and attach the accepted benchmark
   archive, then review the draft and publish it from a maintainer GitHub CLI
   session.
4. Dispatch crates.io publication from the retained acceptance and approve the
   protected `crates-io` deployment after its verification job succeeds.
5. Verify the exact registry version, a fresh Cargo consumer, and Zenodo
   metadata. Record the evidence in the release tracker.

Run `just release-status` from the prepared checkout after an interruption or in
a fresh shell. It shows retained state and the next command without advancing the
release. Routine commands use the registered acquisition owner and the
`reviewed` selection. Set `TAG` again when opening a fresh shell.

## Prepare the environment

Set `TAG=vX.Y.Z`. Start from a clean, current `main` with the
[development environment](../CONTRIBUTING.md#development-environment-setup) and
a working `gh auth status`. Verify [Trusted Publishing](dev/crates-io-publishing.md)
before starting. Land tool and dependency updates separately from the release PR.

## Step 1: Prepare the release PR

### 1. Create the release branch

```bash
git switch -c "release/$TAG"
```

### 2. Update release metadata

```bash
just update-version "$TAG"
```

Review the Cargo manifests and lockfiles, citation version and date, and install
instructions. The updater selects the previous stable published release.

### 3. Generate the release changelog

```bash
just changelog-release "$TAG"
```

Review the generated notes and archives. Regeneration uses the prepared citation
date. To change it deliberately, pass `--date YYYY-MM-DD` to
`just update-version` and regenerate.

### 4. Generate the release performance comparison

```bash
just release-dependency-builds
just performance-release
```

`just performance-release` verifies published dependency provenance before
collecting benchmark samples.

Review both paired trials and the workload inventory. Keep the original checkpoint,
external owner record, and an independent backup. If interrupted, run
`just release-status` and follow the [benchmark recovery procedure](dev/release-recovery.md#benchmark-acquisition).
See [benchmark preparation](BENCHMARKING.md#release-preparation-and-readme-publication)
for the measurement and source contracts.

### 5. Publish the README performance snapshot

Review the declared trial-1 selection in `tooling/performance-publication.toml`.

```bash
just performance-readme --preview
just performance-readme
```

These commands reuse the retained measurements. Include the report, run records,
table, and CSV/SVG together in the preparation commit.

### 6. Run the release checks

```bash
just release-version-check
just security
just ci
```

### 7. Commit the release preparation

Stage only the reviewed release files and inspect the staged diff. The changelog
generator skips the preparation commit when it has the subject shown below.

```bash
git diff --cached
git commit -m "chore(release): release $TAG"
```

### 8. Verify the benchmark evidence, check the Cargo package, and run CodeRabbit

From the clean preparation commit, check its source and generated outputs against
the original Step 1 acquisition. Preflight also reproduces the README table,
CSV, and SVG using the release identities saved by the acquisition owner.
Then verify the default and all-feature Cargo packages and run the local review
against the PR base.

```bash
just release-preflight "$(git rev-parse HEAD)"
just publish-check
just review origin/main
```

If preflight fails, inspect each reported path before proceeding. Restore an
accidental change or follow the acquisition recovery procedure below if the
change belongs in the release. After CodeRabbit finishes, verify each finding
against the release branch.

- For a preparation-only correction, such as a typo in this guide, stage the
  correction and amend the unpushed preparation commit with `git commit --amend --no-edit`.
- For a user-facing fix, make a separate commit with a descriptive subject, then
  run `just changelog-release "$TAG"` so the fix appears in the notes.

Regenerate the notes if another changelog input changes, including the prepared
citation date. Review and stage any regenerated files, then commit them with
`git commit -m "chore(release): prepare $TAG notes"`.

If a correction changes measured code (for example, sampler arithmetic), the
benchmark harness, toolchain, or dependencies, preserve the original run and
follow [benchmark recovery](dev/release-recovery.md#benchmark-acquisition) to
start a fresh acquisition. Repeat steps 4 and 5 to collect both trials and
regenerate the report and README snapshot before repeating the release checks.
The retained run also requires the final tree to differ from the measured tree
only at declared output paths. See the [evidence policy](dev/release-recovery.md#evidence-policy).

After corrections, repeat step 6 and all step 8 commands on the final clean
commit before pushing.

### 9. Push the branch and open the release PR

Once the checks and local review pass, push the reviewed branch.

```bash
git push -u origin "release/$TAG"
```

After the push succeeds, open the PR against `main`.

```bash
gh pr create --base main --head "release/$TAG" --title "chore(release): release $TAG"
```

Record the release checks, Cargo package checks, and local review results in the
PR description. Resolve confirmed findings and wait for required PR checks and
reviews before merging. Apply the [PR push requirements](../AGENTS.md#validation)
to any corrective push.

## Step 2: Publish after the PR is merged

### 1. Synchronize main and wait for CI

Use the clean, reviewed release commit on `main`. Wait for the full main-push CI
run, including native Linux, macOS, and Windows, and for **Clippy SARIF Analysis**
and **Repository Rule SARIF Analysis** on the same commit. See
[exact-run inspection](dev/release-recovery.md#inspect-workflow-runs).

### 2. Verify the final release commit

```bash
just release-evidence
```

Review the retained acquisition against the final commit and its signed native
and Cargo package checks before tagging. Record the original run, attempt, and
artifact identities. See [Reusable CI evidence](#reusable-ci-evidence).

### 3. Create the annotated tag

```bash
just tag "$TAG"
```

The recipe revalidates the existing acceptance, checks final metadata and
published dependencies through managed Cargo, and prints the local annotated
tag and accepted commit. Optional `--commit SHA` pins the reviewed commit.
Use `just tag-preview "$TAG"` to inspect the annotation. Explicit local
replacement requires `--force` and the same checks.

### 4. Push the tag

```bash
git push origin "refs/tags/$TAG"
```

### 5. Wait for the draft GitHub release

Wait for the original `Prepare GitHub release` run to succeed. The tag push
creates the draft and a receipt that binds its release ID. Use
`just release-status` to inspect progress.

### 6. Attach accepted benchmarks and review the draft

```bash
just release-stage
just release-benchmarks "$TAG"
```

Wait for `Release Benchmarks` to succeed. Review the draft title, generated
notes, and attached `markov-chain-monte-carlo-$TAG-criterion-baseline.tar.gz`.
Keep the release in draft state until that review is complete.

### 7. Publish the reviewed draft and approve Cargo publication

```bash
just release-publish "$TAG"
just release-upload "$TAG"
```

The first command confirms the immutable GitHub release. Run the upload command
after that succeeds. It sends the independently retained acceptance and asset
identities to the workflow for this tag. Open the dispatch run and approve the
protected **crates-io** deployment after verification succeeds.

### 8. Verify registry availability and downstream resolution

```bash
just release-verify "$TAG"
```

Check a [fresh Cargo consumer](dev/release-recovery.md#verify-a-fresh-cargo-consumer)
and the new Zenodo version reached from concept DOI `10.5281/zenodo.20033111`.
Record the release commit, run and attempt IDs, artifact IDs and digests,
registry checksum, and consumer results in the
[release tracker](dev/release-flow-integration.md#release-tracking).

## Reusable CI evidence

The final main-push run signs the exact source after native CI and default and
all-feature package verification. Tagging, draft preparation, and attachment
reuse that evidence and the original measurement. See the
[evidence policy and retained records](dev/release-recovery.md#evidence-policy).

## Recovering a failed run

Start with `just release-status`. See [release recovery](dev/release-recovery.md)
for explicit owner and selection arguments, partial state, workflow inspection,
and retries. For an uncertain publish or upload result, inspect GitHub and
crates.io before retrying.
