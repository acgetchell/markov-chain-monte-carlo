# Release recovery and retained evidence

Follow the [release checklist](../RELEASING.md) for a normal release. After
interruption, start in the prepared checkout with `just release-status`. It
inspects existing files and read-only provider state. The command it prints
still performs its own verification. Use the retained phase and the links below
to reconcile unexpected or incomplete state before continuing.

## Benchmark acquisition

The default checkpoint is `target/bench-reports/release-checkpoint`. If you chose
a different checkout-relative path, pass it to `just release-status --checkpoint PATH`
and `just performance-resume PATH`. Keep the original preparation and owner.
RRT reuses complete suite pairs with their original measurement sessions and
recollects an interrupted pair as a unit.

When the owner directory holds the original preparation and both raw files,
run `just performance-capture "$OWNER"` at the measured revision. It restores
missing scratch files from those exact bytes and rejects conflicting files.
If the callback stopped before saving
those files, `just performance-capture "$OWNER" --retained-scratch` is available
from the original acquisition checkout with its verified scratch evidence.
If publication stopped during file replacement, preserve the partial files
separately. Use the original `publication.json` receipt to identify the expected
files and their prior digests. Restore the complete pre-promotion report and
history from an independent backup before retrying.

After restoring the original scratch payload and evidence, retry report promotion
using those exact inputs.

```bash
just performance-doc --payload target/bench-reports/performance.run.json \
  --manifest target/bench-reports/performance.evidence.json
```

At the measured revision, review and republish the README snapshot with
`just performance-readme --preview` and `just performance-readme`.
See [benchmark preparation](../BENCHMARKING.md#release-preparation-and-readme-publication).

### Reproduce the committed README snapshot

After the preparation commit, `HEAD` differs from the measured revision.
For the paired release, set `OWNER` to the independent owner directory shown by
`just release-status`. Read both release labels and measured revisions from its
original preparation. The publisher uses these identities to reproduce the
marked README section, CSV, and SVG from the retained run.

```bash
PREPARATION="$OWNER/preparation.json"
just performance-readme \
  --current-release "$(jq -er '.current.release' "$PREPARATION")" \
  --current-revision "$(jq -er '.current.revision' "$PREPARATION")" \
  --baseline-release "$(jq -er '.baseline.release' "$PREPARATION")" \
  --baseline-revision "$(jq -er '.baseline.revision' "$PREPARATION")"
```

## Preparation corrections

Any correction outside the declared benchmark output paths requires fresh
acquisition, even a typo in the release guide. Preserve the earlier checkpoint,
owner and outputs. In a fresh clone at a new path, reconstruct the pre-acquisition
tree with the intended fixes and prepared release metadata. Keep the report,
run inventory and README snapshot as they stood **before** the superseded
acquisition, including any previously released history.

Run setup in that clone. If changelog inputs changed, regenerate the notes with
`just changelog-release "$TAG"`. Adjust the date with
`just update-version "$TAG" --date YYYY-MM-DD` first when needed. Resume at
step 1.4 of the [release checklist](../RELEASING.md#4-generate-the-release-performance-comparison)
and use the new checkout and its registered owner through publication.

Integrate post-merge corrections in a follow-up PR, then select the exact new
main-push CI evidence. Review any existing remote tag, draft, and registry
state before changing an unpublished candidate. Use a new version for a
published correction.

## Evidence policy

The registered selection lives at
`~/.local/state/markov-chain-monte-carlo/releases/<checkout-sha256>/<tag>/owner.json`.
The original acquisition is under
`~/.local/state/markov-chain-monte-carlo/acquisitions/<tag>/<checkpoint-path-sha256>/`.
Keep both outside candidate control and retain independent backups. The original
payload, evidence, preparation, policy, and publisher receipt preserve their
measured revision, sessions, configuration, and SHA-256 digests.

`just release-evidence` binds them to the exact clean merged commit. It retains
`acceptance.json` in `<owner>/final/<commit>/<selection>/`. The default selection
is `reviewed`. The [evidence policy](../../tooling/release-evidence.toml) binds
the original main-push run and attempt, signer, required jobs and steps, runner
labels, and package. A PR, manual dispatch, partial rerun, development
supplier pin, or expired or changed artifact cannot satisfy this selection.
The independently configured SARIF checks remain publication gates.

To recover a moved checkout or choose a named review, pass the original owner
path and selection explicitly. For staging, leave the optional target empty
unless recovering an independently retrieved receipt.

```bash
just release-status "$OWNER" "$SELECTION"
just release-evidence "$OWNER" "$SELECTION"
just release-archive "$OWNER" "$SELECTION"
just tag-preview "$TAG" "$OWNER" "$SELECTION" --commit "$RELEASE_COMMIT"
just release-stage "$OWNER" "" "$SELECTION"
just release-benchmarks "$TAG" "$OWNER" "$SELECTION"
```

Each line applies to its own phase. `release-archive` is an optional local
inspection step. Staging builds the same archive from the original records.
Use the same owner and selection for tagging, publishing, and upload.

## Inspect workflow runs

Set `RELEASE_COMMIT` to the full reviewed SHA and `TAG` to the prepared tag.
Inspect the exact main-push CI and independently required SARIF workflows.

```bash
for WORKFLOW in ci.yml rust-clippy.yml semgrep-sarif.yml; do
  gh run list --workflow "$WORKFLOW" --event push --branch main --commit "$RELEASE_COMMIT"
done
gh run list --workflow prepare-release.yml --event push --branch "$TAG" --commit "$RELEASE_COMMIT"
gh run list --workflow release-benchmarks.yml --event workflow_dispatch --branch "$TAG" --commit "$RELEASE_COMMIT"
gh run list --workflow publish-crates.yml --event workflow_dispatch --branch "$TAG" --commit "$RELEASE_COMMIT"
gh run watch "$RUN_ID" --exit-status
```

Use the matching run ID and inspect failed jobs before choosing a retry.
The publication checks bind **Clippy SARIF Analysis** and **Repository Rule
SARIF Analysis** to the GitHub Actions app and the exact release commit.

## Recover draft preparation

The tag workflow freezes the original validation and notes handoff before a
checkout-free writer creates the draft. Retain the independent handoff,
draft-receipt, and draft-state artifact IDs, ZIP digests, and producer run and
attempt. Automatic staging retrieves and verifies the original receipt. To
retrieve a receipt from an independently identified original run, use
`just release-assets retrieve`.

```bash
just release-assets retrieve tooling/draft-receipt.toml --tag "$TAG" --commit "$RELEASE_COMMIT" \
  --run-id "$PREPARATION_RUN_ID" --output "$RECEIPT_DIR"
just release-stage "$OWNER" "$RECEIPT_DIR/target.json" "$SELECTION"
```

If the writer fails, preserve its state as a unit (`producer.json`,
`handoff.json`, `draft-receipt.json`, and `target.json` when captured).
A pending journal may correspond to an existing draft. Recover it through the
[shared authenticated recovery procedure](https://github.com/acgetchell/research-repo-tools/blob/v0.3.2/docs/PREPARING_DRAFTS.md#retry-and-recovery).
Work outside the candidate checkout in a reviewed Actions job and keep the
same repository/tag concurrency group. A reviewed `--recover-release-id`
records an existing draft without creating a second one. Automatic preparation
accepts only the original attempt. A rerun or recreated tag cannot replace it.
If no trustworthy journal or history is available, reconcile provider state
before authorizing a write.

## Recover attachment or publication

Keep the original staging journal, transport records, attached archive, and
publication transfer. Staging retries use the same draft ID and identical
bytes. Failed-jobs-only reruns cannot substitute an earlier verification
attempt. Review the precise failure before retrying. GitHub retains workflow
artifacts for 30 days.

Publication freezes the canonical archive ID and digest alongside the four
transport assets in `publication-transfer.json`. Keep other draft writers
idle through publication verification. If publication has an uncertain
result, inspect the captured release ID and its assets before another write.
Once published, run `just release-upload` with the same tag and retained
selection. If dispatch or upload is uncertain, inspect the matching workflow
and verify the exact crate version before retrying.

If native evidence needs renewal, rerun **all** jobs in the original main-push
run. Explicitly review a new selection, for example
`just release-evidence "$OWNER" renewed-ci`. Before publication, use that
selection for staging and attachment after reconciling any existing archive.
After publication, preserve the immutable release and original publication
transfer. For a new dispatch supply the original transfer and the independently
reviewed SHA-256 of the renewed acceptance to the tagged workflow.

```bash
gh workflow run publish-crates.yml --ref "$TAG" -f "tag=$TAG" \
  -F "candidate=@$PUBLICATION_TRANSFER" -f "acceptance=$ACCEPTANCE_SHA256"
```

The verification job freezes the reviewed acceptance. After protected
environment approval, the upload job rechecks that same artifact ID, digest,
and run attempt, then confirms the registry version is absent before obtaining
OIDC credentials. Cargo repackages the revalidated source without building.
The guarantee is signed source. See
[workflow ownership and hosted validation](release-flow-integration.md).

## Verify a fresh Cargo consumer

After `just release-verify "$TAG"` succeeds, check a temporary consumer with
a fresh Cargo home and a supported installed Rust toolchain.

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

Use `just release-visibility "$TAG"` for remote asset and registry visibility.
Both verification recipes accept the captured `target.json` as a second
argument to bind the check to the reviewed release ID and commit. If an
individual registry lookup times out, `RRT_REGISTRY_TIMEOUT_SECONDS=180`
increases the per-lookup budget.
