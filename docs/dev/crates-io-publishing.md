# crates.io deployment setup and evidence

The repository source implements #176, #182, #220 and #223 using la-stack v0.5.4's release process
and the published `research-repo-tools==0.2.9` release contract. This document records the intended
deployment identity and the evidence still needed outside source control.
The routine release commands live in [RELEASING.md](../RELEASING.md).

## Publisher identity

| Setting | Required value |
| --- | --- |
| Registry/package | crates.io / `markov-chain-monte-carlo` |
| GitHub owner/repository | `acgetchell/markov-chain-monte-carlo` |
| Workflow filename | `publish-crates.yml` |
| Deployment environment | `crates-io` |
| Trigger | Explicit maintainer dispatch at the published stable tag, with candidate and acceptance inputs |
| Deployment refs | Selected tag patterns, `v*`; shared gate rejects noncanonical stable tags |
| Approval | Required maintainer reviewer before registry upload; disable administrator bypass |
| OIDC permission | `id-token: write` on the upload job only |

The crate already exists, so configure its Trusted Publisher on crates.io as an
owner using the exact workflow filename, repository and environment above.
Follow [crates.io Trusted Publishing](https://crates.io/docs/trusted-publishing)
for current account and first-publication requirements. The official
[authentication action](https://github.com/rust-lang/crates-io-auth-action)
exchanges GitHub OIDC for a short-lived token and revokes it at job completion.
No persistent crates.io token is required in repository secrets.

The reviewed payloads are
[`crates-io-environment.json`](../../.github/settings/crates-io-environment.json)
and [`crates-io-tags.json`](../../.github/settings/crates-io-tags.json). They require
`acgetchell` (user ID `434849`) review, permit self-review, disable administrator
bypass, and restrict deployment to `v*` tags. Keep `main` protected.
The current [`actions-selected.json`](../../.github/settings/actions-selected.json) declares explicit action and reusable-workflow identities for the shared
checker, including GitHub-owned actions and CodeQL subpaths. The workflow pins the reviewed v1.0.6 authentication commit
`3268eb413133e2960fa4ae0e963ca76380c54cd9`. The proposed allowlist and consumer regression accept exactly that commit and the previous v1.0.5 commit
`c6f97d42243bad5fab37ca0427f495c86d5b1a18`. Mutable tags, wildcards and other SHAs
are not accepted for authentication. This proposed local payload supersedes the original broad GitHub-owned exemption described
in the historical evidence below. It has not been applied by this migration. Re-read live settings before applying any payload and preserve newer
reviewed identities. Publishing workflow updates are excluded from Dependabot automatic
approval. Verify all account settings separately from reviewing source.

On November 3, 2026, the live selected-actions entry was changed from the v1.0.5
authentication SHA to `3268eb413133e2960fa4ae0e963ca76380c54cd9` and read back.
All other live permissions were preserved, including the GitHub-owned exemption.
That exemption already permits the CI evidence attestation action. This narrow
update does not apply the broader proposed selected-actions payload.

## Evidence status

Initial GitHub API inspection on October 22, 2026 found `main` protected and
release immutability enabled. After #206 merged, the owner registered the
crates.io Trusted Publisher. Read-only inspection of the crate's settings
confirmed repository `acgetchell/markov-chain-monte-carlo`, owner ID `434849`,
workflow `publish-crates.yml`, and environment `crates-io`.

The reviewed GitHub environment and allowlist payloads were subsequently applied
and read back through the API on the same date. Environment ID `25076265562`
requires reviewer `acgetchell`, permits self-review, and disables administrator
bypass. Its only deployment policy (ID `64027043`) permits `v*` tags. The Actions
allowlist retains its previous entries and adds the exact authentication-action
commit. `main` remains protected. The optional crates.io setting requiring
Trusted Publishing for all new versions is currently disabled.
The last completed main-branch CI for
`e273e5c48f375518c387b97ddcf0c567167ac858` exposed the actual checks
`build (ubuntu-latest)`, `build (macos-latest)` and `build (windows-latest)` from
GitHub Actions app ID `15368`; these names/provider are declared in publishing
policy, together with `Clippy SARIF Analysis` and `Repository Rule SARIF Analysis`.
They must succeed on the eventual release SHA; this older run is not
release validation.

| Evidence | Status |
| --- | --- |
| Protected default branch and native CI identities | Inspected through GitHub API |
| `crates-io` environment, reviewers and tag restrictions | Applied and verified through GitHub API |
| Selected Actions allowlist addition | Exact reviewed action commit applied and verified; existing entries preserved |
| crates.io Trusted Publisher registration | Verified in the authenticated crate settings after #206 merged |
| Native checks for the final release commit | Pending reviewed release commit |
| Clean default/all-feature Cargo verification | Required in final main-push CI before tagging |
| Required baseline attached to draft | Pending release preparation |
| Approved deployment and successful upload run | Pending first approved release |
| Exact registry version and clean downstream resolution | Pending successful publication |

Record the actual reviewer/branch-policy API evidence, crates.io publisher
settings, release tag, full SHA, run URL and registry/downstream results in #176.
Do not close it based solely on merged source, a GitHub Release, or local checks.
Do not publish a throwaway version to test the workflow.

## Gate and recovery contract

Publication uses the same explicit dispatch and frozen approval protocol as
[la-stack v0.5.4](https://github.com/acgetchell/la-stack/blob/v0.5.4/docs/RELEASING.md).
After `just release-publish "$TAG"`, `just release-upload "$TAG"` sends the
independently retained publication transfer and full reviewed acceptance SHA-256.
Both the actor and triggering actor must be `acgetchell`.

The installed RRT verifier requires the exact remote tag/checkout, protected-main
ancestry, synchronized metadata, generated title/notes, immutable published
release, required original asset IDs/hashes and unambiguous successful exact-SHA
checks. It binds the Step 1 acquisition and final signed main CI evidence.
The first job freezes that full acceptance in an immutable Actions artifact.
After environment approval, the upload job requires the same artifact ID/digest
and current run/attempt, downloads that exact receipt, revalidates it byte for byte,
and proves registry absence before requesting OIDC credentials.

Publication is serialized across all versions of this crate. Registry lookup errors
and an existing exact version block upload. If Cargo times out after upload,
the final verification step still checks visibility with bounded attempts;
inspect its results before retrying anything. This follows
[Cargo's documented post-upload polling behavior](https://doc.rust-lang.org/cargo/commands/cargo-publish.html).
Original main CI owns default/all-feature package builds and the required scientific
checks. Publication verifies retained evidence and performs only Cargo's clean
`--locked --no-verify` repackaging/upload; build scripts do not run with the registry
token. The guarantee is signed source, not byte-identical Cargo archive promotion.
No publication stage acquires benchmark samples.

Preserve the external owner, named acceptance, draft receipt, staging journal and
original asset identities during recovery. See the release guide's
[recovery instructions](../RELEASING.md#recovering-a-failed-run).
