# crates.io deployment setup and evidence

The repository source implements #176 and #182 using Delaunay's recipe interfaces
and the published `research-repo-tools==0.2.3` release contract. This document records the intended
deployment identity and the evidence still needed outside source control.
The routine release commands live in [RELEASING.md](../RELEASING.md).

## Publisher identity

| Setting | Required value |
| --- | --- |
| Registry/package | crates.io / `markov-chain-monte-carlo` |
| GitHub owner/repository | `acgetchell/markov-chain-monte-carlo` |
| Workflow filename | `publish-crates.yml` |
| Deployment environment | `crates-io` |
| Trigger | Stable reviewed GitHub Release, `release: published` |
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
checker, including GitHub-owned actions and CodeQL subpaths. The authentication-action identity uses `@*` in this strict shared schema; its workflow and
consumer regression still require the exact reviewed commit. This proposed local payload supersedes the original broad GitHub-owned exemption described
in the historical evidence below. It has not been applied by this migration. Re-read live settings before applying any payload and preserve newer
reviewed identities. Publishing workflow updates are excluded from Dependabot automatic
approval. Verify all account settings separately from reviewing source.

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
| Clean tagged Cargo packaging dry run | Pending reviewed release tag |
| Required baseline attached to draft | Pending release preparation |
| Approved deployment and successful upload run | Pending first approved release |
| Exact registry version and clean downstream resolution | Pending successful publication |

Record the actual reviewer/branch-policy API evidence, crates.io publisher
settings, release tag, full SHA, run URL and registry/downstream results in #176.
Do not close it based solely on merged source, a GitHub Release, or local checks.
Do not publish a throwaway version to test the workflow.

## Gate and recovery contract

The shared gate requires the configured repository, published stable event and
release ID, matching remote tag/checkout SHA, protected default-branch ancestry,
synchronized metadata and generated notes, required assets, and unambiguous
successful exact-SHA checks from the configured provider. The consumer's thin
`release-gate` recipe additionally requires an immutable release and compares its
title/body with the tag and shared changelog-notes output, as Delaunay does. The
shared gate currently validates local notes without comparing that body. Both
workflow jobs enforce these checks before packaging/upload; the upload job
repeats them after environment approval.

Publication is serialized across all versions of this crate. Registry lookup errors block upload,
and an existing exact version blocks retries. If Cargo times out after upload,
the final verification step still checks visibility with bounded attempts;
inspect its results before retrying anything. This follows
[Cargo's documented post-upload polling behavior](https://doc.rust-lang.org/cargo/commands/cargo-publish.html).
The verification job builds default and all-feature packages without upload
credentials. Cargo repackages that clean, revalidated source with `--no-verify`
after approval; build scripts do not run with the registry token. This does not
promise that the dry-run archive bytes are reused.
