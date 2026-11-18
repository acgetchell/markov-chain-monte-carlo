# Release flow integration

This implements #220, #223 and #231 using published
[la-stack v0.5.4](https://github.com/acgetchell/la-stack/releases/tag/v0.5.4),
commit `560b8dd736cdc83441e888e36c25dc5ddc069fea`, and the registry
`research-repo-tools==0.2.9` distribution pinned in `pyproject.toml` and `uv.lock`.
The executable procedure lives in [RELEASING.md](../RELEASING.md).

## Command and ownership map

| Entry point | Shared work and retained boundary | Acceptance |
| --- | --- | --- |
| `performance-release` / `performance-resume` | Prepare the explicit release pair, acquire once, freeze external owner and publisher receipt before report promotion | Consumer acquisition and failure/retry fixtures; real scientific acquisition belongs to release preparation |
| `performance-readme` | Publish the reviewed trial/row selection from the same retained run | Existing MCMC workload and publication-policy tests |
| `release-evidence` | Independently bind original measured source to clean final HEAD and signed native/package evidence; freeze named acceptance | Installed native verifier and candidate handoff tests; final main-push proof remains required |
| `tag TAG` | Require existing reviewed acceptance before annotated tag creation | Source-bound review/rejection tests; no tag is created during adoption |
| Tag push / Prepare GitHub release | Read-only installed draft planner, checkout-free authenticated writer, original run/attempt receipt and durable journal | Installed CLI path, real writer transport, partial-effect/recovery fixtures |
| `release-stage` / `release-benchmarks TAG` | Original four inert asset IDs/hashes, exact tag dispatch, verification and attachment without measurement | Actual inline writer/receiver, retry/conflict and dispatch tests |
| `release-publish TAG` / `release-upload TAG` | Publish immutable reviewed draft; separate explicit Cargo dispatch with independent transfer and full acceptance SHA-256 | Real shared publishing assertions and operator-selection tests |
| Protected `crates-io` job | Same approval artifact ID/digest/current attempt, full acceptance recheck, registry absence, then OIDC and nonbuilding Cargo upload | Actual shell failure guards and frozen-selection tests; live deployment belongs to #176 |
| `actions-check` / `update-actions` | Offline coupled pin validation and shared update with one policy/scope | Installed-package partial-pin rejection and CI wiring tests |

The release adapter follows the tagged reference protocol. MCMC declarations
retain its repository/crate identity, native Linux/macOS/Windows checks,
repository tooling and independent SARIF gates, scientific tests, and clean
default/all-feature Cargo verification. The measurement policy retains all 22
stepping workloads, two alternating paired trials and 100 raw samples per case.
README publication still selects all 13 unthinned workloads from trial 1.
The diagnostic workspace and frozen v1 evidence/guard remain unchanged.

The supported Cargo guarantee is signed source. The final trusted main-push
attempt owns verification builds; Step 2 consumes its signed evidence.
Cargo later repackages the same clean, revalidated source with
`--locked --no-verify`; byte-identical producer archive promotion is not claimed.

The historical `performance-github-assets` inspector retains its two-file asset
contract. RRT 0.2.9 does not read complete candidate archives through `performance
assets`; the new series retains its full paired report/run instead. This inspector
is outside the reference's release path and cannot certify a new candidate pair.

## Validation and remaining hosted acceptance

Local checks run on macOS arm64. Fixtures replace provider/process boundaries;
they do not establish native Linux/Windows success, cold hosted setup time,
real scientific comparability or a live release handoff. No benchmark campaign,
tag, provider upload, publication or account-setting change is part of this adoption.

For the reviewed adoption revision, the maintainer must retain:

1. Required hosted repository tooling and native Linux/macOS/Windows results.
2. A complete original trusted main-push attempt containing native checks,
   default/all-feature package verification and signed evidence. On clean final
   main, run `just release-evidence` before creating any tag.
3. During the actual release, the Step 1 owner/raw bytes and final acceptance,
   authenticated draft receipt, staged asset IDs/hashes, attachment run/attempt,
   and approval artifact/publication/registry evidence.
4. Cold bootstrap and composed-job elapsed time for the new draft, attachment
   and publication jobs, including setup and dependency authentication. The
   existing 45-minute job budgets remain unchanged; local fixtures do not
   establish their hosted margin. Record timing in #229 and release evidence in #176.

## 220 and #223 remain open until their required real consumer evidence is recorded

or explicitly handed to the linked release tracker. This source change alone
does not establish publication readiness or close the release work.
