# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### ⚠️ Breaking Changes

- Rust 1.98.1 or newer is now required.
- changelog-unreleased now requires TAG and DATE arguments. Remove postprocess-changelog and update-release-version --sync-changelog-date in favor of the shared
  workflow.

### Merged Pull Requests

- Complete shared changelog adoption with v0.1.1 [#162](https://github.com/acgetchell/markov-chain-monte-carlo/pull/162)
- Raise the Rust baseline to 1.98.1 [#161](https://github.com/acgetchell/markov-chain-monte-carlo/pull/161)
- Bump the dependencies group with 4 updates [#159](https://github.com/acgetchell/markov-chain-monte-carlo/pull/159)
- Bump the github-actions group with 5 updates [#158](https://github.com/acgetchell/markov-chain-monte-carlo/pull/158)
- Bump the dependencies group with 2 updates [#156](https://github.com/acgetchell/markov-chain-monte-carlo/pull/156)
- Bump zizmorcore/zizmor-action in the github-actions group [#155](https://github.com/acgetchell/markov-chain-monte-carlo/pull/155)
- Bump the github-actions group with 3 updates [#152](https://github.com/acgetchell/markov-chain-monte-carlo/pull/152)
- Attach benchmark assets before publishing releases [#149](https://github.com/acgetchell/markov-chain-monte-carlo/pull/149)

### Dependencies

- Bump the github-actions group with 3 updates [#152](https://github.com/acgetchell/markov-chain-monte-carlo/pull/152)
  [`43429bf`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/43429bfa9f173484d5f91dc283b61936c49a2676)
- Bump zizmorcore/zizmor-action in the github-actions group [#155](https://github.com/acgetchell/markov-chain-monte-carlo/pull/155)
  [`85e176f`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/85e176f19c008ef929d416d3d6581d6244b8b4d5)
- Bump the dependencies group with 2 updates [#156](https://github.com/acgetchell/markov-chain-monte-carlo/pull/156)
  [`e47608f`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/e47608f2a95805d43670ac046cb4a29c7f357983)
- Bump the github-actions group with 5 updates [#158](https://github.com/acgetchell/markov-chain-monte-carlo/pull/158)
  [`81d1b24`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/81d1b2485f2cfbedf14997b25945a8a96cc95750)
- Bump the dependencies group with 4 updates [#159](https://github.com/acgetchell/markov-chain-monte-carlo/pull/159)
  [`f2a7e0d`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/f2a7e0d5df597d61e44d81c491c7765f418347da)

### Fixed

- Attach benchmark assets before publishing releases [#149](https://github.com/acgetchell/markov-chain-monte-carlo/pull/149)
  [`7c56e6c`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/7c56e6c709121069e5b99ebcb506e0fe198dd167)

  - Require an explicit stable tag backed by a mutable draft release.
  - Upload the durable Criterion baseline before publishing the draft.
  - Align release guidance and rollout boundaries with immutable releases.

### Maintenance

- [**breaking**] Raise the Rust baseline to 1.98.1 [#161](https://github.com/acgetchell/markov-chain-monte-carlo/pull/161)
  [`3d822a3`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/3d822a39d6233212ce91058c47b5abee0fe83707)

  - Align the MSRV, contributor toolchain, and Clippy configuration.
  - Update setup guidance and document the trait-object vtable miscompilation fix.
- [**breaking**] Pilot research-repo-tools for changelog maintenance
  [`6d5bf0e`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/6d5bf0e74659212ef64721a0944e3e2cbd9ed65d)

  - Pin research-repo-tools 0.1.0 from PyPI in the shared tooling group.
  - Delegate changelog generation, archiving, and release-note extraction to shared commands.
  - Remove duplicated changelog processing, date synchronization, and their dedicated tests.
  - Preserve included tooling constraints during dependency updates and skip deleted files in Semgrep.
  - Document migration policies, upstream gaps, and the package upgrade path.
- Complete shared changelog adoption with v0.1.1 [#162](https://github.com/acgetchell/markov-chain-monte-carlo/pull/162)
  [`54cbed4`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/54cbed4ae6716f9b9269762fa29a36bbc88d100f)

  - Pin research-repo-tools 0.1.1 with upstream date, archive, and content-preservation fixes.
  - Regenerate changelog history and archive the 0.1–0.3 series while retaining published release dates.
  - Add strict root and archive validation to the repository checks and CI workflow.
  - Document the adopted policies, resolved pilot gaps, and shared-package upgrade procedure.

## [0.4.2] - 2026-08-31

### ⚠️ Breaking Changes

- Rust 1.98.0 is required. Chain::step, Sampler::step, by-value observing methods, and Sampler's Iterator item now return Step telemetry;
  DiscreteProposalRatio::new requires forward and reverse endpoint weight sums; and Sampler serialization now emits the ChainCheckpoint shape.
- Replace DiscreteProposalRatio::new with from_endpoints,
- Replace `just performance-rerender` with `just performance-doc`. Run `just update-version "$TAG"` before
  `just changelog-unreleased "$TAG"` to prepare matching release metadata.

### Merged Pull Requests

- Streamline release preparation and benchmark publication [#147](https://github.com/acgetchell/markov-chain-monte-carlo/pull/147)
- Preserve sampler invariants and harden release tooling [#146](https://github.com/acgetchell/markov-chain-monte-carlo/pull/146)
- Harden MCMC contracts and release workflows [#145](https://github.com/acgetchell/markov-chain-monte-carlo/pull/145)
- Bump the dependencies group with 4 updates [#144](https://github.com/acgetchell/markov-chain-monte-carlo/pull/144)
- Bump the github-actions group with 4 updates [#143](https://github.com/acgetchell/markov-chain-monte-carlo/pull/143)
- Bump the dependencies group with 3 updates [#140](https://github.com/acgetchell/markov-chain-monte-carlo/pull/140)
- Bump astral-sh/setup-uv in the github-actions group [#139](https://github.com/acgetchell/markov-chain-monte-carlo/pull/139)
- Bump the dependencies group with 3 updates [#138](https://github.com/acgetchell/markov-chain-monte-carlo/pull/138)
- Bump the github-actions group with 5 updates [#137](https://github.com/acgetchell/markov-chain-monte-carlo/pull/137)

### Added

- [**breaking**] Harden MCMC contracts and release workflows [#145](https://github.com/acgetchell/markov-chain-monte-carlo/pull/145)
  [`baef874`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/baef8747db4d9afa4f77dd3229f9548fdd7faa30)

  - Expose by-value transition telemetry while avoiding telemetry overhead in bulk sampling paths.
  - Correct discrete proposal ratios for state-dependent family normalizers and serialize samplers as portable checkpoints.
  - Add durable release-performance evidence and managed dependency and tool updates.
  - Adopt Rust 1.98 and prohibit relaxed algebraic f64 operations.
- [**breaking**] Streamline release preparation and benchmark publication [#147](https://github.com/acgetchell/markov-chain-monte-carlo/pull/147)
  [`ef21cc6`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/ef21cc6430329cefa64497f7eed1ffa8e669bea3)

  - Synchronize versions, UTC dates, and active references from one release tag without upgrading dependencies.
  - Publish README tables and deterministic SVGs from retained measurements, with artifact integrity checks and rollback that preserves file contents.
  - Reuse the notebook extra for development dependencies.
  - Support self-repository Actions references with a narrow actionlint exception and refresh formatting and security tool pins.
  - Update release guidance and move AI tool acknowledgments into CONTRIBUTING.md.

### Dependencies

- Bump the github-actions group with 5 updates [#137](https://github.com/acgetchell/markov-chain-monte-carlo/pull/137)
  [`0e7a910`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/0e7a910e9e2d61bb7ac3aa347277502c7f064c31)
- Bump the dependencies group with 3 updates [#138](https://github.com/acgetchell/markov-chain-monte-carlo/pull/138)
  [`f7fa43d`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/f7fa43d85e396ca21a54b19874d1c082c3fe9046)
- Bump astral-sh/setup-uv in the github-actions group [#139](https://github.com/acgetchell/markov-chain-monte-carlo/pull/139)
  [`ee06e6e`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/ee06e6e287ab37b74c26b0d5d1da1055be2e0b8b)
- Bump the dependencies group with 3 updates [#140](https://github.com/acgetchell/markov-chain-monte-carlo/pull/140)
  [`11c1e0e`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/11c1e0e7f9a0c4086508fda579ef4eaca3444991)
- Bump the github-actions group with 4 updates [#143](https://github.com/acgetchell/markov-chain-monte-carlo/pull/143)
  [`5ea13d1`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/5ea13d15d160b4466f373a309ee0fffd990ebb61)
- Bump the dependencies group with 4 updates [#144](https://github.com/acgetchell/markov-chain-monte-carlo/pull/144)
  [`ae478aa`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/ae478aad57ff880ba9a3d40e46be57d9058757b6)

### Fixed

- [**breaking**] Preserve sampler invariants and harden release tooling [#146](https://github.com/acgetchell/markov-chain-monte-carlo/pull/146)
  [`e58126c`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/e58126c5a4794bbe60b57e9a35079c10bd6d9fb8)

  - Re-score current states before transitions and restore in-place and checked delayed state when post-mutation callbacks unwind.
  - Add Sampler::from_state, support unsized targets, and reduce thinning overhead without changing retained-step semantics.
  - Restore checkpoints between delayed example chunks while preserving caller-owned proposal and RNG state.
  - Expose scored and committed values in delayed-commit mismatch errors.
  - Record CPU-aware benchmark provenance, retain legacy sidecar support, stream benchmark progress, and identify non-reproducible legacy reports.
  - Validate tag versions against Cargo metadata, check current-release dates, and preserve useful Git failure diagnostics.
  - Resolve installed benchmark commands from the invocation directory and expose notebook dependencies through an installable extra.
  - Enforce notebook schemas and explicit paths, isolate execution caches, preserve artifact permissions, and unify Python environment setup.

## [0.4.1] - 2026-08-04

### ⚠️ Breaking Changes

- Rust 1.97.1 is now the minimum supported version.
- Thinned sampler methods now accept ThinningInterval and return their underlying error types, ThinningError is removed, and Step telemetry fields are private.
- `ProposalMut` now requires an `Info` type and mutable proposal hooks. In-place `step_mut` APIs return `Step<Info>` instead of `bool`, and shared `&P`
  forwarding is removed in favor of owned proposals or `&mut P`.

### Merged Pull Requests

- Harden v0.4.1 release preparation [#134](https://github.com/acgetchell/markov-chain-monte-carlo/pull/134)
- Bump ty from 0.0.63 to 0.0.64 in the dependencies group [#133](https://github.com/acgetchell/markov-chain-monte-carlo/pull/133)
- Bump python-multipart from 0.0.27 to 0.0.31 [#132](https://github.com/acgetchell/markov-chain-monte-carlo/pull/132)
- Bump tornado from 6.5.6 to 6.5.7 [#131](https://github.com/acgetchell/markov-chain-monte-carlo/pull/131)
- Bump pillow from 12.2.0 to 12.3.0 [#130](https://github.com/acgetchell/markov-chain-monte-carlo/pull/130)
- Bump cryptography from 47.0.0 to 50.0.0 [#129](https://github.com/acgetchell/markov-chain-monte-carlo/pull/129)
- Bump pydantic-settings from 2.14.0 to 2.14.2 [#128](https://github.com/acgetchell/markov-chain-monte-carlo/pull/128)
- Bump starlette from 1.0.1 to 1.3.1 [#127](https://github.com/acgetchell/markov-chain-monte-carlo/pull/127)
- Add release performance benchmarking [#126](https://github.com/acgetchell/markov-chain-monte-carlo/pull/126)
- Add structured in-place proposal telemetry [#125](https://github.com/acgetchell/markov-chain-monte-carlo/pull/125)
- Make thinning and step telemetry invariant-safe [#124](https://github.com/acgetchell/markov-chain-monte-carlo/pull/124)
- Require Python 3.14 and strengthen validation [#123](https://github.com/acgetchell/markov-chain-monte-carlo/pull/123)
- Bump the github-actions group across 1 directory with 10 updates [#122](https://github.com/acgetchell/markov-chain-monte-carlo/pull/122)
- Bump the dependencies group across 1 directory with 8 updates [#118](https://github.com/acgetchell/markov-chain-monte-carlo/pull/118)
- Require Rust 1.97.1 [#113](https://github.com/acgetchell/markov-chain-monte-carlo/pull/113)
- Bump astral-sh/setup-uv from 8.1.0 to 8.2.0 [#90](https://github.com/acgetchell/markov-chain-monte-carlo/pull/90)
- Prepare reviewer-facing MCMC documentation [#88](https://github.com/acgetchell/markov-chain-monte-carlo/pull/88)
- Bump actions/checkout from 6.0.2 to 6.0.3 [#87](https://github.com/acgetchell/markov-chain-monte-carlo/pull/87)
- Bump github/codeql-action from 4.36.0 to 4.36.1 [#86](https://github.com/acgetchell/markov-chain-monte-carlo/pull/86)
- Bump starlette in the uv group across 1 directory [#83](https://github.com/acgetchell/markov-chain-monte-carlo/pull/83)

### Added

- [**breaking**] Add structured in-place proposal telemetry [#125](https://github.com/acgetchell/markov-chain-monte-carlo/pull/125)
  [`dc1820f`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/dc1820f7178371e362b4fea9304c344b957a5502)

  - Return invariant-bearing `Step<Info>` metadata from in-place single-step APIs.
  - Support stateful proposal hooks with rollback of transition-relevant proposal state.
  - Skip discarded telemetry construction throughout bulk sampling.
  - Keep mutable detailed-balance trials rollback-safe and document fixed-kernel requirements.
- Add release performance benchmarking [#126](https://github.com/acgetchell/markov-chain-monte-carlo/pull/126)
  [`5979880`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/59798807820e0b5926b0fc7374aee9b924607591)

  - Add fixed-seed Criterion workloads and Just commands for local, saved-baseline, release-asset, and curated comparisons.
  - Publish durable release baselines with provenance metadata and archive prior curated reports.
  - Enforce release-pair and report invariants while documenting the prospective two-release rollout.
  - Group and validate the Just command surface with Justfile-sourced workflow pins.

### Changed

- [**breaking**] Make thinning and step telemetry invariant-safe [#124](https://github.com/acgetchell/markov-chain-monte-carlo/pull/124)
  [`0b1e6a9`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/0b1e6a91a5d991431b62f3d3ee4bd1bc0bb1501d)

  - Parse positive thinning intervals once with ThinningInterval and propagate underlying run errors directly.
  - Keep Step telemetry internally consistent through private fields and read-only accessors.
  - Add repository guardrails against raw thinning parameters and public telemetry fields.

### Dependencies

- Bump actions/checkout from 6.0.2 to 6.0.3 [#87](https://github.com/acgetchell/markov-chain-monte-carlo/pull/87)
  [`7817555`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/781755596d63231f907aac87f4571a823b2c898d)
- Bump github/codeql-action from 4.36.0 to 4.36.1 [#86](https://github.com/acgetchell/markov-chain-monte-carlo/pull/86)
  [`40a43a9`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/40a43a96aa89de1f962d9f881187ca0bf5c85ba7)
- Bump starlette in the uv group across 1 directory [#83](https://github.com/acgetchell/markov-chain-monte-carlo/pull/83)
  [`1384a58`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/1384a58784b4e0b1bf38d8813033820c6c27c820)
- Bump astral-sh/setup-uv from 8.1.0 to 8.2.0 [#90](https://github.com/acgetchell/markov-chain-monte-carlo/pull/90)
  [`c7e938c`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/c7e938ce1c53eee1d3d7dcbf55d8a8c3d4200def)
- Bump the dependencies group across 1 directory with 8 updates [#118](https://github.com/acgetchell/markov-chain-monte-carlo/pull/118)
  [`1037c4e`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/1037c4e68d61de0bac0cf651c47a3646bfa24ec6)
- Bump the github-actions group across 1 directory with 10 updates [#122](https://github.com/acgetchell/markov-chain-monte-carlo/pull/122)
  [`1f07a84`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/1f07a8493c039155d64563e5e21b667eb0a7e4a8)
- Bump pydantic-settings from 2.14.0 to 2.14.2 [#128](https://github.com/acgetchell/markov-chain-monte-carlo/pull/128)
  [`1f3a18e`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/1f3a18e160dea620cd4b97b0363accc07b4a97e9)
- Bump python-multipart from 0.0.27 to 0.0.31 [#132](https://github.com/acgetchell/markov-chain-monte-carlo/pull/132)
  [`11a9632`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/11a9632eb4e4241867011013f8de2bac152c9357)
- Bump tornado from 6.5.6 to 6.5.7 [#131](https://github.com/acgetchell/markov-chain-monte-carlo/pull/131)
  [`ba37bb0`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/ba37bb0981629f4d7999376d65e3d1995f08d43f)
- Bump starlette from 1.0.1 to 1.3.1 [#127](https://github.com/acgetchell/markov-chain-monte-carlo/pull/127)
  [`4a384f8`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/4a384f8368d1eb57f2160eceee052a2237821ae6)
- Bump cryptography from 47.0.0 to 50.0.0 [#129](https://github.com/acgetchell/markov-chain-monte-carlo/pull/129)
  [`61568a9`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/61568a9e30dda0af87af8c4ed8664755d717c150)
- Bump pillow from 12.2.0 to 12.3.0 [#130](https://github.com/acgetchell/markov-chain-monte-carlo/pull/130)
  [`341e6bd`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/341e6bd68baf8f1b93fae939e1aee47c4ecfcc3b)
- Bump ty from 0.0.63 to 0.0.64 in the dependencies group [#133](https://github.com/acgetchell/markov-chain-monte-carlo/pull/133)
  [`6994d83`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/6994d835a5ee1897727183d740ea0cf9cedbd783)

### Documentation

- Prepare reviewer-facing MCMC documentation [#88](https://github.com/acgetchell/markov-chain-monte-carlo/pull/88)
  [`cd861ee`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/cd861ee7e0b8604e31e8e19bf39e12bc26298222)

#### docs: prepare reviewer-facing MCMC documentation

- Add a reviewer guide that points scientific and engineering reviewers to the README, scientific basis, proposal validation, roadmap, references, and local
  checks.
- Refocus the README and scientific basis docs around the Metropolis-Hastings contract, externally supplied regularizer terms, and explicit non-claims for
  convergence or learned-proposal training.
- Keep crate-level rustdoc focused on programming contracts while moving project orientation into the README and topic docs.
- Clarify example comments for normal, Ising, and additive-target workflows.
- Pin the markdown and spelling tool versions used by the validation gate.

#### Changed: Update internal RUMDL and TYPOS tooling in CI

#### Changed: Correct example branch type in contributing guide

### Fixed

- Harden v0.4.1 release preparation [#134](https://github.com/acgetchell/markov-chain-monte-carlo/pull/134)
  [`56d2cb5`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/56d2cb5a6a8e9e5909fe2c6da2c76704f4f81004)

  - Preserve existing tags and generated release artifacts when replacement fails.
  - Run release tests with all features and filter development-dependency noise from changelogs.
  - Trim crate contents and align audit, example, and pre-v1 compatibility guidance.

### Maintenance

- [**breaking**] Require Rust 1.97.1 [#113](https://github.com/acgetchell/markov-chain-monte-carlo/pull/113)
  [`55323fc`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/55323fcb47ef154e35113e46b15c9de553b2b8ab)

  - Raise the crate MSRV and contributor toolchain to Rust 1.97.1.
  - Adopt Cargo-owned Clippy warning denial and update dprint and rumdl pins.
  - Set explicit GitHub release titles from version tags.
- Group GitHub Actions updates [`d07bb85`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/d07bb8501ffd767374cceac281d16096d7fd0e78)

  - Consolidate action version bumps into a single weekly Dependabot pull request.
- Automate Dependabot reviews and merges [`b86ca7f`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/b86ca7fe572df8a60a98ec64c590ec7c3c3339e1)

  - Request CodeRabbit reviews and enable guarded squash auto-merge for Dependabot PRs.
- Stagger weekly Dependabot updates [`dcc9a89`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/dcc9a891774d6227324b136861d0c0b646fbbf88)

  - Schedule GitHub Actions, Cargo, and uv updates on Wednesday mornings in Pacific time.
- Finalize review enforcement and refresh badges
  [`d62debd`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/d62debd0347c737bdbd6f426ae90a46d0779e0af)

  - Fail the CodeRabbit commit status when its review does not pass.
  - Serve DOI, crate, download, and license badges through Badgen.
  - Align the local zizmor guard with version 1.29.0.
- Require Python 3.14 and strengthen validation [#123](https://github.com/acgetchell/markov-chain-monte-carlo/pull/123)
  [`c15f859`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/c15f859a9fd5d16e15110e9091d55833feefa7ba)

  - Require Python 3.14 and uv 0.12.1 for locked tooling environments.
  - Parse support-script and notebook inputs into validated types, with source-safe headless notebook execution.
  - Flatten CI into focused validators with a shared release-profile Rust test bucket.
  - Replace the benchmark result helper with the postfix `OrAbort` trait.
- Align dependency automation and tool pins [`771ae2a`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/771ae2acf38fb84bcadfa6aee1dc30ae08905180)

  - Group version and security updates separately across managed ecosystems.
  - Match just, rumdl, and typos-cli pins to the local toolchain.

## [0.4.0] - 2026-05-30

### ⚠️ Breaking Changes

- Public telemetry and detailed-balance structs are now non-exhaustive, so downstream callers should construct them through the provided constructors instead of
  direct struct literals.
- Removed DetailedBalanceError::InvalidLogAcceptanceRatio and changed impossible-endpoint detailed-balance checks from an invalid-acceptance error into a
  zero-flow report.
- DetailedBalanceConfig fields are now private and DetailedBalanceConfig::new returns Result. Detailed-balance batch helpers now return
  DetailedBalanceBatchReport directly instead of Result.

#### test: cover delayed ratio validator diagnostics

- Assert DiscreteProposalRatio accessors and typed Display messages for invalid weights.
- Exercise the delayed proposal fixture info and commit path used by detailed-balance validation.
- Removed infallible statistics ingestion APIs, including push, push_unchecked, Extend&lt;f64&gt;, FromIterator&lt;f64&gt;, and Sum&lt;f64&gt; implementations
  for OnlineStats and BinningAnalysis. Callers must use try_push, try_extend, or try_from_iter and handle StatisticsError.
- DelayedStep/Step no longer expose accepted or proposed fields; use step.outcome plus StepOutcome::is_accepted or StepOutcome::has_proposal instead.

#### feat(sampler): add delayed chunk telemetry observer [#61](https://github.com/acgetchell/markov-chain-monte-carlo/pull/61)

- Add `Sampler::run_delayed_chunk_observing` so delayed runs can stream per-step telemetry with the post-step state while preserving resumable chunk
  checkpoints.
- Document the delayed telemetry workflow and add a runnable example for recording proposal-family and rejection-reason statistics across chunks.

### Merged Pull Requests

- Enforce fallible public Rust examples [#62](https://github.com/acgetchell/markov-chain-monte-carlo/pull/62)
  [#80](https://github.com/acgetchell/markov-chain-monte-carlo/pull/80)
- Add trace recording diagnostics [#79](https://github.com/acgetchell/markov-chain-monte-carlo/pull/79)
- Add additive target composition [#78](https://github.com/acgetchell/markov-chain-monte-carlo/pull/78)
- Expose delayed-step outcome telemetry [#61](https://github.com/acgetchell/markov-chain-monte-carlo/pull/61)
  [#77](https://github.com/acgetchell/markov-chain-monte-carlo/pull/77)
- Expose resumable chunked runs [#60](https://github.com/acgetchell/markov-chain-monte-carlo/pull/60)
  [#76](https://github.com/acgetchell/markov-chain-monte-carlo/pull/76)
- Enforce parse-don't-validate invariants [#75](https://github.com/acgetchell/markov-chain-monte-carlo/pull/75)
- Add delayed proposal ratio helpers [#71](https://github.com/acgetchell/markov-chain-monte-carlo/pull/71)
- Validate delayed commits after acceptance [#70](https://github.com/acgetchell/markov-chain-monte-carlo/pull/70)
- Bump Rust MSRV to 1.96.0 [#65](https://github.com/acgetchell/markov-chain-monte-carlo/pull/65)
  [#69](https://github.com/acgetchell/markov-chain-monte-carlo/pull/69)
- Cache cargo-installed CI tools [#68](https://github.com/acgetchell/markov-chain-monte-carlo/pull/68)
- Use rumdl and dprint for checks [#53](https://github.com/acgetchell/markov-chain-monte-carlo/pull/53)
  [#67](https://github.com/acgetchell/markov-chain-monte-carlo/pull/67)
- Add constructors for fallible stats and public reports [#66](https://github.com/acgetchell/markov-chain-monte-carlo/pull/66)
- Harden repository security and tooling validation [#63](https://github.com/acgetchell/markov-chain-monte-carlo/pull/63)
- Bump idna in the uv group across 1 directory [#58](https://github.com/acgetchell/markov-chain-monte-carlo/pull/58)
- Bump codecov/codecov-action from 6.0.0 to 6.0.1 [#55](https://github.com/acgetchell/markov-chain-monte-carlo/pull/55)
- Bump taiki-e/install-action from 2.77.6 to 2.79.2 [#54](https://github.com/acgetchell/markov-chain-monte-carlo/pull/54)
- Bump taiki-e/install-action from 2.77.0 to 2.77.6 [#52](https://github.com/acgetchell/markov-chain-monte-carlo/pull/52)
- Bump actions-rust-lang/setup-rust-toolchain [#51](https://github.com/acgetchell/markov-chain-monte-carlo/pull/51)
- Bump urllib3 in the uv group across 1 directory [#50](https://github.com/acgetchell/markov-chain-monte-carlo/pull/50)
- Bump taiki-e/install-action from 2.75.23 to 2.77.0 [#49](https://github.com/acgetchell/markov-chain-monte-carlo/pull/49)

### Added

- [**breaking**] Add constructors for fallible stats and public reports [#66](https://github.com/acgetchell/markov-chain-monte-carlo/pull/66)
  [`fee14b4`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/fee14b4ac75e2a0f8ede899541c10109d6ebd990)

  - Add checked bulk constructors for streaming statistics so callers can reject non-finite samples without retaining partial accumulator state.
  - Add constructors for detailed-balance config, reports, failures, batches, and delayed transitions before making those public structs non-exhaustive.
  - Reuse compiled example binaries during local validation to avoid rebuilding examples twice.
  - Refresh the pre-1.0 roadmap around v0.4.0, adaptive diagnostics, learned proposals, and portability.
- [**breaking**] Validate delayed commits after acceptance [#70](https://github.com/acgetchell/markov-chain-monte-carlo/pull/70)
  [`1c4654b`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/1c4654bf9cc94a9bc37131858183854620e46692)

  - Add checked delayed-commit stepping for Chain and Sampler so proposal authors can verify that committed states match the scored plan.
  - Report committed-state NaN, positive-infinity, and score-mismatch failures with dedicated McmcError variants while restoring the previous chain state.
  - Treat undefined detailed-balance acceptance ratios as zero acceptance so impossible bidirectional transitions produce balanced zero-flow reports.
  - Expose unchecked streaming-statistics push methods for callers that already validate measurement streams.
  - Add benchmark coverage for streaming observations into OnlineStats and BinningAnalysis.
- [**breaking**] Add delayed proposal ratio helpers [#71](https://github.com/acgetchell/markov-chain-monte-carlo/pull/71)
  [`ed876f6`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/ed876f67fcd1f3ed0fa79242ba5a10c8ccd7a489)

#### feat!: add delayed proposal ratio helpers

- Add DiscreteProposalRatio for weighted move-family and valid-site Hastings corrections in delayed proposals.
- Document delayed valid-site multiplicities and expose the ratio helper through the public API and scoped preludes.
- Validate DetailedBalanceConfig at construction so detailed-balance checks operate on accepted configuration values.
- Add validator property tests and document the repository convention for proptest integration files.
- Expose resumable chunked runs [#60](https://github.com/acgetchell/markov-chain-monte-carlo/pull/60)
  [#76](https://github.com/acgetchell/markov-chain-monte-carlo/pull/76)
  [`bbcad08`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/bbcad08c22216c4738227f8ae802495e8024796a)

  - Add checkpoint accessors on Sampler so callers can inspect or persist continuation state without unwrapping the inner chain.
  - Add chunked by-value, in-place, and delayed-commit run helpers that preserve sampler counters and RNG streams across repeated chunks.
  - Document chunked sampling as the ergonomic path for workflows that choose each next step budget from the updated state.
- [**breaking**] Expose delayed-step outcome telemetry [#61](https://github.com/acgetchell/markov-chain-monte-carlo/pull/61)
  [#77](https://github.com/acgetchell/markov-chain-monte-carlo/pull/77)
  [`6f846ff`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/6f846ff25e44895f9b6573352a365121127b748f)

#### feat!: expose delayed-step outcome telemetry [#61](https://github.com/acgetchell/markov-chain-monte-carlo/pull/61)

- Replace delayed-step accepted/proposed booleans with StepOutcome so step state is represented by one invariant-bearing value.
- Add StepRejectionReason and DelayedProposal::no_plan_info for no-plan delayed steps that still need domain-specific telemetry.
- Re-export delayed telemetry through the root API and delayed prelude, and update docs and benchmarks to use outcome-based access.
- Add additive target composition [#78](https://github.com/acgetchell/markov-chain-monte-carlo/pull/78)
  [`55140e0`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/55140e086b4f4b724372b797b6856b41b8c7d1f1)

#### feat: add additive target composition

- Add AdditiveTarget for composing model and bias log-weight terms in the target distribution.
- Re-export the adapter through the crate root and scoped preludes for by-value, in-place, delayed, and testing workflows.
- Document additive energy/action semantics and keep proposal-ratio corrections separate from target bias terms.
- Add trace recording diagnostics [#79](https://github.com/acgetchell/markov-chain-monte-carlo/pull/79)
  [`adbb84a`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/adbb84a60a509eaba0d37ac298f2a6196b6495ee)

#### feat: add trace recording diagnostics

- Add ChainId, TraceStepOutcome, TraceRecord, Trace, and TraceRecorder for reusable numeric MCMC traces with accept/reject metadata and CSV export.
- Re-export trace diagnostics through the root API and scoped preludes so examples, doctests, benchmarks, and downstream users can import them consistently.
- Extend the Ising example with energy and magnetization trace export, add an individual just example recipe, and document the notebook workflow for
  inspecting the generated CSV.

#### ci: add notebook validation checks

- Add notebook linting and in-memory execution recipes so tracked notebooks are checked in local validation and full CI simulation.
- Add notebook runtime dependencies and a reusable check-notebooks helper for JSON, code-cell syntax, and nbclient execution.
- Harden the Ising trace notebook with clearer missing-file errors and proposed-move acceptance-rate handling.
- Cover notebook checking and subprocess utility behavior with Python tests.

#### fix(ci): use ASCII notebook checker output

- Replace Unicode status markers with ASCII text so notebook checks run on Windows consoles.

#### fix(tooling): exclude notebook checkpoints from discovery

- Skip `.ipynb_checkpoints` copies in `discover_notebooks` so checkpoint duplicates are never linted or executed.
- Correct the `paths` CLI help text to describe discovery via `discover_notebooks` instead of tracked files.
- Treat `check_notebooks` as a first-party module for Ruff import sorting.

### Changed

- [**breaking**] Enforce parse-don't-validate invariants [#75](https://github.com/acgetchell/markov-chain-monte-carlo/pull/75)
  [`85ee3e0`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/85ee3e036197b6ced9f7559cc394909a7949b4d5)

  - Require OnlineStats and BinningAnalysis to ingest raw f64 samples through fallible APIs, with private finite-sample evidence carried through accumulator
    internals.
  - Store nonzero detailed-balance and proposal-ratio counts as refined types so zero is rejected at construction instead of rechecked downstream.
  - Add Semgrep guardrails for unchecked APIs, infallible statistics ingestion, raw invariant fields, and public unit validators.
  - Refresh docs, property tests, and crate metadata for the stricter public API surface.

### Dependencies

- Bump taiki-e/install-action from 2.75.23 to 2.77.0 [#49](https://github.com/acgetchell/markov-chain-monte-carlo/pull/49)
  [`f29d788`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/f29d78890a110daf70a351ba1d565d75a318c966)
- Bump taiki-e/install-action from 2.77.0 to 2.77.6 [#52](https://github.com/acgetchell/markov-chain-monte-carlo/pull/52)
  [`98d8bf0`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/98d8bf06b993874409ad70152e76f09724af9a9b)
- Bump actions-rust-lang/setup-rust-toolchain [#51](https://github.com/acgetchell/markov-chain-monte-carlo/pull/51)
  [`c508494`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/c50849495da78fbb849ba9ed732a6b925bcdf047)
- Bump urllib3 in the uv group across 1 directory [#50](https://github.com/acgetchell/markov-chain-monte-carlo/pull/50)
  [`0520cc0`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/0520cc0a7dc88934d108657bc4c01211da00d58f)
- Bump idna in the uv group across 1 directory [#58](https://github.com/acgetchell/markov-chain-monte-carlo/pull/58)
  [`88c3618`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/88c3618ec0f040d00125fac0dd0849c2ba3216df)
- Bump codecov/codecov-action from 6.0.0 to 6.0.1 [#55](https://github.com/acgetchell/markov-chain-monte-carlo/pull/55)
  [`e9c1b4a`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/e9c1b4a95a04aaed30f53a9be43afe2d20200874)
- Bump taiki-e/install-action from 2.77.6 to 2.79.2 [#54](https://github.com/acgetchell/markov-chain-monte-carlo/pull/54)
  [`c0230a9`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/c0230a92b387c7463e2c6c44b017ff662814b9ff)

### Fixed

- Escape changelog angle brackets for GitHub rendering
  [`82014b4`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/82014b424d0b5be37c93770e2cdd925ba351ca25)

  - Escape generated changelog commit text so Rust generics are not parsed as HTML tags
  - Regenerate CHANGELOG.md to preserve visible generic type names and version footer links
- Escape changelog angle brackets for GitHub rendering
  [`f712f15`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/f712f1515d8a215359a2551c598d8fa0de974ef5)

  Angle brackets in changelog entries, such as Rust generics (``),
  were parsed as HTML tags by Markdown renderers like GitHub.
  This fix escapes those characters during changelog generation,
  ensuring correct display. The changelog was regenerated,
  also incorporating repository security enhancements from #63.
- Clean regenerated changelog entries [`4147806`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/414780606cc68da0f8a5b99de821af89a94e0671)

  - Repair the historical changelog body for escaped Rust generic examples.
  - Filter changelog-only body noise from generated release notes.
  - Regenerate CHANGELOG.md with the corrected commit preprocessing.
- Enforce fallible public Rust examples [#62](https://github.com/acgetchell/markov-chain-monte-carlo/pull/62)
  [#80](https://github.com/acgetchell/markov-chain-monte-carlo/pull/80)
  [`ca2ba16`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/ca2ba169385a8fa99aced9100a392fb402fa29bf)

#### fix: enforce fallible public Rust examples [#62](https://github.com/acgetchell/markov-chain-monte-carlo/pull/62)

- Replace unwrap and expect flows in doctests, examples, and benchmarks with typed error propagation or contextual benchmark failure handling.
- Add Semgrep guardrails and fixtures that catch unwrap and expect usage in public doctests, examples, and benchmarks.
- Keep generated documentation and tooling metadata aligned with the updated validation surface.

#### fix(semgrep): catch doctest unwrap continuations [#62](https://github.com/acgetchell/markov-chain-monte-carlo/pull/62)

- Detect unwrap and expect calls on rustfmt-style continuation lines in doctests.
- Cover line and block doctest fixtures for public panic-based example flows.
- Document the no-unwrap doctest, example, and benchmark convention.

### Maintenance

- Make changelog generation commit-driven [`a0441a1`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/a0441a139e14ff17ce961f2630fb6c60eff7e665)

  - Regenerate release notes from commit ranges instead of annotated tag messages
  - Preserve Markdown headings when creating annotated release tags
  - Add Zenodo DOI metadata to the README, references, and citation file
- Harden repository security and tooling validation [#63](https://github.com/acgetchell/markov-chain-monte-carlo/pull/63)
  [`a861f86`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/a861f863a5acae1d43a0ab006231c4aa949bb390)

#### ci: harden repository security and tooling validation

- Add zizmor and repository-owned Semgrep SARIF workflows for GitHub Actions and project-rule scanning.
- Replace Codacy with local and GitHub-native security signals, including action pinning, allowlist, and version-comment rules.
- Run full CI through just across Linux, macOS, and Windows with uv-managed Python tools and pinned Cargo-installed tooling.
- Move Rust unit and integration tests to cargo-nextest while keeping doctests on cargo test --doc.
- Add a security policy and document the updated setup, line-length, and validation workflow.

#### ci: allow Rust cache action in workflow policy

- Add swatinem/rust-cache to the repository-owned GitHub Actions allowlist so the Rust toolchain action's cache helper is permitted.
- Cover the allowlist entry in the Semgrep workflow policy fixture.
- ci: simplify Semgrep SARIF concurrency key
- Refresh managed tooling installs [`27bcae6`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/27bcae6faad4e0197241ea0ace49f88cdc65ea25)

  - Replace taiki-e/install-action with cached Cargo installs for audit and coverage workflow tools.
  - Update uv-managed development tools, including Semgrep, Ruff, Ty, and the workflow uv version.
  - Refresh the serde_json dev dependency and remove taiki-e from the repository-owned Actions allowlist.
  - Update changelog
- Clear code scanning workflow findings [`cf946db`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/cf946dbd2e2f4b44bc41343692d98459954ced4d)

  - Align the Clippy SARIF job with the local Clippy policy so dependency-version noise is not uploaded as code scanning alerts.
  - Refresh CodeQL action SHA pins and version comments across CodeQL, Clippy, and Semgrep SARIF workflows.
- Use rumdl and dprint for checks [#53](https://github.com/acgetchell/markov-chain-monte-carlo/pull/53)
  [#67](https://github.com/acgetchell/markov-chain-monte-carlo/pull/67)
  [`d23029c`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/d23029c9763067a1ad5510cd80dcb19c9c7fdc1e)

  - Replace dprint Markdown formatting and yamllint validation with rumdl Markdown checks and dprint Pretty YAML checks.
  - Install and verify rumdl through Cargo-managed local setup and CI tooling.
  - Keep non-Rust formatter widths at 160 columns across Markdown, TOML, YAML, and Python.
  - Add a Semgrep fixture that keeps contributor docs ordering non-mutating checks before mutating fixes.
- Cache cargo-installed CI tools [#68](https://github.com/acgetchell/markov-chain-monte-carlo/pull/68)
  [`ebd98de`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/ebd98de9fb8725f2f261a52ab6c4c6e77ee7a9d7)

#### ci: cache cargo-installed CI tools

- Install Cargo-managed CI tools through taiki-e/cache-cargo-install-action instead of rebuilding them in each matrix job.
- Keep full just ci coverage on Linux, macOS, and Windows while exposing smaller CI subsets for timing and local diagnosis.
- Compile benchmark harnesses with all crate features enabled and allow the pinned cache action in the repository workflow policy.
- Bump Rust MSRV to 1.96.0 [#65](https://github.com/acgetchell/markov-chain-monte-carlo/pull/65)
  [#69](https://github.com/acgetchell/markov-chain-monte-carlo/pull/69)
  [`ba7beb2`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/ba7beb2a65b4d506374569d19a77b336ad74e53d)

#### build: bump Rust MSRV to 1.96.0 [#65](https://github.com/acgetchell/markov-chain-monte-carlo/pull/65)

- Pin Cargo, rustup, Clippy, and contributor docs to Rust 1.96.0.
- Adopt 1.96-compatible numeric updates with fused multiply-add in statistics, examples, and benchmarks.
- Refresh typed error assertions and doctests to use `std::assert_matches!`.
- Clarify that LLVM coverage tools are installed by setup and Codecov rather than the default pinned toolchain.

#### ci: add Windows taplo install fallback [#65](https://github.com/acgetchell/markov-chain-monte-carlo/pull/65)

- Preserve the cached cargo install action as the primary tool installation path across operating systems.
- Fall back to direct taplo installation on Windows only when the cached install step fails.

## Archives

Older releases are archived by minor series:

- [0.3.x](docs/archives/changelog/0.3.md)
- [0.2.x](docs/archives/changelog/0.2.md)
- [0.1.x](docs/archives/changelog/0.1.md)

[Unreleased]: https://github.com/acgetchell/markov-chain-monte-carlo/compare/v0.4.2...HEAD
[0.4.2]: https://github.com/acgetchell/markov-chain-monte-carlo/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/acgetchell/markov-chain-monte-carlo/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/acgetchell/markov-chain-monte-carlo/compare/v0.3.0...v0.4.0
