# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### ⚠️ Breaking Changes

- Release benchmark dispatch uses the required tag input, and release-performance recipes use inferred preparation and complete-run declarations.
- Rust 1.100.0 is required. Release operators must retain the Step 1 acquisition and now run release-upload after release-publish.
- Corrected arithmetic can change seeded sampling trajectories and diagnostic values compared with earlier releases.
- Preserve MCMC numerical precision and adopt RRT 0.3.2 [#239](https://github.com/acgetchell/markov-chain-monte-carlo/pull/239)

### Merged Pull Requests

- Preserve MCMC numerical precision and adopt RRT 0.3.2 [#239](https://github.com/acgetchell/markov-chain-monte-carlo/pull/239)
- Simplify release recovery and adopt RRT 0.3.1 [#236](https://github.com/acgetchell/markov-chain-monte-carlo/pull/236)
- Stabilize sampler arithmetic and adopt RRT 0.3.0 [#234](https://github.com/acgetchell/markov-chain-monte-carlo/pull/234)
- Reuse retained evidence for immutable releases [#233](https://github.com/acgetchell/markov-chain-monte-carlo/pull/233)
- Bump the github-actions group with 3 updates [#232](https://github.com/acgetchell/markov-chain-monte-carlo/pull/232)
- Adopt RRT 0.2.8 and harden diagnostic inputs [#230](https://github.com/acgetchell/markov-chain-monte-carlo/pull/230)
- Allow Dependabot updates to release preparation [#228](https://github.com/acgetchell/markov-chain-monte-carlo/pull/228)
- Bump astral-sh/setup-uv [#226](https://github.com/acgetchell/markov-chain-monte-carlo/pull/226)
- Adopt shared release tooling and validation [#224](https://github.com/acgetchell/markov-chain-monte-carlo/pull/224)
- Bump the github-actions group across 1 directory with 5 updates [#219](https://github.com/acgetchell/markov-chain-monte-carlo/pull/219)

### Dependencies

- Bump the github-actions group across 1 directory with 5 updates [#219](https://github.com/acgetchell/markov-chain-monte-carlo/pull/219)
  [`1371beb`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/1371bebc61bc672401e2d94deb462c040ba876f3)
- Bump astral-sh/setup-uv [#226](https://github.com/acgetchell/markov-chain-monte-carlo/pull/226)
  [`d39678e`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/d39678e5479b9b90ca957f19471db3a8113e8603)
- Bump the github-actions group with 3 updates [#232](https://github.com/acgetchell/markov-chain-monte-carlo/pull/232)
  [`0ae7c9d`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/0ae7c9de9e999132b801c26eede560b5b5fbc34b)

### Fixed

- [**breaking**] Stabilize sampler arithmetic and adopt RRT 0.3.0 [#234](https://github.com/acgetchell/markov-chain-monte-carlo/pull/234)
  [`b10ad50`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/b10ad50a414fafec96e7885676bb9916450fe0c7)

#### fix!: stabilize sampler arithmetic and adopt RRT 0.3.0

- Preserve finite Hastings corrections across cancellation and overflow, and share the acceptance arithmetic with detailed-balance checks.
- Preserve low-order mean terms in streaming variance and binning estimates.
- Delegate authenticated draft preparation and release visibility to research-repo-tools 0.3.0. Bind publication to the captured release and asset identities,
  and include the setup composite in the Actions allowlist gate.
- Refresh locked Rust dependencies and the Semgrep and Markdown tool pins.

---

- Simplify release recovery and adopt RRT 0.3.1 [#236](https://github.com/acgetchell/markov-chain-monte-carlo/pull/236)
  [`14f8c58`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/14f8c58fa932769c1ba0f33f902c65c0af92c1d9)

#### fix: simplify release recovery and adopt RRT 0.3.1

- Resume releases from retained benchmark and CI evidence and tag the reviewed commit with managed Cargo.
- Use shared scanner selection and reporting and upgrade Semgrep's wcmatch dependency to a patched release.
- Harden notebook diagnostics and document tracing across sampling workers.

#### fix: reject contradictory Ising timing cadence

- Reject full-production timing metadata that conflicts with observed trace gaps and preserve unavailable estimates for single draws.
- Refresh the Rust and Python dependency locks and the managed coverage tool.

#### fix: verify every retained release report output

- Report missing or stale run outputs before suggesting release evidence verification.

---

- [**breaking**] Preserve MCMC numerical precision and adopt RRT 0.3.2 [#239](https://github.com/acgetchell/markov-chain-monte-carlo/pull/239)
  [`d513eff`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/d513eff5cb7916fbc05bb1976621e269297a7c47)

#### docs: clarify release preparation and restore historical notes

- Document the package dry runs, review corrections, and release PR handoff.
- Explain how to verify the measured Cargo files and when to collect fresh timings.
- Restore historical changelog detail and support both benchmark snapshot formats in release checks.

#### fix!: preserve small MCMC corrections and adopt RRT 0.3.2

- Preserve near-equal Hastings corrections, tiny standard errors, and quantiles across the normal/subnormal boundary.
- Correct variance estimates in the examples and verify quantile intervals against independent references.
- Use shared benchmark inventory validation, preserve historical reports at v0.5.2, and verify candidate evidence before publication.

#### fix!: preserve subnormal quantiles and correct release recovery [#239](https://github.com/acgetchell/markov-chain-monte-carlo/pull/239)

- Preserve representable type-7 quantiles when interpolation crosses zero near the normal/subnormal boundary.
- Document when standard errors remain positive despite variance underflow and verify zero Hastings correction for equally probable weighted moves.
- Require fresh benchmark acquisition for changes outside declared outputs and reproduce the committed README snapshot from the original measured revisions.

#### fix: round quantile interpolation across the subnormal boundary [#239](https://github.com/acgetchell/markov-chain-monte-carlo/pull/239)

- Preserve interpolation residuals through final ties-to-even rounding when opposite-sign endpoints have an inexact span.
- Cover exact and near-midpoint cutoffs on both sides of the normal boundary with independent regression cases.

---

- Keep archived benchmark data compatible with release checks
  [`edeb3fe`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/edeb3fe9fd11ea06c2c13e328978a4cb1f39f60b)

  - Verify the raw benchmark payload through release preflight and keep its report and evidence manifest in CodeRabbit review.
  - Limit Gitleaks exceptions to encoded Criterion outlier statistics in retained run files.
  - Document the generated benchmark layout and update the Python packaging lock to 26.4.

### Maintenance

- [**breaking**] Adopt shared release tooling and validation [#224](https://github.com/acgetchell/markov-chain-monte-carlo/pull/224)
  [`24ee734`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/24ee73403b3d289f6b8f292c69258544454a9116)
  Closes [#216](https://github.com/acgetchell/markov-chain-monte-carlo/issues/216).

#### ci!: adopt shared release tooling and validation

- Adopt research-repo-tools 0.2.5 and updated managed tools while retaining Semgrep 1.183.0 through a bounded compatibility adapter.
- Allow both reviewed authentication action commits so Dependabot upgrades can land independently.
- Share citation-backed changelog dates and reject incomplete combined R-hat fixture inventories before oracle reproduction.
- Run repository policy checks once while preserving native validation and signed exact-source release evidence across publication approval.
- Infer release-performance identities from prepared metadata and retain complete measurements separately from frozen legacy evidence.
- Prepare verified GitHub drafts automatically after tag pushes and dispatch release benchmarks at the exact reviewed tag.

---

- Allow Dependabot updates to release preparation [#228](https://github.com/acgetchell/markov-chain-monte-carlo/pull/228)
  [`4db292b`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/4db292b029f6f5d9cd950b356f21d31eedb953b1)

  - Include release preparation in the automatic Dependabot approval policy.
  - Keep registry publication updates subject to maintainer review and align policy coverage and documentation.

- Adopt RRT 0.2.8 and harden diagnostic inputs [#230](https://github.com/acgetchell/markov-chain-monte-carlo/pull/230)
  [`5345603`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/5345603ba22482c31fe575dc55e2a042e57f6133)
  Closes [#212](https://github.com/acgetchell/markov-chain-monte-carlo/issues/212), [#218](https://github.com/acgetchell/markov-chain-monte-carlo/issues/218),
  [#222](https://github.com/acgetchell/markov-chain-monte-carlo/issues/222), [#225](https://github.com/acgetchell/markov-chain-monte-carlo/issues/225),
  [#227](https://github.com/acgetchell/markov-chain-monte-carlo/issues/227).

#### chore: adopt RRT 0.2.8 and harden diagnostic inputs

- Use published shared scanner, dependency provenance and isolated release APIs; remove the temporary Semgrep adapter.
- Cache managed Cargo compilation separately from installed tools and overlap independent CI validation while retaining exact-main evidence gates.
- Declare counterbalanced benchmark trials with preserved workloads, raw provenance and an independent README trial selection.
- Represent step telemetry and accepted ESS estimators with algebraic data types and clarify public API contracts.
- Parse scientific fixtures and notebook inputs before computation or output creation, preserving the established numerical references.

---

- [**breaking**] Reuse retained evidence for immutable releases [#233](https://github.com/acgetchell/markov-chain-monte-carlo/pull/233)
  [`113ef7a`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/113ef7ace67d4bbec93c1253884322b44e4ed2d5)
  Closes [#231](https://github.com/acgetchell/markov-chain-monte-carlo/issues/231).

#### ci!: reuse retained evidence for immutable releases

- Adopt the la-stack v0.5.4 release flow. Use one benchmark acquisition for the report, README, and archive, and verify signed main CI evidence before
  tagging.
- Attach the retained archive to the draft, publish the immutable GitHub release, and dispatch the crates.io upload with the reviewed acceptance.
- Pin Rust 1.100 and research-repo-tools 0.2.9, refresh dependencies, and check coupled setup-uv and CodeQL pins offline.
- Reuse synthetic acquisition fixtures and exercise notebook rejection cases without launching a kernel for every input.

## [0.5.2] - 2026-10-24

### ⚠️ Breaking Changes

- Rust 1.99.0 is the minimum supported version. Fused arithmetic can change last-bit results and seeded warmup trajectories.
- Repository tooling and notebooks require Python 3.15; Semgrep retains its separate Python 3.14 environment.
- Use update-version, release-version-check, tag, and release-publish in place of the retired release commands.
- BinningAnalysis uses correctly rounded midpoint arithmetic, which can change floating-point block means for subnormal inputs.
- Rust 1.99.1 is the minimum supported version, superseding the earlier 1.99.0 baseline.

#### chore(release): release v0.5.2

### Merged Pull Requests

- Release v0.5.2 [#217](https://github.com/acgetchell/markov-chain-monte-carlo/pull/217)
- Unify Semgrep with the Python 3.15 environment [#214](https://github.com/acgetchell/markov-chain-monte-carlo/pull/214)
- Preserve binning precision and enforce release checks [#213](https://github.com/acgetchell/markov-chain-monte-carlo/pull/213)
- Complete shared tooling adoption [#211](https://github.com/acgetchell/markov-chain-monte-carlo/pull/211)
- Scope security scanner setup to required tools [#207](https://github.com/acgetchell/markov-chain-monte-carlo/pull/207)
- Standardize reviewed crates.io releases [#206](https://github.com/acgetchell/markov-chain-monte-carlo/pull/206)
- Bump the github-actions group with 3 updates [#204](https://github.com/acgetchell/markov-chain-monte-carlo/pull/204)
- Adopt research-repo-tools v0.2.3 [#203](https://github.com/acgetchell/markov-chain-monte-carlo/pull/203)
- Bump the github-actions group with 4 updates [#200](https://github.com/acgetchell/markov-chain-monte-carlo/pull/200)
- Bump the github-actions group with 2 updates [#199](https://github.com/acgetchell/markov-chain-monte-carlo/pull/199)
- Bump the github-actions group with 4 updates [#198](https://github.com/acgetchell/markov-chain-monte-carlo/pull/198)
- Require Rust 1.99 and adopt fused arithmetic [#197](https://github.com/acgetchell/markov-chain-monte-carlo/pull/197)
- Unify chain analysis around the Ising example [#196](https://github.com/acgetchell/markov-chain-monte-carlo/pull/196)
- Clarify proposal validation and chain analysis [#195](https://github.com/acgetchell/markov-chain-monte-carlo/pull/195)
- Preserve diagnostic failure details [#194](https://github.com/acgetchell/markov-chain-monte-carlo/pull/194)

### Changed

- Preserve diagnostic failure details [#194](https://github.com/acgetchell/markov-chain-monte-carlo/pull/194)
  [`ef31dad`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/ef31dad7475e8140a64ccccd997d09fe435a1528)
  Closes [#184](https://github.com/acgetchell/markov-chain-monte-carlo/issues/184).

  - Verify R-hat, ESS, MCSE, and timing errors retain their reasons and workload locations.
  - Cover unresolved ESS variance in every original chain and split half.
  - Document folded and combined R-hat availability in v0.5.1.

### Dependencies

- Bump the github-actions group with 4 updates [#198](https://github.com/acgetchell/markov-chain-monte-carlo/pull/198)
  [`f3dfa6d`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/f3dfa6dfbdb984b891d035dec709d03ceb62fa01)
- Bump the github-actions group with 2 updates [#199](https://github.com/acgetchell/markov-chain-monte-carlo/pull/199)
  [`8c87656`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/8c876565d52cca082dc1f6570c6849566cf4a0ea)
- Bump the github-actions group with 4 updates [#200](https://github.com/acgetchell/markov-chain-monte-carlo/pull/200)
  [`37d7b42`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/37d7b428b23f694e41cdc8b762bc7e24840c7a3f)
- Bump the github-actions group with 3 updates [#204](https://github.com/acgetchell/markov-chain-monte-carlo/pull/204)
  [`4023a3a`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/4023a3a49ac452b25b578f2db86d485bbf4e21de)

### Documentation

- Clarify proposal validation and chain analysis [#195](https://github.com/acgetchell/markov-chain-monte-carlo/pull/195)
  [`b08c4f2`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/b08c4f264e4c88f1f1398601c5424ce31843ba73)

  - Account for bounded-search success probabilities and delayed commit identity
  - Explain ESS, R-hat, MCSE, and timing assumptions with clearer navigation
  - Credit original methods, independent ArviZ checks, and EABM guidance
  - Link executable examples and regression evidence from both task guides

- Unify chain analysis around the Ising example [#196](https://github.com/acgetchell/markov-chain-monte-carlo/pull/196)
  [`9738851`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/9738851d3c3dc4730360274e372e4652301b06b1)

  - Use baseline and cold Ising runs throughout the diagnostic walkthrough.
  - Explain production counts, rank plots, prefix ESS, and unavailable
    results for discrete observables.
  - Consolidate diagnostic exports and notebook inputs under ising_1d,
    replacing the separate diagnostics executable.
  - Refresh Semgrep and dprint pins and Cargo dependencies.

### Fixed

- [**breaking**] Preserve binning precision and enforce release checks [#213](https://github.com/acgetchell/markov-chain-monte-carlo/pull/213)
  [`873e96f`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/873e96fe164e63c627ebbab8b2688602fa106780)

  - Preserve constant subnormal means when combining binning blocks.
  - Propagate Clippy and SARIF pipeline failures while retaining report uploads.
  - Enforce action pinning and version comments for unnamed and composite steps.
  - Bound scientific reference tolerances and report malformed diagnostic inputs cleanly.
  - Package archived changelogs and clarify clone isolation and retained performance evidence.
  - Preserve scientific regression contracts and portable Bash fixture input.

### Maintenance

- [**breaking**] Require Rust 1.99 and adopt fused arithmetic [#197](https://github.com/acgetchell/markov-chain-monte-carlo/pull/197)
  [`19a3268`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/19a32689ed1cc8e58947e2096ba66e122795fc6c)

  - Align the MSRV, contributor toolchain, and documentation with Rust 1.99.0.
  - Use single-rounding multiply-add in adaptive warmup, normal quantiles, and examples while preserving compensated-sum rounding.
  - Mark borrowed diagnostic accessors as must-use and scope mutable-reference lint expectations to forwarding tests.
  - Refresh locked dependencies and managed uv/dprint pins, and clarify the PyJWT compatibility floor.
  - Run notebook lint and Rust formatting earlier in CI and streamline contributor validation guidance.

- [**breaking**] Adopt research-repo-tools v0.2.3 [#203](https://github.com/acgetchell/markov-chain-monte-carlo/pull/203)
  [`e273e5c`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/e273e5c48f375518c387b97ddcf0c567167ac858)
  Closes [#202](https://github.com/acgetchell/markov-chain-monte-carlo/issues/202).

  - Move shared tooling and notebooks to Python 3.15 while isolating Semgrep in its locked Python 3.14 environment.
  - Delegate Python validation and release preflight to the shared commands while retaining current Ruff and Ty versions.
  - Refresh dependency and managed-tool pins, and cover both Python projects in updates, security scans, and Dependabot.
  - Permit Git mutations only within explicit maintainer authorization and record the Rust artifact-promotion assessment.

- [**breaking**] Standardize reviewed crates.io releases [#206](https://github.com/acgetchell/markov-chain-monte-carlo/pull/206)
  [`ae8b71f`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/ae8b71fc3dfb11a7515b038740a8c8543d23bd44)
  Closes [#182](https://github.com/acgetchell/markov-chain-monte-carlo/issues/182).

  - Align release commands and instructions with the shared repository workflow.
  - Attach scientific benchmark evidence to drafts before reviewed OIDC publication.
  - Require native checks, immutable release identity, and registry absence before upload.
  - Align Semgrep's OpenTelemetry dependencies with the fixed release family.
  - Clarify task-scoped Git approval and deployment setup.

- Scope security scanner setup to required tools [#207](https://github.com/acgetchell/markov-chain-monte-carlo/pull/207)
  [`53e42e1`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/53e42e1163db43a50e2576577a593e6d08be7f7e)

  - Select each scanner's declared installation plan through the shared tooling API.
  - Keep binary scanners independent of Rust and zizmor builds limited to minimal host Rust.
  - Isolate scanner caches and use the tooling-only Python environment for security recipes.
  - Record verified Trusted Publisher, deployment approval, tag restriction, and action allowlist settings.

- Complete shared tooling adoption [#211](https://github.com/acgetchell/markov-chain-monte-carlo/pull/211)
  [`5a49ac5`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/5a49ac59c39deb02fb5a4a7957c4e33aa4a0c63b)
  Closes [#208](https://github.com/acgetchell/markov-chain-monte-carlo/issues/208), [#210](https://github.com/acgetchell/markov-chain-monte-carlo/issues/210).

  - Enforce the shared Actions allowlist and preserve captured benchmark release identities across workflow jobs.
  - Replace duplicated notebook and Just test plumbing with published public helpers.
  - Discover Cargo examples while retaining scientific output contracts, feature coverage, and source provenance.
  - Use verified dprint and rumdl release binaries and separate tool updates from contributor setup.
  - Preserve consumer-owned Python tool requirements and include the staged dependency refresh.
  - Retain the report-inventory guard until equivalent shared validation is published.

- Unify Semgrep with the Python 3.15 environment [#214](https://github.com/acgetchell/markov-chain-monte-carlo/pull/214)
  [`f980683`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/f98068392fa201f8cedb79ac7cbcbe29c1a3c029)

  - Share one Python manifest and lockfile across development tools and Semgrep.
  - Retain patched scanner dependencies and Python 3.15 Windows wheels.
  - Apply the Linux stack-limit workaround to local and SARIF scans.
  - Align scanner commands, dependency automation, and contributor guidance.
  - Refresh the locked cc build dependency for release preparation.

#### Chore

---

- [**breaking**] Release v0.5.2 [#217](https://github.com/acgetchell/markov-chain-monte-carlo/pull/217)
  [`389039f`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/389039f123e989afd20484969eca650017c08d92)

#### build!: require Rust 1.99.1 and align release tooling

- Align Rust declarations, shared Python tools, and release-date handling.

## [0.5.1] - 2026-10-02

### ⚠️ Breaking Changes

- The diagnostics example now exports schema-2 JSON and a
  multi-run CSV layout. Consumers of its previous exports must update readers.

### Merged Pull Requests

- Add pooled ranks and diagnostic plot workflow [#192](https://github.com/acgetchell/markov-chain-monte-carlo/pull/192)
- Add multi-chain ESS and mean/quantile MCSE [#191](https://github.com/acgetchell/markov-chain-monte-carlo/pull/191)
- Replace R-hat fixture script with Python [#190](https://github.com/acgetchell/markov-chain-monte-carlo/pull/190)
- Add folded and combined R-hat diagnostics [#187](https://github.com/acgetchell/markov-chain-monte-carlo/pull/187)
- Bump the github-actions group with 4 updates [#186](https://github.com/acgetchell/markov-chain-monte-carlo/pull/186)
- Add rank-normalized split R-hat diagnostics [#185](https://github.com/acgetchell/markov-chain-monte-carlo/pull/185)

### Added

- Add rank-normalized split R-hat diagnostics [#185](https://github.com/acgetchell/markov-chain-monte-carlo/pull/185)
  [`69c7fee`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/69c7fee9025194ebf8862df0e95a2e46b4110ca6)
  Closes [#183](https://github.com/acgetchell/markov-chain-monte-carlo/issues/183).

  - Add RankNormalizedSplitRhat for borrowed scalar chains with pooled average ranks, normal scores, and original and retained sample counts.
  - Preserve classical SplitRhat behavior and report typed failures for invalid inputs, numerical degeneracy, and excessive rank counts.
  - Document scientific sources and limitations, including the omission of folding and the combined diagnostic.
  - Align README citations with the ecosystem's Zenodo DOI and BibTeX style.

- Add folded and combined R-hat diagnostics [#187](https://github.com/acgetchell/markov-chain-monte-carlo/pull/187)
  [`1737398`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/17373983bc1234709c8c42103f7eff6508ed8cff)

  - Add FoldedRankNormalizedSplitRhat and CombinedRhat with borrowed inputs, typed component results, and sample metadata.
  - Return the combined maximum only when both components succeed, preserving explicit failures for folded degeneracy and precision loss.
  - Document folding conventions and estimator limits, and export reproducible CSV traces and JSON reports from the diagnostics example.
  - Update uv, dprint, and rumdl pins and refresh locked Python dependencies.

- Add multi-chain ESS and mean/quantile MCSE [#191](https://github.com/acgetchell/markov-chain-monte-carlo/pull/191)
  [`c3513b1`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/c3513b148f7e5e95f62278dc8025d750474f3595)
  Closes [#188](https://github.com/acgetchell/markov-chain-monte-carlo/issues/188).

  - Expose mean, bulk, quantile, and tail ESS through borrowed-chain APIs
    that account for between-chain disagreement.
  - Add mean and quantile Monte Carlo standard errors in observable units.
  - Report original and retained sample counts, relative ESS, and measured
    ESS rates with workload timing checks.
  - Preserve typed unavailable outcomes for degenerate inputs and
    numerically unresolved estimates.
  - Document ArviZ estimator conventions and extend the diagnostics
    example with precision and efficiency summaries.

- [**breaking**] Add pooled ranks and diagnostic plot workflow [#192](https://github.com/acgetchell/markov-chain-monte-carlo/pull/192)
  [`9fc2479`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/9fc2479167aaa47c7bbe3ef5e43eb76393c9dc89)
  Closes [#189](https://github.com/acgetchell/markov-chain-monte-carlo/issues/189).

#### feat!: add pooled ranks and diagnostic plot workflow

- Expose pooled average ranks with original chain identity and borrowed views.
- Add reproducible rank and prefix ESS plots, MCSE reports, and numeric exports
  with explicit availability, timing scope, and provenance.
- Document independent ArviZ/SciPy checks and expand ESS/MCSE coverage.
- Remove redundant Semgrep rules and refresh managed dependencies while
  retaining the Windows-compatible Semgrep pin.

### Dependencies

- Bump the github-actions group with 4 updates [#186](https://github.com/acgetchell/markov-chain-monte-carlo/pull/186)
  [`acb593c`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/acb593cc2b1a9ba32118dfa3a4f0b589d5ba0e84)

### Fixed

- Replace R-hat fixture script with Python [#190](https://github.com/acgetchell/markov-chain-monte-carlo/pull/190)
  [`eecdf65`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/eecdf65eead0097adf8788d151598fa99da7e7b4)

## [0.5.0] - 2026-09-28

### ⚠️ Breaking Changes

- Rust 1.98.1 or newer is now required.
- changelog-unreleased now requires TAG and DATE arguments. Remove postprocess-changelog and update-release-version --sync-changelog-date in favor of the shared
  workflow.
- Remove the check-notebooks, tag-release, update-python-dev-pins, and update-tool-pins entry points. Use the documented Just recipes or research-repo-tools CLI
  instead. just update now upgrades uv through its installation owner and declared Cargo tools before updating dependencies. Executed notebooks now mirror
  repository paths under target/notebooks.
- Remove the release-check executable and subprocess_utils module. Use just release-check or research-repo-tools release check
  --final-release, and research_repo_tools.process for process helpers.
- Remove the archive-performance, bench-compare,
  publish-performance-readme, and update-release-version executables.
  Use the documented Just recipes or research-repo-tools CLI instead. New performance evidence and reports use docs/performance/v1 and the shared
  schemas; explicit report inputs now use --payload and --manifest.

### Merged Pull Requests

- Clarify scientific documentation and shared tooling workflows [#180](https://github.com/acgetchell/markov-chain-monte-carlo/pull/180)
- Complete the scalar diagnostics workflow [#179](https://github.com/acgetchell/markov-chain-monte-carlo/pull/179)
- Adopt research-repo-tools v0.1.7 [#178](https://github.com/acgetchell/markov-chain-monte-carlo/pull/178)
- Add optional tracing for long-running simulations [#177](https://github.com/acgetchell/markov-chain-monte-carlo/pull/177)
- Add bounded adaptive Metropolis-Hastings warmup [#175](https://github.com/acgetchell/markov-chain-monte-carlo/pull/175)
- Add continuous proposal diagnostics [#174](https://github.com/acgetchell/markov-chain-monte-carlo/pull/174)
- Add reference benchmark distributions [#173](https://github.com/acgetchell/markov-chain-monte-carlo/pull/173)
- Add ESS, ESS-rate, and split R-hat diagnostics [#172](https://github.com/acgetchell/markov-chain-monte-carlo/pull/172)
- Add autocorrelation and integrated time diagnostics [#171](https://github.com/acgetchell/markov-chain-monte-carlo/pull/171)
- Align security audits and enforce Python typing guards [#170](https://github.com/acgetchell/markov-chain-monte-carlo/pull/170)
- Replace local tooling with research-repo-tools v0.1.5 [#169](https://github.com/acgetchell/markov-chain-monte-carlo/pull/169)
- Adopt research-repo-tools v0.1.3 for shared tooling [#167](https://github.com/acgetchell/markov-chain-monte-carlo/pull/167)
- Adopt research-repo-tools v0.1.2 for maintenance [#165](https://github.com/acgetchell/markov-chain-monte-carlo/pull/165)
- Pilot research-repo-tools for changelog maintenance [#162](https://github.com/acgetchell/markov-chain-monte-carlo/pull/162)
- Raise the Rust baseline to 1.98.1 [#161](https://github.com/acgetchell/markov-chain-monte-carlo/pull/161)
- Bump the dependencies group with 4 updates [#159](https://github.com/acgetchell/markov-chain-monte-carlo/pull/159)
- Bump the github-actions group with 5 updates [#158](https://github.com/acgetchell/markov-chain-monte-carlo/pull/158)
- Bump the dependencies group with 2 updates [#156](https://github.com/acgetchell/markov-chain-monte-carlo/pull/156)
- Bump zizmorcore/zizmor-action in the github-actions group [#155](https://github.com/acgetchell/markov-chain-monte-carlo/pull/155)
- Bump the github-actions group with 3 updates [#152](https://github.com/acgetchell/markov-chain-monte-carlo/pull/152)
- Attach benchmark assets before publishing releases [#149](https://github.com/acgetchell/markov-chain-monte-carlo/pull/149)

### Added

- Add autocorrelation and integrated time diagnostics [#171](https://github.com/acgetchell/markov-chain-monte-carlo/pull/171)
  [`a83e746`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/a83e746e5f4a3da2822e1e10316f0d9b18edb0b6)
  Closes [#73](https://github.com/acgetchell/markov-chain-monte-carlo/issues/73).

  - Expose scalar ACF and Geyer initial monotone sequence estimates with typed errors, sample counts, and retained lag windows.
  - Add borrowed observable selection by name for individual trace chains.
  - Extend the Ising example and notebook with energy and magnetization diagnostics, including buffered CSV exports with explicit flushing.
  - Document estimator assumptions, truncation behavior, numerical limits, and trace mutation guarantees.
  - Add diagnostic benchmarks and reproducible arima and ferromorphic comparisons, retaining native estimators without new runtime dependencies.
  - Allow apply_patch when the preferred editing tools are unavailable.

- Add ESS, ESS-rate, and split R-hat diagnostics [#172](https://github.com/acgetchell/markov-chain-monte-carlo/pull/172)
  [`3d6dd13`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/3d6dd13362fab1af43b3fe979acf18d3f794a775)
  Closes [#74](https://github.com/acgetchell/markov-chain-monte-carlo/issues/74).

  - Add single-chain ESS and measured ESS per second to integrated autocorrelation time summaries.
  - Provide classical split R-hat over borrowed chains with typed failures and sample-count metadata.
  - Extend the Ising example to four seeded chains with JSON diagnostics and explicit timing and warmup metadata.
  - Add independent notebook calculations and CSV summaries, using timing only for matching samples, and document estimator limits.
  - Reserve full CI for final commit/push readiness and reuse focused validation during review.

- Add reference benchmark distributions [#173](https://github.com/acgetchell/markov-chain-monte-carlo/pull/173)
  [`6a7f459`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/6a7f4599f272a0baa58e58c8fbc36ae4317a92ad)
  Closes [#20](https://github.com/acgetchell/markov-chain-monte-carlo/issues/20).

  - Add an optional dependency-free benchmarks feature with fixed 2D Rosenbrock, Neal's funnel, Gaussian mixture, and banana targets
  - Expose analytical means and covariances with documented density conventions and numerical limits
  - Add a seeded sampling example reporting moment errors, mean ESS, and measured ESS per second
  - Enable all features in documentation, integration-test, and example workflows

- Add continuous proposal diagnostics [#174](https://github.com/acgetchell/markov-chain-monte-carlo/pull/174)
  [`3d28989`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/3d289897b8c24e85eedc2dd12bcd23c3e02594e3)
  Closes [#42](https://github.com/acgetchell/markov-chain-monte-carlo/issues/42).

  - Add verify_proposal_density to compare Hastings ratios with independent log densities, with typed support and numerical errors.
  - Add verify_proposal_bins to check sampled bin masses using simultaneous Hoeffding bounds and reports that retain their decision thresholds.
  - Document sampling assumptions, error budgets, and when to choose continuous diagnostics instead of exact-hit detailed-balance checks.
  - Refresh tooling and dependency pins, and document using just update when repository tooling falls behind stable releases.

- Add bounded adaptive Metropolis-Hastings warmup [#175](https://github.com/acgetchell/markov-chain-monte-carlo/pull/175)
  [`c9879ce`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/c9879ce2494f7f5d76d8e45b21c7c817bd34dbd6)
  Closes [#10](https://github.com/acgetchell/markov-chain-monte-carlo/issues/10).

  - Add AdaptiveScale and TunableProposal for acceptance-rate tuning across by-value, in-place, and delayed proposal workflows
  - Preserve tuning across warmup chunks and keep production sampling at the final fixed scale
  - Skip unused proposal metadata during in-place and delayed warmup
  - Document adaptation limits and add a normal-distribution example
  - Update zerocopy, platformdirs, and typos-cli patch versions

- Add optional tracing for long-running simulations [#177](https://github.com/acgetchell/markov-chain-monte-carlo/pull/177)
  [`a3066c2`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/a3066c244906d4a9995538bc5ca3b1e25a3d9521)
  Closes [#21](https://github.com/acgetchell/markov-chain-monte-carlo/issues/21).

  - Emit per-step outcomes, acceptance rates, step counts, and log probabilities across all sampling kernels.
  - Add DEBUG spans around sampling, observation, thinning, and warmup loops.
  - Compile out instrumentation when the tracing feature is disabled.
  - Document subscriber setup, filtering, and metric semantics.

- Complete the scalar diagnostics workflow [#179](https://github.com/acgetchell/markov-chain-monte-carlo/pull/179)
  [`1c16afa`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/1c16afa83f2afa832d30afd3c61fff987ccb2b2d)
  Closes [#13](https://github.com/acgetchell/markov-chain-monte-carlo/issues/13).

  - Add a sequential-chain example covering ACF, mean ESS, and classical split R-hat.
  - Document diagnostic contracts, limitations, and unavailable results.
  - Integrate coverage reporting and managed-tool cleanup from research-repo-tools v0.1.7.
  - Make tracing test capture tolerate retained subscriber references.

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

### Documentation

- Clarify scientific documentation and shared tooling workflows [#180](https://github.com/acgetchell/markov-chain-monte-carlo/pull/180)
  [`1afa54c`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/1afa54c88aea55e054322aeb38ee1c3f115f6289)
  Closes [#154](https://github.com/acgetchell/markov-chain-monte-carlo/issues/154).

  - Align README structure with delaunay and la-stack, separating API navigation, scientific explanations, and bibliographic provenance.
  - Rename task guides, preserve citation anchors, and clarify statistical assumptions and diagnostic limitations.
  - Synchronize published API links with releases, keep current guide links on main, and preserve unique rustdoc tracing anchors.
  - Enable research-repo-tools v0.1.7 notebook install restrictions and authenticated shared setup.
  - Retire historical performance reports and legacy adapters, establishing the next release as the baseline for a new shared performance series.

- Clarify multi-chain support and refresh v0.5.0 notes
  [`ce52eb2`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/ce52eb2c2966c63bc84d0203582acf5f9b2691bb)

  - Distinguish available sequential-chain workflows and diagnostics from planned parallel orchestration, tempering, and learned-proposal integrations.
  - Clarify support for supplied learned target terms while keeping model and proposal-policy training outside the crate's scope.
  - Regenerate release notes to include the latest documentation and workflow fixes.

- Clarify release verification and tagging workflow
  [`e7d104c`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/e7d104cad42936675dad2ef1cf8bfcca5298ad59)

  - Separate post-merge release steps into runnable blocks with recovery guidance.
  - Require verification of the reviewed commit and final CI before tagging.
  - Add tag-preview and replace just tag with the explicit tag-release recipe.
  - Use an explicit release date and include changelog validation.

### Fixed

- Attach benchmark assets before publishing releases [#149](https://github.com/acgetchell/markov-chain-monte-carlo/pull/149)
  [`7c56e6c`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/7c56e6c709121069e5b99ebcb506e0fe198dd167)

  - Require an explicit stable tag backed by a mutable draft release.
  - Upload the durable Criterion baseline before publishing the draft.
  - Align release guidance and rollout boundaries with immutable releases.

- Preserve README hero image across release updates
  [`2783be9`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/2783be965643819647a44c726df7c031f82b3ce7)

  - Pin the Ising trace image to an existing commit so it renders before the release tag exists.
  - Preserve documentation asset links during release metadata updates.
  - Clarify why the quick-start example requires a direct rand dependency and how callers create and seed the sampler's generator.

- Keep documentation links stable and simplify Just workflows
  [`7e2946f`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/7e2946f6c7fc81bca4a9d4d0984d969475b75dbf)

  - Keep active repository links on main and API links on docs.rs latest, preserving those destinations across release updates.
  - Generate complete recipe help from bare just and remove redundant help, setup, formatting, testing, and benchmark entry points.
  - Add just publish and just release-verify to simplify release instructions.
  - Show runnable examples and command discovery in Quick start, with contributor validation and security requirements in CONTRIBUTING.
  - Update benchmark guidance to use canonical commands and repair the UC Davis community principles link.

### Maintenance

- [**breaking**] Raise the Rust baseline to 1.98.1 [#161](https://github.com/acgetchell/markov-chain-monte-carlo/pull/161)
  [`3d822a3`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/3d822a39d6233212ce91058c47b5abee0fe83707)

  - Align the MSRV, contributor toolchain, and Clippy configuration.
  - Update setup guidance and document the trait-object vtable miscompilation fix.

- [**breaking**] Pilot research-repo-tools for changelog maintenance [#162](https://github.com/acgetchell/markov-chain-monte-carlo/pull/162)
  [`6fef368`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/6fef368ceb8dfbf921ddc5bbaf5495ba4247e684)
  Closes [#157](https://github.com/acgetchell/markov-chain-monte-carlo/issues/157).

#### build!: pilot research-repo-tools for changelog maintenance

- Pin research-repo-tools 0.1.0 from PyPI in the shared tooling group.
- Delegate changelog generation, archiving, and release-note extraction to shared commands.
- Remove duplicated changelog processing, date synchronization, and their dedicated tests.
- Preserve included tooling constraints during dependency updates and skip deleted files in Semgrep.
- Document migration policies, upstream gaps, and the package upgrade path.

---

- [**breaking**] Adopt research-repo-tools v0.1.2 for maintenance [#165](https://github.com/acgetchell/markov-chain-monte-carlo/pull/165)
  [`496df7f`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/496df7f79fef4aadbe849c56fc171130c3ae3357)
  Closes [#160](https://github.com/acgetchell/markov-chain-monte-carlo/issues/160), [#163](https://github.com/acgetchell/markov-chain-monte-carlo/issues/163).

  - Centralize declared toolchain setup and checked execution across local recipes and CI.
  - Share dependency updates, release metadata, annotated tagging, Semgrep fixture validation, and opt-in CodeRabbit review commands.
  - Replace local notebook tooling with shared checks, cleanup, fresh-kernel execution, and provenance reports.
  - Preserve MCMC release policies, scientific workflows, and pinned SARIF fallbacks while removing duplicated maintenance scripts.
  - Refresh tool pins and dependency locks and document the migration.

- [**breaking**] Adopt research-repo-tools v0.1.3 for shared tooling [#167](https://github.com/acgetchell/markov-chain-monte-carlo/pull/167)
  [`b717f48`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/b717f4818479725a9646e2c164ada0579538002b)
  Closes [#164](https://github.com/acgetchell/markov-chain-monte-carlo/issues/164).

  - Replace duplicated process, Criterion, archive, and publication mechanics with supported shared APIs.
  - Declare MCMC release policies and use validated shared release plans.
  - Preserve historical evidence bytes and require exact tagged artifacts for README publication.
  - Manage SARIF converters through the shared toolchain and remove redundant local helpers and duplicated generic tests.

- [**breaking**] Replace local tooling with research-repo-tools v0.1.5 [#169](https://github.com/acgetchell/markov-chain-monte-carlo/pull/169)
  [`2ad831a`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/2ad831a8114f873da1cd8216a623cdaca6136d0e)

  - Delegate benchmark measurement, evidence publication, release preparation, and common validation mechanics to shared commands.
  - Remove the local tooling package, support scripts, and duplicated generic tests; retain a dependency-only uv environment and MCMC policy checks.
  - Add shared performance reports and evidence companions while preserving historical artifact bytes and provenance meaning.
  - Replace workflow helpers with shared environment export and release-asset handling while keeping write credentials separate from benchmark execution.
  - Document declarative scientific policies and revised maintainer workflows.
  - Pin uv to 0.12.18 and update the Codecov action to v7.1.1.

- Align security audits and enforce Python typing guards [#170](https://github.com/acgetchell/markov-chain-monte-carlo/pull/170)
  [`778ea3d`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/778ea3df554fba99c3aa9b710c0c95c807c83dc6)
  Closes [#150](https://github.com/acgetchell/markov-chain-monte-carlo/issues/150), [#153](https://github.com/acgetchell/markov-chain-monte-carlo/issues/153).

  - Upgrade research-repo-tools and release benchmark writers to v0.1.6.
  - Share zizmor configuration across local and CI audits, with token discovery, offline fallback, and guarded SARIF uploads.
  - Enforce complete annotations and strict type-checking imports with full Ruff and Ty checks across Python sources and Semgrep fixtures.
  - Discover notebooks throughout the repository and retain narrow exceptions for intentional negative fixtures.
  - Exclude Semgrep fixtures from CodeRabbit review and disable its docstring-percentage gate.

- Adopt research-repo-tools v0.1.7 [#178](https://github.com/acgetchell/markov-chain-monte-carlo/pull/178)
  [`c35e118`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/c35e1187c38dd1790fe279df36ff7b2feb21453c)

  - Inherit the shared Python baseline and add adoption commands that synchronize package pins, Python mirrors, environments, and kernels.
  - Replace personal-token Dependabot review requests with shared approvals and native auto-merge.
  - Add managed OSV and Gitleaks scans with scheduled workflows, redacted reports, and README status badges.
  - Align release tooling pins and remove duplicated Python selectors.
  - Remove redundant tooling tests and obsolete CSV migration policy while retaining scientific checks and historical evidence.

- Synchronize v0.5.0 release artifacts for September 28
  [`83f729c`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/83f729c62dd959dc9afa2ca9c4146b0515465fae)

  - Set the changelog and citation release dates to 2026-09-28.
  - Refresh generated notes with multi-chain scope clarifications and the updated release verification and tagging workflow.

## Archives

Older releases are archived by minor series:

- [0.4.x](docs/archives/changelog/0.4.md)
- [0.3.x](docs/archives/changelog/0.3.md)
- [0.2.x](docs/archives/changelog/0.2.md)
- [0.1.x](docs/archives/changelog/0.1.md)

[Unreleased]: https://github.com/acgetchell/markov-chain-monte-carlo/compare/v0.5.2...HEAD
[0.5.2]: https://github.com/acgetchell/markov-chain-monte-carlo/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/acgetchell/markov-chain-monte-carlo/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/acgetchell/markov-chain-monte-carlo/compare/v0.4.2...v0.5.0
