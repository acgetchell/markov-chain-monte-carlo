# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.2] - 2026-08-31

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

### Added

- [**breaking**] Add structured in-place proposal telemetry [#125](https://github.com/acgetchell/markov-chain-monte-carlo/pull/125)
  [`dc1820f`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/dc1820f7178371e362b4fea9304c344b957a5502)

  - Return invariant-bearing `Step&lt;Info&gt;` metadata from in-place single-step APIs.
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

### Documentation

- Prepare reviewer-facing MCMC documentation [#88](https://github.com/acgetchell/markov-chain-monte-carlo/pull/88)
  [`cd861ee`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/cd861ee7e0b8604e31e8e19bf39e12bc26298222)

  - docs: prepare reviewer-facing MCMC documentation
  - Add a reviewer guide that points scientific and engineering reviewers to the README, scientific basis, proposal validation, roadmap, references, and local
    checks.
  - Refocus the README and scientific basis docs around the Metropolis-Hastings contract, externally supplied regularizer terms, and explicit non-claims for
    convergence or learned-proposal training.
  - Keep crate-level rustdoc focused on programming contracts while moving project orientation into the README and topic docs.
  - Clarify example comments for normal, Ising, and additive-target workflows.
  - Pin the markdown and spelling tool versions used by the validation gate.
  - Changed: Update internal RUMDL and TYPOS tooling in CI
  - Changed: Correct example branch type in contributing guide

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

  - feat!: add delayed proposal ratio helpers
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

  - feat!: expose delayed-step outcome telemetry [#61](https://github.com/acgetchell/markov-chain-monte-carlo/pull/61)
  - Replace delayed-step accepted/proposed booleans with StepOutcome so step state is represented by one invariant-bearing value.
  - Add StepRejectionReason and DelayedProposal::no_plan_info for no-plan delayed steps that still need domain-specific telemetry.
  - Re-export delayed telemetry through the root API and delayed prelude, and update docs and benchmarks to use outcome-based access.
- Add additive target composition [#78](https://github.com/acgetchell/markov-chain-monte-carlo/pull/78)
  [`55140e0`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/55140e086b4f4b724372b797b6856b41b8c7d1f1)

  - feat: add additive target composition
  - Add AdditiveTarget for composing model and bias log-weight terms in the target distribution.
  - Re-export the adapter through the crate root and scoped preludes for by-value, in-place, delayed, and testing workflows.
  - Document additive energy/action semantics and keep proposal-ratio corrections separate from target bias terms.
- Add trace recording diagnostics [#79](https://github.com/acgetchell/markov-chain-monte-carlo/pull/79)
  [`adbb84a`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/adbb84a60a509eaba0d37ac298f2a6196b6495ee)

  - feat: add trace recording diagnostics
  - Add ChainId, TraceStepOutcome, TraceRecord, Trace, and TraceRecorder for reusable numeric MCMC traces with accept/reject metadata and CSV export.
  - Re-export trace diagnostics through the root API and scoped preludes so examples, doctests, benchmarks, and downstream users can import them consistently.
  - Extend the Ising example with energy and magnetization trace export, add an individual just example recipe, and document the notebook workflow for
    inspecting the generated CSV.
  - ci: add notebook validation checks
  - Add notebook linting and in-memory execution recipes so tracked notebooks are checked in local validation and full CI simulation.
  - Add notebook runtime dependencies and a reusable check-notebooks helper for JSON, code-cell syntax, and nbclient execution.
  - Harden the Ising trace notebook with clearer missing-file errors and proposed-move acceptance-rate handling.
  - Cover notebook checking and subprocess utility behavior with Python tests.
  - fix(ci): use ASCII notebook checker output
  - Replace Unicode status markers with ASCII text so notebook checks run on Windows consoles.
  - fix(tooling): exclude notebook checkpoints from discovery
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

### Fixed

- Escape changelog angle brackets for GitHub rendering
  [`82014b4`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/82014b424d0b5be37c93770e2cdd925ba351ca25)

  - Escape generated changelog commit text so Rust generics are not parsed as HTML tags
  - Regenerate CHANGELOG.md to preserve visible generic type names and version footer links
- Escape changelog angle brackets for GitHub rendering
  [`f712f15`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/f712f1515d8a215359a2551c598d8fa0de974ef5)

  Angle brackets in changelog entries, such as Rust generics (`Chain&lt;S&gt;`), were parsed as HTML tags by Markdown renderers like GitHub. This fix escapes
  those characters during changelog generation, ensuring correct display. The changelog was regenerated, also incorporating repository security enhancements
  from #63.
- Clean regenerated changelog entries [`4147806`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/414780606cc68da0f8a5b99de821af89a94e0671)

  - Repair the historical changelog body for escaped Rust generic examples.
  - Filter changelog-only body noise from generated release notes.
  - Regenerate CHANGELOG.md with the corrected commit preprocessing.
- Enforce fallible public Rust examples [#62](https://github.com/acgetchell/markov-chain-monte-carlo/pull/62)
  [#80](https://github.com/acgetchell/markov-chain-monte-carlo/pull/80)
  [`ca2ba16`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/ca2ba169385a8fa99aced9100a392fb402fa29bf)

  - fix: enforce fallible public Rust examples [#62](https://github.com/acgetchell/markov-chain-monte-carlo/pull/62)
  - Replace unwrap and expect flows in doctests, examples, and benchmarks with typed error propagation or contextual benchmark failure handling.
  - Add Semgrep guardrails and fixtures that catch unwrap and expect usage in public doctests, examples, and benchmarks.
  - Keep generated documentation and tooling metadata aligned with the updated validation surface.
  - fix(semgrep): catch doctest unwrap continuations [#62](https://github.com/acgetchell/markov-chain-monte-carlo/pull/62)
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

  - ci: harden repository security and tooling validation
  - Add zizmor and repository-owned Semgrep SARIF workflows for GitHub Actions and project-rule scanning.
  - Replace Codacy with local and GitHub-native security signals, including action pinning, allowlist, and version-comment rules.
  - Run full CI through just across Linux, macOS, and Windows with uv-managed Python tools and pinned Cargo-installed tooling.
  - Move Rust unit and integration tests to cargo-nextest while keeping doctests on cargo test --doc.
  - Add a security policy and document the updated setup, line-length, and validation workflow.
  - ci: allow Rust cache action in workflow policy
  - Add swatinem/rust-cache to the repository-owned GitHub Actions allowlist so the Rust toolchain action's cache helper is permitted.
  - Cover the allowlist entry in the Semgrep workflow policy fixture.
  - ci: simplify Semgrep SARIF concurrency key
- Refresh managed tooling installs [`27bcae6`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/27bcae6faad4e0197241ea0ace49f88cdc65ea25)

  - Replace taiki-e/install-action with cached Cargo installs for audit and coverage workflow tools.
  - Update uv-managed development tools, including Semgrep, Ruff, Ty, and the workflow uv version.
  - Refresh the serde_json dev dependency and remove taiki-e from the repository-owned Actions allowlist.
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

  - ci: cache cargo-installed CI tools
  - Install Cargo-managed CI tools through taiki-e/cache-cargo-install-action instead of rebuilding them in each matrix job.
  - Keep full just ci coverage on Linux, macOS, and Windows while exposing smaller CI subsets for timing and local diagnosis.
  - Compile benchmark harnesses with all crate features enabled and allow the pinned cache action in the repository workflow policy.
- Bump Rust MSRV to 1.96.0 [#65](https://github.com/acgetchell/markov-chain-monte-carlo/pull/65)
  [#69](https://github.com/acgetchell/markov-chain-monte-carlo/pull/69)
  [`ba7beb2`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/ba7beb2a65b4d506374569d19a77b336ad74e53d)

  - build: bump Rust MSRV to 1.96.0 [#65](https://github.com/acgetchell/markov-chain-monte-carlo/pull/65)
  - Pin Cargo, rustup, Clippy, and contributor docs to Rust 1.96.0.
  - Adopt 1.96-compatible numeric updates with fused multiply-add in statistics, examples, and benchmarks.
  - Refresh typed error assertions and doctests to use `std::assert_matches!`.
  - Clarify that LLVM coverage tools are installed by setup and Codecov rather than the default pinned toolchain.
  - ci: add Windows taplo install fallback [#65](https://github.com/acgetchell/markov-chain-monte-carlo/pull/65)
  - Preserve the cached cargo install action as the primary tool installation path across operating systems.
  - Fall back to direct taplo installation on Windows only when the cached install step fails.

## [0.3.0] - 2026-05-05

### Added

- Add delayed-commit proposal API [#36](https://github.com/acgetchell/markov-chain-monte-carlo/pull/36)
  [`65445f4`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/65445f4e48143a47a217f573bc82c37421092a3d)

  - add DelayedProposal for accept-before-mutation workflows
  - add Step/DelayedStep telemetry and orthogonal DelayedStepError variants
  - implement Chain::step_delayed with no-plan, rejection, acceptance, and commit-failure handling
  - extend Sampler with delayed stepping and generic proposal-handle storage
  - add focused by-value, in-place, and delayed prelude modules
  - update doctests, examples, README snippets, and property-test imports
  - add tests for delayed acceptance, rejection, no-plan, invalid numerics, proposal-stage errors, commit atomicity, and run_delayed error stopping
- Add observable measurement framework [#37](https://github.com/acgetchell/markov-chain-monte-carlo/pull/37)
  [`82cbaf5`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/82cbaf58a71e1355e370eee43450d22f8c02df91)

  - add Observable and TryObservable traits for infallible and fallible measurements
  - add ObservedStepError to keep sampling and observation failures orthogonal
  - add SampleBuffer for collected observation outputs
  - integrate observing APIs across by-value, in-place, and delayed Sampler paths
  - expose minimal workflow preludes for by-value, in-place, and delayed usage
  - add doctests, unit tests, and documentation for the new measurement APIs
  - simplify bounds, imports, and test names across touched code
- Add streaming statistics and error bars [#38](https://github.com/acgetchell/markov-chain-monte-carlo/pull/38)
  [`4575047`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/457504708de375dc23ea86de56505c2cdf67419d)

  - Add OnlineStats and BinningAnalysis for streaming mean, variance, standard error, and blocked autocorrelation-aware estimates
  - Add StatisticsError variants for invalid samples and non-finite accumulator state
  - Add TryAccumulator and ObservedStreamError for streaming observations into fallible sinks
  - Add sampler APIs for streaming by-value, in-place, and delayed observations into accumulators
  - Export new statistics and streaming types through minimal workflow preludes
  - Document usage in README and doctests
  - Ignore local .codex workspace metadata
- Add sampler thinning support [#40](https://github.com/acgetchell/markov-chain-monte-carlo/pull/40)
  [`c03d07e`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/c03d07e8bec3e9b5e67c22273689971f60fad85a)

  - add typed ThinningError and thinned result aliases
  - add state-collecting thinning APIs for by-value, in-place, and delayed samplers
  - add observing, streaming, and fallible thinning variants across sampler workflows
  - re-export thinning types through the public API and appropriate preludes
  - document thinning behavior in README and public doctests
  - cover zero intervals, interval &gt; steps boundaries, and thinned observation behavior
- Add serde checkpointing support [#41](https://github.com/acgetchell/markov-chain-monte-carlo/pull/41)
  [`d976bc6`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/d976bc67f3373cf2ead697508c127b5ae205d2bc)

  - Add optional `serde` feature and derive serialization for `Chain&lt;S&gt;`
  - Derive `Serialize` for `Sampler` when stored handles support it
  - Document chain checkpointing as the portable resume path
  - Add serde-gated tests for checkpoint roundtrip, resumed sampling, sampler serialization, and non-serializable state construction
  - Mark serde checkpointing as complete in the README
- Add detailed-balance proposal diagnostics [#43](https://github.com/acgetchell/markov-chain-monte-carlo/pull/43)
  [`cefc035`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/cefc0356ed29f6a273271230f6b59ffbd5f0886c)

  - Add detailed-balance verification APIs for by-value, in-place, delayed, and batch proposal checks.
  - Add typed reports and errors, scoped testing prelude exports, public doctests, and a runnable detailed_balance example.
  - Document proposal validation, scientific basis, roadmap, and refreshed README usage guidance.
  - Add Semgrep guardrails and fixtures to keep examples, benches, and doctests on typed errors.
  - Improve git-cliff and agent commit guidance, then regenerate CHANGELOG.md.
  - Bump serde_json dev-dependency to 1.0.149.
- [**breaking**] Validate sampler construction and checkpoint restores [#46](https://github.com/acgetchell/markov-chain-monte-carlo/pull/46)
  [`e355a6d`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/e355a6de20234ef5b74a61b8fe2a576793fcda7a)

  - Add ChainCheckpoint as the portable restore format and recompute cached log-probabilities through Chain::from_checkpoint.
  - Make Sampler::new validate the chain against the sampler target and return Result.
  - Add sampler-level reset and replacement helpers so callers do not need mutable access to the underlying Chain.
  - Report checkpoint and current-state cache failures with dedicated McmcError variants.
  - Refresh README organization and run coverage with all crate features enabled in local CI.

### Changed

- Docs/cargo rdme readme refresh [#44](https://github.com/acgetchell/markov-chain-monte-carlo/pull/44)
  [`0524993`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/052499374aa2e30485c1a429a2d3efd39c31731a)

### Documentation

- Include README in rustdoc [`ac8cd20`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/ac8cd2082d5b450099fc4759fdf3004a65f7061c)

  - Use README.md as the user-facing docs.rs landing page through rustdoc inclusion
  - Remove cargo-rdme generation, CI installation, setup checks, and justfile recipes
  - Keep src/lib.rs focused on semantic and API contract documentation
  - Update contributor, release, agent, and development docs for the new documentation layout

### Fixed

- Harden MCMC acceptance and proposal invariants [#35](https://github.com/acgetchell/markov-chain-monte-carlo/pull/35)
  [`8ad2703`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/8ad27038dd9c79380ac1d77d35834b481bb51a85)

  - keep Metropolis-Hastings acceptance decisions in log space
  - explicitly reject arithmetic-created NaN acceptance ratios
  - document log_prob semantics and numerical behavior at the crate level
  - strengthen ProposalMut::propose_mut(None) contract
  - remove unnecessary S: Clone bound from by-value Proposal APIs
  - make Sampler chain storage private and expose chain_ref/chain_mut accessors
  - update examples, README, and organization docs for by-value proposal wording
  - add tests for extreme log-domain acceptance, no-move rollback, state-dependent log_q_ratio, and -inf edge cases

### Maintenance

- Add changelog and Python tooling [#39](https://github.com/acgetchell/markov-chain-monte-carlo/pull/39)
  [`02deb42`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/02deb4291f2bc89311cdb5c254aa284d81ddc4bb)

  - add git-cliff changelog generation with post-processing and release tag helpers
  - add Ruff, Ty, Pytest, and dprint configuration for repository tooling
  - wire Python, Markdown, and Semgrep checks into justfile workflows
  - add Python Semgrep rules and fixtures for script/test hygiene
  - update release, tooling, code organization, README, references, and agent docs
  - regenerate CHANGELOG.md from local git history

### Removed

- Remove pre-release warning from README [`d48bb16`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/d48bb1604f63cad95020a8f027312c48f902cc3f)

## [0.2.1] - 2026-04-30

### Dependencies

- Bump rand from 0.10.0 to 0.10.1 [#25](https://github.com/acgetchell/markov-chain-monte-carlo/pull/25)
  [`70cec05`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/70cec057f0070c8a4ebf5084f4f6eb8a2ed4eaa5)

### Documentation

- Add release and code organization guides [`8f0282a`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/8f0282a0b054660856c8f18e9de4768ea7694e1d)

  - Add docs/code_organization.md with the crate layout, module responsibilities, testing structure, and development conventions
  - Add docs/RELEASING.md with the simplified manual release workflow for this crate
  - Link the new developer docs from README.md
  - Record the documentation additions in CHANGELOG.md

### Maintenance

- Upgrade Rust tooling and validation workflows [#32](https://github.com/acgetchell/markov-chain-monte-carlo/pull/32)
  [`8194a00`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/8194a006e9c81a953549de5caa4194cd34a38dc2)

  - Update Rust toolchain/MSRV to 1.95.0
  - Replace tarpaulin coverage with cargo-llvm-cov for local HTML and CI Cobertura reports
  - Add CodeRabbit, CodeQL, Codecov, Codacy/OpenGrep, Taplo, rustfmt, clippy, typos, and Semgrep configuration
  - Add uv-managed Semgrep tooling with pyproject.toml and uv.lock
  - Expand and sort justfile workflows, including lint groups, setup-tools, Semgrep checks, and coverage recipes
  - Add citation/reference metadata and README sections for contributing, citation, references, and AI tooling disclosure
  - Document Rust/tooling workflow in docs/dev/rust.md and update AGENTS.md guidance

## [0.2.0] - 2026-04-06

### Added

- Add Sampler API and split into modules [#9](https://github.com/acgetchell/markov-chain-monte-carlo/pull/9)
  [`d32c9cc`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/d32c9cc1271f1c3d138c6155b6b34315482885f7)

  - feat: add Sampler API and split into modules
  - Add `Sampler&lt;S, T, P, R&gt;` ergonomic wrapper bundling Chain + target +
    proposal + RNG with `step()`/`run()`, `step_mut()`/`run_mut()`, and
    `Iterator` impl for the clone-based path
  - Split crate into modules: `chain`, `error`, `sampler`, `traits`
  - Make Chain bookkeeping fields private; add `log_prob()`, `accepted()`,
    `rejected()`, `total_steps()`, `reset_counters()` accessors
  - Add `McmcError::InfiniteInitialLogProb` and `InfiniteProposedLogProb`
    for +∞ detection with automatic rollback
  - Derive `Copy` on `McmcError`; add `#[must_use]` on `Chain`, `Sampler`,
    and all query methods
  - Add asymmetric proposal tests, -∞/+∞ edge-case tests, error Display
    tests, and doctests for all public Chain/Sampler methods
  - Update examples to use `Sampler` with `reset_counters()` for
    production-only acceptance rates
  - Add `ising_1d` to justfile `examples` and `validate-examples` recipes
  - Update README, CHANGELOG, and crate-level docs
  - Changed: encapsulate Chain state and provide accessor methods

  Make the final public field in Chain private to ensure internal
  consistency between the state and its cached log-probability. Add
  state(), state_mut(), and into_state() accessors. Update examples and
  tests to use the new API. Also mark McmcError as non-exhaustive and
  implement Debug for Sampler.

  - feat: harden API with safe state replacement, +∞ log q-ratio detection, and Debug
  - Replace `state_mut()` with `replace_state()` that recomputes and
    validates `log_prob`, preventing stale-cache bugs
  - Add `into_state()` to consume the chain and recover the state
  - Add `McmcError::InfiniteLogQRatio` for +∞ log q-ratio detection,
    completing the symmetric NaN/+∞ error matrix for all computed values
  - Add `+∞` checks on `log_q_ratio` in both `step` and `step_mut`
    (with rollback in the mut path)
  - Implement `Debug` for `Sampler` (prints chain state)
  - Add tests for all new error paths, accessors, and Debug output

### Dependencies

- Bump proptest in the dependencies group [#6](https://github.com/acgetchell/markov-chain-monte-carlo/pull/6)
  [`a5c29af`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/a5c29af3588d6f38f18a9f7a9349903147bdcaf4)

## [0.1.0] - 2026-03-24

### Added

- Add GitHub Actions CI/CD workflows and justfile linting recipes
  [`51b5377`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/51b53775026f736e7359d601558842f70cbc526c)
- Add cargo-tarpaulin coverage recipes and CI linting dependencies
  [`74c7db5`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/74c7db5e46031cfe142c4b619592f7bb2dfd886d)

  Introduce coverage analysis using cargo-tarpaulin with recipes for local HTML reports and CI XML output. Update the CI workflow to install required linting
  tools on Linux and macOS runners and reorganize the justfile for better logical grouping.
- Add CI tooling, error handling, prelude, and project infrastruc… [#2](https://github.com/acgetchell/markov-chain-monte-carlo/pull/2)
  [`6584b92`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/6584b9257b3a0e0cf92f31347ca2536be8fe2444)

  - feat: add CI tooling, error handling, prelude, and project infrastructure
  - Add McmcError enum with NaN detection for log-probabilities and
    proposal ratios; Chain::new and Chain::step now return Result
  - Add prelude module for convenience re-exports
  - Simplify trait bounds: remove unnecessary State bounds from Target,
    Proposal, and Chain struct definition
  - Add rust-toolchain.toml pinned to MSRV 1.94.0
  - Add BSD-3-Clause LICENSE and update Cargo.toml
  - Add README badges (crates.io, docs.rs, CI, codecov, audit, clippy)
  - Add AGENTS.md with project guidance for AI assistants
  - Add doc test with full Metropolis–Hastings example
  - Install yamllint in CI workflow for Linux/macOS; keep actionlint
    as local-only recipe
  - Add coverage and coverage-ci justfile recipes
  - chore: minor project hygiene fixes
  - Add /coverage to .gitignore
  - Consolidate duplicate editing tools policy in AGENTS.md
  - Wire validate-examples into `just ci`
  - Remove redundant "cargo" component from rust-toolchain.toml
  - Link to crates instead of repos in README.md
- Add in-place mutation API (ProposalMut) for non-Clone state spaces [#4](https://github.com/acgetchell/markov-chain-monte-carlo/pull/4)
  [`ada42eb`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/ada42eb1d5c5719aebd4a8ccbe4bf7e1d907a8e6)

  - feat: add in-place mutation API (ProposalMut) for non-Clone state spaces

  API changes:
  - Remove State marker trait; Chain&lt;S&gt; now works with any S
  - Add ProposalMut&lt;S&gt; trait with associated Undo type for cheap rollback
  - Add Chain::step_mut for zero-copy Metropolis-Hastings
  - Move Clone bound from State to Proposal&lt;S&gt; and Chain::step

  New files:
  - examples/ising_1d.rs: 1-D Ising model demonstrating ProposalMut
  - tests/proptest_chain.rs: property-based tests for MH invariants
    (log_prob consistency, step/step_mut equivalence, counts invariant)

### Changed

- Initial commit: scaffold MCMC crate [`5d7f706`](https://github.com/acgetchell/markov-chain-monte-carlo/commit/5d7f706da2d7c41af619a2f5669cdcd56dae94ba)

[0.4.2]: https://github.com/acgetchell/markov-chain-monte-carlo/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/acgetchell/markov-chain-monte-carlo/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/acgetchell/markov-chain-monte-carlo/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/acgetchell/markov-chain-monte-carlo/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/acgetchell/markov-chain-monte-carlo/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/acgetchell/markov-chain-monte-carlo/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/acgetchell/markov-chain-monte-carlo/tree/v0.1.0
