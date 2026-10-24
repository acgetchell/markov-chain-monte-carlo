# Shared maintenance adoption

MCMC pins the published `research-repo-tools==0.2.3` registry distribution.
The tooling group and shared notebook extra use the same exact version; `uv.lock`
records PyPI wheel/sdist hashes. No sibling checkout, editable package, local wheel,
or private import is needed. This completes the local extraction planned in #166
after #164; hosted CI and the first live release-asset pair remain separate evidence.

## v0.2.3 adoption

The managed adoption command advances both shared-package pins and the root Python
baseline to 3.15, refreshes the lock, and rebuilds the environment and notebook kernel.
Semgrep initially used an independent Python 3.14 project. Following Delaunay,
it now shares the root Python 3.15 environment and lockfile, retaining its exact pin,
PyJWT compatibility floor, patched MCP 1.x and OpenTelemetry family, and Python 3.15
Windows wheels. Its source distribution cannot substitute for the native engine.
Local scans and fixtures retain the Linux stack-limit workaround, as does SARIF.

`just update-python-dependencies` refreshes the shared lock. Local scans, rule fixtures,
SARIF, OSV, Dependabot and the dependency approval file policy use the root manifest
and lock. Registry-only release jobs pin v0.2.3 and the declared
uv version, with their resolution cutoff aligned to the adopted root lock.

Python check, fix, and typecheck recipes now call the shared Python gates, including
strict Ty diagnostics and the complete source/fixture inventory. Ruff and Ty remain
consumer-owned at their existing locked versions. Release preflight delegates tag,
version, metadata and release-note validation to `release check TAG`; the local
recipe retains only the current-version default and MCMC crate-name assertion.

## Remaining v0.2.3 capability adoption (#208 and #210)

This follow-up consumes the same exact published package and notebook extra. Ruff, Ty, and pytest retain their consumer-owned requirements and locked
versions; Python-tool inheritance remains disabled. Existing staged dependency updates are independent of this migration.

- The shared Actions YAML checker now enforces the strict committed selected-actions policy in local checks and CI. Explicit GitHub-owned action identities
  and CodeQL subpaths replace the broad exemption and duplicated Semgrep allowlist. Existing SHA and version-comment rules remain. The shared checker accepts
  workflows, so a focused consumer test passes the actual setup composite's steps through the same policy. This is the only composite adaptation; no local
  allowlist parser or production wrapper was added. The authentication workflow and its regression retain the exact reviewed commit.
- Trusted benchmark preflight captures the repository, tag, release ID, full commit, and title assertion. Job outputs preserve that target independently of
  measured checkout files. Checkout/HEAD/tag checks bind measurement to the captured commit; registry-only upload uses the original `--target` outside the
  checkout. Focused integrations exercise the installed capture/upload commands, missing or malformed handoffs, changed identities, and checkout mismatches.
- Notebook tests use public `notebook_testing.isolated_project`, `NotebookProject.execute`, and returned reports. They retain explicit MCMC environment
  cleanup, source-byte preservation, injected scientific assertion cells, trace roots, timing/ACF/ESS/R-hat semantics, figures, and exact retained evidence.
- Command and publication tests use public `just_inspect.inspect_justfile` and `dry_run`. Four generic helpers (`notebook_project`, `execute`, `_run_just`,
  and `_recipes`) are removed. One consumer fixture retains notebook selection and MCMC environment cleanup. The command/notebook files retain all 56
  collected cases and add three policy cases, for 59; no scientific cases were removed. The separate publishing file adds ten handoff cases to its existing
  17. Generic copying, interpreter setup, TOML slicing, subprocess discovery, Just JSON parsing, and sidecar reconstruction now belong to the package.
- Shared `validation cargo-examples` discovers binary targets and compiler-reported executable paths. The separate Ising selection prepares notebook inputs
  first; the remaining plan discovers all other targets. Both retain all three features, development builds, 300-second execution and 1800-second build
  budgets, unchanged scientific markers, and build-time revision/dirty-state provenance. CI checks Cargo metadata coverage and runs each example once.
- dprint and rumdl move to checksum-verified managed release binaries at their existing versions. `tools-sync-binaries` supports installation/repair;
  `update-tools` upgrades tools without repeating setup. Contributor setup remains explicit when declarations change.

The report-inventory classifier remains locally owned under #209. Its prerequisite,
[research-repo-tools #133](https://github.com/acgetchell/research-repo-tools/issues/133), is still open and absent from published v0.2.3 as checked on
October 23, 2026. Retain initialized reproduction and empty/orphan/path regressions until an equivalent shared interface is published. This change does not
acquire release benchmark measurements, regenerate retained performance evidence, modify the README, publish a release, or apply GitHub settings.
Native Linux/Windows and hosted workflow execution still require CI on the eventual adoption revision; local validation records macOS arm64 only.

On October 23, 2026, the uncommitted adoption worktree based on `53e42e1163db43a50e2576577a593e6d08be7f7e` passed `just ci` on native macOS arm64 with
installed registry `research-repo-tools==0.2.3` and Python 3.15.1. This includes 160 Python consumer cases, 415 Rust nextest cases, 216 doctests,
both scientific notebooks, all eight Cargo examples, benchmark compilation, and repository validators. Zizmor 1.31.1 used authenticated online audits.
Focused installed-command
handoff, composite-policy, inventory, and notebook integrations preceded the aggregate. Jupyter execution required local kernel sockets outside the sandbox;
no regression was skipped or weakened. Both pre-existing staged uv lockfiles remain byte-for-byte unchanged. Linux/Windows and live release handoff remain
pending their native/hosted execution; no throwaway release or release benchmark measurement was used as a handoff test.

## Artifact promotion assessment (#202)

This adoption retains native Cargo build/package verification and the existing
registry publication controls. Trusted main-artifact promotion is deferred:
shared immutable-selection/download primitives do not establish native package
validity, scientific correctness, benchmark freshness, or source/harness provenance.
The current release process still measures and attaches the tagged benchmark
baseline before publishing the draft GitHub Release.

Rust artifact reuse needs a separately reviewed producer, native validation and
promotion contract bound to the exact commit, artifact IDs and digests. The
reviewed publication flow remains tracked in #176; #182 still owns adopting the
shared release-guide procedure and the remaining publishing/verification recipes.
This migration does not publish packages or replace released artifacts.

## Reviewed release adoption (#176 and #182)

The v0.5.2 implementation uses Delaunay's release procedure and recipe interfaces,
backed by the already-published v0.2.3 shared implementation. `update-version TAG`,
`release-version-check`, `tag TAG`, `release-package`, `release-publish TAG` and
`release-verify TAG` replace transitional wrappers. CI delegates final metadata
checks to the shared CLI without a local Cargo version/name parser. Each crate
retains its scientific validation, feature selections and benchmark evidence.

Benchmark preparation now attaches its asset and leaves the release in draft.
Human-reviewed GitHub publication triggers a separate crates.io workflow using
the shared event gate, actual native CI check/provider identities, clean native
Cargo package verification and environment-scoped OIDC. The consumer also binds
the published GitHub title/notes to the tag and shared changelog output, and
requires an immutable published release. Registry checks block
already-published versions and uncertain lookup results; post-upload verification
still runs after a Cargo visibility timeout. No second manual upload is needed.

The existing exact package pins, lock and inherited Python baseline already
satisfy the published-package prerequisite and remain in place. Live deployment
settings and first-release evidence are tracked separately in
[crates.io deployment setup](crates-io-publishing.md); #176 remains open until
they are verified. Trusted main-artifact promotion remains deferred.

## Ownership inventory

| Consumer surface | Disposition and replacement |
| --- | --- |
| Four production Python modules (1,991 lines) | Removed; public shared performance, publication, and release CLI |
| Release update wrapper and callback | Replaced by canonical-stable policy, DOI/link rules, and version-independent examples |
| Release discovery, worktrees, binary patches, untracked files | Shared `performance measure`; explicit Git mutation opt-in, publication-order selection |
| Stepping execution, sample selection, source/host/toolchain capture | Shared measurement; `tooling/benchmark.toml` retains MCMC command, inventories, and compatibility policy |
| Criterion comparison, CSV/evidence serialization, reports, SVG/table rendering | Shared models and renderers; scientific prose and row selection in `tooling/` |
| Historical CSV/provenance interpretation | Retired with the old performance series during #154; new series uses native shared evidence |
| Authenticated release assets | Shared `performance assets`; only native shared archives are supported after the reset |
| Archival, index/link updates, immutable pairs, promotion | Shared `performance promote`; new paths under `docs/performance/v1/` |
| Current-version checks, future-release preparation, tagged blobs, stale inputs | Shared `performance publish`; construct reviewed publication TOML when the first new pair exists |
| Release workflow inline Python, packaging, draft checks, retry/upload/publish shell | Shared `performance baseline/release-draft/release-upload`; separate read-only benchmark and registry-only writer jobs |
| Setup composite inline environment exporter | Shared `toolchain export`; uv/cache inputs remain thin workflow configuration |
| Just file discovery, batching, prerequisites and metadata validation | Shared `files run` and `validation require/cargo-metadata` |
| Example discovery, builds, executable paths and output loops | Shared `validation cargo-examples`; scientific contracts in `tooling/examples.toml` and `tooling/ising-example.toml` |
| Notebook parser, kernel/lint/execution, cleanup and provenance | Shared notebook CLI; fast/slow selection, Ising trace ordering and figure destination remain in Just |
| SARIF and coverage | Native report producers; shared `coverage report` validates and summarizes Cobertura, with no local parser or converter |
| Obsolete managed installations | Shared `toolchain clean`; `tools-clean` previews by default and accepts other consumers' retention roots |
| Rust test/check gates, CodeQL, audit, cache/permission policy | Native tools/actions and thin Just composition retained |
| Dependabot approval and auto-merge | SHA-pinned shared workflow; local repository and file policy |
| Dependency and secret scans | Shared OSV/Gitleaks commands and managed binaries; local lockfile scope, schedules, and badges |
| Local package/build/console scripts and wheel/entry-point tests | Obsolete and removed; dependency-only uv environment |
| Production-only Python portability rules/fixtures | Obsolete locally after deletion; shared package owns byte transport and text publication |
| Generic parser, process, worktree, rendering, rollback and upload tests | Shared upstream; local duplicates removed |
| Scientific workload, notebook, release policy and representative integration tests | Retained under `tests/tooling/` |

No executable production Python remains. Shell is limited to native command composition,
explicit slow-notebook selection, coverage-directory creation, and normal CI pipelines.
There is no replacement callback, workflow heredoc, copied renderer, or local framework.

## Environment and release policy

The uv pin advances from 0.12.17 to the installed stable 0.12.18, matching the shared
v0.1.5 release's tool baseline. Rust remains 1.98.1 and all Cargo-tool pins remain
unchanged. Setup/cache keys still use OS, architecture, tool declarations and lock identity.

The environment has `tool.uv.package=false`, no build backend, console scripts, or
self-referencing extras. Its placeholder version is independent of the Rust release.
The notebook group directly selects the shared notebook extra plus Matplotlib and Polars.

Release checks remain offline. Three fixed DOI assertions, required publication files,
active README source links, and historical artifact exclusions remain declared in
`pyproject.toml`. The current command, `just update-version TAG --previous-release PREVIOUS_TAG --date RELEASE_DATE --offline --dry-run`,
previews the complete release plan offline. No callback advances benchmark examples.

## Performance series reset

The maintainer requested a fresh start during #154. Earlier release-performance reports, the original CSV/provenance/SVG files, converted shared companions,
and the old README comparison have been removed. The legacy asset-layout adapter and publication selection were also retired. This supersedes the original
migration's historical-byte preservation requirement. No old measurements are relabeled as current data.

At the reset, the checkout could produce local Criterion baselines without Git changes; the next tagged release was to establish the shared baseline. The series
now starts at v0.5.0, with a retained comparison of the prepared v0.5.1 tree against that baseline. `performance-check` explicitly accepts an empty release
inventory and rejects orphaned evidence without its report. Actual report validation still runs through shared `performance promote --check`.

New release evidence and generated navigation belong under `docs/performance/v1/`, with `performance.md` as the configured current report.
`tooling/performance-readme.toml` selects that retained pair for README publication. The diagnostic-backend experiment remains
separate under `experiments/` because it records the current backend decision. See [benchmarking](../BENCHMARKING.md) for the current workflow.

## Regression ownership and reduction

The baseline at `b717f48` contains 1,991 production Python lines, 2,987 test Python
lines, and 217 collected tests. Counts exclude notebook cells, embedded workflow
programs, and deliberately invalid Semgrep fixtures.

| Python surface | Before extraction | After initial extraction |
| --- | ---: | ---: |
| Production lines | 1,991 | 0 |
| Test lines (including initializer) | 2,987 | 535 |
| Collected tests | 217 | 28 |

Retained tests cover:

- `test_benchmark_contracts.py`: workload names, fixture construction outside timed
  regions, fresh batches, stepping configuration, and same-host eligibility policy.
- `test_notebooks.py`: explicit trace/root semantics, unchanged notebook source,
  and selected figure destinations exercised through shared execution.
- `test_release_policy.py`: MCMC's fixed concept DOI, active source links and
  historical exclusions, using copies of the actual metadata and release policy.
- `test_commands.py`: actual scan exclusions, Python policy, scientific CI dependencies,
  release writer pins and credential separation; review remains outside validation.
- `test_performance_evidence.py`: the empty first-report state, rejection of orphaned release artifacts, and delegation to shared report validation.

Generic implementation tests now run upstream. Wheel/entry-point tests are obsolete;
existing release-workflow tests are replaced by a consumer wiring/permission check.
The shared upstream suite owns saved-sample comparison, notebook lint/provenance,
final-changelog enforcement, and review argument/failure matrices. Local tests do not
repeat those suites; representative integrations exercise MCMC's actual configuration
and content. Native notebook linting remains part of the regular validation gate.
The scientific benchmark lifecycle and notebook tests were retained, not traded for
a smaller count.

## v0.1.7 adoption and further reduction

The official `toolchain adopt` preview and apply commands updated both exact registry
pins and `uv.lock`, preserving all other locked dependency versions. The environment
opts into `toolchain.inherit-python`; the installed package owns the Python baseline
and checks `.python-version`, the dependency-only runtime requirement and package pins.
Ruff and Ty infer their targets from project metadata, replacing two redundant targets
and local pin assertions. `python-typecheck` runs the shared drift check before Ty.

`shared-python-plan VERSION` and `shared-python-update VERSION` expose the upstream standalone preview and apply commands without starting the old environment.
A future package release can therefore carry its Python baseline, both package pins, lockfile, environment, and notebook kernel into MCMC together.
The registry-only release jobs select a package-compatible interpreter without a numeric Python override. Their separate package pins and resolution cutoffs
remain reviewed release inputs and must be aligned when adopting another shared release.

Consumer Python tests shrink from 788 to 664 lines and from 40 to 29 collected cases,
including the new Dependabot caller/token and security-workflow boundary checks.
Generic recipe shape/help checks, shared review argument checks, required-file failure
cases and document/SVG renderer checks are removed. Release-policy fixtures now copy
the actual metadata instead of constructing a second release and editing its policy.
`performance-check` owns offline report reproduction. The initial adoption removed the unused CSV conversion configuration; #154 subsequently retired
that historical evidence series and replaced its preservation tests with first-report state checks.

At that adoption, the registry-only release writer jobs used v0.1.7 and uv 0.12.19. Their resolution cutoff
was after both v0.1.7 PyPI artifacts were uploaded. Native Rust, scientific notebook,
benchmark lifecycle, project-rule fixtures and consumer security boundaries remain
locally owned. The existing batched native Semgrep scan stays in place: the shared
per-file JSON/SARIF scan is not a replacement for its command and performance contract.
Dependabot approval now calls the separately SHA-pinned shared workflow. The caller
owns only repository identity and dependency-file policy; signed-head eligibility,
approval and native auto-merge live upstream. The personal-token CodeRabbit request
is removed. See [Dependabot automation](DEVELOPING.md#dependabot-automation) for GitHub
settings, the required-check boundary, and token removal after merging the caller.

OSV-Scanner and Gitleaks use exact managed binary pins and shared report handling.
OSV covers all maintained Python and Rust lockfiles; Gitleaks covers full reachable history
and current tracked/nonignored files. Dedicated security workflows retain native
JSON/SARIF reports, with Gitleaks findings redacted, and supply README badges.
The consumer test checks lockfile coverage, full-history checkout, and failure
propagation; scanner implementation and generic report tests remain upstream.

The #13 completion audit also exposes `tools-clean` without changing the existing build cleanup command, and `coverage-report` for saved Cobertura XML.
`coverage-ci` now calls that shared parser after generation; the workflow no longer duplicates directory setup and shell file-existence checks.
The scalar diagnostics example joins the shared example-output inventory and final CI selection. Numerical diagnostic tests remain Rust-owned;
consumer checks protect example inventory coverage and the actual Codecov recipe, while generic XML parsing and tool-retention tests stay upstream.

## Documentation and remaining v0.1.7 wiring (#154)

The documentation ownership refactor retains the exact v0.1.7 pins and shared performance workflow; the maintainer-requested performance reset is described
above. At that point, active README guide links followed `main` through a fixed-value release rule, while source, image, and metadata links used the current
tag. The v0.5.0 release preparation briefly changed active guides to the declared release tag, breaking navigation before that tag existed.
Active repository links now follow `main`, and API links use docs.rs `latest`, independently of release metadata. See the current
[documentation ownership and public-link policy](../code_organization.md#documentation-ownership-and-names).

Notebook linting now opts into shared `notebooks.prohibit-installs`, so literal dependency-install cells fail the locked-environment policy. The setup
composite supplies the job's GitHub token only to shared setup for authenticated scanner release-metadata reads; it retains the caller's permissions.
Consumer checks exercise install rejection, release-link revision boundaries, and token scope. The shared package owns their generic implementations.

## Audit and Python policy adoption (#150 and #153)

The v0.1.6 update consumes the shared zizmor command and complete Python selection.
The local and SARIF recipes verify the existing zizmor 1.30.1 Cargo pin and explicit
regular persona. Shared token discovery and reported offline fallback serve local
contributors; the SARIF workflow requires online audits, fails on findings through
plain output, and generates SARIF separately even after findings. Fork and Dependabot
runs retain the audit gate but skip privileged upload. This replaces zizmor-action,
so an action-specific scanner-input rule is no longer needed.

`python-check` is the direct CI dependency for both consumer and fixture linting.
It applies the complete configured Ruff policy and Ty to every selected `.py` and
`.pyi`; notebook discovery covers every source `.ipynb`. Missing annotations,
annotation-only imports and quoted annotations are governed by ANN, strict TC and UP.
The existing tests and scientific notebook satisfy this policy. The deliberate
exception fixture has exact per-file rule exceptions, including only ANN201 from
the typing rules. CodeRabbit excludes Semgrep fixtures and disables its duplicate
docstring-percentage check. Local Ruff, Ty and Semgrep still validate those fixtures.

Three consumer checks protect this wiring and policy. Generic authentication,
discovery and failure matrices remain upstream. Both registry-only release writer
steps also use v0.1.6 with a resolution cutoff after its PyPI publication.

For the v0.1.5 extraction, local macOS arm64 validation on 2026-09-23 passed `just check` and `just ci`,
including 274 Rust nextest tests, 188 doctests, notebook execution,
example validation and benchmark compilation. After test deduplication, `just check`
and all 28 retained Python tests passed again. The real offline release dry-run
for the next Cargo version proposed only Cargo metadata, citation and README changes;
it left the dependency-only Python environment and historical artifacts unchanged.

The v0.1.6 adoption also passed both gates on macOS arm64 on 2026-09-23, including
31 Python tests, 274 Rust nextest tests, 188 doctests, notebook execution, examples
and benchmark compilation. Authenticated online, explicit offline and unauthenticated
fallback scans reported no findings; the online SARIF command produced a valid
2.1.0 report from zizmor 1.30.1. The registry-only release command resolved under
its new cutoff. Workflow wiring is covered locally; hosted SARIF upload and the
native Linux/Windows runs still require GitHub Actions evidence.

Use `just check` during iteration and `just ci` for final readiness without immediately
repeating the smaller gate. The full gate remains configured for native Linux,
macOS and Windows. A local macOS arm64 pass does not prove native Linux or Windows
success. Live worktree measurement requires maintainer execution under the repository's
Git policy; live asset publication requires the next release pair. Local fixture evidence
must not be presented as either. No CodeRabbit review is part of these gates.
