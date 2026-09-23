# Shared maintenance adoption

MCMC pins the published `research-repo-tools==0.1.6` registry distribution.
The tooling group and shared notebook extra use the same exact version; `uv.lock`
records PyPI wheel/sdist hashes. No sibling checkout, editable package, local wheel,
or private import is needed. This completes the local extraction planned in #166
after #164; hosted CI and the first live release-asset pair remain separate evidence.

## Ownership inventory

| Consumer surface | Disposition and replacement |
| --- | --- |
| Four production Python modules (1,991 lines) | Removed; public shared performance, publication, and release CLI |
| Release update wrapper and callback | Replaced by canonical-stable policy, DOI/link rules, and version-independent examples |
| Release discovery, worktrees, binary patches, untracked files | Shared `performance measure`; explicit Git mutation opt-in, publication-order selection |
| Stepping execution, sample selection, source/host/toolchain capture | Shared measurement; `tooling/benchmark.toml` retains MCMC command, inventories, and compatibility policy |
| Criterion comparison, CSV/evidence serialization, reports, SVG/table rendering | Shared models and renderers; scientific prose and row selection in `tooling/` |
| Historical CSV/provenance interpretation | Shared `performance convert` with bounded `legacy-csv.toml`; no local reader or writer |
| Authenticated legacy release assets | Shared `performance assets` and `legacy-baseline.toml`; exact historical layout declared locally |
| Archival, index/link updates, immutable pairs, promotion | Shared `performance promote`; new paths under `docs/performance/v1/` |
| Current-version checks, future-release preparation, tagged blobs, stale inputs | Shared `performance publish`; independently reviewed pins, inventories and references in publication TOML |
| Release workflow inline Python, packaging, draft checks, retry/upload/publish shell | Shared `performance baseline/release-draft/release-upload`; separate read-only benchmark and registry-only writer jobs |
| Setup composite inline environment exporter | Shared `toolchain export`; uv/cache inputs remain thin workflow configuration |
| Just file discovery, batching, prerequisites and metadata validation | Shared `files run` and `validation require/cargo-metadata` |
| Example binary suffix detection and output loops | Shared `validation run`; six scientific output contracts in `tooling/examples.toml` |
| Notebook parser, kernel/lint/execution, cleanup and provenance | Shared notebook CLI; fast/slow selection, Ising trace ordering and figure destination remain in Just |
| SARIF and coverage | Native Semgrep, clippy-sarif, sarif-fmt and cargo-llvm-cov; no local parser or converter |
| Rust test/check gates, CodeQL, audit, Dependabot, cache/permission policy | Native tools/actions and thin Just composition retained |
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
30 active README source links, and historical artifact exclusions remain declared in
`pyproject.toml`. `just update-version TAG --previous-release PREVIOUS --date DATE --dry-run`
previews the complete release plan offline. No callback advances benchmark examples.

## Evidence transition

Historical `docs/PERFORMANCE.md`, `docs/archive/performance/`, and the README performance
section are unchanged. Shared companions retain every v0.4.2 point and marginal bound,
both sample inventories and coverage gaps, original hashes, release/source identities,
and opaque legacy records. Missing confidence levels remain unknown.
Legacy source digests are never promoted to shared fingerprints.

New shared reports use `docs/performance/v1/current.md` and pair-specific
`.comparison.json`, `.evidence.json`, and CSV files beside it. Future archival and
promotion operate only on that shared path. Report formatting may differ; numerical
meaning and scientific limitations are unchanged.

The legacy CSV configuration is a bounded migration aid. Retire it and its conversion
test once consumers no longer invoke conversion and all retained evidence uses verified
companions. Retire the legacy baseline layout once ordinary release pairs use shared
assets. Originals remain immutable after both retirements; no local Python is retained
for either boundary.

The existing historical README cannot be regenerated into a different format under
its already published tag. The checked-in selection records that historical pair but
fails future-release eligibility. Before the next publication, measure the new release
and update the explicit selection and independently verified provenance pins. Shared
publication then checks the current Cargo version, report identity, measured inputs,
and exact existing-tag blobs. See [benchmarking](../BENCHMARKING.md).

## Regression ownership and reduction

The baseline at `b717f48` contains 1,991 production Python lines, 2,987 test Python
lines, and 217 collected tests. Counts exclude notebook cells, embedded workflow
programs, and deliberately invalid Semgrep fixtures.

| Python surface | Before | After |
| --- | ---: | ---: |
| Production lines | 1,991 | 0 |
| Test lines (including initializer) | 2,987 | 535 |
| Collected tests | 217 | 28 |

Retained tests cover:

- `test_benchmark_contracts.py`: workload names, fixture construction outside timed
  regions, fresh batches, stepping configuration, and same-host eligibility policy.
- `test_notebooks.py`: explicit trace/root semantics, unchanged notebook source,
  and selected figure destinations exercised through shared execution.
- `test_release_policy.py`: MCMC DOI/reference requirements, active source links,
  historical exclusions, and the independent non-package environment.
- `test_commands.py`: pinned registry/configuration, canonical recipe exposure,
  shared review modes, actual scan exclusions, scientific CI dependencies and credential separation.
- `test_performance_evidence.py`: every historical value/bound and coverage label,
  opaque source identity, verified companions, offline reproduction,
  and representative document/SVG publication.

Generic implementation tests now run upstream. Wheel/entry-point tests are obsolete;
existing release-workflow tests are replaced by a consumer wiring/permission check.
The v0.1.5 upstream suite owns saved-sample comparison, notebook lint/provenance,
final-changelog enforcement, and review argument/failure matrices. Local tests do not
repeat those suites; representative integrations exercise MCMC's actual configuration
and content. Native notebook linting remains part of the regular validation gate.
The scientific benchmark lifecycle and notebook tests were retained, not traded for
a smaller count.

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

Run `just check` and `just ci`. The full gate remains configured for native Linux,
macOS and Windows. A local macOS arm64 pass does not prove native Linux or Windows
success. Live worktree measurement requires maintainer execution under the repository's
Git policy; live asset publication requires the next release pair. Local fixture evidence
must not be presented as either. No CodeRabbit review is part of these gates.
