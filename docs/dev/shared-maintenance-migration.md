# Shared maintenance adoption

MCMC uses the published `research-repo-tools==0.1.3` distribution from PyPI. The
exact tooling pin is included by dev; the optional notebook extra uses the same
version. `uv.lock` records registry artifacts. Setup and CI need no sibling
checkout or local wheel.

## Ownership

| Workflow | Owner after migration |
| --- | --- |
| Changelog generation, archives, notes | Shared CLI; the [pilot record](shared-changelog-pilot.md) records the original comparison |
| Annotated local tags | Shared `changelog tag`; thin `tag` and `tag-force` recipes |
| Branch and uncommitted CodeRabbit reviews (#163) | Shared CLI; consumer tests check recipe forwarding with local stubs |
| Exact Python development pins and stable-uv preflight | Shared `deps update-python` and `deps check-uv` |
| uv owner upgrades and pin reconciliation | Shared `deps update-uv` |
| Rust/Python/Cargo setup, version checks, and execution (#160) | Shared `setup` and `toolchain check/run/sync` |
| Managed Cargo upgrades | Shared `toolchain upgrade`, resolving [upstream #19](https://github.com/acgetchell/research-repo-tools/issues/19) |
| Dependency workflow composition | Thin consumer recipes following the published template; Cargo requirements/lock and full Python lock refresh retained |
| Release metadata (#164) | Shared release plans and configured rules; a small MCMC baseline-command adapter |
| Semgrep fixture harness | Shared CLI; MCMC owns rules, fixtures, and expected counts |
| Notebook checks, cleanup, and fresh-kernel execution | Shared CLI; MCMC owns selection, input preparation, scientific content, and figure promotion |
| Clippy SARIF converters | Shared toolchain catalog and setup; upstream #25 is resolved in v0.1.3 |
| Performance mechanics (#164) | Shared Criterion parsing/comparison, identity comparison, byte hashing, extraction, and transactional publication |
| Scientific policy and retained artifacts | MCMC workload selection, measurement, legacy schema/digest adapters, and historical renderers |

The v0.1.3 migration removes the local subprocess module and its generic tests.
Both SARIF converters now use the same exact declarations, setup, and cache as
the other Cargo tools. Markdown and coverage keep their existing shared commands.

## Toolchain and policy changes

The authority is `[tool.uv].required-version` for uv, `.python-version` for
Python, `rust-toolchain.toml` for Rust/components/targets, and
`[tool.research-repo-tools.toolchain.cargo]` for supported Cargo tools.
`llvm-tools-preview` is now declared with the other Rust components.

Shared setup installs isolated Rust/Cargo versions and managed Python, supplies
Just through its pinned `rust-just` dependency, configures user shell PATH, and
synchronizes dev. `just tools-check` never installs or synchronizes anything.
Recipes needing Cargo tools use checked `toolchain run` paths. CI uses the same
declarations and a shared setup action, caching by OS, architecture, and inputs.

`just update` upgrades uv through its supported installation owner, upgrades
declared Cargo tools, runs setup, then updates Cargo/Python dependencies.
Unsupported uv owners receive manual guidance. Upgrading user-level uv affects
other checkouts with exact uv pins. Managed Cargo upgrades publish pins only after
successful installation/verification and retain old installations on failure.
The old cargo-update dependency and Just pin reconciler have been removed.
The shared package and its Just dependency remain explicit package-pin changes.

## Consumer release policy

Required publication files, three fixed concept DOI assertions, 30 active README
links, and the current benchmark-command tag are declared as release rules in
`pyproject.toml`. Update the selected paths and counts when active documentation
changes. Performance report/archive links remain excluded from source-link
updates. Fixed assertions fail before any repair, including when all three DOI
surfaces agree on the wrong value.

`just release-check` invokes `research-repo-tools release check --final-release`
directly; there is no local checker or compatibility entry point.
`update-release-version` uses `plan_release` and
`apply_release`; it owns only stable-tag spelling and the callback advancing
MCMC performance-command baselines. The callback reads the shared candidate,
returns bytes, and never stages or publishes its own tree. Keeping the baseline
update in that callback lets ordinary checks remain offline. An explicit previous
release and date permit an offline `--dry-run`.

## Performance adoption and retained adapters

| Local module | Remaining purpose |
| --- | --- |
| `bench_compare.py` | Existing command defaults, nanosecond legacy-schema adapter, and historical Markdown formatting/scientific prose |
| `archive_performance.py` | MCMC release-pair selection, measurement/worktrees, authenticated GitHub asset selection, CSV and v1/v2 provenance parsing, archive naming, and scientific eligibility |
| `publish_performance_readme.py` | Current-release eligibility, future-release preparation policy, retained table/SVG renderers, and scientific interpretation |
| `update_release_version.py` | Stable-tag command contract and documented performance-baseline callback |

Criterion estimates and complete comparison inventories use the public shared
API with an explicit nanosecond unit. The retained CSV has no confidence-level
column; its adapter preserves the recorded marginal bounds and existing report
meaning. Generic parsing, ratios, hashes, deterministic JSON, harness identity
comparison, bounded extraction, process diagnostics, and rollback use supported
shared interfaces. Live inherited Cargo output uses shared executable resolution
with a direct standard-library process because the public captured runners do
not stream output.

The shared evidence envelope is a different schema. MCMC retains its CSV and
provenance parsers and uses the public byte transaction for their publication.
Loaded payload and sidecar bytes survive save and promotion unchanged, including
v1 JSON, field order, whitespace, and LF/CRLF. The legacy source digest framing
also remains unchanged; replacing it with the shared fingerprint format would
relabel scientific evidence. A future envelope or digest migration needs distinct
schema identifiers and filenames.

Authenticated `gh release download` remains local: shared HTTPS retrieval requires
an independently trusted digest, which the historical release contract does not
provide. Extraction uses shared archive limits and rejects nonportable names,
links, and aliases before writing. Benchmark execution, source inventory, CPU
capture, and judgments about workload/hardware compatibility stay consumer-owned.

README publication snapshots the exact validated CSV, JSON, report, and Cargo
metadata bytes, then uses shared marker validation, candidate planning, tag
verification, stale-input checks, and multi-output publication. Existing tags
must contain exact stored blob bytes; Git filters and newline conversion cannot
make different bytes match. `.gitattributes` disables text conversion for retained
reports and evidence on checkout. MCMC's explicit future-release working-tree
allowance remains a small adapter; it never skips checks for an existing tag.

Historical table and Matplotlib SVG renderers remain because shared renderers
intentionally produce a different layout. The v0.4.2 retained CSV, JSON, report,
and SVG are exercised as consumer fixtures. No retained evidence or figure is
regenerated in this migration. New files created by shared transactions start
owner-only; existing modes are preserved. Publication is recoverable on caught
failures, not crash-atomic across files.

## Notebook migration

The local notebook parser, extracted-code linting, executor, cleanup engine, and
`check-notebooks` entry point are removed. Native Ruff/ty checks read the original
notebook. Source cell IDs and code are preserved. The scientific notebook's input
validation, acceptance statistics, explicit path semantics, and figures remain
consumer-owned.

`just notebook-sync` registers a project-local kernel. Fast/slow selection and
the Ising input dependency stay in Just. Lint/cleanup include tracked and
non-ignored notebooks, excluding checkpoints. Execution artifacts mirror the
root-relative notebook path under `target/notebooks/`; a sibling report records
source/lock hashes, environment versions, status, and failing cell identity.
Runtime state uses private temporary directories. Figure promotion remains
`just notebook-ising-figure`.

## Regression ownership and verification

The upstream suite owns common parsers, changelog/tag behavior, dependency pin
resolution, review execution, installation mechanics, archive extraction, marker
validation, notebook structure/cleanup, and subprocess contracts. MCMC does not
retest those implementations. Its tests exercise local Just argument forwarding
and failure propagation, CI wiring, configured release policies, installed
consumer scripts, and scientific notebook path/figure behavior. Performance tests
retain evidence-schema and rendering fixtures, measurement selection, historical
archive policy, and checks that local code publishes all outputs together.
Live CodeRabbit review is separate, explicitly authorized work.

Four Python modules remain. Most production code is the performance workflow:
benchmark execution and source capture, release-pair policy, authenticated asset
selection, and the legacy evidence adapters above. The v0.1.3 CLI does not own
that orchestration and its comparison/publication formats differ. Removing those
adapters requires an explicit format migration or additional upstream APIs; the
historical byte contract cannot be replaced by the shared envelope implicitly.

This is the current migration boundary, not the intended permanent ownership.
Worktree lifecycle, working-tree snapshots, host/toolchain capture, Criterion
execution, release-asset retrieval, and report archival belong in shared tooling
when exposed through supported APIs. MCMC should supply benchmark commands,
source paths, release-selection preferences, workload meanings, and scientific
acceptance rules. Existing evidence can remain immutable while future runs adopt
the shared format and renderers; exact reproduction of the old layout need not
remain a permanent local implementation requirement. A legacy reader is a
transition aid, not a reason to keep generating the old schema indefinitely.

The intended endpoint is configuration, thin Just recipes, scientific notebooks,
and a small consumer integration suite. Once the last local command disappears,
the installable Python tooling package and its wheel/entry-point tests can also
be removed. Generic workflow regressions should move with their implementation
to the shared repository.

Run `just check` and `just ci` before handoff. The hosted matrix runs the full
consumer gate on Linux, macOS, and Windows; local macOS results do not establish
native Linux or Windows success. For package upgrades, change the exact tooling
and notebook-extra pins together, refresh the lock, read the release notes, and
recheck these consumer boundaries before removing retained adapters.
