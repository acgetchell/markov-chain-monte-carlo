# Shared maintenance adoption

MCMC uses the published `research-repo-tools==0.1.2` distribution from PyPI. The
exact tooling pin is included by dev; the optional notebook extra uses the same
version. `uv.lock` records registry artifacts. Setup and CI need no sibling
checkout or local wheel.

## Ownership

| Workflow | Owner after migration |
| --- | --- |
| Changelog generation, archives, notes | Shared CLI; the [pilot record](shared-changelog-pilot.md) records the original comparison |
| Annotated local tags | Shared `changelog tag`; thin `tag` and `tag-force` recipes |
| Branch and uncommitted CodeRabbit reviews (#163) | Shared CLI; consumer integration tests use local stubs |
| Exact Python development pins and stable-uv preflight | Shared `deps update-python` and `deps check-uv` |
| uv owner upgrades and pin reconciliation | Shared `deps update-uv` |
| Rust/Python/Cargo setup, version checks, and execution (#160) | Shared `setup` and `toolchain check/run/sync` |
| Managed Cargo upgrades | Shared `toolchain upgrade`, resolving [upstream #19](https://github.com/acgetchell/research-repo-tools/issues/19) |
| Dependency workflow composition | Thin consumer recipes following the published template; Cargo requirements/lock and full Python lock refresh retained |
| Release metadata | Shared public CLI composed with consumer policy and transaction handling |
| Semgrep fixture harness | Shared CLI; MCMC owns rules, fixtures, and expected counts |
| Notebook checks, cleanup, and fresh-kernel execution | Shared CLI; MCMC owns selection, input preparation, scientific content, and figure promotion |
| Clippy SARIF converters | Existing exact CI pins/installers retained pending [upstream #25](https://github.com/acgetchell/research-repo-tools/issues/25) |
| Rust library, scientific validation, benchmarks, evidence, and publication | MCMC |

The SARIF converters `clippy-sarif` and `sarif-fmt` are rejected by the published
v0.1.2 catalog. Removing their retained installation would break Code Scanning.
The follow-up is tracked with v0.1.2 consumer acceptance but requires a subsequent
package release now that v0.1.2 is published.

Markdown already uses rumdl, and coverage uses cargo-llvm-cov without a duplicated
summary implementation. No extra gate is added merely because the package offers
one. `subprocess_utils.py` remains necessary for performance workflows.

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

`update-release-version` invokes the shared public CLI in a temporary tree,
then applies MCMC performance-command pairs and non-evidence README links,
validates the fixed concept DOI and required publication files, and publishes
the complete candidate through the existing rollback mechanism. An explicit
previous release and date permit an offline `--dry-run` preview.

`release-check` composes the shared final-release check with these MCMC policies.
Measured reports, historical evidence, and changelog archives keep their original
versions. Tagging uses the declared date policy; oversized tag messages link to
their source changelog, including archived locations. Production consumer code
uses only the shared public CLI.

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

The upstream suite owns common parsers, installation mechanics, safe publication,
notebook structure/cleanup, and subprocess contracts. MCMC retains integration
tests for the pinned registry package, Just/CI wiring, review scope/failure
propagation, release policies, installed consumer scripts, and scientific notebook
path/figure behavior. Live CodeRabbit review is separate, explicitly authorized
work.

Run `just check` and `just ci` before handoff. The hosted matrix runs the full
consumer gate on Linux, macOS, and Windows; local macOS results do not establish
native Linux or Windows success. For package upgrades, change the exact tooling
and notebook-extra pins together, refresh the lock, read the release notes, and
recheck these consumer boundaries before removing retained fallbacks.
