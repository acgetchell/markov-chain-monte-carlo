# Tooling Scripts

Python utilities used by local repository workflows.

## Benchmark Reports

```bash
just bench-compare [baseline]
just performance-doc [measurements-path]
just performance-local
just performance-github-assets [current-tag baseline-tag]
just performance-readme
just performance-release [current-tag baseline-tag]
```

`bench-compare` discovers Criterion samples below `target/criterion`, compares `new` with a saved baseline, and writes Markdown under
`target/bench-reports/`. For current-working-tree comparisons, `archive-performance` resolves the latest stable release and measures both revisions in isolated
worktrees. GitHub Release comparisons resolve two published tags and consume their durable assets without local measurements. Each measurement-producing
`performance-*` command writes deterministic CSV plus structured JSON provenance below `target/bench-reports/`, reloads those files, and renders Markdown from
the validated artifact. Release-promotion runs infer or accept a release pair, measure it in isolated worktrees, and safely promote one curated report into
`docs/PERFORMANCE.md` while preserving prior reports and the promoted CSV/JSON evidence under `docs/archive/performance/`. With no path,
`performance-doc` resolves the tracked evidence for the current curated report; an explicit path can rerender another saved pair. Both modes avoid GitHub
access, Git worktrees, and Cargo. Native Criterion archives attached to GitHub Releases remain the richer raw evidence for historical reanalysis.
The legacy v0.4.1 curated report predates the tracked pair, so no-argument rerendering becomes available after the next release promotion; use an explicit
generated CSV path for a pre-migration repair.

`performance-readme` uses the same validated retained CSV/JSON pair to generate the marked README table and a deterministic SVG beside the evidence, with
tag-pinned links. It does not run benchmarks or discover releases. Read-only Git checks require existing tags to contain the exact report, CSV, JSON, and
rendered SVG; repaired or same-version evidence that differs stays local. A new-release working-tree comparison can target its future tag; keep local release
tags current and commit every linked artifact before creating that tag. Missing evidence fails before publication with `just performance-release` recovery
guidance; invalid digests remain integrity errors. Both README and SVG are prepared before replacement, with rollback on publication failure.

Installed `bench-compare` and `archive-performance` commands resolve relative paths from the current directory by default. Run them from the repository root or
pass `--repo-root` explicitly.

The benchmark command contracts and interpretation limits live in [`docs/BENCHMARKING.md`](../docs/BENCHMARKING.md).

## Dependency and Tool Updates

```bash
just update
```

`just update` upgrades uv through its supported installation owner and reconciles the exact project pin, upgrades declared Cargo tools, runs shared setup, and
updates Cargo and Python dependencies. The included shared tooling pin is retained. Just is supplied by the pinned shared package; it is no longer upgraded
through Cargo.

Cargo tool versions live in `tool.research-repo-tools.toolchain.cargo`. The shared updater installs exact locked candidates in isolated directories, verifies
them, and only then publishes their TOML pins. Failed candidates leave prior declarations usable. The two unsupported SARIF helpers remain pinned in CI pending
[upstream #25](https://github.com/acgetchell/research-repo-tools/issues/25).

`research-repo-tools deps update-python` resolves direct exact dev pins as one compatible set, preserving ranged requirements and included tooling constraints.
Its recipe preflights stable uv, refreshes the full Python lock, and synchronizes dev with managed tools. Use `just update-dependencies`, `just update-tools`,
or the individual Cargo/Python recipes for narrower work. See [the migration record](../docs/dev/shared-maintenance-migration.md).

## Release Metadata

```bash
TAG=vX.Y.Z
just update-version "$TAG"
just changelog-unreleased "$TAG" "$DATE"
just release-check
```

`update-release-version` infers the previous stable published GitHub release and runs shared release preparation in a temporary tree. It then applies MCMC's
performance-command and README-link policies, validates the fixed concept DOI, and replaces the complete result transactionally. It preserves dependency
versions and existing performance artifact links. Same-tag reruns on the same UTC
day leave contents unchanged; another UTC day updates citation and existing target changelog dates. Set `$DATE` to the prepared `CITATION.cff` date and pass it
explicitly to changelog generation. No benchmark measurement or dependency upgrade is part of metadata preparation.

For an offline preview, run `uv run --locked update-release-version vX.Y.Z --previous-release vA.B.C --date YYYY-MM-DD --dry-run`.
The same validation runs for previews and actual updates. `release-check` combines the shared final-release check with required MCMC publication surfaces,
performance commands, release-pinned README links, and the fixed concept DOI.

`release-check` treats `Cargo.toml` as the release-version source of truth and verifies the Rust and Python lockfiles, Python project metadata,
`CITATION.cff`, the latest generated changelog release, and intentional current-version references in active documentation. It also checks that the citation
release date matches the changelog and that the stable concept DOI agrees across citation metadata, the README badge, and `REFERENCES.md`.

## Notebooks

```bash
just notebook-lint
just notebook-check
just notebook-check-slow
just notebook-clear-outputs-all
```

`just notebook-lint` uses the shared native notebook checks for structure, stable cell IDs, output hygiene, Ruff rules/formatting, and ty types.
`just notebook-check` generates the Ising trace and executes only the fast notebook set in fresh headless kernels. Source notebooks are unchanged; executed
copies and provenance reports mirror root-relative paths under `target/notebooks/`. Temporary runtime state is private to each run.

The scientific notebook content, fast/slow selection, input preparation, and tracked figure promotion remain in MCMC. `just notebook-check-slow` adds explicitly
configured heavier notebooks; `just notebook-clear-outputs-all` deliberately clears source outputs and execution metadata. `just notebook-sync` registers the
kernel inside the locked project environment.

The installable `markov-chain-monte-carlo-tooling[notebook]` extra supplies the shared notebook extra plus MCMC's plotting/data dependencies. The repository's
notebook group selects it. The former `check-notebooks` entry point is replaced by `research-repo-tools notebooks`; use the Just recipes for repository
selection and input preparation.

## Changelog

The published `research-repo-tools==0.1.2` package owns generation, normalization,
archiving, and release-note parsing. Its exact pin lives in the `tooling` dependency
group included by `dev`; `uv.lock` resolves it from PyPI for local setup and CI.

```bash
just changelog-preview
just changelog-check
just changelog
just changelog-unreleased "$TAG" "$DATE"
just changelog-archive
just release-notes "$TAG"
```

`changelog-release TAG DATE` and its `changelog-unreleased` alias require an explicit ISO date matching the prepared citation. Generation uses the packaged
git-cliff template with this repository's owner/name and validates candidates with `rumdl.toml`. It keeps Unreleased and the newest minor series at the root and
rotates older series into `docs/archives/changelog/MAJOR.MINOR.md`. Preview publishes nothing. `changelog-check` validates the root and all archives
independently of note extraction. Release notes work from either location and include referenced links.

Common parser/formatter regressions belong upstream. This repository keeps only
consumer integration coverage in `test_shared_changelog.py`; it no longer ships a
changelog postprocessor or a copy of the git-cliff template. See the
[pilot comparison](../docs/dev/shared-changelog-pilot.md) for intentional policy
changes, resolved upstream issues, and the upgrade procedure.

## Release Tags

```bash
just tag v0.3.0
just tag-force v0.3.0
```

`just tag` directly invokes the shared CLI to extract the matching version section from the root changelog or an archive, validates the tag as `vX.Y.Z` SemVer,
and creates an annotated git tag from that changelog content. If the section exceeds GitHub's tag annotation limit, the tag message falls back to a short link
to the source changelog, including its archive path. The tag must match the package version and the heading date must match `CITATION.cff`. The configured
`declared` date policy permits tagging an already prepared release on a later day.

`just tag-force` replaces only the local tag. Follow the normal tag push and recovery guidance in the
[release procedure](../docs/RELEASING.md#after-the-pr-merges).

## Tests

```bash
just test-python
```

The Python tooling tests live in `scripts/tests/` and run through `uv`.
