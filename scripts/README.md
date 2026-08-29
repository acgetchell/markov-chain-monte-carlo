# Tooling Scripts

Python utilities used by local repository workflows.

## Benchmark Reports

```bash
just bench-compare [baseline]
just performance-local
just performance-github-assets [current-tag baseline-tag]
just performance-release [current-tag baseline-tag]
just performance-rerender [measurements-path]
```

`bench-compare` discovers Criterion samples below `target/criterion`, compares `new` with a saved baseline, and writes Markdown under
`target/bench-reports/`. For current-working-tree comparisons, `archive-performance` resolves the latest stable release and measures both revisions in isolated
worktrees. GitHub Release comparisons resolve two published tags and consume their durable assets without local measurements. Each measurement-producing
`performance-*` command writes deterministic CSV plus structured JSON provenance below `target/bench-reports/`, reloads those files, and renders Markdown from
the validated artifact. Release-promotion runs infer or accept a release pair, measure it in isolated worktrees, and safely promote one curated report into
`docs/PERFORMANCE.md` while preserving prior reports and the promoted CSV/JSON evidence under `docs/archive/performance/`. With no path,
`performance-rerender` resolves the tracked evidence for the current curated report; an explicit path can rerender another saved pair. Both modes avoid GitHub
access, Git worktrees, and Cargo. Native Criterion archives attached to GitHub Releases remain the richer raw evidence for historical reanalysis.
The legacy v0.4.1 curated report predates the tracked pair, so no-argument rerendering becomes available after the next release promotion; use an explicit
generated CSV path for a pre-migration repair.

The benchmark command contracts and interpretation limits live in [`docs/BENCHMARKING.md`](../docs/BENCHMARKING.md).

## Dependency and Tool Updates

```bash
just update
```

`update-python-dev-pins` resolves exact entries in `dependency-groups.dev` as one compatible set, leaves ranged requirements unchanged, applies all exact
pin changes in one uv transaction, and restores `pyproject.toml` and `uv.lock` if the mutation fails or changes unrelated manifest content.
`update-tool-pins` reconciles the root justfile with the Cargo-installed tools managed by `just setup` and the active uv version. The aggregate recipe also
updates Cargo requirements and lockfiles, upgrades those managed Cargo tools, refreshes the uv lock, and syncs the development environment.

## Release Metadata

```bash
just release-check
```

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

`just notebook-lint` validates notebook JSON and stable cell IDs, rejects outputs and execution counts, compiles each code cell with cell-aware diagnostics,
and checks extracted code with Ruff and Ty. `just notebook-check` generates required example artifacts and executes only the fast notebook set headlessly.
Executed notebooks and runtime caches are written under `target/notebooks/`, leaving source notebooks unchanged. `just notebook-check-slow` adds only the
explicitly configured heavier notebook set; `just notebook-clear-outputs-all` intentionally clears source outputs and counts in place.

## Changelog

```bash
just changelog
just changelog-unreleased v0.3.0
```

`just changelog` runs `git-cliff -o CHANGELOG.md` in offline mode and then `postprocess-changelog` to apply lightweight markdown hygiene. Configuration lives in
`cliff.toml` at the repository root.

Changelog entries are generated from local git metadata:

- unreleased sections use squash commit bodies when available
- tagged historical releases use annotated tag notes when those notes contain release bullets
- release-prep commits and CI/action dependency churn are filtered out

Put user-facing release-note bullets in squash commit bodies or annotated tag messages. Do not hand-edit generated changelog content; details that appear only
in old manual changelog edits cannot be recovered by the generator later.

## Release Tags

```bash
just tag v0.3.0
just tag-force v0.3.0
```

`tag-release` extracts the matching version section from `CHANGELOG.md`, validates the tag as `vX.Y.Z` SemVer, and creates an annotated git tag from that
changelog content. If the section exceeds GitHub's tag annotation limit, the tag message falls back to a short link to `CHANGELOG.md`.

## Tests

```bash
just test-python
```

The Python tooling tests live in `scripts/tests/` and run through `uv`.
