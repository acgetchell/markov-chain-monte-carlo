# Tooling Scripts

Python utilities used by local repository workflows.

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
