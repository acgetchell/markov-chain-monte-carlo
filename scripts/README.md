# Tooling Scripts

Python utilities used by the local release workflow.

## Changelog

```bash
just changelog
just changelog-unreleased v0.3.0
```

`just changelog` runs `git-cliff -o CHANGELOG.md` in offline mode and then `postprocess-changelog` to apply lightweight markdown hygiene. Configuration lives in `cliff.toml` at the repository root.

Changelog entries are generated from local git metadata:

- unreleased sections use squash commit bodies when available
- tagged historical releases use annotated tag notes when those notes contain release bullets
- release-prep commits and CI/action dependency churn are filtered out

Put user-facing release-note bullets in squash commit bodies or annotated tag messages. Do not hand-edit generated changelog content; details that appear only in old manual changelog edits cannot be recovered by the generator later.

## Release Tags

```bash
just tag v0.3.0
just tag-force v0.3.0
```

`tag-release` extracts the matching version section from `CHANGELOG.md`, validates the tag as `vX.Y.Z` SemVer, and creates an annotated git tag from that changelog content. If the section exceeds GitHub's tag annotation limit, the tag message falls back to a short link to `CHANGELOG.md`.

## Tests

```bash
just test-python
```

The Python tooling tests live in `scripts/tests/` and run through `uv`.
