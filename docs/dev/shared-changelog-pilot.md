# Shared changelog pilot (#157)

The pilot adopts published `research-repo-tools==0.1.0` for changelog generation,
normalization, archiving, and release-note extraction. The migration remains open
pending fixes for the upstream failures and policy changes below. Local validation of the
consumer integration does not mean those package failures are resolved.

## Ownership and commands

The `tooling` dependency group pins the package; `dev` includes that group.
`uv.lock` resolves the package from PyPI. Existing `just setup`, `python-sync`,
and CI use that locked environment without a sibling checkout or local wheel.
The local development-pin updater retains included tooling constraints and does
not automatically upgrade the pilot package.

| Command | Shared operation |
| --- | --- |
| `just changelog` | Generate, normalize, and archive completed minor series |
| `just changelog-preview` | Validate and print the candidate without publishing files |
| `just changelog-release TAG DATE` | Generate a prospective release with an explicit ISO date |
| `just changelog-unreleased TAG DATE` | Alias for `changelog-release` |
| `just changelog-archive` | Archive existing notes without regenerating history |
| `just release-notes TAG` | Extract notes and referenced links from root or archive |

Preview accepts shared generation arguments, for example
`just changelog-preview --tag v0.5.0 --date 2026-09-17`.

Generation uses the packaged git-cliff template, configured owner/repository
links, and this repository's `rumdl.toml`. Archives use
`docs/archives/changelog/MAJOR.MINOR.md`; generation keeps Unreleased and the newest
minor series at the root. Add generated archive files to the checkout inventory
when publishing them. This pilot leaves the checked-in changelog unchanged.

Removed ownership: `cliff.toml`, `scripts/postprocess_changelog.py`, its package
entry point, and its complete test suite. The local tag helper delegates note
extraction to the supported shared CLI; its duplicated section parser and parser
tests are removed. The unused post-generation date-sync helper and its dedicated
test are also removed; prospective generation receives its date explicitly. The package's internal Python parsers are not imported.

Consumer tests cover the exact pin, included-group constraints, recipe wiring,
and tag-wrapper extraction from root/archive files with reference links. Common
parser, formatting, and transaction regressions belong in the shared package.
The history trial below is recorded evidence, not an extra full-history CI job.

Benchmark evidence, scientific notebooks, local release metadata, tagging policy,
and dependency/tool setup still have local owners and tests. Remaining maintenance
migration belongs to [#160](https://github.com/acgetchell/markov-chain-monte-carlo/issues/160).
Python validation cannot be removed while those local scripts remain.

## Recorded consumer comparison

Tested on macOS on 2026-09-17 with Python 3.14.7, git-cliff 2.14.1, rumdl 0.2.73,
and MCMC history at `3d822a39d6233212ce91058c47b5abee0fe83707`. Version 0.1.0
installed directly from PyPI into a disposable uv environment. No editable shared
checkout or locally built wheel was used.

- The original checkout passes the shared read-only release check at 0.4.2.
- Dry-run generation succeeds, leaving the source changelog and archive paths
  untouched. It adds Unreleased, complete breaking-change summaries, merged-PR
  summaries, and dependency groups previously filtered by the local template.
- A first actual generation in a disposable consumer succeeds. The root retains
  the 0.4 series; older releases are archived. Shared extraction finds 0.4.2,
  0.3.0, 0.2.1, 0.2.0, and 0.1.0 after rotation.
- Prospective generation with an explicit 2026-09-17 date succeeds in a fresh
  disposable consumer. Invalid-date failure preserves all generated Markdown
  files. The repeated-generation conflict also leaves their bytes unchanged.
- The package adopts common heading normalization, complete SemVer tag matching,
  and code-aware rendering rather than the old template's blanket angle-bracket
  escaping and repository-specific dependency filters. Detailed parser edge-case
  regressions remain owned upstream.

### Known failures

1. [research-repo-tools#11](https://github.com/acgetchell/research-repo-tools/issues/11):
   generation changes the existing 0.4.2 date from 2026-08-31 to 2026-09-01.
   Both local and shared release checks reject that candidate against the
   unchanged citation date. Using `--tag v0.4.2 --date 2026-08-31` instead fails
   with a duplicate heading; those arguments are intended for prospective tags.
   The old MCMC git-cliff configuration also produces September 1 on ordinary
   regeneration: this is not established as a newly introduced regression. The
   old prospective workflow separately synchronized its heading to the citation.
2. [research-repo-tools#12](https://github.com/acgetchell/research-repo-tools/issues/12):
   repeating generation against the same history and generated archives fails
   with `conflicting retained release 0.2.0`. A first successful generation is
   insufficient evidence that routine regeneration works.

Do not erase retained archives or rewrite published citation metadata to bypass
these failures. They are acceptance gaps for completing #157, not reasons to
restore duplicated consumer implementations.

## Agreed common policies

The maintainer selected these policies for the shared package:

- Include all dependency updates, including CI/tooling, as concise entries in a
  distinct Dependencies section. Accept the shared inclusion policy; do not
  restore the local package-name allowlist.
- Preserve authored squash-entry structure and wording. Embedded conventional
  headings remain under their parent; do not promote them into additional change
  entries or infer semantic equivalence to remove differently worded content.
  Track the implementation and upstream regressions in
  [research-repo-tools#13](https://github.com/acgetchell/research-repo-tools/issues/13).
- Extract a valid requested release despite unrelated document problems when its
  boundaries and required links remain unambiguous. Reject duplicate target
  versions, ambiguous boundaries, and conflicting required links. Keep strict
  whole-document validation separate from extraction. Track this change in
  [research-repo-tools#14](https://github.com/acgetchell/research-repo-tools/issues/14).

Version 0.1.0 does not yet implement the latter two policies. Keep their fixes in
the shared package and adopt a published release; do not add local parser or
normalizer workarounds. Code/link preservation, full breaking descriptions, and
complete SemVer matching remain accepted improvements.

## Upgrade and completion

The consumer migration passed `just check` and `just ci` on macOS on 2026-09-17,
including 444 Python tests, 274 Rust tests, 188 doctests, notebook execution,
benchmark compilation, and example validation. These gates validate the retained
checkout and integration; the two generation failures above remain unresolved.
Zizmor ran in its default offline mode. Native Linux and Windows runs were not
performed in this local trial.

After an upstream release includes the fixes and agreed policy changes:

1. Install its exact published version in a disposable uv environment and repeat
   preview, generation, repeated generation, prospective-date, and archive-note
   comparisons using this history. Verify failure cases preserve artifacts.
2. Change the `tooling` pin in `pyproject.toml` and the matching consumer assertion,
   run `uv lock` and `uv sync --locked`, and review package changes.
3. Run focused consumer tests, `just check`, and `just ci`. Record the actual
   platform; local macOS results do not establish native Windows or Linux results.
4. Regenerate and review root/archive content, run release validation, update the
   checkout inventory, and close #157 only when its acceptance criteria pass.
