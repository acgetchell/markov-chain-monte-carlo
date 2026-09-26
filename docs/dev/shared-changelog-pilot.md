# Shared changelog adoption (#157)

This record describes the original `research-repo-tools==0.1.1` changelog adoption,
which resolved four gaps found during the 0.1.0 pilot. The current pin is v0.1.7;
the [maintenance migration](shared-maintenance-migration.md) covers its broader
setup, release, review, and notebook ownership and consumer verification.

## Ownership and commands

The `tooling` dependency group pins the package; `dev` includes that group.
`uv.lock` resolves it from PyPI. Local setup and CI use the locked environment
without a sibling checkout or local wheel. The local development-pin updater
retains included tooling constraints and does not automatically upgrade this pin.

| Command | Shared operation |
| --- | --- |
| `just changelog` | Generate, normalize, and archive completed minor series |
| `just changelog-preview` | Validate and print the candidate without publishing files |
| `just changelog-release TAG DATE` | Generate a prospective release with an explicit ISO date |
| `just changelog-unreleased TAG DATE` | Alias for `changelog-release` |
| `just changelog-archive` | Archive existing notes without regenerating history |
| `just changelog-check` | Strictly validate the root changelog and every archive |
| `just release-notes TAG` | Extract notes and required links from root or archive |

Preview accepts generation arguments, for example
`just changelog-preview --tag v0.5.0 --date 2026-09-19`.
Generation uses the packaged git-cliff template, configured owner/repository
links, and the repository's `rumdl.toml`. Unreleased and the newest minor series
remain at the root. Older series are retained in
`docs/archives/changelog/MAJOR.MINOR.md`. These files are generated, not hand-edited.

`changelog-check` is part of `just check` and `just ci`. Successful extraction
does not establish that the entire history is valid; the strict check does.

Removed local ownership includes `cliff.toml`, the changelog postprocessor and
its package entry point/test suite, the release-note section parser and its
parser tests, and the post-generation date-sync helper and its dedicated test.
The tag recipes call the supported shared CLI rather than importing
internal package parsers. Consumer tests cover the pin, included-group
constraints, recipe wiring, and root/archive note extraction with reference
links. Common algorithm and parser regressions remain upstream.

Repository-specific benchmark evidence, scientific notebook content, and release
policies retain local owners and tests. Shared maintenance and notebook
infrastructure are covered by
[#160](https://github.com/acgetchell/markov-chain-monte-carlo/issues/160) and the
[maintenance migration record](shared-maintenance-migration.md).

## Common policies and resolved pilot gaps

- Include all dependency updates, including CI/tooling, as concise entries in a
  distinct Dependencies section. The former package-name allowlist is removed.
- Preserve authored squash-entry structure and wording; embedded conventional
  headings remain within their parent rather than becoming additional entries.
  Do not infer semantic equivalence to discard differently worded content.
  This implements [upstream #13](https://github.com/acgetchell/research-repo-tools/issues/13).
- Extract an unambiguous requested release despite unrelated document problems.
  Reject duplicate target versions, ambiguous boundaries, and conflicting required
  links; keep full-document validation separate.
  This implements [upstream #14](https://github.com/acgetchell/research-repo-tools/issues/14).
- Retain declared historical release dates.
  [Upstream #11](https://github.com/acgetchell/research-repo-tools/issues/11)
  fixes the observed 0.4.2 August 31/September 1 mismatch. Ordinary generation
  under the old local template also exhibited this mismatch; it was not solely
  introduced by shared tooling.
- Repeated generation through the configured formatter preserves retained archives
  without false conflicts. [Upstream #12](https://github.com/acgetchell/research-repo-tools/issues/12)
  resolves the 0.2.0 retained-release conflict found in 0.1.0.

Full breaking descriptions, PR summaries, complete SemVer matching, and literal
Rust code/link preservation are accepted improvements. Historical formatting and
dependency inclusion can differ from the old local renderer; those differences
do not justify duplicating shared policy here.

## Recorded 0.1.1 consumer comparison

Tested on macOS on 2026-09-19 with Python 3.14.7, git-cliff 2.14.1, rumdl 0.2.73,
and MCMC history at `6d5bf0e`. Installed the exact 0.1.1 release directly from
PyPI in a disposable uv environment before updating the repository pin.
The host had uv 0.12.16, so validation used an isolated installation of the
repository-pinned uv 0.12.15 without changing the host or repository tool pin.

- Shared read-only release checking passes at 0.4.2 before regeneration.
- Generation preserves the declared `2026-08-31` date for 0.4.2, keeps the 0.4
  series at the root, and creates archives for 0.1, 0.2, and 0.3.
- Two actual generations produce identical SHA-256 hashes for the root and every
  archive. Strict changelog and local release checks pass after generation.
- Prospective preview emits `0.5.0 - 2026-09-19` with an explicit date. Preview,
  invalid-date rejection, and a missing formatter configuration leave every
  existing root/archive hash unchanged.
- A disposable authored-content fixture retains nested `fix:`/`feat:` details,
  literal `Chain<S>`, a Rust code fence, and the required API reference link.
- Requested-release extraction succeeds alongside unrelated misordered history;
  the independent strict check rejects that same document.
- Extraction rejects a duplicate requested version across root and archive, and
  all seven historical MCMC releases can be extracted after regeneration.
- Notes for a fixture release are identical before and after archiving. Consumer
  integration tests also cover required reference links in both locations.

The 0.1.0 comparison found the now-resolved date and regeneration failures; its
successful consumer gates were never evidence that those generation paths worked.

## Completion validation

On 2026-09-19, the focused consumer/recipe checks passed (43 tests), followed by
successful `just check` and `just ci` runs on macOS with the pinned uv. The full
gate includes 444 Python tests, Rust nextest tests, 188 doctests, notebook
execution, benchmark compilation, and example validation. Zizmor ran in its
default offline mode. Native Linux and Windows CI remain checks for the pushed
PR; they were not run locally. No unresolved shared-package gap was found in the
completion comparison. The changes are ready for the maintainer's commit/push
and PR review; #157 can close when the adoption is merged.

## Upgrading the shared package

1. Install the candidate exact published version in a disposable uv environment.
2. Repeat preview, generation, repeated generation, prospective-date, failure
   preservation, and root/archive extraction comparisons.
3. Update the `tooling` pin and matching consumer assertion, then run `uv lock`
   and `uv sync --locked`. Review package policy changes and generated history.
4. Run focused consumer checks, `just check`, and `just ci`; record the actual
   platform. Local macOS results do not establish native Windows or Linux results.
5. Update the checkout inventory whenever generation adds or removes archives.
