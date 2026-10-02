# Analyzing Chains

Version 0.5.0 provides all three diagnostics requested in [#13](https://github.com/acgetchell/markov-chain-monte-carlo/issues/13):
autocorrelation, effective sample size (ESS), and Gelman–Rubin R-hat using classical split chains. These APIs require no optional Cargo features.
ACF and integrated time landed through #73; mean ESS, ESS per second, and split R-hat through #74. Parallel execution (#12) is separate:
independently initialized chains can run sequentially and still supply R-hat inputs.
The working source also adds rank-based R-hat, multi-chain mean/bulk/tail/quantile ESS, and original-unit mean/quantile MCSE for v0.5.1.
These capabilities require no optional features; source validation and registry publication are separate.

## Run the complete workflow

```bash
just diagnostic-plots
```

[`examples/diagnostics.rs`](../examples/diagnostics.rs) samples four standard-normal chains with distinct seeds and dispersed starts, discards warmup,
and retains every production step. It prints lag-one ACF and single-chain mean ESS, then multi-chain mean/bulk/tail ESS, quantile ESS and MCSE at
0.05/0.5/0.95, mean MCSE, and classical, rank-normalized, folded, and combined R-hat. MCSE uses position units; the standard-normal target's spread is one.
ESS rates divide by measured sequential production wall-clock segments, including transitions and recording but excluding allocation, warmup,
diagnostics, and export. They illustrate timing contracts and are not benchmark evidence.
The same borrowed chain slices feed the estimators. Its schema-2 `target/diagnostics.json` contains exact draws, ranks, and all diagnostic results for five
scenarios: ordinary sampling, shifted or rescaled chain 3, slow mixing with a narrow proposal, and the discrete observable `floor(abs(position))`.
The transformations create deliberate comparison cases; shifted/rescaled chains no longer share a target. Run/observable/chain IDs, seeds, starts,
proposal widths, RNG and crate versions, estimator reference, warmup, cadence, and original/used counts travel with the results.
The named recipe captures the build's source revision and dirty state; the report embeds the Cargo lockfile and records the declared Rust baseline.
Retain the source checkout/diff for dirty builds; a version/revision alone does not
reconstruct local edits. Direct Cargo builds without those environment values report null source provenance.
The example-owned CSV now uses `run_id,observable,chain_id,draw,value`; the self-contained JSON supersedes the earlier R-hat-only schema 1.
Each rerun replaces these artifacts. `status`, nullable values, and reasons preserve unavailable results; message/debug strings are versioned evidence,
not a stable error-matching API. Use the Rust variants for failure-specific handling.
The fixed seeds make the demonstration reproducible within the pinned toolchain and dependencies. Its output is an illustration, not a convergence gate.
`just examples` and `just ci` check its successful diagnostic output through the shared `tooling/examples.toml` validator.

`just diagnostic-plots-data` regenerates only Rust CSV/JSON. `just example diagnostics` also runs the consumer without plotting.
The separate [`examples/ising_1d.rs`](../examples/ising_1d.rs) remains the physical-model trace example. `just notebook-check` executes both notebooks.

## Original-chain ranks and prefix efficiency

`PooledRanks::from_chains(&chains)` returns owned plot data with borrowed `chain(index)` slices in the original draw order. Pair input positions with
application `ChainId`s; the API does not invent new identifiers. Ranks are one-based pooled averages for exact ties, including signed zeros. Constant,
single, or unequal-length finite chains are valid plot inputs. Empty inputs/chains and nonfinite draws are rejected with `PooledRankError`; pooled counts
must fit `usize` and the shared `2^50` rank bound. No graphics or Python dependency enters the library.

Plot ranks include **all supplied production draws**, including an odd middle draw. Split R-hat/ESS omit that draw before their internal ranking.
An explicit half-chain plot must select and label the halves itself. Prefixes of 64, 127, 256, 512, and 1,024 original draws per chain are analyzed afresh:
each prefix gets its own pooled ranks, original-pool quantile cutoffs, and estimates. Reusing full-run ranks or quantile cutoffs would change the question.
Efficiency plots use `S = 2 M floor(N/2)`, the total retained split count, on the horizontal axis; CSVs also preserve original `N` and `M N`.

[`notebooks/diagnostic_plots.ipynb`](../notebooks/diagnostic_plots.ipynb) consumes only exported Rust results. It saves `rank_overlays.png`,
`ess_efficiency.png`, and `traces_acf.png` under `target/notebooks/diagnostics/`, alongside `rank_bins.csv`, `efficiency.csv`, `summary.csv`, and
`blocked_errors.csv`. The exact input JSON is copied there, and `manifest.json` records its SHA-256 and plotting-package versions. The shared executor
also records notebook and environment provenance. Plots never substitute favorable numbers for unavailable results; CSV status/reason columns and the
JSON preserve them. Quantile point/interval fields are unavailable together when quantile MCSE cannot be estimated; their ESS result remains separate.

Rank overlays compare each original chain's bin proportions with the **pooled bin reference**, using common edges. Deterministic ties can make this
reference nonuniform for discrete data even when chains agree. Neither uniform-looking continuous ranks nor agreement with discrete pooled bins proves
convergence. See [Vehtari et al., Section 4.5](../REFERENCES.md#ref-14) for rank and efficiency plots.

![Original-chain rank proportions for ordinary sampling, location and scale disagreement, slow mixing, and a tied discrete observable.](assets/diagnostic_rank_overlays.png)

Each row uses the full 1,024-draw prefix from four original chains. Colored lines identify the chains; the dashed black line is the pooled reference.
Location and scale changes alter chain 3's rank distribution. The discrete row illustrates why agreement with the pooled reference matters more than
flat bins when values tie.

Relative ESS is ESS/S, not ESS/second; it can exceed one for antithetic observations. Only full, directly sampled runs have measured production timing.
Shorter prefixes and post-sampling transforms carry explicit unavailable timing; whole-run seconds are never attached to them. Original-unit mean and
quantile MCSE are reported separately from pooled sample standard deviation. Blocked errors preserve each chain's block size, block count, and used draws;
the coarsest level is not automatically a reliable uncertainty estimate.

![Bulk and tail ESS and relative ESS across independently recomputed prefixes of the five diagnostic scenarios.](assets/diagnostic_ess_efficiency.png)

Each point recomputes diagnostics for that prefix. The left column shows bulk/tail ESS; the right divides each by the retained split count `S`.
These curves compare information as draws accumulate, not wall-clock throughput. They are seeded demonstrations, not release performance measurements.
Regenerate both tracked figures with `just diagnostic-plots-figures`; the matching JSON, CSVs, and provenance remain in `target/notebooks/diagnostics/`.

For an already exported report, run the shared notebook executor from the repository root, with `MCMC_DIAGNOSTICS_PATH` selecting that JSON and
`MCMC_NOTEBOOK_OUTPUT_DIR` selecting a separate output directory if desired. `MCMC_REPO_ROOT` controls default paths. Explicit paths never fall back to
another report. Change the notebook's `rank_prefix` selection to another exported prefix to inspect its independently recomputed ranks.

To render a saved report without regenerating the sampled data:

```bash
just notebook-sync
MCMC_DIAGNOSTICS_PATH=/path/to/diagnostics.json \
MCMC_NOTEBOOK_OUTPUT_DIR=target/notebooks/saved-diagnostics \
uv run --locked --group dev --group notebook research-repo-tools notebooks execute notebooks/diagnostic_plots.ipynb
```

Trace axes use recorded-draw indices and label the report's positive recording interval. Unavailable ACF chains remain in the legend, and
`blocked_errors.csv` retains an explicit status/reason row when a chain has no blocked-error estimate.

## Export and notebook workflow

`just example ising_1d` records four sequential chains with distinct seeds and initial states. In addition to CSV traces and ACF/time estimates, it writes
`target/ising_1d_diagnostics.json` (schema version 1). The report names the estimators and observables, identifies original chains and seeds, records sample
counts, discarded warmup, recording interval, per-chain measured seconds, and the timing scope. ESS/rate and R-hat results carry status and nullable values;
unavailable estimates include an error message. Successful R-hat records use the estimator's original and half lengths to report omitted-middle counts.
On failure, half lengths and omitted-middle counts are null; the original per-chain length is also null if the inputs have unequal lengths or no chains.
These are example-owned exports, independent of the optional checkpoint `serde` feature. R-hat chain selection does not require timing metadata.
The `error` and `rate_error` strings are display-only diagnostics; use status and nullable values for availability, or the Rust error variants for
failure-specific handling. Do not parse message text as a stable error category.

Production timing includes sampling and observation/recording, and excludes warmup, export, and diagnostics. Rates vary with build profile and machine and
are illustrative, not benchmark evidence. The notebook computes diagnostics from the trace and consumes companion timing only for matching full production
samples. External traces require explicit `MCMC_DIAGNOSTICS_PATH` for rates; additional notebook warmup removal makes full-run timing unavailable. The
notebook exports ESS and R-hat tables beside its figures, using the same availability rules for R-hat count metadata. Preserve the source JSON with those
tables to retain the original timing and warmup scope.

This workflow uses a Cargo example with parameters set in Rust source. Run it from the repository root; reruns replace the generated files under `target/`.
Console output is a human-readable summary. Consume CSV/JSON for analysis, preserving the source report alongside exported tables. JSON reporting uses a
development dependency, and notebook dependencies belong to the contributor environment; ordinary library use requires neither.

## Choose the quantity

For a `Trace`, select one observable from one chain with `trace.observable_values(id, "energy")?.copied().collect::<Vec<_>>()`.
Selection borrows the trace, checks the column name, and preserves insertion order. Use `records_for_chain(id)` to verify step ordering and constant
spacing before estimating an ACF; keep rejected and no-proposal steps. Do not concatenate chains into one series.

| Question | API | Meaning |
| --- | --- | --- |
| Do original chains occupy similar parts of the pooled distribution? | `PooledRanks::from_chains(&chains)?.chain(index)` | One-based average ranks in original draw order, including odd middle draws |
| How correlated are draws at a given lag? | `Autocorrelation::estimate(samples, max_lag)?.values()` | ACF from lag zero through the inclusive maximum lag |
| How much serial correlation affects a scalar mean? | `acf.integrated_time()?` | Geyer's initial monotone sequence estimate in recorded-sample intervals |
| How much information does this chain provide about that mean? | `time.effective_sample_size()` | `N / tau`, for this observable and chain |
| How efficiently did the measured run produce that information? | `time.effective_sample_size_per_second(elapsed)?` | Mean ESS divided by measured seconds |
| How much mean information do comparable chains provide together? | `EssEstimate::estimate(&chains, EssEstimator::Mean)?` | Raw-scale split ESS incorporating between-half disagreement |
| How much bulk or quantile information do those chains provide? | `EssEstimator::Bulk` or `EssEstimator::Quantile(p)` | Rank-normalized or quantile-indicator split ESS |
| How much tail information is available? | `TailEss::estimate(&chains)?` | Both 0.05/0.95 components and their available minimum |
| What is the relative information and measured throughput? | `ess.relative()`, `ess.per_second(timing)?` | ESS/retained count and ESS/workload wall seconds |
| How precise is the estimated mean? | `MeanMcse::estimate(&chains)?` | Original-unit pooled standard deviation / sqrt(raw mean ESS) |
| How precise is an estimated quantile? | `QuantileMcse::estimate(&chains, p)?` | Original-unit beta/order-statistic MCSE and uncertainty bounds |
| Do within-chain and between-chain variations agree? | `SplitRhat::estimate(&chains)?.value()` | Classical split R-hat from equally long borrowed scalar slices |
| Do the chains' ranked locations agree? | `RankNormalizedSplitRhat::estimate(&chains)?.value()` | Pooled rank-normalized split component, without folding |
| Do the chains' scales agree? | `FoldedRankNormalizedSplitRhat::estimate(&chains)?.value()` | Ranked absolute deviations from the pooled median |
| Do both rank-based components agree? | `CombinedRhat::estimate(&chains)?.value()` | Maximum, or `None` if either component is unavailable |

Keep chains separate for ACF and ESS. Concatenation introduces artificial transitions between chains, and summing per-chain ESS does not produce a
diagnostic that accounts for disagreement between chains. ESS is observable-specific; mean ESS is not bulk or tail ESS.

For original-unit precision, use the new APIs directly:

```rust
use markov_chain_monte_carlo::{EssEstimate, EssEstimator, MeanMcse, MonteCarloError, QuantileMcse, TailEss};

fn precision(chains: &[&[f64]]) -> Result<(), MonteCarloError> {
    let mean_ess = EssEstimate::estimate(chains, EssEstimator::Mean)?;
    let mean_error = MeanMcse::estimate(chains)?;
    let median_error = QuantileMcse::estimate(chains, 0.5)?;
    let tails = TailEss::estimate(chains)?;
    println!("Mean ESS={}, ESS/S={}, mean MCSE={}", mean_ess.value(), mean_ess.relative(), mean_error.value());
    println!("Median={}, MCSE={}, bounds={:?}", median_error.quantile(), median_error.value(), median_error.interval());
    println!("Tail minimum={:?}, lower={:?}, upper={:?}", tails.value(), tails.lower(), tails.upper());
    Ok(())
}
```

Mean MCSE measures error in the mean estimate, rather than posterior standard deviation; it requires finite population moments.
Quantile MCSE approximates uncertainty in the selected quantile. Tied/discrete observations can leave that method unavailable even with computable
indicator ESS. [Binning](scientific_basis.md#binning-analysis) remains a separate blocked mean-error method.

For a scientific efficiency comparison, record what timing includes. The Ising workflow times production transitions and recording, excluding warmup
and diagnostics. Other timing scopes are possible, but must be comparable across the runs being compared.
For multi-chain rates, construct `DiagnosticTiming::try_new(elapsed, original_chain_count, original_draws_per_chain)` from the measured workload and pass
`Some(&timing)`. `None`, zero time, or mismatched counts leave rates unavailable. Matching counts do not prove trace identity. Parallel runs require actual
concurrent wall time; never sum their chain durations. A selected prefix needs its own measurement.

## Input and estimator contracts

- Discard warmup, preserve temporal order, and use a fixed recording interval. Keep rejected steps and no-proposal self-loops.
- Compare the same observable, units, target, recording interval, and warmup policy across independently initialized chains.
  Slices cannot establish these assumptions; separate seeds alone do not prove independence or mixing.
- ACF needs at least two finite observations and `max_lag < N`. It uses the sample mean and biased autocovariances with the same implicit divisor `N`
  at every lag. Lag zero is exactly one. The direct implementation costs `O(N * (max_lag + 1))` time.
- Integrated time pairs adjacent ACF values starting at lag zero and requires a nonpositive pair to establish truncation. It monotonizes the preceding
  positive pairs. An arbitrary lag cutoff is not treated as a valid truncation. ESS can exceed `N` for anticorrelated draws.
- Split R-hat needs at least two original chains of equal length, each with at least four draws. Each is split into first and last halves;
  an odd middle draw is omitted from moments but still checked for finiteness. The count minima make the formula defined, not reliable.
- Classical R-hat uses unbiased within-half variances and variation of the half means. Values below one are retained. It assumes finite marginal mean
  and variance, and does not rank-normalize or fold observations; scale and tail differences can be missed.
- `RankNormalizedSplitRhat` uses the same variance ratio after pooling retained draws, averaging tied ranks, and transforming them to normal scores.
  Signed zeros tie. Its distinct result type identifies the estimator, while `chain_count()`, `samples_per_chain()`, and `samples_per_split_chain()`
  report original chains, original length, and retained half length. It omits the folded scale-sensitive component and combined maximum.
  See the [scientific contract and pinned reference fixtures](scientific_basis.md#rank-normalized-split-r-hat).
- `CombinedRhat` preserves both the location and folded components. Folding uses the median of all original draws before splitting, including odd middle
  draws. Use `rank_normalized()` and `folded()` to inspect typed component results and `value()` for their optional maximum.
  See [folding arithmetic, conventions, and reference evidence](scientific_basis.md#folded-and-combined-r-hat).
- Multi-chain ESS shares those shape/count checks, but allows constant individual halves. It uses biased within-half autocovariances and between-half
  variation. Bulk uses retained pooled ranks; quantile cutoffs include original odd middle draws. Relative ESS uses only retained draws, and can exceed
  one under antithetic sampling. The reference regularizes time at `1/log10(S)`; `is_regularized()` reports whether that bound changed the result.
  See [splitting, truncation, regularization, and fixtures](scientific_basis.md#multi-chain-effective-sample-size).
- Mean MCSE uses all-original-draw variance and raw mean ESS. Quantile MCSE uses beta/order-statistic bounds with a linear type-7 point estimate and
  probabilities strictly in `(0,1)`. See [numerical conventions and interpretation limits](scientific_basis.md#monte-carlo-standard-errors).

The [Stan reference on ESS](https://mc-stan.org/docs/2_29/reference-manual/effective-sample-size.html) describes the integrated-time relationship and
initial monotone sequence method. The crate exposes both single-chain and split multi-chain ESS with separately documented finite-lag contracts.
The [classical split R-hat reference](https://mc-stan.org/docs/2_29/reference-manual/notation-for-samples-chains-and-draws.html) gives the variance ratio.
Neither a large ESS nor R-hat near one establishes convergence or exploration of every mode. Inspect traces, use dispersed starts, and assess multiple
observables; consider rank-normalized and folded diagnostics when raw moments are insufficient.

## Keep unavailable results explicit

`AutocorrelationError` distinguishes short, nonfinite, constant, and invalid-lag inputs. `TruncationNotFound` means the available lag window does not
establish the integrated-time truncation: increase the lag budget or collect a longer trace, then reassess. `NonPositiveTime` also leaves ESS unavailable.
`EssRateError` rejects a zero measured duration.

`SplitRhatError` identifies too few chains or samples, unequal lengths, nonfinite observations, constant halves, numerically unresolved within-half
variances, and a nonrepresentable final estimate. All R-hat components reject even one constant half. Rank normalization additionally rejects retained
counts above `2^50` or overflowing `usize` with `TooManyRankedSamples` to preserve exact rank offsets. This error reports the chain count, retained half
length, and effective count limit as typed fields. Preserve errors with chain/observable context; do not replace them with zero ESS or R-hat one.
The example prints unavailable diagnostics explicitly, while its fixed demonstration inputs are expected to produce successful estimates in CI.

`CombinedRhat::estimate` returns `Err` for invalid input or unsupported counts. Otherwise it returns a report with both component `Result`s, original
counts, and an optional combined value. A binary/count observable may be valid yet have constant folded deviations, making its combined value `None`.
`UnresolvedFolding` reports overflow-avoidance scaling that would lose subnormal precision. Preserve the successful component for interpretation, but
never substitute it for the missing maximum. Deciding whether a structurally constant observable belongs in a consumer gate remains caller policy.

`MonteCarloError::Input` preserves shared shape/finite/count failures, while `InvalidProbability` rejects unsupported quantile requests.
`ConstantSamples`, `NoWithinChainVariation`, `DegenerateIndicator`, `UnresolvedVariance`, and `NumericalFailure` describe unavailable ESS or MCSE.
`CollapsedQuantileInterval` does not establish exactness or zero uncertainty. `TailEss` preserves successful components even when the other fails;
its minimum/relative/rate remain unavailable. Keep original and retained count metadata and typed reasons alongside observable/chain identity.

## Validation evidence

`tests/autocorrelation.rs` and `tests/convergence.rs` cover hand-calculated estimates, independent and correlated synthetic draws, separated locations,
within-chain drift, degenerate inputs, and numerical extremes. `tests/proptest_autocorrelation.rs` and `tests/proptest_convergence.rs` check algebraic
invariants through the public APIs. `just test-integration` runs these with the other integration tests; `just test-doc` checks the API examples.
The Ising notebook is an additional end-to-end consumer, not an independent numerical oracle.

`tests/pooled_ranks.rs` covers ordering, ties, signed zeros, original identity, odd middles, unequal lengths, constant and invalid inputs, immutability,
and owned results through root/prelude imports. `tests/fixtures/diagnostic_plots.json` retains five regimes at three prefixes, independently calculated
with SciPy 1.16.2 and ArviZ 0.22.0 from the exact ESS fixture inputs. Reproduce and cross-check exported results with:

```bash
uv run --script tests/fixtures/generate_diagnostic_plots.py
uv run --script tests/fixtures/generate_diagnostic_plots.py --report target/diagnostics.json
```

The isolated generator pins Python 3.13.7 and NumPy 2.2.6 as well. It verifies provenance, rank arrays, availability, and numeric tolerances without
rewriting retained evidence. Notebook execution is evidence of artifact consumption and rendering; it does not replace these independent numerical checks.

`tests/rank_normalized_rhat.rs` adds pinned ArviZ component fixtures, deterministic Cauchy location shifts, same-distribution controls, preserved ties and
ordering under monotone transforms, signed zeros, extreme finite inputs, and borrowed-input checks. Its scale-only fixture deliberately distinguishes
the rank-normalized split component from the combined rank/folded maximum.

`tests/combined_rhat.rs` checks both components and their maximum against pinned posterior fixtures, with explicit folded degeneracy and odd-length
conventions. Release availability is separate from source implementation: v0.5.1 publication and a clean registry consumer build remain release checks.

`tests/ess.rs` compares all new numerical methods against thirteen pinned ArviZ regimes, retaining intentional sentinel-policy differences.
`tests/proptest_ess.rs` checks affine units, chain/time reordering, and tied ranks under nonlinear monotone transforms. `tests/public_api.rs` and doctests
exercise public imports, components, ratios, rates, and use after borrowed input storage drops. Reproduce the independent corpus with
`uv run --script tests/fixtures/generate_ess.py`; versions, input draws, tolerances, and provenance are retained beside it.
