# Analyzing Chains

Use scalar diagnostics to assess mixing, estimate Monte Carlo precision, and compare sampling efficiency.
The library provides autocorrelation, single- and multi-chain effective sample size (ESS),
classical and rank-based R-hat, and mean/quantile Monte Carlo standard errors (MCSE), without optional Cargo features.
Independently initialized chains can run sequentially; parallel execution is not required.

For practical guidance, see Martin, Abril-Pla, and Deklerk's **[Exploratory Analysis of Bayesian Models (EABM)][eabm]**,
especially [Chapter 4: MCMC Diagnostics][eabm-diagnostics] on interpreting traces, rank plots, R-hat, ESS, and MCSE.

**Independent numerical checks:** the crate's multi-chain ESS, MCSE, and rank-normalized R-hat methods are checked against [ArviZ 0.22.0][arviz]
using pinned reference fixtures and reproducible generators.
[posterior 1.7.0][posterior] supplies additional combined R-hat references.
The [validation evidence](#validation-evidence) records the comparisons and intentional differences in handling unavailable results.

## Contents

- [Choose the quantity](#choose-the-quantity)
- [Input and estimator contracts](#input-and-estimator-contracts)
- [Run the complete workflow](#run-the-complete-workflow)
- [Original-chain ranks and prefix efficiency](#original-chain-ranks-and-prefix-efficiency)
- [Export and notebook workflow](#export-and-notebook-workflow)
- [Keep unavailable results explicit](#keep-unavailable-results-explicit)
- [Validation evidence](#validation-evidence)

## Choose the quantity

Start with traces and original-chain rank plots, combined rank/folded R-hat, and bulk/tail ESS.
Then estimate MCSE for the specific mean or quantile you intend to report.
[Vehtari et al.][rank-methods] motivate this combination: classical R-hat can miss scale disagreement and heavy-tail problems,
and bulk ESS does not directly measure precision of a raw mean.

| Question | API | Interpretation |
| --- | --- | --- |
| Do original chains explore similar regions? | `PooledRanks::from_chains(&chains)?.chain(index)` | Pooled average ranks in original draw order |
| How correlated are draws at each lag? | `Autocorrelation::estimate(samples, max_lag)?.values()` | ACF from lag zero through the inclusive maximum lag |
| How much information does one chain provide about a mean? | `acf.integrated_time()?.effective_sample_size()` | Single-chain mean ESS, `N / tau` |
| How much mean information do comparable chains provide together? | `EssEstimate::estimate(&chains, EssEstimator::Mean)?` | Raw-scale split ESS, including between-half disagreement |
| How well are the bulk and selected quantiles explored? | `EssEstimator::Bulk`, `EssEstimator::Quantile(p)`, `TailEss::estimate(&chains)?` | Rank-normalized bulk or quantile-indicator ESS; tail ESS is the minimum of both 0.05/0.95 components |
| How precise are the summaries in observable units? | `MeanMcse::estimate(&chains)?`, `QuantileMcse::estimate(&chains, p)?` | Estimated Monte Carlo error of the mean or quantile |
| Do within-half and between-half variations agree? | `SplitRhat::estimate(&chains)?.value()` | Classical split variance ratio |
| Do chains disagree in ranked location or folded deviations? | `RankNormalizedSplitRhat`, `FoldedRankNormalizedSplitRhat` | Rank-based components; folding adds scale sensitivity |
| What is the combined rank-based check? | `CombinedRhat::estimate(&chains)?.value()` | Maximum of both components; `None` if either is unavailable |
| What fraction of retained draws is effective? | `ess.relative()` | ESS divided by retained split count |
| How much effective information was produced per second? | `ess.per_second(timing)?`, `time.effective_sample_size_per_second(elapsed)?` | ESS divided by measured workload wall time |

For a `Trace`, select one observable from one chain with `trace.observable_values(id, "energy")?.copied().collect::<Vec<_>>()`.
Selection borrows the trace, checks the column name, and preserves insertion order.
Use `records_for_chain(id)` to verify step ordering and constant spacing; keep rejected and no-proposal steps.
Keep chains separate: concatenation creates artificial transitions, and summing per-chain ESS omits between-chain disagreement.
ESS is specific to the observable and summary; mean, bulk, and tail ESS are different quantities.

For original-unit precision:

```rust
use markov_chain_monte_carlo::{
    EssEstimate, EssEstimator, MeanMcse, MonteCarloError, QuantileMcse, TailEss,
};

fn precision(chains: &[&[f64]]) -> Result<(), MonteCarloError> {
    let mean_ess = EssEstimate::estimate(chains, EssEstimator::Mean)?;
    let mean_error = MeanMcse::estimate(chains)?;
    let median_error = QuantileMcse::estimate(chains, 0.5)?;
    let tails = TailEss::estimate(chains)?;
    println!(
        "Mean ESS={}, ESS/S={}, mean MCSE={}",
        mean_ess.value(), mean_ess.relative(), mean_error.value()
    );
    println!(
        "Median={}, MCSE={}, bounds={:?}",
        median_error.quantile(), median_error.value(), median_error.interval()
    );
    println!(
        "Tail minimum={:?}, lower={:?}, upper={:?}",
        tails.value(), tails.lower(), tails.upper()
    );
    Ok(())
}
```

Mean MCSE estimates uncertainty in the sampled mean, in observable units.
It is distinct from target/posterior standard deviation and requires finite population moments and a suitable Markov chain central limit theorem.
A finite result from heavy-tailed data does not establish those assumptions.
Use raw mean ESS in this calculation; substituting bulk ESS changes its meaning.
See the [mean MCSE contract](scientific_basis.md#monte-carlo-standard-errors) and [Stan's explanation][stan-diagnostics].

Quantile MCSE follows [Vehtari et al., Section 4.4][rank-methods], with beta/order-statistic bounds for uncertainty in the estimated quantile.
These bounds describe Monte Carlo error, rather than a target/posterior interval.
Interpretation needs adequate local information; the usual quantile approximation assumes positive continuous density near the quantile.
Tied/discrete observations can leave MCSE unavailable even when indicator ESS is computable.
[Binning](scientific_basis.md#binning-analysis), following [Flyvbjerg and Petersen][blocking], is a separate blocked mean-error method.

For rates, record the measured workload and what its timing includes.
Construct `DiagnosticTiming::try_new(elapsed, original_chain_count, original_draws_per_chain)` and pass `Some(&timing)` to multi-chain rates.
Absent timing, zero duration, or mismatched counts leave the rate unavailable.
Matching counts do not prove trace identity.
Sequential production segments can be summed; parallel chains require their actual concurrent wall time.
Prefixes need their own measurement, and comparisons need matching timing scopes.

## Input and estimator contracts

Discard warmup and finish adaptation before analyzing fixed-kernel production draws.
Preserve temporal order and a fixed recording interval, including rejection and no-proposal self-loops.
Compare the same observable, units, target, cadence, and warmup policy across independently initialized chains.
Separate seeds and finite slices cannot establish independence, stationarity, or mixing.

The integrated-time interpretation uses [Geyer's initial sequence estimators][initial-sequence].
Their justification assumes a stationary reversible chain, finite observable variance, and summable autocorrelations.
Finding a truncation pair in a finite trace does not establish those conditions.
See [autocorrelation](scientific_basis.md#autocorrelation-estimator-contract)
and [ESS assumptions](scientific_basis.md#ess-and-wall-clock-efficiency) for the complete contracts.

| Estimator | Input and arithmetic choices |
| --- | --- |
| ACF and single-chain ESS | At least two finite, varying observations; `max_lag < N`. Biased autocovariances use divisor `N` at every lag. Integrated time requires a nonpositive adjacent pair to truncate its initial monotone sequence. |
| Split R-hat components | At least two equal-length finite chains, each with at least four draws. First/last halves omit an odd middle draw, which is still checked for finiteness. Any constant half makes the affected component unavailable. |
| Multi-chain ESS | Same shape/count minima, with constant individual halves allowed when the estimator remains defined. Bulk ranks use retained draws; quantile cutoffs use all original draws. |
| Mean and quantile MCSE | Mean MCSE uses all-original-draw sample variance and raw mean ESS. Quantile MCSE uses probabilities strictly in `(0,1)`, a linear type-7 point estimate, and beta/order-statistic bounds. |

These count minima make formulas defined; they do not establish statistical reliability.
Classical split R-hat builds on [Gelman and Rubin's multiple-sequence diagnostic][original-rhat],
with the implemented split variance ratio specified by [Stan][stan-diagnostics].
Rank normalization and folding follow [Vehtari et al.][rank-methods].
Folding uses the median of all original draws **before splitting**, including an odd middle draw.
Classical and rank-based R-hat preserve values below one.
See the scientific basis for [classical](scientific_basis.md#classical-split-r-hat), [rank-normalized](scientific_basis.md#rank-normalized-split-r-hat),
and [combined](scientific_basis.md#folded-and-combined-r-hat) conventions.

For `M` original chains of length `N`, multi-chain ESS uses `S = 2 M floor(N/2)` retained draws.
Relative ESS is ESS/S and can exceed one under antithetic sampling.
The pooled estimator regularizes integrated time at `1/log10(S)`; `is_regularized()` reports when this changes the result.
Its finite-lag conventions differ from single-chain ESS.
See [multi-chain ESS and its pinned references](scientific_basis.md#multi-chain-effective-sample-size).

Neither large ESS nor R-hat near one proves convergence or exploration of every mode.
Use dispersed starts, inspect multiple observables and traces, and assess precision for the summaries you need.
All chains can agree while missing the same region of the target distribution.

## Run the complete workflow

From the repository root:

```bash
just diagnostic-plots
```

[`examples/diagnostics.rs`](../examples/diagnostics.rs) runs four standard-normal chains with distinct seeds and dispersed starts.
It discards warmup, retains every production step, and computes ACF, single- and multi-chain ESS, mean/quantile MCSE, and all R-hat components.
Quantile summaries use probabilities 0.05, 0.5, and 0.95.
MCSE has position units; the target standard deviation is one.

The report includes five scenarios: ordinary sampling, shifted or rescaled chain 3,
a narrow proposal with slow mixing, and the tied discrete observable `floor(abs(position))`.
Shifted/rescaled chains deliberately no longer share a target.
The fixed seeds support reproduction with pinned tools and dependencies; the results illustrate diagnostics and are not a convergence gate.

| Command | Result |
| --- | --- |
| `just diagnostic-plots` | Regenerate Rust exports and render the rank/efficiency notebook |
| `just diagnostic-plots-data` | Regenerate Rust CSV/JSON with build-time source provenance |
| `just example diagnostics` | Run the Rust example without plotting |
| `just example ising_1d` | Run the physical-model trace example |
| `just notebook-check` | Lint and execute both analysis notebooks |
| `just diagnostic-plots-figures` | Regenerate the two tracked rank/ESS figures below |

The diagnostics example measures sequential production segments, including transitions and recording.
Allocation, warmup, diagnostics, and export are excluded.
ESS/second illustrates this timing contract; it varies with build profile and machine and is not benchmark evidence.
The [example validator](../tooling/examples.toml) checks successful demonstration output through `just examples` and `just ci`.

## Original-chain ranks and prefix efficiency

`PooledRanks::from_chains(&chains)` returns owned plot data with borrowed `chain(index)` slices in original draw order.
Pair input positions with your application's `ChainId`s.
Ranks are one-based pooled averages for exact ties, including signed zeros.
Constant, single-draw, or unequal-length finite chains are valid plot inputs, although they may be unsuitable for ESS or R-hat.
Empty/nonfinite inputs and unsupported counts return [`PooledRankError`][rank-errors].

Plot ranks include **all original production draws**, including an odd middle draw.
Split R-hat/ESS omit that draw before their internal ranking.
The example independently recomputes ranks, quantile cutoffs, and diagnostics for prefixes of 64, 127, 256, 512, and 1,024 draws per chain.
Reusing full-run ranks or cutoffs would change the prefix analysis.
Efficiency plots use retained split count `S` on the horizontal axis; CSVs also preserve original `N` and `M N`.

Rank overlays compare each chain's bin proportions with the **pooled bin reference**, using common edges.
Ties can make the discrete pooled reference nonuniform even when chains agree.
Neither uniform-looking ranks nor agreement with pooled bins proves convergence.
[Vehtari et al., Section 4.5][rank-methods] motivate rank plots and prefix efficiency curves.

![Original-chain rank proportions for ordinary sampling, location and scale disagreement, slow mixing, and a tied discrete observable.](assets/diagnostic_rank_overlays.png)

Each row uses the full 1,024-draw prefix from four original chains.
Colored lines identify chains; the dashed black line is the pooled reference.
The discrete row illustrates why tied data need that reference instead of an expectation of flat bins.

![Bulk and tail ESS and relative ESS across independently recomputed prefixes of the five diagnostic scenarios.](assets/diagnostic_ess_efficiency.png)

Each point recomputes diagnostics for its prefix.
The left column shows bulk/tail ESS; the right divides each by retained split count `S`.
These curves compare information as draws accumulate.
Only full, directly sampled runs have measured production timing; prefixes and post-sampling transformations leave ESS/second unavailable.

[`notebooks/diagnostic_plots.ipynb`](../notebooks/diagnostic_plots.ipynb) renders exported Rust results without recomputing estimators.
It writes rank, efficiency, and trace/ACF figures plus `rank_bins.csv`, `efficiency.csv`, `summary.csv`,
and `blocked_errors.csv` under `target/notebooks/diagnostics/`.
Unavailable results retain status/reason fields, including missing ACF legend entries and blocked-error rows.
Blocked errors preserve each chain's block size, block count, and used draws; the coarsest level is not automatically reliable.

To render a saved report without regenerating data:

```bash
just notebook-sync
MCMC_DIAGNOSTICS_PATH=/path/to/diagnostics.json \
MCMC_NOTEBOOK_OUTPUT_DIR=target/notebooks/saved-diagnostics \
uv run --locked --group dev --group notebook research-repo-tools notebooks execute notebooks/diagnostic_plots.ipynb
```

Explicit input paths never fall back to another report.
`MCMC_NOTEBOOK_OUTPUT_DIR` selects the output directory; `MCMC_REPO_ROOT` controls default paths.
Change the notebook's `rank_prefix` to inspect another exported prefix.
Trace axes use recorded-draw indices and label the positive recording interval.

## Export and notebook workflow

The diagnostics example writes schema-2 `target/diagnostics.json` with exact draws, ranks, results, and run/observable/chain identities.
It retains seeds, starts, proposal widths, warmup, cadence, original/used counts, RNG/crate versions, and estimator references.
Its CSV columns are `run_id,observable,chain_id,draw,value`.

The named build recipes capture source revision and dirty state; the report embeds the Cargo lockfile and declared Rust baseline.
Retain the source checkout/diff for dirty builds: a version or revision alone cannot reconstruct local edits.
Direct Cargo builds without the provenance environment values report null source provenance.
The plotting notebook copies the exact input JSON and records its SHA-256 and plotting versions in `manifest.json`;
the shared executor also records notebook/environment provenance.

For a physical observable, [`examples/ising_1d.rs`](../examples/ising_1d.rs) records four sequential chains with distinct seeds and initial states.
Its schema-1 `target/ising_1d_diagnostics.json` accompanies the CSV trace and identifies estimators, observables, chains, warmup,
recording interval, counts, per-chain seconds, and production timing scope.
Production timing includes sampling and observation/recording and excludes warmup, export, and diagnostics.

[`notebooks/ising_trace_analysis.ipynb`](../notebooks/ising_trace_analysis.ipynb) computes diagnostics from that trace.
It uses companion timing only for matching full production samples.
External traces need an explicit `MCMC_DIAGNOSTICS_PATH` for rates; further warmup removal leaves full-run timing unavailable.
Preserve the source JSON with exported ESS/R-hat tables to retain timing and warmup scope.
This notebook is an additional consumer, rather than an independent numerical oracle.

Both workflows set example parameters in Rust source and replace generated files under `target/` on reruns.
Console output is a summary; use CSV/JSON for analysis.
The exports are example-owned formats, independent of optional checkpoint `serde` support.
JSON and plotting dependencies are development dependencies; ordinary library use requires neither.

## Keep unavailable results explicit

Preserve status, nullable values, typed errors, and observable/chain context.
Do not replace unavailable ESS with zero, R-hat with one, or a collapsed quantile interval with zero uncertainty.
Display/debug strings in exports are versioned evidence; use Rust variants for failure-specific handling.

| Outcome | Response |
| --- | --- |
| `AutocorrelationError::TruncationNotFound` | Increase the lag budget or collect more draws, then reassess; the current window does not establish truncation |
| Unavailable R-hat component | Retain its reason and original count metadata; a successful component does not replace the missing combined maximum |
| Unavailable tail component | Retain each component; the minimum, relative ESS, and rate require both to succeed |
| Collapsed quantile interval | Retain unavailability; tied order statistics do not establish exactness |
| Missing or mismatched timing | Preserve ESS while leaving ESS/second unavailable |

`CombinedRhat::estimate` returns an error for invalid inputs/counts, otherwise a report with both component results and an optional maximum.
A valid binary/count observable can have constant folded deviations and therefore no combined value.
Whether a structurally constant observable belongs in an application gate remains caller policy.

Exports retain original and retained counts.
When Ising R-hat fails, half lengths and omitted-middle counts are null; original per-chain length is also null for unequal lengths or no chains.
Unavailable quantile MCSE leaves its exported point/interval fields null while preserving the separate indicator ESS result.

See [`AutocorrelationError`][acf-errors], [`SplitRhatError`][rhat-errors], and [`MonteCarloError`][mc-errors] for complete failure contracts,
and [numerical conventions](scientific_basis.md#diagnostics) for count/precision limits.

## Validation evidence

The existing examples and tests cover this workflow; reference agreement and successful rendering answer different validation questions.

| Evidence | Coverage |
| --- | --- |
| [`examples/diagnostics.rs`](../examples/diagnostics.rs), [`examples/ising_1d.rs`](../examples/ising_1d.rs) | Complete scalar and physical-model workflows |
| [`tests/autocorrelation.rs`](../tests/autocorrelation.rs), [`tests/convergence.rs`](../tests/convergence.rs) | Hand-calculated estimates, analytic AR(1) behavior, drift, separated chains, trace selection, self-loops, and numerical extremes |
| [ACF properties](../tests/proptest_autocorrelation.rs), [R-hat properties](../tests/proptest_convergence.rs) | Exact integer-moment oracles and public-API invariants |
| [`tests/rank_normalized_rhat.rs`](../tests/rank_normalized_rhat.rs), [`tests/combined_rhat.rs`](../tests/combined_rhat.rs) | Pinned ArviZ/posterior comparisons, Cauchy and scale disagreement, folded degeneracy, ties, and odd-length conventions |
| [`tests/ess.rs`](../tests/ess.rs), [ESS properties](../tests/proptest_ess.rs) | Thirteen pinned ArviZ regimes, ESS/MCSE units, timing mismatches, antithetic regularization, and explicit sentinel-policy differences |
| [`tests/pooled_ranks.rs`](../tests/pooled_ranks.rs), [plot fixtures](../tests/fixtures/diagnostic_plots.json) | Original identity/order, odd middles, ties/signed zeros, and independently reranked prefixes |
| [`tests/public_api.rs`](../tests/public_api.rs), API doctests | Public imports, components, ratios/rates, and ownership contracts |
| [`tests/tooling/test_notebooks.py`](../tests/tooling/test_notebooks.py) | Report/count consistency, missing results, cadence labels, matching timing, and artifact consumption |

[ArviZ developers][arviz] supply independent numerical reference implementations for multi-chain ESS/MCSE and rank-normalized R-hat.
The [ArviZ software paper by Kumar et al.][arviz-paper] credits the package underlying these pinned comparisons.
The [Stan posterior contributors][posterior] supply the combined R-hat fixtures.
Pinned versions, exact inputs, tolerances, and intentional policy differences are retained with the fixture generators;
agreement establishes the tested conventions, rather than convergence of an arbitrary chain.

Run the Rust integration tests with `just test-integration`, API examples with `just test-doc`,
and notebook consumer tests with `uv run --locked --group dev pytest -q tests/tooling/test_notebooks.py`.
To reproduce independent references and cross-check a freshly exported report:

```bash
uv run --script tests/fixtures/generate_ess.py
uv run --script tests/fixtures/generate_combined_rhat.py
uv run --script tests/fixtures/generate_diagnostic_plots.py
uv run --script tests/fixtures/generate_diagnostic_plots.py --report target/diagnostics.json
```

These isolated scripts verify retained evidence without rewriting it.
They pin Python 3.13.7, NumPy 2.2.6, SciPy 1.16.2, and ArviZ 0.22.0.
The combined-fixture script cross-checks retained posterior 1.7.0 results using ArviZ with posterior's fold-before-split convention;
it does not rerun the original R oracle.

[acf-errors]: https://docs.rs/markov-chain-monte-carlo/latest/markov_chain_monte_carlo/enum.AutocorrelationError.html
[arviz]: ../REFERENCES.md#ref-19
[arviz-paper]: ../REFERENCES.md#ref-23
[blocking]: ../REFERENCES.md#ref-7
[eabm]: ../REFERENCES.md#ref-24
[eabm-diagnostics]: https://arviz-devs.github.io/EABM/Chapters/MCMC_diagnostics.html
[initial-sequence]: ../REFERENCES.md#ref-8
[mc-errors]: https://docs.rs/markov-chain-monte-carlo/latest/markov_chain_monte_carlo/enum.MonteCarloError.html
[original-rhat]: ../REFERENCES.md#ref-22
[posterior]: ../REFERENCES.md#ref-20
[rank-errors]: https://docs.rs/markov-chain-monte-carlo/latest/markov_chain_monte_carlo/enum.PooledRankError.html
[rank-methods]: ../REFERENCES.md#ref-14
[rhat-errors]: https://docs.rs/markov-chain-monte-carlo/latest/markov_chain_monte_carlo/enum.SplitRhatError.html
[stan-diagnostics]: ../REFERENCES.md#ref-13
