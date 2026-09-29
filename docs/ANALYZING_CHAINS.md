# Analyzing Chains

Version 0.5.0 provides all three diagnostics requested in [#13](https://github.com/acgetchell/markov-chain-monte-carlo/issues/13):
autocorrelation, effective sample size (ESS), and Gelman–Rubin R-hat using classical split chains. These APIs require no optional Cargo features.
ACF and integrated time landed through #73; mean ESS, ESS per second, and split R-hat through #74. Parallel execution (#12) is separate:
independently initialized chains can run sequentially and still supply R-hat inputs.

## Run the complete workflow

```bash
just example diagnostics
```

[`examples/diagnostics.rs`](../examples/diagnostics.rs) samples four standard-normal chains with distinct seeds and dispersed starts, discards warmup,
and retains every production step. It prints lag-one ACF and mean ESS for each chain, then classical and rank-normalized split R-hat across the four chains.
The same borrowed chain slices feed both estimators. Neither includes the folded component or combined maximum.
The fixed seeds make the demonstration reproducible within the pinned toolchain and dependencies. Its output is an illustration, not a convergence gate.
`just examples` and `just ci` check its successful diagnostic output through the shared `tooling/examples.toml` validator.

For observable names, chain identifiers, CSV/JSON export, measured ESS per second, and notebook plots, use
[`examples/ising_1d.rs`](../examples/ising_1d.rs) and `just notebook-check`. The Ising example writes under `target/`; the notebook consumes those artifacts.

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
| How correlated are draws at a given lag? | `Autocorrelation::estimate(samples, max_lag)?.values()` | ACF from lag zero through the inclusive maximum lag |
| How much serial correlation affects a scalar mean? | `acf.integrated_time()?` | Geyer's initial monotone sequence estimate in recorded-sample intervals |
| How much information does this chain provide about that mean? | `time.effective_sample_size()` | `N / tau`, for this observable and chain |
| How efficiently did the measured run produce that information? | `time.effective_sample_size_per_second(elapsed)?` | Mean ESS divided by measured seconds |
| Do within-chain and between-chain variations agree? | `SplitRhat::estimate(&chains)?.value()` | Classical split R-hat from equally long borrowed scalar slices |
| Do the chains' ranked locations agree? | `RankNormalizedSplitRhat::estimate(&chains)?.value()` | Pooled rank-normalized split component, without folding |

Keep chains separate for ACF and ESS. Concatenation introduces artificial transitions between chains, and summing per-chain ESS does not produce a
diagnostic that accounts for disagreement between chains. ESS is observable-specific; mean ESS is not bulk or tail ESS.

For a scientific efficiency comparison, record what timing includes. The Ising workflow times production transitions and recording, excluding warmup
and diagnostics. Other timing scopes are possible, but must be comparable across the runs being compared.

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

The [Stan reference on ESS](https://mc-stan.org/docs/2_29/reference-manual/effective-sample-size.html) describes the integrated-time relationship and
initial monotone sequence method. Stan's combined multi-chain ESS is distinct from this crate's single-chain mean ESS.
The [classical split R-hat reference](https://mc-stan.org/docs/2_29/reference-manual/notation-for-samples-chains-and-draws.html) gives the variance ratio.
Neither a large ESS nor R-hat near one establishes convergence or exploration of every mode. Inspect traces, use dispersed starts, and assess multiple
observables; consider rank-normalized and folded diagnostics when raw moments are insufficient.

## Keep unavailable results explicit

`AutocorrelationError` distinguishes short, nonfinite, constant, and invalid-lag inputs. `TruncationNotFound` means the available lag window does not
establish the integrated-time truncation: increase the lag budget or collect a longer trace, then reassess. `NonPositiveTime` also leaves ESS unavailable.
`EssRateError` rejects a zero measured duration.

`SplitRhatError` identifies too few chains or samples, unequal lengths, nonfinite observations, constant halves, numerically unresolved within-half
variances, and a nonrepresentable final estimate. Both R-hat estimators reject even one constant half. Rank normalization additionally rejects retained
counts above `2^50` or overflowing `usize` with `TooManyRankedSamples` to preserve exact rank offsets. This error reports the chain count, retained half
length, and effective count limit as typed fields. Preserve errors with chain/observable context; do not replace them with zero ESS or R-hat one.
The example prints unavailable diagnostics explicitly, while its fixed demonstration inputs are expected to produce successful estimates in CI.

## Validation evidence

`tests/autocorrelation.rs` and `tests/convergence.rs` cover hand-calculated estimates, independent and correlated synthetic draws, separated locations,
within-chain drift, degenerate inputs, and numerical extremes. `tests/proptest_autocorrelation.rs` and `tests/proptest_convergence.rs` check algebraic
invariants through the public APIs. `just test-integration` runs these with the other integration tests; `just test-doc` checks the API examples.
The Ising notebook is an additional end-to-end consumer, not an independent numerical oracle.

`tests/rank_normalized_rhat.rs` adds pinned ArviZ component fixtures, deterministic Cauchy location shifts, same-distribution controls, preserved ties and
ordering under monotone transforms, signed zeros, extreme finite inputs, and borrowed-input checks. Its scale-only fixture deliberately distinguishes
the rank-normalized split component from the combined rank/folded maximum.
