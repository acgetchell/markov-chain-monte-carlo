# Scalar sampling diagnostics

The unreleased checkout provides all three diagnostics requested in [#13](https://github.com/acgetchell/markov-chain-monte-carlo/issues/13):
autocorrelation, effective sample size (ESS), and Gelman–Rubin R-hat using classical split chains. These APIs require no optional Cargo features.
ACF and integrated time landed through #73; mean ESS, ESS per second, and split R-hat through #74. Parallel execution (#12) is separate:
independently initialized chains can run sequentially and still supply R-hat inputs.

## Run the complete workflow

```bash
just example diagnostics
```

[`examples/diagnostics.rs`](../examples/diagnostics.rs) samples four standard-normal chains with distinct seeds and dispersed starts, discards warmup,
and retains every production step. It prints lag-one ACF and mean ESS for each chain, then classical split R-hat across the four chains.
The fixed seeds make the demonstration reproducible within the pinned toolchain and dependencies. Its output is an illustration, not a convergence gate.
`just examples` and `just ci` check its successful diagnostic output through the shared `tooling/examples.toml` validator.

For observable names, chain identifiers, CSV/JSON export, measured ESS per second, and notebook plots, use
[`examples/ising_1d.rs`](../examples/ising_1d.rs) and `just notebook-check`. The Ising example writes under `target/`; the notebook consumes those artifacts.

## Choose the quantity

| Question | API | Meaning |
| --- | --- | --- |
| How correlated are draws at a given lag? | `Autocorrelation::estimate(samples, max_lag)?.values()` | ACF from lag zero through the inclusive maximum lag |
| How much serial correlation affects a scalar mean? | `acf.integrated_time()?` | Geyer's initial monotone sequence estimate in recorded-sample intervals |
| How much information does this chain provide about that mean? | `time.effective_sample_size()` | `N / tau`, for this observable and chain |
| How efficiently did the measured run produce that information? | `time.effective_sample_size_per_second(elapsed)?` | Mean ESS divided by measured seconds |
| Do within-chain and between-chain variations agree? | `SplitRhat::estimate(&chains)?.value()` | Classical split R-hat from equally long borrowed scalar slices |

Keep chains separate for ACF and ESS. Concatenation introduces artificial transitions between chains, and summing per-chain ESS does not produce a
diagnostic that accounts for disagreement between chains. ESS is observable-specific; mean ESS is not bulk or tail ESS.

For a scientific efficiency comparison, record what timing includes. The Ising workflow times production transitions and recording, excluding warmup
and diagnostics. Other timing scopes are possible, but must be comparable across the runs being compared.

## Input and estimator contracts

- Discard warmup, preserve temporal order, and use a fixed recording interval. Keep rejected steps and no-proposal self-loops.
- Compare the same observable, units, target, recording interval, and warmup policy across independently initialized chains.
  Slices cannot establish these assumptions; separate seeds alone do not prove independence or mixing.
- ACF needs at least two finite observations and `max_lag < N`. It uses the sample mean and biased autocovariances with the same implicit divisor `N`
  at every lag. Lag zero is exactly one. The direct implementation costs `O(N * max_lag)` time.
- Integrated time pairs adjacent ACF values starting at lag zero and requires a nonpositive pair to establish truncation. It monotonizes the preceding
  positive pairs. An arbitrary lag cutoff is not treated as a valid truncation. ESS can exceed `N` for anticorrelated draws.
- Split R-hat needs at least two original chains of equal length, each with at least four draws. Each is split into first and last halves;
  an odd middle draw is omitted from moments but still checked for finiteness. The count minima make the formula defined, not reliable.
- R-hat uses unbiased within-half variances and variation of the half means. Values below one are retained. It assumes finite marginal mean and variance,
  and does not rank-normalize or fold observations; scale and tail differences can be missed.

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
variances, and a nonrepresentable final estimate. Preserve these errors with the chain/observable context; do not replace them with zero ESS or R-hat one.
The example prints unavailable diagnostics explicitly, while its fixed demonstration inputs are expected to produce successful estimates in CI.

## Validation evidence

`tests/autocorrelation.rs` and `tests/convergence.rs` cover hand-calculated estimates, independent and correlated synthetic draws, separated locations,
within-chain drift, degenerate inputs, and numerical extremes. `tests/proptest_autocorrelation.rs` and `tests/proptest_convergence.rs` check algebraic
invariants through the public APIs. `just test-integration` runs these with the other integration tests; `just test-doc` checks the API examples.
The Ising notebook is an additional end-to-end consumer, not an independent numerical oracle.
