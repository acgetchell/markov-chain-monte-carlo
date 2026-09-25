# Scientific Basis and Scope

This crate implements Metropolis-Hastings sampling primitives for user-defined state spaces. It provides the transition machinery, numerical checks, rollback
contracts, diagnostics, and streaming estimators needed for robust scientific code, but the scientific validity of a chain also depends on the caller's target
distribution, proposal kernel, state representation, and analysis choices.

## Metropolis-Hastings Contract

For a current state `x` and proposed state `y`, the crate uses the standard Metropolis-Hastings acceptance probability:

```text
alpha(x, y) = min(1, exp(log pi(y) - log pi(x) + log q(x | y) - log q(y | x)))
```

The `Target<S>` implementation supplies `log pi(s)` up to an additive constant. Additive constants are fine because Metropolis-Hastings only uses differences,
but arbitrary scores or logits sample a different distribution. The proposal implementation supplies either symmetric proposals or an explicit log proposal
ratio through:

- `Proposal::log_q_ratio(current, proposed)`
- `ProposalMut::log_q_ratio(state, token)`
- `DelayedProposal::log_q_ratio(state, plan) -> Result<f64, Self::Error>`

These ratios must describe the same concrete transition that was proposed. For combinatorial systems, this usually means accounting for move-kind probabilities,
site counts, reverse-site counts, and invalid-move handling.

Detailed balance, or a valid Metropolis-Hastings correction for a non-symmetric proposal, is a property of the user-provided target+proposal pair. The crate
checks transition mechanics; domain code still owns irreducibility, aperiodicity, burn-in, autocorrelation, convergence, and observable interpretation.

## Additive Target Terms

Bias potentials, umbrella-sampling weights, softened constraints, auxiliary energy/action terms, and externally supplied learned regularizer terms should be
included in the target distribution itself. For separate model and bias terms, use `AdditiveTarget` or an equivalent `Target` implementation that returns the
combined log weight:

```text
log pi(state) = log pi_model(state) + log pi_bias(state)
```

When the model is written as an action or energy, each component should return the negative component action:

```text
log pi(state) = -S_model(state) - S_bias(state)
```

The Metropolis-Hastings target contribution is therefore:

```text
log pi(y) - log pi(x) = -(Delta S_model + Delta S_bias)
```

Proposal asymmetry is not folded into the bias term. Keep it in the appropriate `log_q_ratio` implementation so the full acceptance ratio remains:

```text
log_alpha = -(Delta S_model + Delta S_bias) + log q(x | y) - log q(y | x)
```

The runnable [`examples/additive_target_bias.rs`](../examples/additive_target_bias.rs) demonstrates this split on a two-state target: a flat model term is
combined with a bias weight through `AdditiveTarget`, while the symmetric flip proposal keeps the proposal-ratio correction at its default zero value.

Externally supplied learned regularizer terms use the same contract as physics actions: return an unnormalized log weight, or return `-E(state)` when the term
is written in energy form. This crate currently provides sampler mechanics and target composition, not training for learned energies or learned proposal
policies.

## Adaptive Warmup

`AdaptiveScale` tunes one positive proposal width through `TunableProposal`. `Sampler::warm_up`, `warm_up_mut`, and `warm_up_delayed` run the existing
Metropolis-Hastings transitions and change the scale only after each completed transition. All sampling and Hastings-ratio calculations for a transition
must use the same scale. Every fixed scale must define a valid kernel for the same target, and increasing width should generally lower acceptance.

For completed warmup step `n`, starting at one, the implemented update is:

```text
log_scale = clamp(log_scale + n^(-0.6) * (I_accepted - target_acceptance), log_min, log_max)
scale = clamp(exp(log_scale), min_scale, max_scale)
```

This is a bounded acceptance-indicator Robbins-Monro scale update, related to the acceptance-probability scaling updates discussed in
[Andrieu and Thoms (2008)](https://doi.org/10.1007/s11222-008-9110-y), section 5.1.2. It does not estimate a covariance matrix or implement the full
Haario adaptive Metropolis algorithm. The gain exponent is fixed at 0.6; the target rate and bounds are explicit caller choices, not universal optima.

Rejection and no-proposal self-loops both contribute zero acceptance, matching chain counters. Errors do not advance tuning, and earlier completed work
is retained. A zero-step call has no effects. Keep the same tuner, proposal, and RNG across warmup chunks; counter resets do not restart the schedule.
Chain checkpoints contain neither the tuner nor the tuned proposal parameters. The completed-step count saturates at `usize::MAX`, freezing further updates.

Discard adaptive draws. Ordinary `step`, `run`, observation, thinning, and iterator methods perform no tuning, so production uses the final fixed kernel.
This avoids claiming validity for indefinite adaptation, but freezing does not itself make the starting state stationary. Choose additional burn-in and
assess mixing, ESS, and convergence for the frozen kernel. A bounded scale and a plausible acceptance rate do not establish irreducibility or adequate
exploration; targets with strong correlations or multiple modes may need different proposals.

## What the Crate Checks

The library enforces several local invariants:

- Acceptance decisions are computed in log space.
- Log-space acceptance avoids underflow in tail probabilities.
- `NaN` and positive-infinite target log-probabilities or proposal ratios are rejected.
- In-place proposals roll back target state and proposal-internal transition state on rejection or invalid proposed values.
- Delayed proposals separate planning, scoring, acceptance, and commit so mutations happen only after acceptance.
- Sampling counters and cached log-probabilities stay synchronized through library-owned transitions.

These checks protect the mechanics of a single transition. They do not prove that a user-defined proposal explores the full intended state space.

## Diagnostics

The crate includes diagnostics that help users test assumptions:

- Observables measure derived quantities during sampling.
- `OnlineStats` provides one-pass summary statistics.
- `BinningAnalysis` estimates uncertainty for correlated samples.
- `Autocorrelation` estimates the ACF and integrated autocorrelation time for one regularly sampled scalar observable after burn-in.
- Thinning helpers collect every k-th state or observation while still advancing the chain on every step.
- Detailed-balance helpers empirically compare forward and reverse transition flows for representative discrete transitions.
- `verify_proposal_density` compares reported Hastings ratios with independently supplied forward/reverse log densities.
- `verify_proposal_bins` compares independent proposal histograms with reference bin probabilities using a simultaneous Hoeffding bound.

Detailed-balance checks are especially useful for new proposal kernels, but they remain empirical tests over selected transitions. Passing them does not
establish irreducibility, aperiodicity, or adequate mixing.

For in-place proposals, every concrete hypothetical proposal is undone before the next trial, and no-proposal telemetry is consumed. The proposal must still
represent a fixed kernel: freeze online adaptation before validation, and keep any transition-relevant mutable state inside the `undo` contract.

Continuous density checks test the supplied pair's ratio; binned checks test the generator's mass over a fixed partition. Neither is a continuous
detailed-balance or convergence test. Bin checks require independent proposal draws from a fixed endpoint, exhaustive bins, and reference probabilities
derived independently of the generator. They are not calibrated for correlated chain output. See the
[continuous-proposal workflow](proposal_validation.md#continuous-proposals) for the bound, error budget, and limits of coarsened flow comparisons.

### Autocorrelation estimator contract

`Autocorrelation::estimate(&samples, max_lag)` accepts a finite scalar slice, whether collected in memory or read from an exported CSV. Select one chain and
one observable, preserve time order, and use a constant recording interval. For a `Trace`, collect a named column with
`trace.observable_values(id, "energy")?.copied().collect::<Vec<_>>()`. Selection borrows the trace, checks the column name, and preserves insertion order;
use `records_for_chain(id)` to verify step ordering and spacing. Keep repeated values from rejected and no-proposal steps.
Remove burn-in before analysis;
never concatenate independent chains into one time series.

The sample-mean-centered ACF uses biased autocovariances with divisor `N` at every lag, normalized by the lag-zero covariance. Direct summation costs
`O(N * (max_lag + 1))`, with `O(N + max_lag)` memory. The inclusive lag limit bounds work. Shifted, scaled centering and compensated summation reduce numerical
error and avoid overflow for extreme finite values, but cannot recover variation already lost when observations were rounded to `f64`.

`integrated_time()` uses the initial monotone sequence: pair lags `(0, 1), (2, 3), ...`, discard the first nonpositive pair and everything after it, and make
the retained pair sums nonincreasing by taking cumulative minima. The estimate is `tau = -1 + 2 * sum(retained_pairs)`, with independent-sample convention
`tau = 1`. The result includes the largest retained lag and sample count; an unpaired final lag is unused. This follows
[Geyer's initial sequence estimators](https://www.stat.umn.edu/geyer/mcmc/library/mcmc/html/initseq.html), whose justification assumes a stationary
reversible chain, finite observable variance, and summable autocorrelations.

Constant, nonfinite, and fewer-than-two-sample inputs are explicit errors. Missing truncation within the lag budget and nonpositive time estimates are also
errors, not silently accepted sums or clamped values. Positive times below one remain possible for anticorrelated samples. Times are in recorded-sample
intervals: multiply by the recording interval for transition-step units, or divide by the number of spins to express per-step Ising results in sweeps.

For a trace recorded every `d` transitions, the retained series has `rho_retained[k] = rho_original[d*k]`. Multiplying its estimated time by `d` changes
units; it does not reconstruct the correlations at omitted lags or the unthinned chain's integrated time. For example, an AR(1) process with coefficient
`1/2` has true time `3`; retaining every second value gives time `5/3` in retained-sample intervals, or `10/3` in transition-step units.
The cutoff and final positivity tests use rounded `f64` values, so pair sums or time estimates near zero can change classification under roundoff or underflow.

A found window is not evidence of convergence or adequate length. Finite-sample centering biases the ACF, noisy tails can end the window prematurely, and a
short trace can miss slow modes. Compare longer production runs, larger lag budgets, and independent chains. Energy and magnetization need separate estimates.
The Ising example exports Rust estimates under `target/`; its notebook independently computes the same formulas from the trace and flags short runs with
fewer than 50 estimated correlation times as a heuristic caution, not a pass/fail convergence test; see the
[emcee discussion of trace length](https://emcee.readthedocs.io/en/stable/tutorials/autocorr/).

### ESS and wall-clock efficiency

`IntegratedAutocorrelationTime::effective_sample_size()` estimates the effective sample size for one scalar mean as `N / tau`. It uses the same retained
sample count and recorded-sample time units as the ACF. The stationarity, reversibility, finite-variance, and summable-correlation assumptions above apply.
Positive estimates with `tau < 1` give ESS greater than `N`; there is no cap. This is not rank-normalized bulk ESS, tail ESS, or a pooled multi-chain estimator.
Short, constant, nonfinite, nonpositive-time, and missing-truncation failures remain the typed ACF/time errors; no ESS is produced for those inputs.
In particular, at least four samples and a complete nonpositive pair are necessary for this initial-sequence estimate, but never sufficient to trust it.

`effective_sample_size_per_second(elapsed: Duration)` divides by measured wall seconds and rejects zero duration with `EssRateError`. Missing timing should
remain unavailable; sample indices are not a clock. Measure the exact production workload supplying the analyzed draws, including intervening transitions
when thinning. Document whether warmup, observation, I/O, and analysis are included. Never reuse a full-run duration for a selected subset of draws.
Per-chain rates do not automatically describe aggregate parallel throughput, which requires the wall time of the complete concurrent workload.

### Classical split R-hat

`SplitRhat::estimate(&[&chain_a, &chain_b, ...])` takes borrowed slices for a single comparable observable. Supply at least two original chains, each with
at least four finite draws and equal lengths, after the same warmup policy. Keep their targets, units, recording intervals, and observation definitions
consistent; use dispersed starts and independent random streams. The scalar API checks lengths and values but cannot check these scientific prerequisites.

For original length `N`, use the first and last `n = floor(N/2)` draws of each chain; omit the middle draw when `N` is odd. If there are `m` split chains,
`W` is their mean unbiased sample variance and `B/n` is the unbiased variance of their means (denominator `m-1`). The reported value is
`sqrt(((n-1)/n * W + B/n) / W)`, without a floor at one. Result metadata reports original chain count, supplied length, and used half length.
This follows the [classical split R-hat definition](https://mc-stan.org/docs/2_29/reference-manual/notation-for-samples-chains-and-draws.html).

`SplitRhatError` distinguishes insufficient chains, short chains, unequal lengths, nonfinite values (including omitted middle draws), any constant half, and
numerically unresolved variance or result. Rejecting even one constant half is a conservative stuck-chain policy. Common scaling avoids overflow and local
centering preserves small within-half changes; enormous scale disparities can still underflow variance. `UnresolvedVariance` identifies the original chain
and half whose varying samples lost resolvable variance. `NumericalFailure` is reserved for an unrepresentable final estimate after all half variances pass.

This is a raw-moment diagnostic with finite marginal mean/variance assumptions. Splitting helps expose within-chain drift, but classical R-hat can miss scale
differences and heavy-tail problems. It does not implement the rank-normalized and folded improvements described by
[Vehtari et al.](https://arxiv.org/abs/1903.08008). A value near one does not establish convergence or exploration of all modes; combine it with ESS, trace
inspection, dispersed starts, and longer runs. Do not apply modern rank-normalized thresholds as a guarantee for this estimator.

### Export and notebook workflow

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

## User Responsibilities

Domain code should still validate:

- State invariants before and after domain-specific moves
- Proposal irreducibility or known connected components for the intended state space
- Correct proposal ratios for asymmetric moves
- Burn-in, autocorrelation, effective sample size, and convergence behavior
- Reproducible random-number seeding and independent streams for parallel chains
- Scientific interpretation of observables and uncertainty estimates

For constrained triangulations, graphs, or other combinatorial systems, the strongest checks usually combine domain-specific invariant tests with this crate's
transition-level diagnostics.

## References

See [`REFERENCES.md`](../REFERENCES.md) for canonical background references on Metropolis-Hastings, MCMC, and the example models used by this repository.
