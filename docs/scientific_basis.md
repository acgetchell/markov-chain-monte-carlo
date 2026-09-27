# Scientific Basis and Scope

This crate implements Metropolis-Hastings sampling primitives for user-defined state spaces. It provides the transition machinery, numerical checks, rollback
contracts, diagnostics, and streaming estimators needed for robust scientific code, but the scientific validity of a chain also depends on the caller's target
distribution, proposal kernel, state representation, and analysis choices.

## Contents

- [Scope and API selection](#scope-and-api-selection)
- [Shared conventions](#shared-conventions)
- [Metropolis-Hastings contract](#metropolis-hastings-contract)
- [What the crate checks](#what-the-crate-checks)
- [Target composition and tuning](#target-composition-and-tuning)
  - [Adaptive warmup](#adaptive-warmup)
  - [Additive target terms](#additive-target-terms)
- [Diagnostics](#diagnostics)
  - [Autocorrelation estimator contract](#autocorrelation-estimator-contract)
  - [Binning analysis](#binning-analysis)
  - [Classical split R-hat](#classical-split-r-hat)
  - [ESS and wall-clock efficiency](#ess-and-wall-clock-efficiency)
  - [Online statistics](#online-statistics)
  - [Proposal validation](#proposal-validation)
- [Export and notebook workflow](#export-and-notebook-workflow)
- [User responsibilities](#user-responsibilities)
- [References](#references)

## Scope and API selection

Use this crate when application code owns the state representation, target log weight, and proposal kernel. It supplies Metropolis-Hastings transitions and
scalar diagnostics for those definitions. It does not infer a model, train a learned target or proposal, or establish that a chain explores the intended state
space. State invariants and move availability remain application-specific.

Choose the proposal contract by how an application can construct a transition. All three paths use the same acceptance rule:

| State and mutation needs | API | Application responsibility |
| --- | --- | --- |
| An independently constructed proposed value is practical | `Proposal<S>` | Generate the proposed state and report its reverse/forward log ratio |
| A large state supports a local move with rollback | `ProposalMut<S>` | Undo both state changes and transition-relevant proposal changes on rejection |
| A move can be planned and scored before mutation | `DelayedProposal<S>` | Score the plan consistently, then commit only an accepted move |

`Target<S>` supplies the common target contract. `Chain<S>` owns chain state and counters; `Sampler` coordinates repeated transitions, observations, and
explicit warmup. The [crate API documentation](https://docs.rs/markov-chain-monte-carlo/latest/markov_chain_monte_carlo/) owns programming contracts and
worked usage. The [proposal-validation guide](VALIDATING_PROPOSALS.md) owns kernel checks, and the
[chain-analysis guide](ANALYZING_CHAINS.md) owns diagnostic recipes and exports. Bibliographic records belong in
[`REFERENCES.md`](../REFERENCES.md#method-to-source-index); this page owns the scientific explanations and their limits.

## Shared conventions

- `log` means natural logarithm. `pi(s)` is the intended target weight or density relative to the application's common reference measure, and
  `q(y | x)` is the proposal probability or density for moving from `x` to `y`.
- `Target::log_prob` returns an unnormalized log weight. A state-independent additive constant cancels in differences; arbitrary scores or logits generally
  define a different target. An action or dimensionless energy `S` contributes `-S`; physical temperature and unit conversions belong in the target.
- A proposal reports `log q(x | y) - log q(y | x)`. This correction describes the same concrete move that was generated, including move selection and
  reverse availability. It is distinct from any bias or additional target term.
- Numerical values use `f64` arithmetic, including deliberate fused multiply-add operations. Log-space acceptance avoids exponentiating tiny
  probabilities, but does not make target differences exact or eliminate finite-precision limitations. Target log weights and proposal log ratios may be
  finite or negative infinity; `NaN` and positive infinity produce errors. A `NaN` assembled acceptance ratio, including `-inf - (-inf)`, rejects the move.
- Production diagnostics refer to one observable after discarded warmup unless a method explicitly compares multiple chains. Preserve repeated states from
  rejection and no-proposal self-loops, temporal order, and constant recording intervals. Thinning advances through all intervening transitions.

## Metropolis-Hastings Contract

For a current state `x` and proposed state `y`, the crate uses the acceptance probability of
[Metropolis et al.](../REFERENCES.md#ref-1) and [Hastings](../REFERENCES.md#ref-2):

```text
alpha(x, y) = min(1, exp(log pi(y) - log pi(x) + log q(x | y) - log q(y | x)))
```

The target and proposal use the [shared conventions](#shared-conventions). The proposal supplies a zero ratio for a symmetric kernel or an explicit
log proposal ratio through:

- `Proposal::log_q_ratio(current, proposed)`
- `ProposalMut::log_q_ratio(state, token)`
- `DelayedProposal::log_q_ratio(state, plan) -> Result<f64, Self::Error>`

These ratios must describe the same concrete transition that was proposed. For combinatorial systems, this usually means accounting for move-kind probabilities,
site counts, reverse-site counts, and invalid-move handling.

Detailed balance, or a valid Metropolis-Hastings correction for a non-symmetric proposal, is a property of the user-provided target+proposal pair. The crate
checks transition mechanics; domain code still owns irreducibility, aperiodicity, burn-in, autocorrelation, convergence, and observable interpretation.

## What the Crate Checks

The library enforces several local invariants within the proposal contracts:

- Acceptance decisions are computed in log space, without exponentiating tail probabilities.
- Delayed proposals separate planning, scoring, acceptance, and commit so mutations happen only after acceptance.
- In-place proposals invoke rollback on rejection or invalid proposed values; the caller's undo implementation must restore target state and
  proposal-internal transition state.
- Invalid target log-probabilities and proposal ratios produce errors under the shared numerical conventions.
- Sampling counters and cached log-probabilities stay synchronized through library-owned transitions.

These checks protect the mechanics of a single transition. They do not prove that a user-defined proposal explores the full intended state space.

## Target composition and tuning

Both methods use the acceptance contract above: target composition changes the sampled distribution, while tuning changes the proposal used to explore it.

### Adaptive Warmup

`AdaptiveScale` tunes one positive proposal width through `TunableProposal`. `Sampler::warm_up`, `warm_up_mut`, and `warm_up_delayed` run the existing
Metropolis-Hastings transitions and change the scale only after each completed transition. All sampling and Hastings-ratio calculations for a transition
must use the same scale. Every fixed scale must define a valid kernel for the same target, and increasing width should generally lower acceptance.

For completed warmup step `n`, starting at one, the implemented update is:

```text
log_scale = clamp(log_scale + n^(-0.6) * (I_accepted - target_acceptance), log_min, log_max)
scale = clamp(exp(log_scale), min_scale, max_scale)
```

This is a bounded acceptance-indicator Robbins-Monro scale update, related to the acceptance-probability scaling updates discussed in
[Andrieu and Thoms (2008)](../REFERENCES.md#ref-12), section 5.1.2. It does not estimate a covariance matrix or implement the full
Haario adaptive Metropolis algorithm. The gain exponent is fixed at 0.6; the target rate and bounds are explicit caller choices, not universal optima.

Rejection and no-proposal self-loops both contribute zero acceptance, matching chain counters. Errors do not advance tuning, and earlier completed work
is retained. A zero-step call has no effects. Keep the same tuner, proposal, and RNG across warmup chunks; counter resets do not restart the schedule.
Chain checkpoints contain neither the tuner nor the tuned proposal parameters. The completed-step count saturates at `usize::MAX`, freezing further updates.

Discard adaptive draws. Ordinary `step`, `run`, observation, thinning, and iterator methods perform no tuning, so production uses the final fixed kernel.
This avoids claiming validity for indefinite adaptation, but freezing does not itself make the starting state stationary. Choose additional burn-in and
assess mixing, ESS, and convergence for the frozen kernel. A bounded scale and a plausible acceptance rate do not establish irreducibility or adequate
exploration; targets with strong correlations or multiple modes may need different proposals.

### Additive Target Terms

Bias potentials, umbrella-sampling weights, softened constraints, auxiliary energy/action terms, and externally supplied learned regularizer terms should be
included in the target distribution itself. For separate model and bias terms, `AdditiveTarget` or an equivalent `Target` implementation returns the combined
log weight:

```text
log pi(state) = log pi_model(state) + log pi_bias(state)
```

When the model is written as an action or dimensionless energy, each component returns the negative component action:

```text
log pi(state) = -S_model(state) - S_bias(state)
```

The Metropolis-Hastings target contribution is therefore:

```text
log pi(y) - log pi(x) = -(Delta S_model + Delta S_bias)
```

Proposal asymmetry remains in `log_q_ratio`, distinct from the bias term, so the acceptance ratio is:

```text
log_alpha = -(Delta S_model + Delta S_bias) + log q(x | y) - log q(y | x)
```

This composition follows directly from the [Metropolis-Hastings contract](#metropolis-hastings-contract), rather than introducing a different sampling
algorithm. The runnable [`examples/additive_target_bias.rs`](../examples/additive_target_bias.rs) demonstrates it for a two-state target.
Externally supplied learned regularizer terms use the same log-weight contract; this crate does not train those terms.

## Diagnostics

Observables measure derived quantities during sampling. The diagnostics below test specific assumptions or estimate scalar uncertainty; passing a
diagnostic does not establish convergence, irreducibility, aperiodicity, or adequate mixing. Shared sampling conventions apply before choosing a method.

### Autocorrelation estimator contract

`Autocorrelation::estimate(&samples, max_lag)` accepts a finite scalar slice for one chain and observable, whether collected in memory or read from an
exported CSV. Discard burn-in, preserve time order and a constant recording interval, and retain self-loops. Never concatenate independent chains into one
time series. The [chain-analysis guide](ANALYZING_CHAINS.md) covers selecting a trace column and checking its recording interval.

The sample-mean-centered ACF uses biased autocovariances with divisor `N` at every lag, normalized by the lag-zero covariance. Direct summation costs
`O(N * (max_lag + 1))`, with `O(N + max_lag)` memory. The inclusive lag limit bounds work. Shifted, scaled centering and
[Kahan compensated summation](../REFERENCES.md#ref-16) reduce numerical error and avoid overflow for extreme finite values, but cannot recover variation already
lost when observations were rounded to `f64`.

`integrated_time()` uses the initial monotone sequence: pair lags `(0, 1), (2, 3), ...`, discard the first nonpositive pair and everything after it, and make
the retained pair sums nonincreasing by taking cumulative minima. The estimate is `tau = -1 + 2 * sum(retained_pairs)`, with independent-sample convention
`tau = 1`. The result includes the largest retained lag and sample count; an unpaired final lag is unused. This follows
[Geyer's initial sequence estimators](../REFERENCES.md#ref-8), whose justification assumes a stationary
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
[emcee discussion of trace length](../REFERENCES.md#ref-15). That discussion uses a different windowing procedure and does not validate this crate's estimator
for a particular run length.

### Binning analysis

`BinningAnalysis` accumulates neighboring samples into a hierarchy of nonoverlapping blocks with sizes `1, 2, 4, ...`. Each level applies
[Welford accumulation](#online-statistics) to completed block means and pairs adjacent means to feed the next level. For block size `b`, there are
`floor(N / b)` completed blocks. A trailing incomplete block contributes only at finer levels where it is complete, so different levels can summarize
different retained prefixes. The original samples are not stored; memory grows as `O(log N)`.

At a level with `k >= 2` completed block means and their sample variance `s_b^2`, the reported standard error is `sqrt(s_b^2 / k)`. This is the blocking
approach of [Flyvbjerg and Petersen](../REFERENCES.md#ref-7): progressively average nearby correlated measurements, then examine whether uncertainty
estimates stabilize as block size increases. It assumes a stationary series with finite variance and sufficient decay of correlations for block means to
become approximately independent. Block size one gives the naive error that ignores autocorrelation.

The crate returns per-level estimates and offers the coarsest level with at least two completed blocks as a convenience. It does not test for a plateau
or select a scientifically sufficient block size. Coarse levels have fewer blocks and noisier variance estimates; two blocks merely make the formula
defined. A plateau in a short run can miss slow modes and does not prove equilibration. Inspect neighboring levels, retain enough blocks, and compare
longer runs and independently initialized chains before interpreting correlated uncertainty. Fallible updates reject nonfinite samples or accumulator
arithmetic and leave the current sample unapplied; earlier successful samples remain.

### Classical split R-hat

`SplitRhat::estimate(&[&chain_a, &chain_b, ...])` takes borrowed slices for a single comparable observable. Supply at least two original chains, each with
at least four finite draws and equal lengths, after the same warmup policy. Keep their targets, units, recording intervals, and observation definitions
consistent; use dispersed starts and independent random streams. The scalar API checks lengths and values but cannot check these scientific prerequisites.

For original length `N`, use the first and last `n = floor(N/2)` draws of each chain; omit the middle draw when `N` is odd. If there are `m` split chains,
`W` is their mean unbiased sample variance and `B/n` is the unbiased variance of their means (denominator `m-1`). The reported value is
`sqrt(((n-1)/n * W + B/n) / W)`, without a floor at one. Result metadata reports original chain count, supplied length, and used half length.
This follows the [classical split R-hat definition](../REFERENCES.md#ref-13).

`SplitRhatError` distinguishes insufficient chains, short chains, unequal lengths, nonfinite values (including omitted middle draws), any constant half, and
numerically unresolved variance or result. Rejecting even one constant half is a conservative stuck-chain policy. Common scaling avoids overflow and local
centering preserves small within-half changes; enormous scale disparities can still underflow variance. `UnresolvedVariance` identifies the original chain
and half whose varying samples lost resolvable variance. `NumericalFailure` is reserved for an unrepresentable final estimate after all half variances pass.

This is a raw-moment diagnostic with finite marginal mean/variance assumptions. Splitting helps expose within-chain drift, but classical R-hat can miss scale
differences and heavy-tail problems. It does not implement the rank-normalized and folded improvements described by
[Vehtari et al.](../REFERENCES.md#ref-14). A value near one does not establish convergence or exploration of all modes; combine it with ESS, trace
inspection, dispersed starts, and longer runs. Do not apply modern rank-normalized thresholds as a guarantee for this estimator.

### ESS and wall-clock efficiency

`IntegratedAutocorrelationTime::effective_sample_size()` estimates the effective sample size for one scalar mean as `N / tau`, using
[Geyer's integrated-time estimate](../REFERENCES.md#ref-8) and the [integrated-time/ESS relationship](../REFERENCES.md#ref-13).
It uses the same retained sample count and recorded-sample time units as the ACF. The stationarity, reversibility, finite-variance, and summable-correlation
assumptions [above](#autocorrelation-estimator-contract) apply. Positive estimates with `tau < 1` give ESS greater than `N`; there is no cap.
This is not rank-normalized bulk ESS, tail ESS, or a pooled multi-chain estimator. Short, constant, nonfinite, nonpositive-time, and missing-truncation
failures remain the typed ACF/time errors; no ESS is produced for those inputs. In particular, at least four samples and a complete nonpositive pair are
necessary for this initial-sequence estimate, but never sufficient to trust it.

`effective_sample_size_per_second(elapsed: Duration)` divides by measured wall seconds and rejects zero duration with `EssRateError`. Missing timing should
remain unavailable; sample indices are not a clock. Measure the exact production workload supplying the analyzed draws, including intervening transitions
when thinning. Document whether warmup, observation, I/O, and analysis are included. Never reuse a full-run duration for a selected subset of draws.
Per-chain rates do not automatically describe aggregate parallel throughput, which requires the wall time of the complete concurrent workload.

### Online statistics

`OnlineStats` uses [Welford's recurrence](../REFERENCES.md#ref-6) to accumulate the mean and centered sum of squares in constant memory. For a new sample `x`
and updated count `n`, the update is:

```text
delta = x - mean_old
mean = mean_old + delta / n
M2 = M2_old + delta * (x - mean)
```

The implementation uses a fused multiply-add for the `M2` update. It avoids subtracting two separately accumulated large sums to obtain a variance, but
finite inputs can still overflow intermediate arithmetic. A failed update leaves the accumulator unchanged; batch accumulation retains earlier successful
samples. These are numerical and mutation policies of this crate, rather than guarantees that the statistical estimator is appropriate for a given chain.

Population variance is `M2 / n` for at least one sample, and sample variance is `M2 / (n - 1)` for at least two. The latter is unbiased for independent,
identically distributed samples with finite variance; this statement does not extend to arbitrary correlated MCMC output. The standard error
`sqrt(sample_variance / n)` ignores autocorrelation. Use [binning](#binning-analysis) or [autocorrelation analysis](#autocorrelation-estimator-contract) to
assess correlated uncertainty. Neither accumulation nor taking a square root adds a convergence guarantee.

### Proposal validation

Detailed-balance helpers empirically compare forward and reverse transition flows for representative discrete transitions under the
[Metropolis-Hastings contract](#metropolis-hastings-contract). These selected comparisons are useful for new proposal kernels, but they do not prove
reversibility over the entire state space or establish irreducibility, aperiodicity, or adequate mixing.

For in-place proposals, every concrete hypothetical proposal is undone before the next trial, and no-proposal telemetry is consumed. The proposal must
represent a fixed kernel: freeze online adaptation before validation, and keep transition-relevant mutable state inside the `undo` contract.

`verify_proposal_density` compares a supplied pair's reported Hastings ratio with independently supplied forward and reverse log densities.
`verify_proposal_bins` compares independent proposal histograms with reference bin probabilities, using
[Hoeffding's independent-sum inequality](../REFERENCES.md#ref-11) and a union bound. Bin checks require independent proposal draws from a fixed endpoint,
exhaustive bins, and reference probabilities derived independently of the generator. They are not calibrated for correlated chain output.
Neither continuous helper is a continuous detailed-balance or convergence test. The
[continuous-proposal workflow](VALIDATING_PROPOSALS.md#continuous-proposals) owns the tolerance formula, error budget, and limits of coarsened flow comparisons.

## Export and notebook workflow

The [chain-analysis guide](ANALYZING_CHAINS.md#export-and-notebook-workflow) owns the runnable Ising workflow, trace-column selection, JSON availability
rules, timing scope, and notebook exports. Those example outputs illustrate the diagnostics; their timings are not benchmark evidence.

## User Responsibilities

Domain code should still validate:

- Aperiodicity and proposal irreducibility, or known connected components for the intended state space
- Burn-in, autocorrelation, effective sample size, and convergence behavior
- Correct proposal ratios for asymmetric moves
- Reproducible random-number seeding and independent streams for parallel chains
- Scientific interpretation of observables and uncertainty estimates
- State invariants before and after domain-specific moves

For constrained triangulations, graphs, or other combinatorial systems, the strongest checks usually combine domain-specific invariant tests with this crate's
transition-level diagnostics.

## References

See the [method-to-source index](../REFERENCES.md#method-to-source-index) for algorithm provenance, implementation context, and general background.
The [reviewer guide](reviewer_guide.md) maps the scientific claims to implementation and validation evidence.
