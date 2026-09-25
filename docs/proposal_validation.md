# Proposal Validation Guide

This guide summarizes practical checks for proposal kernels built on `Proposal`, `ProposalMut`, and `DelayedProposal`.

## Why Proposal Validation Matters

Metropolis-Hastings correctness depends on the pair formed by a target distribution and a proposal kernel. The library can apply the acceptance rule, reject
invalid floating-point values, and roll back failed moves, but user code still owns the scientific meaning of each proposal.

For every proposal family, validate that:

- Proposed states preserve domain invariants.
- Invalid moves are reported without corrupting state.
- Proposal ratios describe the same concrete move that was proposed.
- Reverse moves are possible wherever the target chain requires them.
- Representative forward/reverse transitions satisfy detailed balance after the Metropolis-Hastings correction.

## Interpreting Empirical Checks

`DetailedBalanceConfig::new(samples, tolerance, min_hits)` controls a statistical diagnostic, not a proof. The helper draws `samples` proposals from each
endpoint independently. For a selected transition, it counts exact endpoint hits (or matching delayed plans), weights those hits by their
Metropolis-Hastings acceptance probabilities, and estimates the accepted transition probability in each direction. It then compares the log flows

```text
log pi(current) + log P(current -> proposed)
log pi(proposed) + log P(proposed -> current)
```

through `log_balance_residual`, their difference. When both flows are impossible, the residual is defined as zero. Choose representative transitions whose
forward and reverse proposal probabilities are large enough to observe reliably; rare transitions need more samples or a domain-specific analytic check.

The three configuration values have distinct roles:

- `samples` is the number of proposal draws in each direction. More samples reduce Monte Carlo noise but do not expand transition coverage.
- `min_hits` is a data-adequacy guard. If either direction produces fewer matching proposals, the helper returns `InsufficientHits` instead of interpreting
  an unstable estimate.
- `tolerance` is the sole pass/fail threshold: the check succeeds only when `abs(log_balance_residual) <= tolerance`. Set it before looking at the result,
  based on the sampling budget and the scientific sensitivity of the test.

`log_balance_standard_error` is an approximate Monte Carlo standard error obtained from the observed acceptance-weight variance, and `z_score()` reports
the residual divided by that estimate when it is finite and positive. Both are diagnostic context; neither changes the configured tolerance decision.

Use caller-owned, explicitly seeded RNGs so failures reproduce. For stochastic kernels, repeat representative checks with several fixed seeds or raise the
sample budget to confirm the conclusion is stable rather than tuning the tolerance to one draw. A passing local check does not establish ergodicity,
convergence, mixing quality, or correctness for transitions that were not tested.

## By-Value Proposals

Use `Proposal<S>` when proposed states are cheap to create by value. This is the simplest path for small numeric states or small discrete systems.

Useful checks:

- Unit-test deterministic edge cases.
- Property-test invariants of proposed states.
- Use `verify_detailed_balance` for representative discrete transitions.
- Use `verify_detailed_balance_many` for a small grid or graph of transitions.

For continuous proposals, exact endpoint hits are usually too rare for the current detailed-balance helper. See [Continuous Proposals](#continuous-proposals).

## In-Place Proposals

Use `ProposalMut<S>` when cloning the full state is expensive. The proposal mutates state and returns an undo token that must restore the exact previous state
and any proposal-internal state that can affect later transitions on rejection.

Treat `Info`, `info`, and `no_proposal_info` as telemetry only. They must not change future transition behavior: bulk `Sampler::run_mut*` methods skip these
hooks when they do not return `Step` values. Drive `Sampler::step_mut` explicitly when every transition needs metadata.

Useful checks:

- Verify that `propose_mut` returning `None` instead of `Option<Undo>::Some` leaves the state unchanged.
- Verify `undo` restores the exact previous state for every successful proposal.
- Verify rejection also restores proposal-internal transition state.
- Test invalid log-probability and invalid log-ratio paths.
- Use `verify_detailed_balance_mut` on small representative states that implement `Clone + PartialEq`.
- Use `verify_detailed_balance_mut_many` for batches of local moves.

The detailed-balance helper clones endpoints so it can resample transitions from a clean state. After every concrete hypothetical proposal it calls `undo`,
including proposals that do not match the requested destination and proposals whose density is invalid. It also consumes no-proposal telemetry. Each trial
therefore begins from a clean endpoint and rollback-governed proposal transition state. The cloning is intentional test overhead, not a production sampling
requirement.

These diagnostics assume a fixed proposal kernel. Freeze online adaptation before calling them; proposal state outside the `undo` contract makes repeated
transition estimates nonstationary and invalidates the detailed-balance check.

## Delayed Proposals

Use `DelayedProposal<S>` when a concrete move can be planned and scored before mutating state. This is useful for combinatorial systems where rejected moves
should avoid mutation entirely.

Useful checks:

- Verify each plan identifies a concrete transition, not only a move class.
- Test planning failure, proposed-log-probability failure, and log-q-ratio failure paths.
- Verify `commit` is failure-atomic when it can fail.
- Use `verify_detailed_balance_delayed` for a specific planned transition.
- Use `verify_detailed_balance_delayed_many` for batches.

Delayed detailed-balance checks use plan predicates because plans are the transition descriptors. The helper does not assume endpoint equality is enough to
identify a move.

For local combinatorial kernels, include valid-site multiplicities in `DelayedProposal::log_q_ratio` whenever they affect concrete transition probabilities.
For a move that chooses a move kind and then chooses uniformly among valid concrete sites, the ratio for a successful transition is:

```text
log_q_ratio = log q(current | proposed) - log q(proposed | current)
            = log reverse_move_weight - log reverse_weight_sum
            - log forward_move_weight + log forward_weight_sum
            + log valid_forward_sites - log valid_reverse_sites
```

The weight-sum terms cancel only when the total family weight is the same at both endpoints. Add any non-uniform concrete-site selection terms as well. A
bounded search that returns `Ok(None)` after failing to find a valid site is an ordinary self-loop/rejection, but it does not repair an omitted Hastings
correction for successful moves.

Use `DiscreteProposalRatio::from_counts(forward_sites, reverse_sites)?` for the common equal-normalized-family-probability case, or
`DiscreteProposalRatio::from_endpoints(forward, reverse)` with named `DiscreteProposalEndpoint` values when inverse move families have different selection
weights or endpoint totals. Construction rejects invalid successful-forward inputs immediately. A zero reverse-site count is valid and computes to `-inf`,
while a successful plan with zero forward sites is reported as an invalid proposal-ratio input.

## Continuous Proposals

The `verify_detailed_balance*` helpers are designed for discrete, quantized, or exactly comparable transitions. Continuous proposals almost never resample the
exact same endpoint. Use the separate `verify_proposal_density` and `verify_proposal_bins` APIs at the crate root or in `prelude::testing` instead.

| Question | Diagnostic | Evidence required |
| --- | --- | --- |
| Does a discrete transition balance after MH correction? | `verify_detailed_balance*` | Enough exact forward/reverse hits |
| Does a continuous move report the correct Hastings ratio? | `verify_proposal_density` | Independently derived log densities for the same concrete move |
| Does the generator put the right mass in chosen regions? | `verify_proposal_bins` | Independent draws and independently derived bin probabilities |

### Check the density ratio

For a selected pair `x -> y`, call:

```rust
use markov_chain_monte_carlo::{ProposalDensityError, verify_proposal_density};

fn main() -> Result<(), ProposalDensityError> {
    // Independence proposal q(y | x) = 2*y on (0, 1).
    let (x, y) = (0.25_f64, 0.75_f64);
    let reported_log_ratio = x.ln() - y.ln(); // Use your proposal's actual method here.
    let report = verify_proposal_density(
        (2.0 * y).ln(), // log q(y | x)
        (2.0 * x).ln(), // log q(x | y)
        reported_log_ratio,
        1e-12,
    )?;
    assert!(report.residual().abs() <= 1e-12);
    Ok(())
}
```

The expected ratio is `reverse_log_density - forward_log_density`. The residual is reported minus expected, with an absolute tolerance in natural-log
units. Densities must share a reference measure and include Jacobians and state-dependent normalizers. Positive log densities are valid. A successful
forward move requires finite forward log density; a zero reverse density is `-inf` and requires a reported ratio of `-inf`. Matching negative infinities
have residual zero; a support mismatch always fails. Invalid values return typed errors. `ExpectedLogRatioOverflow` and `ResidualOverflow` distinguish the
two finite subtractions and retain the offending operands; neither produces a partial report. Both successful reports and reports carried by
`ProposalDensityError::Violation` retain the original threshold as `report.tolerance()`, so a saved report includes its decision context.

Use `Proposal::log_q_ratio(&x, &y)`, `ProposalMut::log_q_ratio(&proposed_state, &undo_token)`, or
`DelayedProposal::log_q_ratio(&current_state, &plan)?` as appropriate. Evaluate the same concrete move described by the densities, including auxiliary
variables when the proposal's correction is defined on an augmented space. Capture an in-place diagnostic result, undo the move, and only then propagate
an error. For delayed proposals, check the planned move without committing. The diagnostic itself performs no mutation or rollback.

Test both orientations, support boundaries, and representative scales. Derive the density oracle independently: computing it from the production ratio
only checks the formula against itself. This test does not establish normalization or that the generator samples the stated density.

### Check generated bin masses

Choose disjoint bins covering all outcomes before sampling. Draw repeatedly from a fixed endpoint with frozen proposal parameters and count generated
states or deltas. Include tail, invalid-value, and no-proposal outcomes in explicit bins. Never drop observations outside a convenient range. For an
in-place kernel, restore both endpoint and proposal transition state after each draw; for delayed kernels, classify plans without committing.

```rust
use markov_chain_monte_carlo::{ProposalBinsError, verify_proposal_bins};
use rand::{RngExt, SeedableRng, distr::Open01, rngs::StdRng};

fn main() -> Result<(), ProposalBinsError> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut counts = [0; 3];
    for _ in 0..20_000 {
        // Replace with your proposal call from the same fixed endpoint each time.
        let y = rng.sample::<f64, _>(Open01).sqrt();
        let bin = if !(y > 0.0 && y < 1.0) { 2 } else { usize::from(y >= 0.5) };
        counts[bin] += 1;
    }
    // CDF(y)=y^2 gives masses 1/4, 3/4, and zero outside support.
    let report = verify_proposal_bins(&counts, &[0.25, 0.75, 0.0], 1e-6)?;
    assert!(report.max_residual() <= report.tolerance());
    Ok(())
}
```

`Open01` excludes zero before taking the square root, preserving this proposal's open support and finite log ratios. An ordinary `random::<f64>()` draw
includes zero and can produce an invalid positive-infinite ratio for a move from a positive current state.

With `n` independent draws, `k` bins, and preselected error budget `alpha`, the simultaneous absolute probability tolerance is
`sqrt(log(2*k/alpha) / (2*n))`. Each bin indicator is bounded in `[0, 1]`; applying Hoeffding's independent-sum inequality and a union bound yields
`P(any bin exceeds tolerance) <= alpha` under the supplied probabilities, apart from floating-point arithmetic. This is a conservative finite-sample
bound, not a p-value. The supplied probabilities must sum to one within a `1e-12` roundoff allowance and are not renormalized. Derive them independently
from a CDF, analytical integration, or a separately justified reference; the bound does not include uncertainty in estimated reference probabilities.

The report includes the sample count, bin count, largest absolute residual, first worst bin, tolerance, and error budget. A tolerance of at least one
returns `UninformativeBound`; increase the sample budget. Any observation in a zero-probability bin fails immediately. Rare positive-probability bins
may pass with no hits, and discrepancies within a bin are invisible. Choose bins and sample sizes for the errors you need to detect; passing a coarse
partition is weak evidence about fine-scale behavior.

Count errors identify different corrective actions: `NoSamples` requires collecting draws, `SampleCountOverflow` identifies the bin and operands whose
addition overflowed `usize`, and `SampleCountTooLarge` reports a representable total above the exact-conversion limit of `2^53`. Shape, rate, and probability
validation happen first; integer overflow is checked before the precision limit. Fix invalid inputs before interpreting support or statistical failures.

Fix sample sizes and bins in advance. Correlated chain output, adaptation, data-dependent bin selection, or repeatedly sampling until a check passes
invalidates the stated error budget. For several endpoints, partitions, or seeds, allocate per-call budgets whose sum is at most the desired total
false-positive rate. Use caller-owned seeded RNGs so failures reproduce within the repository's reproducibility boundary.

### Limits of coarsened flow checks

These APIs implement complementary density and generator checks, not an empirical continuous detailed-balance test. A balance comparison between regions
`A` and `B` would need the integrated stationary flows `integral_A pi(dx) P(x, B)` and `integral_B pi(dy) P(y, A)`, including target mass and justified
within-region sampling. Substituting a bin predicate into an exact-hit test at one representative endpoint does not estimate those flows. That requires
a separate integration design. Neither these diagnostics nor exact-hit checks prove ergodicity, mixing, or convergence.

## Suggested Test Stack

For a new scientific proposal, combine:

- Ordinary unit tests for hand-picked transitions
- Property tests for state invariants and rollback behavior
- Detailed-balance checks over representative discrete transitions
- Independent density-ratio and sampled-bin checks for continuous proposals
- Regression tests for known asymmetric move ratios
- Long-run observable checks with `OnlineStats` or `BinningAnalysis`

No single test proves scientific validity. The goal is to make local proposal mistakes loud before they contaminate long simulations.

## References

The proposal-ratio and acceptance conventions in this guide follow Metropolis et al. (1953) and Hastings (1970); the bin-mass bound uses Hoeffding (1963).
See the canonical entries in
[`REFERENCES.md`](../REFERENCES.md#background-references).
