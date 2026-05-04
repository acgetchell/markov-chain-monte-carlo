# Proposal Validation Guide

This guide summarizes practical checks for proposal kernels built on `Proposal`, `ProposalMut`, and `DelayedProposal`.

## Why Proposal Validation Matters

Metropolis-Hastings correctness depends on the pair formed by a target distribution and a proposal kernel. The library can apply the acceptance rule, reject invalid floating-point values, and roll back failed moves, but user code still owns the scientific meaning of each proposal.

For every proposal family, validate that:

- Proposed states preserve domain invariants.
- Invalid moves are reported without corrupting state.
- Proposal ratios describe the same concrete move that was proposed.
- Reverse moves are possible wherever the target chain requires them.
- Representative forward/reverse transitions satisfy detailed balance after the Metropolis-Hastings correction.

## By-Value Proposals

Use `Proposal<S>` when proposed states are cheap to create by value. This is the simplest path for small numeric states or small discrete systems.

Useful checks:

- Unit-test deterministic edge cases.
- Property-test invariants of proposed states.
- Use `verify_detailed_balance` for representative discrete transitions.
- Use `verify_detailed_balance_many` for a small grid or graph of transitions.

For continuous proposals, exact endpoint hits are usually too rare for the current detailed-balance helper. See [Continuous Proposals](#continuous-proposals).

## In-Place Proposals

Use `ProposalMut<S>` when cloning the full state is expensive. The proposal mutates state and returns an undo token that must restore the exact previous state on rejection.

Useful checks:

- Verify `propose_mut(None)` paths leave the state unchanged.
- Verify `undo` restores the exact previous state for every successful proposal.
- Test invalid log-probability and invalid log-ratio paths.
- Use `verify_detailed_balance_mut` on small representative states that implement `Clone + PartialEq`.
- Use `verify_detailed_balance_mut_many` for batches of local moves.

The detailed-balance helper clones endpoints so it can resample transitions from a clean state. That cloning is intentional test overhead, not a production sampling requirement.

## Delayed Proposals

Use `DelayedProposal<S>` when a concrete move can be planned and scored before mutating state. This is useful for combinatorial systems where rejected moves should avoid mutation entirely.

Useful checks:

- Verify each plan identifies a concrete transition, not only a move class.
- Test planning failure, proposed-log-probability failure, and log-q-ratio failure paths.
- Verify `commit` is failure-atomic when it can fail.
- Use `verify_detailed_balance_delayed` for a specific planned transition.
- Use `verify_detailed_balance_delayed_many` for batches.

Delayed detailed-balance checks use plan predicates because plans are the transition descriptors. The helper does not assume endpoint equality is enough to identify a move.

## Continuous Proposals

The current detailed-balance helpers are designed for discrete, quantized, or exactly comparable transitions. Continuous proposals almost never resample the exact same endpoint, so exact-hit diagnostics become uninformative.

Reasonable follow-up designs include:

- Bin or coarsen continuous proposals before checking transition flow.
- Compare forward/reverse proposal densities for explicitly supplied pairs.
- Add kernel-specific diagnostics that sample paired local neighborhoods.
- Combine analytic proposal-ratio tests with distributional tests over generated deltas.

Continuous-proposal diagnostics should be developed as a separate API so the existing exact-hit helpers stay honest about their assumptions. That follow-up is tracked in [#42](https://github.com/acgetchell/markov-chain-monte-carlo/issues/42).

## Suggested Test Stack

For a new scientific proposal, combine:

- Ordinary unit tests for hand-picked transitions
- Property tests for state invariants and rollback behavior
- Detailed-balance checks over representative discrete transitions
- Regression tests for known asymmetric move ratios
- Long-run observable checks with `OnlineStats` or `BinningAnalysis`

No single test proves scientific validity. The goal is to make local proposal mistakes loud before they contaminate long simulations.
