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
is written in energy form. This crate currently provides sampler mechanics and target composition, not training for learned energies or adaptive proposal
policies.

## What the Crate Checks

The library enforces several local invariants:

- Acceptance decisions are computed in log space.
- Log-space acceptance avoids underflow in tail probabilities.
- `NaN` and positive-infinite target log-probabilities or proposal ratios are rejected.
- In-place proposals roll back on rejection or invalid proposed values.
- Delayed proposals separate planning, scoring, acceptance, and commit so mutations happen only after acceptance.
- Sampling counters and cached log-probabilities stay synchronized through library-owned transitions.

These checks protect the mechanics of a single transition. They do not prove that a user-defined proposal explores the full intended state space.

## Diagnostics

The crate includes diagnostics that help users test assumptions:

- Observables measure derived quantities during sampling.
- `OnlineStats` provides one-pass summary statistics.
- `BinningAnalysis` estimates uncertainty for correlated samples.
- Thinning helpers collect every k-th state or observation while still advancing the chain on every step.
- Detailed-balance helpers empirically compare forward and reverse transition flows for representative discrete transitions.

Detailed-balance checks are especially useful for new proposal kernels, but they remain empirical tests over selected transitions. Passing them does not
establish irreducibility, aperiodicity, or adequate mixing.

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
