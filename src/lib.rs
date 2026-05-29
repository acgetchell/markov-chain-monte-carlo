#![cfg_attr(any(doc, doctest), doc = include_str!("../README.md"))]

//! ---
//! # Documentation map
//!
//! The README above is included verbatim and serves as the user-facing
//! introduction to the crate: overview, feature list, installation, quick-start
//! example, API selection, examples, project links, citation, and contribution
//! pointers.
//!
//! Everything below this line specifies the semantic and API contract of the
//! `markov-chain-monte-carlo` crate. It is intended for users who need deeper
//! detail about Metropolis-Hastings correctness, numerical behavior, proposal
//! workflows, checkpoints, observables, and streaming statistics.
//!
//! This crate's documentation is intentionally layered by audience and intent:
//!
//! - **README.md** (included above):
//!   user-facing overview, feature list, and quick-start examples.
//! - **Crate-level documentation (`lib.rs`)** (this document):
//!   the programming contract of the sampler APIs, including acceptance
//!   semantics, proposal responsibilities, checkpoint restore behavior, and
//!   measurement utilities.
//! - **`docs/scientific_basis.md`**:
//!   deeper discussion of the Metropolis-Hastings contract and scope.
//! - **`docs/proposal_validation.md`**:
//!   proposal-author testing patterns and `verify_detailed_balance*` usage.
//!
//! # API contract
//!
//! [`Target::log_prob`] should return an unnormalized natural log-probability
//! or log-density.  Additive constants are fine because Metropolis-Hastings
//! only uses differences, but arbitrary scores or logits will sample a
//! different distribution.
//!
//! # Scientific basis and scope
//!
//! This crate implements Metropolis-Hastings sampling for user-defined state
//! spaces.  The transition rule uses target log-probability differences and
//! proposal probability ratios:
//!
//! ```text
//! alpha(x, y) = min(1, exp(log pi(y) - log pi(x) + log q(x | y) - log q(y | x)))
//! ```
//!
//! The library is built around the standard MCMC contract:
//!
//! - `Target<S>` returns an unnormalized natural log probability, log
//!   density, or negative action.
//! - Proposal implementations must describe the same concrete transition in
//!   both the generated move and `log_q_ratio`.
//! - Detailed balance, or a valid Metropolis-Hastings correction, is a
//!   property of the user-provided target+proposal pair.
//! - Irreducibility, aperiodicity, burn-in, autocorrelation, and convergence
//!   are domain-specific analysis questions.
//!
//! What the crate provides:
//!
//! - Log-space acceptance calculations to avoid underflow in tail
//!   probabilities.
//! - Explicit rejection of `NaN` and positive-infinite log probabilities or
//!   proposal ratios.
//! - Rollback-safe in-place proposals for large states where cloning is
//!   expensive.
//! - Delayed-commit proposals for workflows that need to score a concrete
//!   move before mutating state.
//! - Empirical detailed-balance checks for representative discrete
//!   transitions.
//! - Streaming statistics and binning analysis for correlated-sample
//!   uncertainty estimates.
//!
//! What the crate does not prove:
//!
//! - That a proposal is ergodic on a domain-specific state space.
//! - That a chain has mixed enough for a given scientific observable.
//! - That a triangulation, graph, or other combinatorial state satisfies
//!   external validity constraints.
//! - That a chosen model is scientifically appropriate for a downstream
//!   study.
//!
//! For a fuller discussion, see
//! [docs/scientific_basis.md](https://github.com/acgetchell/markov-chain-monte-carlo/blob/main/docs/scientific_basis.md).
//!
//! # Additive target terms
//!
//! Bias potentials, energy-based model terms, learned regularizers,
//! auxiliary actions, umbrella-sampling weights, and other target modifiers
//! should be sampled as part of the target distribution, not applied as ad hoc
//! rejection filters after a proposal has been generated.  Use
//! [`AdditiveTarget`] when model and bias terms are easiest to express as
//! separate log-weight components.
//!
//! If a downstream model is written in action form, implement each component
//! with the same sign convention: return `-S_component(state)`.  Then the
//! acceptance calculation uses the combined action delta naturally:
//!
//! ```text
//! log pi(y) - log pi(x) = -(Delta S_model + Delta S_bias)
//! ```
//!
//! The Hastings correction remains independent and is still supplied through
//! [`Proposal::log_q_ratio`], [`ProposalMut::log_q_ratio`], or
//! [`DelayedProposal::log_q_ratio`]:
//!
//! ```text
//! log_alpha = -(Delta S_model + Delta S_bias) + log q(x | y) - log q(y | x)
//! ```
//!
//! # Numerical semantics
//!
//! The core Metropolis-Hastings acceptance calculation is performed in log
//! space using `f64`.  Domain-specific code may use exact arithmetic internally
//! for predicates or invariant checks, but targets and proposal ratios cross the
//! crate boundary as log weights:
//!
//! - finite values represent unnormalized log probability mass/density
//! - `f64::NEG_INFINITY` represents an impossible or zero-probability state
//! - `NaN` log-probabilities and log proposal ratios are rejected with
//!   [`McmcError`]
//! - `+∞` log-probabilities and log proposal ratios are rejected with
//!   [`McmcError`]
//! - acceptance ratios that become `NaN` during arithmetic, such as
//!   `-∞ - (-∞)`, are treated as rejection
//!
//! # Long runs and parallelism
//!
//! `Chain`, `Sampler`, proposal values, and RNGs are ordinary per-instance
//! values; the crate does not use global mutable state.  Run independent chains
//! in parallel by giving each worker its own chain, proposal state, and RNG
//! stream.  This keeps reproducibility and RNG stream splitting under caller
//! control.
//!
//! Bulk observing methods return a [`SampleBuffer`], which stores one output
//! per step.  For production runs with many samples, use compact observables or
//! single-step observing loops when retaining every measurement is unnecessary.
//! [`OnlineStats`] and [`BinningAnalysis`] provide constant-memory statistics
//! for those streaming measurement loops.  Samplers also provide
//! `*_with_thinning` variants to collect cloned states or measurements only
//! every k-th completed step while still advancing the chain on every step.
//! For workflows that choose the next step budget from the updated state, use
//! [`Sampler::run_chunk`], [`Sampler::run_mut_chunk`], or
//! [`Sampler::run_delayed_chunk`].  They run the next chunk on the same RNG
//! stream and return a checkpoint-compatible view containing the current state
//! and counters.
//!
//! # Resumable chunked runs
//!
//! Chunked runs advance the chain by a chosen number of steps, then return a
//! checkpoint-compatible continuation so a caller can inspect the updated state,
//! choose the next chunk length, and resume without losing RNG state or
//! counters.  Reusing the same [`Sampler`] preserves the RNG stream, so a
//! sequence of chunks reproduces an equivalent one-shot run with the same seed.
//!
//! Measurements stay on the caller side: keep domain-specific statistics in your
//! own buffers and accumulate them across chunks rather than having the sampler
//! own a measurement buffer.  The delayed observing variant
//! [`Sampler::run_delayed_chunk_observing`] hands each step's [`DelayedStep`]
//! telemetry and post-step state to a callback while still returning the
//! continuation, so the chain keeps ownership of the accept/reject draw and
//! counters.  Between chunks a caller can size the next chunk from the current
//! state and stop on an elapsed-time budget.
//!
//! ```
//! use core::convert::Infallible;
//! use markov_chain_monte_carlo::prelude::delayed::*;
//! use rand::{Rng, SeedableRng, rngs::StdRng};
//!
//! # struct Flat;
//! # impl Target<i32> for Flat {
//! #     fn log_prob(&self, _: &i32) -> f64 { 0.0 }
//! # }
//! # struct Advance;
//! # impl DelayedProposal<i32> for Advance {
//! #     type Plan = i32;
//! #     type Info = i32;
//! #     type Error = Infallible;
//! #     fn propose_plan<R: Rng + ?Sized>(&mut self, _: &i32, _: &mut R) -> Result<Option<i32>, Self::Error> { Ok(Some(1)) }
//! #     fn proposed_log_prob<T: Target<i32>>(&self, s: &i32, p: &i32, t: &T) -> Result<f64, Self::Error> { Ok(t.log_prob(&(*s + *p))) }
//! #     fn info(&self, plan: &i32) -> i32 { *plan }
//! #     fn commit<R: Rng + ?Sized>(&mut self, s: &mut i32, p: i32, _: &mut R) -> Result<(), Self::Error> { *s += p; Ok(()) }
//! # }
//! let mut rng = StdRng::seed_from_u64(42);
//! let mut proposal = Advance;
//! let chain = Chain::new(0, &Flat).map_err(DelayedStepError::Mcmc)?;
//! let mut sampler = Sampler::new(chain, &Flat, &mut proposal, &mut rng)
//!     .map_err(DelayedStepError::Mcmc)?;
//!
//! // Domain-specific measurements stay outside the generic sampler.
//! let mut samples: Vec<i32> = Vec::new();
//! let mut next_chunk = 4;
//!
//! for _ in 0..3 {
//!     let continuation = sampler.run_delayed_chunk_observing(next_chunk, |_step, state| {
//!         samples.push(*state);
//!     })?;
//!     // Size the next chunk from the updated state and resume on the same RNG
//!     // stream.  A real caller can also break here on an elapsed-time budget.
//!     next_chunk = usize::try_from(**continuation.state()).unwrap_or(1).max(1);
//! }
//!
//! assert_eq!(sampler.chain_ref().total_steps(), samples.len());
//! # Ok::<(), DelayedStepError<Infallible>>(())
//! ```
//!
//! # Proposal validation
//!
//! The [`verify_detailed_balance`] family of helpers gives proposal authors a
//! test-facing diagnostic for representative discrete transitions.  Use
//! [`verify_detailed_balance`] for by-value [`Proposal`] implementations,
//! [`verify_detailed_balance_mut`] for rollback-based [`ProposalMut`]
//! implementations, and [`verify_detailed_balance_delayed`] for
//! [`DelayedProposal`] plans.  The companion batch helpers collect all
//! per-transition failures in a [`DetailedBalanceBatchReport`], which is useful
//! when checking a small graph, move table, or list of local states.
//!
//! These helpers are empirical diagnostics for exact endpoint hits, not a proof
//! of ergodicity or convergence.  They are intended for tests, examples, and
//! proposal-development checks over discrete or otherwise exactly comparable
//! states.
//!
//! Enable the optional `serde` feature to serialize [`Chain<S>`] checkpoints
//! when `S` implements serde's traits.  Restore checkpoint data with
//! [`Chain::from_checkpoint`] so the cached log-probability is recomputed from
//! the target used for resumed sampling.  [`Sampler`] also derives
//! serialization when all stored handles support it, but targets, proposals,
//! and RNG streams are reconstructed by the caller for portable resumes.
//!
//! ```
//! # #[cfg(feature = "serde")] {
//! use approx::assert_relative_eq;
//! use markov_chain_monte_carlo::prelude::*;
//!
//! struct Normal;
//! impl Target<f64> for Normal {
//!     fn log_prob(&self, state: &f64) -> f64 { -0.5 * state * state }
//! }
//!
//! let chain = Chain::new(1.0, &Normal)
//!     .expect("normal target returns a finite log probability");
//! let checkpoint = chain.checkpoint();
//! let checkpoint = serde_json::to_string(&checkpoint)?;
//! let checkpoint: ChainCheckpoint<f64> = serde_json::from_str(&checkpoint)?;
//! let restored = Chain::from_checkpoint(checkpoint, &Normal)
//!     .expect("normal target returns a finite checkpoint log probability");
//! assert_relative_eq!(
//!     restored.log_prob(),
//!     Normal.log_prob(restored.state()),
//!     epsilon = 1e-12
//! );
//! # }
//! # Ok::<(), serde_json::Error>(())
//! ```
//!
//! # In-place mutation with rollback
//!
//! For state spaces where cloning is expensive, use [`ProposalMut`] with
//! [`Chain::step_mut`].  The proposal mutates the state in place and returns
//! a small undo token for rollback on rejection:
//!
//! ```
//! use markov_chain_monte_carlo::prelude::in_place::*;
//! use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};
//!
//! /// A lattice of spins (not Clone — only mutated in place).
//! struct SpinChain { spins: Vec<i8> }
//!
//! /// Energy = −Σ s_i · s_{i+1}  (1-D Ising, no field).
//! struct Ising;
//! impl Target<SpinChain> for Ising {
//!     fn log_prob(&self, state: &SpinChain) -> f64 {
//!         let s = &state.spins;
//!         let energy: f64 = s.windows(2)
//!             .map(|w| -f64::from(w[0]) * f64::from(w[1]))
//!             .sum();
//!         -energy  // log_prob = −E  (T = 1)
//!     }
//! }
//!
//! /// Flip one random spin; undo token is the site index.
//! struct SpinFlip;
//! impl ProposalMut<SpinChain> for SpinFlip {
//!     type Undo = usize;
//!     fn propose_mut<R: Rng + ?Sized>(&self, state: &mut SpinChain, rng: &mut R) -> Option<usize> {
//!         if state.spins.is_empty() { return None; }
//!         let idx = rng.random_range(0..state.spins.len());
//!         state.spins[idx] *= -1;
//!         Some(idx)
//!     }
//!     fn undo(&self, state: &mut SpinChain, idx: usize) {
//!         state.spins[idx] *= -1;  // flipping twice = identity
//!     }
//! }
//!
//! fn main() -> Result<(), McmcError> {
//!     let mut rng = StdRng::seed_from_u64(42);
//!     let state = SpinChain { spins: vec![1; 20] };
//!     let mut chain = Chain::new(state, &Ising)?;
//!
//!     for _ in 0..1000 {
//!         chain.step_mut(&Ising, &SpinFlip, &mut rng)?;
//!     }
//!
//!     assert!(chain.acceptance_rate() > 0.0);
//!     Ok(())
//! }
//! ```
//!
//! # Delayed commit proposals
//!
//! Use [`DelayedProposal`] with [`Chain::step_delayed`] when a proposal can
//! plan and score a move before mutating the state, then commit only after the
//! Metropolis-Hastings decision accepts it.
//!
//! The plan should describe a concrete transition, such as a move kind plus the
//! local site or handle needed to apply it.  If no valid site can be selected,
//! return `Ok(None)` from [`DelayedProposal::propose_plan`]; that is an ordinary
//! rejection, while [`DelayedProposal::commit`] errors are reserved for
//! exceptional failures applying an already accepted concrete move.
//!
//! Delayed steps return [`DelayedStep`] telemetry with a [`StepOutcome`].  Use
//! [`Step::rejection_reason`] when you only need rejected-step categories.
//! Implement
//! [`DelayedProposal::no_plan_info`] when a no-plan self-loop should still
//! report proposal-family metadata.
//!
//! Use [`DiscreteProposalRatio`] when a delayed combinatorial proposal chooses
//! a move family and then samples uniformly among that family's valid concrete
//! sites.
//!
//! ```
//! use core::convert::Infallible;
//! use markov_chain_monte_carlo::prelude::delayed::*;
//! use rand::{Rng, SeedableRng, rngs::StdRng};
//!
//! struct TargetLine;
//! impl Target<i32> for TargetLine {
//!     fn log_prob(&self, state: &i32) -> f64 {
//!         -f64::from(state.abs())
//!     }
//! }
//!
//! struct MoveRight;
//! impl DelayedProposal<i32> for MoveRight {
//!     type Plan = i32;
//!     type Info = i32;
//!     type Error = Infallible;
//!
//!     fn propose_plan<R: Rng + ?Sized>(
//!         &mut self,
//!         _state: &i32,
//!         _rng: &mut R,
//!     ) -> Result<Option<i32>, Self::Error> {
//!         Ok(Some(1))
//!     }
//!
//!     fn proposed_log_prob<T: Target<i32>>(
//!         &self,
//!         state: &i32,
//!         plan: &i32,
//!         target: &T,
//!     ) -> Result<f64, Self::Error> {
//!         Ok(target.log_prob(&(*state + *plan)))
//!     }
//!
//!     fn info(&self, plan: &i32) -> i32 {
//!         *plan
//!     }
//!
//!     fn commit<R: Rng + ?Sized>(
//!         &mut self,
//!         state: &mut i32,
//!         plan: i32,
//!         _rng: &mut R,
//!     ) -> Result<(), Self::Error> {
//!         *state += plan;
//!         Ok(())
//!     }
//! }
//!
//! fn main() -> Result<(), DelayedStepError<Infallible>> {
//!     let target = TargetLine;
//!     let mut proposal = MoveRight;
//!     let mut rng = StdRng::seed_from_u64(42);
//!     let mut chain = Chain::new(-1, &target).map_err(DelayedStepError::Mcmc)?;
//!
//!     let step = chain.step_delayed(&target, &mut proposal, &mut rng)?;
//!     assert_eq!(step.outcome, StepOutcome::Accepted);
//!     assert_eq!(*chain.state(), 0);
//!     Ok(())
//! }
//! ```
//!
//! # Ergonomic sampling with [`Sampler`]
//!
//! [`Sampler`] bundles a chain with its target, proposal, and RNG so you
//! don't have to pass them on every step:
//!
//! ```
//! use markov_chain_monte_carlo::prelude::by_value::*;
//! use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};
//!
//! # #[derive(Clone)] struct Scalar(f64);
//! # struct Normal;
//! # impl Target<Scalar> for Normal {
//! #     fn log_prob(&self, s: &Scalar) -> f64 { -0.5 * s.0 * s.0 }
//! # }
//! # struct Walk;
//! # impl Proposal<Scalar> for Walk {
//! #     fn propose<R: Rng + ?Sized>(&self, c: &Scalar, r: &mut R) -> Scalar {
//! #         Scalar(c.0 + r.random_range(-1.0..1.0))
//! #     }
//! # }
//! let mut rng = StdRng::seed_from_u64(42);
//! let chain = Chain::new(Scalar(0.0), &Normal)?;
//! let mut sampler = Sampler::new(chain, &Normal, &Walk, &mut rng)?;
//!
//! // Burn-in
//! sampler.run(1000)?;
//! sampler.reset_counters();
//!
//! // Production
//! sampler.run(10_000)?;
//! assert!(sampler.chain_ref().acceptance_rate() > 0.0);
//! # Ok::<(), McmcError>(())
//! ```
//!
//! # Observables and measurements
//!
//! Use [`Observable`] or a closure with [`Sampler::run_observing`] to compute
//! derived quantities during sampling without storing full state histories:
//!
//! ```
//! use markov_chain_monte_carlo::prelude::by_value::*;
//! use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};
//!
//! # #[derive(Clone)] struct Scalar(f64);
//! # struct Normal;
//! # impl Target<Scalar> for Normal {
//! #     fn log_prob(&self, s: &Scalar) -> f64 { -0.5 * s.0 * s.0 }
//! # }
//! # struct Walk;
//! # impl Proposal<Scalar> for Walk {
//! #     fn propose<R: Rng + ?Sized>(&self, c: &Scalar, r: &mut R) -> Scalar {
//! #         Scalar(c.0 + r.random_range(-1.0..1.0))
//! #     }
//! # }
//! let mut rng = StdRng::seed_from_u64(42);
//! let chain = Chain::new(Scalar(0.0), &Normal)?;
//! let mut sampler = Sampler::new(chain, &Normal, &Walk, &mut rng)?;
//! let mut energy = |state: &Scalar| 0.5 * state.0 * state.0;
//!
//! let samples: SampleBuffer<f64> = sampler.run_observing(256, &mut energy)?;
//! assert_eq!(samples.len(), 256);
//! # Ok::<(), McmcError>(())
//! ```
//!
//! # Streaming statistics
//!
//! Use [`OnlineStats`] for Welford mean and variance updates, and
//! [`BinningAnalysis`] for autocorrelation-aware standard-error estimates:
//!
//! ```
//! use markov_chain_monte_carlo::prelude::*;
//!
//! let mut energy = OnlineStats::new();
//! energy.try_extend([1.0, 2.0, 3.0, 4.0])?;
//!
//! assert_eq!(energy.mean(), Some(2.5));
//!
//! let mut bins = BinningAnalysis::new();
//! bins.try_extend([1.0, 2.0, 3.0, 4.0])?;
//! assert!(bins.standard_error().is_some());
//! # Ok::<(), StatisticsError>(())
//! ```
//!
//! `Sampler` can also stream observations directly into these accumulators:
//!
//! ```
//! use core::convert::Infallible;
//! use markov_chain_monte_carlo::prelude::by_value::*;
//! use rand::{Rng, SeedableRng, rngs::StdRng};
//!
//! # struct T;
//! # impl Target<f64> for T { fn log_prob(&self, x: &f64) -> f64 { -0.5 * x * x } }
//! # struct P;
//! # impl Proposal<f64> for P {
//! #     fn propose<R: Rng + ?Sized>(&self, current: &f64, _rng: &mut R) -> f64 {
//! #         current + 1.0
//! #     }
//! # }
//! let mut rng = StdRng::seed_from_u64(42);
//! let chain = Chain::new(0.0, &T).map_err(ObservedStreamError::Step)?;
//! let mut sampler = Sampler::new(chain, &T, &P, &mut rng).unwrap();
//! let mut coordinate = |state: &f64| *state;
//! let mut stats = OnlineStats::new();
//!
//! sampler.run_observing_into(4, &mut coordinate, &mut stats)?;
//! assert_eq!(stats.count(), 4);
//! # Ok::<(), ObservedStreamError<McmcError, Infallible, StatisticsError>>(())
//! ```

mod chain;
mod error;
mod observable;
mod sampler;
mod statistics;
mod testing;
mod traits;

pub use chain::{
    Chain, ChainCheckpoint, DelayedStep, DelayedStepError, Step, StepOutcome, StepRejectionReason,
};
pub use error::McmcError;
pub use observable::{
    Observable, ObservedStepError, ObservedStreamError, SampleBuffer, TryAccumulator, TryObservable,
};
pub use sampler::{
    ObservedDelayedIntoRunResult, ObservedDelayedStep, ObservedDelayedStepResult,
    ObservedIntoRunResult, Sampler, ThinnedObservedDelayedIntoRunResult,
    ThinnedObservedIntoRunResult, ThinnedRunResult, ThinningError, TryObservedDelayedIntoRunResult,
    TryObservedDelayedRunResult, TryObservedDelayedStepResult, TryObservedIntoRunResult,
    TryObservedMutStepResult, TryObservedRunResult, TryObservedStepResult,
    TryThinnedObservedDelayedIntoRunResult, TryThinnedObservedIntoRunResult,
    TryThinnedObservedRunResult,
};
pub use statistics::{BinningAnalysis, BinningEstimate, OnlineStats, StatisticsError};
pub use testing::{
    DetailedBalanceBatchReport, DetailedBalanceConfig, DetailedBalanceDelayedTransition,
    DetailedBalanceDirection, DetailedBalanceError, DetailedBalanceFailure, DetailedBalanceReport,
    DetailedBalanceState, verify_detailed_balance, verify_detailed_balance_delayed,
    verify_detailed_balance_delayed_many, verify_detailed_balance_many,
    verify_detailed_balance_mut, verify_detailed_balance_mut_many,
};
pub use traits::{
    AdditiveTarget, DelayedProposal, DiscreteProposalRatio, DiscreteProposalRatioError, Proposal,
    ProposalMut, Target,
};

/// Convenience re-exports for common usage.
///
/// The top-level prelude contains only the shared sampling foundation:
///
/// ```
/// use markov_chain_monte_carlo::prelude::*;
///
/// fn accepts_target<T: Target<f64>>(_: &T) {}
/// fn accepts_stats(_: OnlineStats, _: BinningAnalysis) {}
/// fn accepts_stream_result(_: ObservedIntoRunResult<McmcError, StatisticsError>) {}
/// ```
///
/// Workflow-specific preludes are available when tests, examples, or
/// benchmarks should import only one proposal API.  Modules that exercise
/// several workflows can import shared types from the top-level prelude and
/// individual proposal traits from workflow preludes:
///
/// ```
/// use markov_chain_monte_carlo::prelude::{Chain, Sampler, Target};
/// use markov_chain_monte_carlo::prelude::by_value::Proposal;
/// use markov_chain_monte_carlo::prelude::delayed as delayed_prelude;
/// use markov_chain_monte_carlo::prelude::in_place as in_place_prelude;
/// use markov_chain_monte_carlo::prelude::testing as testing_prelude;
///
/// fn needs_by_value<T: Target<f64>, P: Proposal<f64>>(_: &T, _: &P) {}
/// fn accepts_chain(_: &Chain<f64>) {}
/// fn accepts_sampler<T, P, R: ?Sized>(_: &Sampler<'_, f64, T, P, R>) {}
/// fn needs_in_place<
///     T: in_place_prelude::Target<f64>,
///     P: in_place_prelude::ProposalMut<f64>,
/// >(_: &T, _: &P) {}
/// fn needs_delayed<
///     T: delayed_prelude::Target<f64>,
///     P: delayed_prelude::DelayedProposal<f64>,
/// >(_: &T, _: &P) {}
/// fn needs_testing<T: testing_prelude::Target<f64>>(_: &T) {}
/// let _: Option<testing_prelude::DetailedBalanceConfig> = None;
/// ```
pub mod prelude {
    pub use crate::{
        AdditiveTarget, BinningAnalysis, BinningEstimate, Chain, ChainCheckpoint, McmcError,
        Observable, ObservedIntoRunResult, ObservedStepError, ObservedStreamError, OnlineStats,
        SampleBuffer, Sampler, StatisticsError, Target, ThinnedObservedIntoRunResult,
        ThinnedRunResult, ThinningError, TryAccumulator, TryObservable, TryObservedIntoRunResult,
        TryThinnedObservedIntoRunResult, TryThinnedObservedRunResult,
    };

    /// Prelude for by-value proposals.
    ///
    /// This imports the shared sampling types plus [`crate::Proposal`], without
    /// importing the in-place or delayed proposal traits.
    pub mod by_value {
        pub use crate::{
            AdditiveTarget, BinningAnalysis, BinningEstimate, Chain, ChainCheckpoint,
            DiscreteProposalRatio, DiscreteProposalRatioError, McmcError, Observable,
            ObservedIntoRunResult, ObservedStepError, ObservedStreamError, OnlineStats, Proposal,
            SampleBuffer, Sampler, StatisticsError, Target, ThinnedObservedIntoRunResult,
            ThinnedRunResult, ThinningError, TryAccumulator, TryObservable,
            TryObservedIntoRunResult, TryThinnedObservedIntoRunResult, TryThinnedObservedRunResult,
        };
    }

    /// Prelude for in-place proposals with rollback.
    ///
    /// This imports the shared sampling types plus [`crate::ProposalMut`], without
    /// importing the by-value or delayed proposal traits.
    pub mod in_place {
        pub use crate::{
            AdditiveTarget, BinningAnalysis, BinningEstimate, Chain, ChainCheckpoint,
            DiscreteProposalRatio, DiscreteProposalRatioError, McmcError, Observable,
            ObservedIntoRunResult, ObservedStepError, ObservedStreamError, OnlineStats,
            ProposalMut, SampleBuffer, Sampler, StatisticsError, Target,
            ThinnedObservedIntoRunResult, ThinnedRunResult, ThinningError, TryAccumulator,
            TryObservable, TryObservedIntoRunResult, TryThinnedObservedIntoRunResult,
            TryThinnedObservedRunResult,
        };
    }

    /// Prelude for delayed-commit proposals.
    ///
    /// This imports the shared sampling types plus delayed-step telemetry and
    /// errors, without importing the by-value or in-place proposal traits.
    pub mod delayed {
        pub use crate::{
            AdditiveTarget, BinningAnalysis, BinningEstimate, Chain, ChainCheckpoint,
            DelayedProposal, DelayedStep, DelayedStepError, DiscreteProposalRatio,
            DiscreteProposalRatioError, McmcError, Observable, ObservedDelayedIntoRunResult,
            ObservedDelayedStep, ObservedDelayedStepResult, ObservedStepError, ObservedStreamError,
            OnlineStats, SampleBuffer, Sampler, StatisticsError, StepOutcome, StepRejectionReason,
            Target, ThinnedObservedDelayedIntoRunResult, ThinnedRunResult, ThinningError,
            TryAccumulator, TryObservable, TryObservedDelayedIntoRunResult,
            TryThinnedObservedDelayedIntoRunResult, TryThinnedObservedRunResult,
        };
    }

    /// Prelude for proposal validation and detailed-balance diagnostics.
    ///
    /// This imports the target and proposal traits plus the public
    /// [`crate::verify_detailed_balance`] helpers, without importing sampler
    /// execution types.  Use this prelude in tests, examples, and benchmarks
    /// that validate proposal kernels with [`crate::DetailedBalanceConfig`] and
    /// inspect [`crate::DetailedBalanceReport`] values.
    pub mod testing {
        pub use crate::{
            AdditiveTarget, DelayedProposal, DetailedBalanceBatchReport, DetailedBalanceConfig,
            DetailedBalanceDelayedTransition, DetailedBalanceDirection, DetailedBalanceError,
            DetailedBalanceFailure, DetailedBalanceReport, DetailedBalanceState,
            DiscreteProposalRatio, DiscreteProposalRatioError, Proposal, ProposalMut, Target,
            verify_detailed_balance, verify_detailed_balance_delayed,
            verify_detailed_balance_delayed_many, verify_detailed_balance_many,
            verify_detailed_balance_mut, verify_detailed_balance_mut_many,
        };
    }
}

#[cfg(test)]
mod public_api_smoke_tests {
    use core::convert::Infallible;

    use rand::{Rng, rngs::StdRng};
    #[cfg(feature = "serde")]
    use serde_json::{Error as JsonError, json, to_value};

    use super::{
        AdditiveTarget, BinningAnalysis, BinningEstimate, Chain, ChainCheckpoint, DelayedStep,
        DetailedBalanceBatchReport, DetailedBalanceConfig, DetailedBalanceDelayedTransition,
        DetailedBalanceDirection, DetailedBalanceError, DetailedBalanceFailure,
        DetailedBalanceReport, DetailedBalanceState, McmcError, Observable, ObservedDelayedStep,
        OnlineStats, Proposal, ProposalMut, SampleBuffer, Sampler, StatisticsError, Step,
        StepOutcome, StepRejectionReason, Target, ThinningError,
        prelude::{self, by_value, delayed, in_place, testing},
    };

    #[cfg_attr(feature = "serde", derive(serde::Serialize))]
    struct Smoke;

    impl Target<f64> for Smoke {
        fn log_prob(&self, _: &f64) -> f64 {
            0.0
        }
    }

    impl Proposal<f64> for Smoke {
        fn propose<R: Rng + ?Sized>(&self, current: &f64, _: &mut R) -> f64 {
            *current
        }
    }

    impl ProposalMut<f64> for Smoke {
        type Undo = ();

        fn propose_mut<R: Rng + ?Sized>(&self, _: &mut f64, _: &mut R) -> Option<Self::Undo> {
            Some(())
        }

        fn undo(&self, _: &mut f64, (): Self::Undo) {}
    }

    impl delayed::DelayedProposal<f64> for Smoke {
        type Plan = ();
        type Info = ();
        type Error = Infallible;

        fn propose_plan<R: Rng + ?Sized>(
            &mut self,
            _: &f64,
            _: &mut R,
        ) -> Result<Option<Self::Plan>, Self::Error> {
            Ok(Some(()))
        }

        fn proposed_log_prob<T: delayed::Target<f64>>(
            &self,
            state: &f64,
            (): &Self::Plan,
            target: &T,
        ) -> Result<f64, Self::Error> {
            Ok(target.log_prob(state))
        }

        fn info(&self, (): &Self::Plan) -> Self::Info {}

        fn commit<R: Rng + ?Sized>(
            &mut self,
            _: &mut f64,
            (): Self::Plan,
            _: &mut R,
        ) -> Result<(), Self::Error> {
            Ok(())
        }
    }

    #[test]
    fn public_reexports_compile() {
        fn needs_target<T: prelude::Target<f64>>() {}
        fn needs_by_value<P: by_value::Proposal<f64>>() {}
        fn needs_in_place<P: in_place::ProposalMut<f64>>() {}
        fn needs_delayed<P: delayed::DelayedProposal<f64>>() {}
        fn needs_testing_target<T: testing::Target<f64>>() {}
        fn needs_testing_proposal<P: testing::Proposal<f64>>() {}
        fn needs_observable<O: Observable<f64, Output = f64>>(_: &mut O) {}

        needs_target::<Smoke>();
        needs_by_value::<Smoke>();
        needs_in_place::<Smoke>();
        needs_delayed::<Smoke>();
        needs_testing_target::<Smoke>();
        needs_testing_proposal::<Smoke>();

        let mut observable = |state: &f64| *state;
        needs_observable(&mut observable);

        let _: Option<AdditiveTarget<Smoke, Smoke>> = None;
        let _: Option<Chain<f64>> = None;
        let _: Option<ChainCheckpoint<f64>> = None;
        let _: Option<Step<()>> = None;
        let _: Option<DelayedStep<()>> = None;
        let _: Option<StepOutcome> = None;
        let _: Option<StepRejectionReason> = None;
        let _: Option<McmcError> = None;
        let _: Option<SampleBuffer<f64>> = None;
        let _: Option<Sampler<'_, f64, Smoke, Smoke, StdRng>> = None;
        let _: Option<ThinningError<McmcError>> = None;
        let _: Option<ObservedDelayedStep<(), f64>> = None;
        let _: Option<BinningAnalysis> = None;
        let _: Option<BinningEstimate> = None;
        let _: Option<OnlineStats> = None;
        let _: Option<StatisticsError> = None;
        let _: Option<DetailedBalanceConfig> = None;
        let _: Option<DetailedBalanceDirection> = None;
        let _: Option<DetailedBalanceError> = None;
        let _: Option<DetailedBalanceFailure> = None;
        let _: Option<DetailedBalanceDelayedTransition<'_, f64, ()>> = None;
        let _: Option<DetailedBalanceBatchReport> = None;
        let _: Option<DetailedBalanceReport> = None;
        let _: Option<DetailedBalanceState> = None;
        let _: Option<prelude::AdditiveTarget<Smoke, Smoke>> = None;
        let _: Option<prelude::ThinnedRunResult<(), McmcError>> = None;
        let _: Option<prelude::TryThinnedObservedRunResult<f64, McmcError, Infallible>> = None;
        let _: Option<by_value::AdditiveTarget<Smoke, Smoke>> = None;
        let _: Option<by_value::DiscreteProposalRatio> = None;
        let _: Option<by_value::DiscreteProposalRatioError> = None;
        let _: Option<by_value::ThinnedObservedIntoRunResult<McmcError, Infallible>> = None;
        let _: Option<in_place::AdditiveTarget<Smoke, Smoke>> = None;
        let _: Option<in_place::DiscreteProposalRatio> = None;
        let _: Option<in_place::DiscreteProposalRatioError> = None;
        let _: Option<
            in_place::TryThinnedObservedIntoRunResult<McmcError, Infallible, Infallible>,
        > = None;
        let _: Option<delayed::AdditiveTarget<Smoke, Smoke>> = None;
        let _: Option<delayed::DiscreteProposalRatio> = None;
        let _: Option<delayed::DiscreteProposalRatioError> = None;
        let _: Option<delayed::StepOutcome> = None;
        let _: Option<delayed::StepRejectionReason> = None;
        let _: Option<delayed::ThinnedObservedDelayedIntoRunResult<Infallible, Infallible>> = None;
        let _: Option<delayed::McmcError> = None;
        let _: Option<testing::DetailedBalanceConfig> = None;
        let _: Option<testing::DetailedBalanceError> = None;
        let _: Option<testing::DetailedBalanceReport> = None;
        let _: Option<testing::AdditiveTarget<Smoke, Smoke>> = None;
        let _: Option<testing::DiscreteProposalRatio> = None;
        let _: Option<testing::DiscreteProposalRatioError> = None;
        let _: Option<
            delayed::TryObservedDelayedIntoRunResult<Infallible, Infallible, Infallible>,
        > = None;
    }

    #[cfg(feature = "serde")]
    #[test]
    fn sampler_serializes_handles() -> Result<(), JsonError> {
        let target = Smoke;
        let proposal = Smoke;
        let mut rng = Smoke;
        let Ok(chain) = Chain::new(1.0, &target) else {
            unreachable!("Smoke target always returns a finite log-probability");
        };
        let Ok(sampler) = Sampler::new(chain, &target, proposal, &mut rng) else {
            unreachable!("Smoke target always returns a finite current log-probability");
        };

        let checkpoint = to_value(&sampler)?;

        assert_eq!(checkpoint["chain"]["state"], json!(1.0));
        assert_eq!(checkpoint["chain"]["accepted"], json!(0));
        assert_eq!(checkpoint["chain"]["rejected"], json!(0));
        assert!(checkpoint.get("target").is_some());
        assert!(checkpoint.get("proposal").is_some());
        assert!(checkpoint.get("rng").is_some());
        Ok(())
    }
}
