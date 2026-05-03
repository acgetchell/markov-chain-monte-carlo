//! Markov Chain Monte Carlo (MCMC) framework.
//!
//! 🚧 **Pre-release (0.x)** — This crate is under active development and
//! not yet ready for production use. APIs may change without notice.
//!
//! This crate aims to provide a composable, zero-cost abstraction for MCMC
//! methods over arbitrary state spaces, including discrete and combinatorial
//! systems (e.g., triangulations).
//!
//! [`Target::log_prob`] should return an unnormalized natural log-probability
//! or log-density.  Additive constants are fine because Metropolis-Hastings
//! only uses differences, but arbitrary scores or logits will sample a
//! different distribution.
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
//! for those streaming measurement loops.
//!
//! # Example
//!
//! Sample from a standard normal distribution using Metropolis–Hastings:
//!
//! ```
//! use markov_chain_monte_carlo::prelude::by_value::*;
//! use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};
//!
//! #[derive(Clone)]
//! struct Scalar(f64);
//!
//! struct Normal;
//! impl Target<Scalar> for Normal {
//!     fn log_prob(&self, state: &Scalar) -> f64 {
//!         -0.5 * state.0 * state.0
//!     }
//! }
//!
//! struct RandomWalk { width: f64 }
//! impl Proposal<Scalar> for RandomWalk {
//!     fn propose<R: Rng + ?Sized>(&self, current: &Scalar, rng: &mut R) -> Scalar {
//!         let delta: f64 = rng.random_range(-self.width..self.width);
//!         Scalar(current.0 + delta)
//!     }
//! }
//!
//! fn main() -> Result<(), McmcError> {
//!     let mut rng = StdRng::seed_from_u64(42);
//!     let mut chain = Chain::new(Scalar(0.0), &Normal)?;
//!     let proposal = RandomWalk { width: 1.0 };
//!
//!     for _ in 0..1000 {
//!         chain.step(&Normal, &proposal, &mut rng)?;
//!     }
//!
//!     assert!(chain.acceptance_rate() > 0.2);
//!     Ok(())
//! }
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
//!     assert!(step.accepted);
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
//! let mut sampler = Sampler::new(chain, &Normal, &Walk, &mut rng);
//!
//! // Burn-in
//! sampler.run(1000)?;
//! sampler.chain_mut().reset_counters();
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
//! let mut sampler = Sampler::new(chain, &Normal, &Walk, &mut rng);
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
//! energy.extend([1.0, 2.0, 3.0, 4.0]);
//!
//! assert_eq!(energy.mean(), Some(2.5));
//!
//! let mut bins = BinningAnalysis::new();
//! bins.extend([1.0, 2.0, 3.0, 4.0]);
//! assert!(bins.standard_error().is_some());
//! ```
//!
//! `Sampler` can also stream observations directly into these accumulators:
//!
//! ```
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
//! let chain = Chain::new(0.0, &T)?;
//! let mut sampler = Sampler::new(chain, &T, &P, &mut rng);
//! let mut coordinate = |state: &f64| *state;
//! let mut stats = OnlineStats::new();
//!
//! sampler.run_observing_into(4, &mut coordinate, &mut stats)?;
//! assert_eq!(stats.count(), 4);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod chain;
mod error;
mod observable;
mod sampler;
mod statistics;
mod traits;

pub use chain::{Chain, DelayedStep, DelayedStepError, Step};
pub use error::McmcError;
pub use observable::{
    Observable, ObservedStepError, ObservedStreamError, SampleBuffer, TryAccumulator, TryObservable,
};
pub use sampler::{
    ObservedDelayedIntoRunResult, ObservedDelayedStep, ObservedDelayedStepResult,
    ObservedIntoRunResult, Sampler, TryObservedDelayedIntoRunResult, TryObservedDelayedRunResult,
    TryObservedDelayedStepResult, TryObservedIntoRunResult, TryObservedMutStepResult,
    TryObservedRunResult, TryObservedStepResult,
};
pub use statistics::{BinningAnalysis, BinningEstimate, OnlineStats, StatisticsError};
pub use traits::{DelayedProposal, Proposal, ProposalMut, Target};

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
/// ```
pub mod prelude {
    pub use crate::{
        BinningAnalysis, BinningEstimate, Chain, McmcError, Observable, ObservedIntoRunResult,
        ObservedStepError, ObservedStreamError, OnlineStats, SampleBuffer, Sampler,
        StatisticsError, Target, TryAccumulator, TryObservable, TryObservedIntoRunResult,
    };

    /// Prelude for by-value proposals.
    ///
    /// This imports the shared sampling types plus [`crate::Proposal`], without
    /// importing the in-place or delayed proposal traits.
    pub mod by_value {
        pub use crate::{
            BinningAnalysis, BinningEstimate, Chain, McmcError, Observable, ObservedIntoRunResult,
            ObservedStepError, ObservedStreamError, OnlineStats, Proposal, SampleBuffer, Sampler,
            StatisticsError, Target, TryAccumulator, TryObservable, TryObservedIntoRunResult,
        };
    }

    /// Prelude for in-place proposals with rollback.
    ///
    /// This imports the shared sampling types plus [`crate::ProposalMut`], without
    /// importing the by-value or delayed proposal traits.
    pub mod in_place {
        pub use crate::{
            BinningAnalysis, BinningEstimate, Chain, McmcError, Observable, ObservedIntoRunResult,
            ObservedStepError, ObservedStreamError, OnlineStats, ProposalMut, SampleBuffer,
            Sampler, StatisticsError, Target, TryAccumulator, TryObservable,
            TryObservedIntoRunResult,
        };
    }

    /// Prelude for delayed-commit proposals.
    ///
    /// This imports the shared sampling types plus delayed-step telemetry and
    /// errors, without importing the by-value or in-place proposal traits.
    pub mod delayed {
        pub use crate::{
            BinningAnalysis, BinningEstimate, Chain, DelayedProposal, DelayedStep,
            DelayedStepError, Observable, ObservedDelayedIntoRunResult, ObservedDelayedStep,
            ObservedDelayedStepResult, ObservedStepError, ObservedStreamError, OnlineStats,
            SampleBuffer, Sampler, StatisticsError, Target, TryAccumulator, TryObservable,
            TryObservedDelayedIntoRunResult,
        };
    }
}

#[cfg(test)]
mod public_api_smoke_tests {
    use core::convert::Infallible;

    use rand::{Rng, rngs::StdRng};

    use super::{
        BinningEstimate, Chain, DelayedStep, McmcError, Observable, ObservedDelayedStep,
        OnlineStats, Proposal, ProposalMut, SampleBuffer, Sampler, StatisticsError, Step, Target,
        prelude::{self, by_value, delayed, in_place},
    };

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
        fn needs_observable<O: Observable<f64, Output = f64>>(_: &mut O) {}

        needs_target::<Smoke>();
        needs_by_value::<Smoke>();
        needs_in_place::<Smoke>();
        needs_delayed::<Smoke>();

        let mut observable = |state: &f64| *state;
        needs_observable(&mut observable);

        let _: Option<Chain<f64>> = None;
        let _: Option<Step<()>> = None;
        let _: Option<DelayedStep<()>> = None;
        let _: Option<McmcError> = None;
        let _: Option<SampleBuffer<f64>> = None;
        let _: Option<Sampler<'_, f64, Smoke, Smoke, StdRng>> = None;
        let _: Option<ObservedDelayedStep<(), f64>> = None;
        let _: Option<BinningEstimate> = None;
        let _: Option<OnlineStats> = None;
        let _: Option<StatisticsError> = None;
    }
}
