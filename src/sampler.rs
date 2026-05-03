//! Ergonomic sampling wrapper that bundles a chain with its components.

use std::fmt;

use rand::Rng;

use crate::{
    Chain, DelayedProposal, DelayedStep, DelayedStepError, McmcError, Observable,
    ObservedStepError, Proposal, ProposalMut, SampleBuffer, Target, TryObservable,
};

/// Delayed-step telemetry paired with a measurement from the resulting state.
pub type ObservedDelayedStep<I, O> = (DelayedStep<I>, O);

/// Result returned by delayed observing steps.
pub type ObservedDelayedStepResult<I, O, E> =
    Result<ObservedDelayedStep<I, O>, DelayedStepError<E>>;

/// Result returned by fallible by-value observing steps.
pub type TryObservedStepResult<O, E> = Result<O, ObservedStepError<McmcError, E>>;

/// Result returned by fallible in-place observing steps.
pub type TryObservedMutStepResult<O, E> = Result<(bool, O), ObservedStepError<McmcError, E>>;

/// Result returned by fallible by-value or in-place observing runs.
pub type TryObservedRunResult<O, E> = Result<SampleBuffer<O>, ObservedStepError<McmcError, E>>;

/// Result returned by fallible delayed observing steps.
pub type TryObservedDelayedStepResult<I, O, P, E> =
    Result<ObservedDelayedStep<I, O>, ObservedStepError<DelayedStepError<P>, E>>;

/// Result returned by fallible delayed observing runs.
pub type TryObservedDelayedRunResult<O, P, E> =
    Result<SampleBuffer<O>, ObservedStepError<DelayedStepError<P>, E>>;

/// Bundles a [`Chain`] with its target, proposal, and RNG for ergonomic
/// sampling.
///
/// `Sampler` owns the chain and stores a proposal handle.  In typical use that
/// handle is a shared borrow (`&P`) for by-value and in-place proposals, or a
/// mutable borrow (`&mut P`) for delayed-commit proposals.  It also borrows the
/// target and RNG.
/// It provides [`step`](Self::step) / [`step_mut`](Self::step_mut) for
/// single Metropolis–Hastings steps and [`run`](Self::run) /
/// [`run_mut`](Self::run_mut) for bulk sampling.
///
/// For the by-value proposal path (`P: Proposal<S>`), `Sampler` also implements
/// [`Iterator`], yielding `Result<(), McmcError>` on each step.
///
/// # Example
///
/// ```
/// use markov_chain_monte_carlo::prelude::by_value::*;
/// use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};
///
/// #[derive(Clone)]
/// struct Scalar(f64);
///
/// struct Normal;
/// impl Target<Scalar> for Normal {
///     fn log_prob(&self, s: &Scalar) -> f64 { -0.5 * s.0 * s.0 }
/// }
///
/// struct Walk;
/// impl Proposal<Scalar> for Walk {
///     fn propose<R: Rng + ?Sized>(&self, c: &Scalar, r: &mut R) -> Scalar {
///         Scalar(c.0 + r.random_range(-1.0..1.0))
///     }
/// }
///
/// let mut rng = StdRng::seed_from_u64(42);
/// let chain = Chain::new(Scalar(0.0), &Normal)?;
/// let mut sampler = Sampler::new(chain, &Normal, &Walk, &mut rng);
///
/// // Burn-in
/// sampler.run(1000)?;
/// sampler.chain_mut().reset_counters();
///
/// // Production sampling
/// sampler.run(10_000)?;
/// assert!(sampler.chain_ref().acceptance_rate() > 0.0);
/// # Ok::<(), McmcError>(())
/// ```
#[must_use]
pub struct Sampler<'a, S, T, P, R: ?Sized> {
    /// The MCMC chain being sampled.
    chain: Chain<S>,
    target: &'a T,
    proposal: P,
    rng: &'a mut R,
}

impl<S: fmt::Debug, T, P, R: ?Sized> fmt::Debug for Sampler<'_, S, T, P, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Sampler")
            .field("chain", &self.chain)
            .finish_non_exhaustive()
    }
}

// --- Construction and decomposition (no trait bounds) ---

impl<'a, S, T, P, R: ?Sized> Sampler<'a, S, T, P, R> {
    /// Create a new sampler from its components.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::by_value::*;
    /// use rand::{Rng, SeedableRng, rngs::StdRng};
    ///
    /// # struct T;
    /// # impl Target<f64> for T { fn log_prob(&self, x: &f64) -> f64 { -0.5 * x * x } }
    /// # struct P;
    /// # impl Proposal<f64> for P {
    /// #     fn propose<R: Rng + ?Sized>(&self, current: &f64, _rng: &mut R) -> f64 {
    /// #         current + 1.0
    /// #     }
    /// # }
    /// let mut rng = StdRng::seed_from_u64(42);
    /// let chain = Chain::new(0.0, &T)?;
    /// let sampler = Sampler::new(chain, &T, &P, &mut rng);
    ///
    /// assert_eq!(sampler.chain_ref().total_steps(), 0);
    /// # Ok::<(), McmcError>(())
    /// ```
    pub const fn new(chain: Chain<S>, target: &'a T, proposal: P, rng: &'a mut R) -> Self {
        Self {
            chain,
            target,
            proposal,
            rng,
        }
    }

    /// Shared reference to the inner chain.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::by_value::*;
    /// use rand::{Rng, SeedableRng, rngs::StdRng};
    ///
    /// # struct T;
    /// # impl Target<f64> for T { fn log_prob(&self, x: &f64) -> f64 { -0.5 * x * x } }
    /// # struct P;
    /// # impl Proposal<f64> for P {
    /// #     fn propose<R: Rng + ?Sized>(&self, current: &f64, _rng: &mut R) -> f64 {
    /// #         current + 1.0
    /// #     }
    /// # }
    /// let mut rng = StdRng::seed_from_u64(42);
    /// let chain = Chain::new(1.0, &T)?;
    /// let sampler = Sampler::new(chain, &T, &P, &mut rng);
    ///
    /// assert!((*sampler.chain_ref().state() - 1.0).abs() < f64::EPSILON);
    /// # Ok::<(), McmcError>(())
    /// ```
    pub const fn chain_ref(&self) -> &Chain<S> {
        &self.chain
    }

    /// Mutable reference to the inner chain.
    ///
    /// This permits operations such as [`Chain::reset_counters`] or
    /// [`Chain::replace_state`] without exposing `Sampler`'s fields as part of
    /// its public representation.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::by_value::*;
    /// use rand::{Rng, SeedableRng, rngs::StdRng};
    ///
    /// # struct T;
    /// # impl Target<f64> for T { fn log_prob(&self, x: &f64) -> f64 { -0.5 * x * x } }
    /// # struct P;
    /// # impl Proposal<f64> for P {
    /// #     fn propose<R: Rng + ?Sized>(&self, current: &f64, _rng: &mut R) -> f64 {
    /// #         current + 1.0
    /// #     }
    /// # }
    /// let mut rng = StdRng::seed_from_u64(42);
    /// let chain = Chain::new(1.0, &T)?;
    /// let mut sampler = Sampler::new(chain, &T, &P, &mut rng);
    ///
    /// sampler.chain_mut().replace_state(2.0, &T)?;
    /// assert!((*sampler.chain_ref().state() - 2.0).abs() < f64::EPSILON);
    /// # Ok::<(), McmcError>(())
    /// ```
    pub const fn chain_mut(&mut self) -> &mut Chain<S> {
        &mut self.chain
    }

    /// Shared reference to the stored proposal handle.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::by_value::*;
    /// use rand::{Rng, SeedableRng, rngs::StdRng};
    ///
    /// # struct T;
    /// # impl Target<f64> for T { fn log_prob(&self, x: &f64) -> f64 { -0.5 * x * x } }
    /// struct Walk { width: f64 }
    /// impl Proposal<f64> for Walk {
    ///     fn propose<R: Rng + ?Sized>(&self, current: &f64, _rng: &mut R) -> f64 {
    ///         current + self.width
    ///     }
    /// }
    ///
    /// let proposal = Walk { width: 1.0 };
    /// let mut rng = StdRng::seed_from_u64(42);
    /// let chain = Chain::new(0.0, &T)?;
    /// let sampler = Sampler::new(chain, &T, &proposal, &mut rng);
    ///
    /// assert!((sampler.proposal_ref().width - 1.0).abs() < f64::EPSILON);
    /// # Ok::<(), McmcError>(())
    /// ```
    #[must_use]
    pub const fn proposal_ref(&self) -> &P {
        &self.proposal
    }

    /// Mutable reference to the stored proposal handle.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::by_value::*;
    /// use rand::{Rng, SeedableRng, rngs::StdRng};
    ///
    /// # struct T;
    /// # impl Target<f64> for T { fn log_prob(&self, x: &f64) -> f64 { -0.5 * x * x } }
    /// struct Walk { width: f64 }
    /// impl Proposal<f64> for Walk {
    ///     fn propose<R: Rng + ?Sized>(&self, current: &f64, _rng: &mut R) -> f64 {
    ///         current + self.width
    ///     }
    /// }
    ///
    /// let mut rng = StdRng::seed_from_u64(42);
    /// let chain = Chain::new(0.0, &T)?;
    /// let mut sampler = Sampler::new(chain, &T, Walk { width: 1.0 }, &mut rng);
    /// sampler.proposal_mut().width = 0.5;
    ///
    /// assert!((sampler.proposal_ref().width - 0.5).abs() < f64::EPSILON);
    /// # Ok::<(), McmcError>(())
    /// ```
    pub const fn proposal_mut(&mut self) -> &mut P {
        &mut self.proposal
    }

    /// Consume the sampler and return the inner chain.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::by_value::*;
    /// use rand::{SeedableRng, rngs::StdRng};
    ///
    /// # struct T;
    /// # impl Target<f64> for T { fn log_prob(&self, x: &f64) -> f64 { -0.5 * x * x } }
    /// # struct P;
    /// # impl Proposal<f64> for P {
    /// #     fn propose<R: rand::Rng + ?Sized>(&self, c: &f64, _r: &mut R) -> f64 { c + 1.0 }
    /// # }
    /// let mut rng = StdRng::seed_from_u64(42);
    /// let chain = Chain::new(1.0_f64, &T)?;
    /// let sampler = Sampler::new(chain, &T, &P, &mut rng);
    /// let chain = sampler.into_chain();
    /// assert!((chain.log_prob() - (-0.5)).abs() < 1e-12);
    /// # Ok::<(), McmcError>(())
    /// ```
    pub fn into_chain(self) -> Chain<S> {
        self.chain
    }
}

// --- By-value stepping ---

impl<S, T: Target<S>, P: Proposal<S>, R: Rng + ?Sized> Sampler<'_, S, T, P, R> {
    /// Perform a single by-value Metropolis–Hastings step.
    ///
    /// Delegates to [`Chain::step`].
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::by_value::*;
    /// use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};
    ///
    /// # #[derive(Clone)] struct S(f64);
    /// # struct T;
    /// # impl Target<S> for T { fn log_prob(&self, s: &S) -> f64 { -0.5 * s.0 * s.0 } }
    /// # struct P;
    /// # impl Proposal<S> for P {
    /// #     fn propose<R: Rng + ?Sized>(&self, c: &S, r: &mut R) -> S {
    /// #         S(c.0 + r.random_range(-1.0..1.0))
    /// #     }
    /// # }
    /// let mut rng = StdRng::seed_from_u64(42);
    /// let chain = Chain::new(S(0.0), &T)?;
    /// let mut sampler = Sampler::new(chain, &T, &P, &mut rng);
    /// sampler.step()?;
    /// assert_eq!(sampler.chain_ref().total_steps(), 1);
    /// # Ok::<(), McmcError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`McmcError`] on NaN or +∞ log-probability or NaN log q-ratio.
    pub fn step(&mut self) -> Result<(), McmcError> {
        self.chain.step(self.target, &self.proposal, self.rng)
    }

    /// Run `steps` by-value Metropolis–Hastings steps.
    ///
    /// Stops and returns the first error encountered.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::by_value::*;
    /// use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};
    ///
    /// # #[derive(Clone)] struct S(f64);
    /// # struct T;
    /// # impl Target<S> for T { fn log_prob(&self, s: &S) -> f64 { -0.5 * s.0 * s.0 } }
    /// # struct P;
    /// # impl Proposal<S> for P {
    /// #     fn propose<R: Rng + ?Sized>(&self, c: &S, r: &mut R) -> S {
    /// #         S(c.0 + r.random_range(-1.0..1.0))
    /// #     }
    /// # }
    /// let mut rng = StdRng::seed_from_u64(42);
    /// let chain = Chain::new(S(0.0), &T)?;
    /// let mut sampler = Sampler::new(chain, &T, &P, &mut rng);
    ///
    /// sampler.run(1000)?;
    /// assert_eq!(sampler.chain_ref().total_steps(), 1000);
    /// # Ok::<(), McmcError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`McmcError`] on the first step that fails.
    pub fn run(&mut self, steps: usize) -> Result<(), McmcError> {
        for _ in 0..steps {
            self.step()?;
        }
        Ok(())
    }

    /// Perform one by-value step and observe the resulting chain state.
    ///
    /// The observable is invoked after the step completes, including rejected
    /// proposals, so the returned value always corresponds to the current chain
    /// state after one counted Metropolis-Hastings step.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::by_value::*;
    /// use rand::{Rng, SeedableRng, rngs::StdRng};
    ///
    /// # struct T;
    /// # impl Target<f64> for T { fn log_prob(&self, x: &f64) -> f64 { -0.5 * x * x } }
    /// # struct P;
    /// # impl Proposal<f64> for P {
    /// #     fn propose<R: Rng + ?Sized>(&self, current: &f64, _rng: &mut R) -> f64 {
    /// #         current + 1.0
    /// #     }
    /// # }
    /// let mut rng = StdRng::seed_from_u64(42);
    /// let chain = Chain::new(0.0, &T)?;
    /// let mut sampler = Sampler::new(chain, &T, &P, &mut rng);
    /// let mut energy = |state: &f64| 0.5 * state * state;
    ///
    /// let measurement = sampler.step_observing(&mut energy)?;
    /// assert!(measurement >= 0.0);
    /// # Ok::<(), McmcError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`McmcError`] if the step fails.  In that case the observable is
    /// not invoked.
    pub fn step_observing<O: Observable<S> + ?Sized>(
        &mut self,
        observable: &mut O,
    ) -> Result<O::Output, McmcError> {
        self.step()?;
        Ok(observable.observe(self.chain.state()))
    }

    /// Run by-value steps and collect one observation after each step.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::by_value::*;
    /// use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};
    ///
    /// # #[derive(Clone)] struct S(f64);
    /// # struct T;
    /// # impl Target<S> for T { fn log_prob(&self, s: &S) -> f64 { -0.5 * s.0 * s.0 } }
    /// # struct P;
    /// # impl Proposal<S> for P {
    /// #     fn propose<R: Rng + ?Sized>(&self, c: &S, r: &mut R) -> S {
    /// #         S(c.0 + r.random_range(-1.0..1.0))
    /// #     }
    /// # }
    /// let mut rng = StdRng::seed_from_u64(42);
    /// let chain = Chain::new(S(0.0), &T)?;
    /// let mut sampler = Sampler::new(chain, &T, &P, &mut rng);
    /// let mut coordinate = |state: &S| state.0;
    ///
    /// let samples = sampler.run_observing(128, &mut coordinate)?;
    /// assert_eq!(samples.len(), 128);
    /// # Ok::<(), McmcError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`McmcError`] on the first step that fails.
    pub fn run_observing<O: Observable<S> + ?Sized>(
        &mut self,
        steps: usize,
        observable: &mut O,
    ) -> Result<SampleBuffer<O::Output>, McmcError> {
        let mut samples = SampleBuffer::with_capacity(steps);
        for _ in 0..steps {
            samples.push(self.step_observing(observable)?);
        }
        Ok(samples)
    }

    /// Perform one by-value step and fallibly observe the resulting state.
    ///
    /// Step failures are returned as [`ObservedStepError::Step`]. Measurement
    /// failures are returned as [`ObservedStepError::Observation`] after the
    /// chain has completed the step.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::by_value::*;
    /// use rand::{Rng, SeedableRng, rngs::StdRng};
    ///
    /// # struct T;
    /// # impl Target<f64> for T { fn log_prob(&self, x: &f64) -> f64 { -0.5 * x * x } }
    /// # struct P;
    /// # impl Proposal<f64> for P {
    /// #     fn propose<R: Rng + ?Sized>(&self, current: &f64, _rng: &mut R) -> f64 {
    /// #         current + 1.0
    /// #     }
    /// # }
    /// let mut rng = StdRng::seed_from_u64(42);
    /// let chain = Chain::new(0.0, &T).map_err(ObservedStepError::Step)?;
    /// let mut sampler = Sampler::new(chain, &T, &P, &mut rng);
    /// let mut finite = |state: &f64| {
    ///     if state.is_finite() { Ok(*state) } else { Err("non-finite") }
    /// };
    ///
    /// let sample = sampler.try_step_observing(&mut finite)?;
    /// assert!(sample.is_finite());
    /// # Ok::<(), ObservedStepError<McmcError, &'static str>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`ObservedStepError`] when either the sampling step or the
    /// observable fails.
    pub fn try_step_observing<O: TryObservable<S> + ?Sized>(
        &mut self,
        observable: &mut O,
    ) -> TryObservedStepResult<O::Output, O::Error> {
        self.step().map_err(ObservedStepError::Step)?;
        observable
            .try_observe(self.chain.state())
            .map_err(ObservedStepError::Observation)
    }

    /// Run by-value steps and collect fallible observations.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::by_value::*;
    /// use rand::{Rng, SeedableRng, rngs::StdRng};
    ///
    /// # struct T;
    /// # impl Target<f64> for T { fn log_prob(&self, x: &f64) -> f64 { -0.5 * x * x } }
    /// # struct P;
    /// # impl Proposal<f64> for P {
    /// #     fn propose<R: Rng + ?Sized>(&self, current: &f64, _rng: &mut R) -> f64 {
    /// #         current + 1.0
    /// #     }
    /// # }
    /// let mut rng = StdRng::seed_from_u64(42);
    /// let chain = Chain::new(0.0, &T).map_err(ObservedStepError::Step)?;
    /// let mut sampler = Sampler::new(chain, &T, &P, &mut rng);
    /// let mut nonnegative = |state: &f64| {
    ///     if *state >= 0.0 { Ok(*state) } else { Err("negative state") }
    /// };
    ///
    /// let samples = sampler.try_run_observing(2, &mut nonnegative)?;
    /// assert_eq!(samples.len(), 2);
    /// # Ok::<(), ObservedStepError<McmcError, &'static str>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`ObservedStepError`] on the first step or observation failure.
    pub fn try_run_observing<O: TryObservable<S> + ?Sized>(
        &mut self,
        steps: usize,
        observable: &mut O,
    ) -> TryObservedRunResult<O::Output, O::Error> {
        let mut samples = SampleBuffer::with_capacity(steps);
        for _ in 0..steps {
            samples.push(self.try_step_observing(observable)?);
        }
        Ok(samples)
    }
}

// --- In-place stepping ---

impl<S, T: Target<S>, P: ProposalMut<S>, R: Rng + ?Sized> Sampler<'_, S, T, P, R> {
    /// Perform a single in-place Metropolis–Hastings step with rollback.
    ///
    /// Delegates to [`Chain::step_mut`].  Returns `Ok(true)` if accepted,
    /// `Ok(false)` if rejected.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::in_place::*;
    /// use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};
    ///
    /// # struct S(f64);
    /// # struct T;
    /// # impl Target<S> for T { fn log_prob(&self, s: &S) -> f64 { -0.5 * s.0 * s.0 } }
    /// # struct P;
    /// # impl ProposalMut<S> for P {
    /// #     type Undo = f64;
    /// #     fn propose_mut<R: Rng + ?Sized>(&self, s: &mut S, r: &mut R) -> Option<f64> {
    /// #         let old = s.0; s.0 += r.random_range(-1.0..1.0); Some(old)
    /// #     }
    /// #     fn undo(&self, s: &mut S, old: f64) { s.0 = old; }
    /// # }
    /// let mut rng = StdRng::seed_from_u64(42);
    /// let chain = Chain::new(S(0.0), &T)?;
    /// let mut sampler = Sampler::new(chain, &T, &P, &mut rng);
    /// let accepted = sampler.step_mut()?;
    /// assert_eq!(sampler.chain_ref().total_steps(), 1);
    /// # Ok::<(), McmcError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`McmcError`] on NaN or +∞ log-probability or NaN log q-ratio
    /// (state is rolled back before the error is returned).
    pub fn step_mut(&mut self) -> Result<bool, McmcError> {
        self.chain.step_mut(self.target, &self.proposal, self.rng)
    }

    /// Run `steps` in-place Metropolis–Hastings steps.
    ///
    /// Stops and returns the first error encountered.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::in_place::*;
    /// use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};
    ///
    /// # struct Spins(Vec<i8>);
    /// # struct Ising;
    /// # impl Target<Spins> for Ising {
    /// #     fn log_prob(&self, s: &Spins) -> f64 {
    /// #         s.0.windows(2).map(|w| f64::from(w[0]) * f64::from(w[1])).sum()
    /// #     }
    /// # }
    /// # struct Flip;
    /// # impl ProposalMut<Spins> for Flip {
    /// #     type Undo = usize;
    /// #     fn propose_mut<R: Rng + ?Sized>(&self, s: &mut Spins, r: &mut R) -> Option<usize> {
    /// #         let i = r.random_range(0..s.0.len()); s.0[i] *= -1; Some(i)
    /// #     }
    /// #     fn undo(&self, s: &mut Spins, i: usize) { s.0[i] *= -1; }
    /// # }
    /// let mut rng = StdRng::seed_from_u64(42);
    /// let chain = Chain::new(Spins(vec![1; 20]), &Ising)?;
    /// let mut sampler = Sampler::new(chain, &Ising, &Flip, &mut rng);
    ///
    /// sampler.run_mut(1000)?;
    /// assert_eq!(sampler.chain_ref().total_steps(), 1000);
    /// # Ok::<(), McmcError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`McmcError`] on the first step that fails.
    pub fn run_mut(&mut self, steps: usize) -> Result<(), McmcError> {
        for _ in 0..steps {
            self.step_mut()?;
        }
        Ok(())
    }

    /// Perform one in-place step and observe the resulting chain state.
    ///
    /// Returns the step acceptance flag together with the measurement.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::in_place::*;
    /// use rand::{Rng, SeedableRng, rngs::StdRng};
    ///
    /// # struct S(f64);
    /// # struct T;
    /// # impl Target<S> for T { fn log_prob(&self, s: &S) -> f64 { -0.5 * s.0 * s.0 } }
    /// # struct P;
    /// # impl ProposalMut<S> for P {
    /// #     type Undo = f64;
    /// #     fn propose_mut<R: Rng + ?Sized>(&self, s: &mut S, _r: &mut R) -> Option<f64> {
    /// #         let old = s.0; s.0 += 1.0; Some(old)
    /// #     }
    /// #     fn undo(&self, s: &mut S, old: f64) { s.0 = old; }
    /// # }
    /// let mut rng = StdRng::seed_from_u64(42);
    /// let chain = Chain::new(S(0.0), &T)?;
    /// let mut sampler = Sampler::new(chain, &T, &P, &mut rng);
    /// let mut coordinate = |state: &S| state.0;
    ///
    /// let (_accepted, sample) = sampler.step_mut_observing(&mut coordinate)?;
    /// assert!(sample >= 0.0);
    /// # Ok::<(), McmcError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`McmcError`] if the step fails.  In that case the observable is
    /// not invoked.
    pub fn step_mut_observing<O: Observable<S> + ?Sized>(
        &mut self,
        observable: &mut O,
    ) -> Result<(bool, O::Output), McmcError> {
        let accepted = self.step_mut()?;
        Ok((accepted, observable.observe(self.chain.state())))
    }

    /// Run in-place steps and collect one observation after each step.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::in_place::*;
    /// use rand::{Rng, SeedableRng, rngs::StdRng};
    ///
    /// # struct S(f64);
    /// # struct T;
    /// # impl Target<S> for T { fn log_prob(&self, s: &S) -> f64 { -0.5 * s.0 * s.0 } }
    /// # struct P;
    /// # impl ProposalMut<S> for P {
    /// #     type Undo = f64;
    /// #     fn propose_mut<R: Rng + ?Sized>(&self, s: &mut S, _r: &mut R) -> Option<f64> {
    /// #         let old = s.0; s.0 += 1.0; Some(old)
    /// #     }
    /// #     fn undo(&self, s: &mut S, old: f64) { s.0 = old; }
    /// # }
    /// let mut rng = StdRng::seed_from_u64(42);
    /// let chain = Chain::new(S(0.0), &T)?;
    /// let mut sampler = Sampler::new(chain, &T, &P, &mut rng);
    /// let mut coordinate = |state: &S| state.0;
    ///
    /// let samples = sampler.run_mut_observing(16, &mut coordinate)?;
    /// assert_eq!(samples.len(), 16);
    /// # Ok::<(), McmcError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`McmcError`] on the first step that fails.
    pub fn run_mut_observing<O: Observable<S> + ?Sized>(
        &mut self,
        steps: usize,
        observable: &mut O,
    ) -> Result<SampleBuffer<O::Output>, McmcError> {
        let mut samples = SampleBuffer::with_capacity(steps);
        for _ in 0..steps {
            let (_, sample) = self.step_mut_observing(observable)?;
            samples.push(sample);
        }
        Ok(samples)
    }

    /// Perform one in-place step and fallibly observe the resulting state.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::in_place::*;
    /// use rand::{Rng, SeedableRng, rngs::StdRng};
    ///
    /// # struct S(f64);
    /// # struct T;
    /// # impl Target<S> for T { fn log_prob(&self, s: &S) -> f64 { -0.5 * s.0 * s.0 } }
    /// # struct P;
    /// # impl ProposalMut<S> for P {
    /// #     type Undo = f64;
    /// #     fn propose_mut<R: Rng + ?Sized>(&self, s: &mut S, _r: &mut R) -> Option<f64> {
    /// #         let old = s.0; s.0 += 1.0; Some(old)
    /// #     }
    /// #     fn undo(&self, s: &mut S, old: f64) { s.0 = old; }
    /// # }
    /// let mut rng = StdRng::seed_from_u64(42);
    /// let chain = Chain::new(S(0.0), &T).map_err(ObservedStepError::Step)?;
    /// let mut sampler = Sampler::new(chain, &T, &P, &mut rng);
    /// let mut finite = |state: &S| {
    ///     if state.0.is_finite() { Ok(state.0) } else { Err("non-finite") }
    /// };
    ///
    /// let (_accepted, sample) = sampler.try_step_mut_observing(&mut finite)?;
    /// assert!(sample.is_finite());
    /// # Ok::<(), ObservedStepError<McmcError, &'static str>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`ObservedStepError`] when either the sampling step or the
    /// observable fails.
    pub fn try_step_mut_observing<O: TryObservable<S> + ?Sized>(
        &mut self,
        observable: &mut O,
    ) -> TryObservedMutStepResult<O::Output, O::Error> {
        let accepted = self.step_mut().map_err(ObservedStepError::Step)?;
        let sample = observable
            .try_observe(self.chain.state())
            .map_err(ObservedStepError::Observation)?;
        Ok((accepted, sample))
    }

    /// Run in-place steps and collect fallible observations.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::in_place::*;
    /// use rand::{Rng, SeedableRng, rngs::StdRng};
    ///
    /// # struct S(f64);
    /// # struct T;
    /// # impl Target<S> for T { fn log_prob(&self, s: &S) -> f64 { -0.5 * s.0 * s.0 } }
    /// # struct P;
    /// # impl ProposalMut<S> for P {
    /// #     type Undo = f64;
    /// #     fn propose_mut<R: Rng + ?Sized>(&self, s: &mut S, _r: &mut R) -> Option<f64> {
    /// #         let old = s.0; s.0 += 1.0; Some(old)
    /// #     }
    /// #     fn undo(&self, s: &mut S, old: f64) { s.0 = old; }
    /// # }
    /// let mut rng = StdRng::seed_from_u64(42);
    /// let chain = Chain::new(S(0.0), &T).map_err(ObservedStepError::Step)?;
    /// let mut sampler = Sampler::new(chain, &T, &P, &mut rng);
    /// let mut finite = |state: &S| {
    ///     if state.0.is_finite() { Ok(state.0) } else { Err("non-finite") }
    /// };
    ///
    /// let samples = sampler.try_run_mut_observing(2, &mut finite)?;
    /// assert_eq!(samples.len(), 2);
    /// # Ok::<(), ObservedStepError<McmcError, &'static str>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`ObservedStepError`] on the first step or observation failure.
    pub fn try_run_mut_observing<O: TryObservable<S> + ?Sized>(
        &mut self,
        steps: usize,
        observable: &mut O,
    ) -> TryObservedRunResult<O::Output, O::Error> {
        let mut samples = SampleBuffer::with_capacity(steps);
        for _ in 0..steps {
            let (_, sample) = self.try_step_mut_observing(observable)?;
            samples.push(sample);
        }
        Ok(samples)
    }
}

// --- Delayed-commit stepping ---

impl<S, T: Target<S>, P: DelayedProposal<S>, R: Rng + ?Sized> Sampler<'_, S, T, P, R> {
    /// Perform a single delayed-commit Metropolis-Hastings step.
    ///
    /// Delegates to [`Chain::step_delayed`].
    ///
    /// ```
    /// use core::convert::Infallible;
    /// use markov_chain_monte_carlo::prelude::delayed::*;
    /// use rand::{Rng, SeedableRng, rngs::StdRng};
    ///
    /// struct TargetLine;
    /// impl Target<i32> for TargetLine {
    ///     fn log_prob(&self, state: &i32) -> f64 { -f64::from(state.abs()) }
    /// }
    ///
    /// struct MoveRight;
    /// impl DelayedProposal<i32> for MoveRight {
    ///     type Plan = i32;
    ///     type Info = i32;
    ///     type Error = Infallible;
    ///
    ///     fn propose_plan<R: Rng + ?Sized>(
    ///         &mut self,
    ///         _state: &i32,
    ///         _rng: &mut R,
    ///     ) -> Result<Option<i32>, Self::Error> {
    ///         Ok(Some(1))
    ///     }
    ///
    ///     fn proposed_log_prob<T: Target<i32>>(
    ///         &self,
    ///         state: &i32,
    ///         plan: &i32,
    ///         target: &T,
    ///     ) -> Result<f64, Self::Error> {
    ///         Ok(target.log_prob(&(*state + *plan)))
    ///     }
    ///
    ///     fn info(&self, plan: &i32) -> i32 { *plan }
    ///
    ///     fn commit<R: Rng + ?Sized>(
    ///         &mut self,
    ///         state: &mut i32,
    ///         plan: i32,
    ///         _rng: &mut R,
    ///     ) -> Result<(), Self::Error> {
    ///         *state += plan;
    ///         Ok(())
    ///     }
    /// }
    ///
    /// let target = TargetLine;
    /// let mut proposal = MoveRight;
    /// let mut rng = StdRng::seed_from_u64(42);
    /// let chain = Chain::new(-1, &target)?;
    /// let mut sampler = Sampler::new(chain, &target, &mut proposal, &mut rng);
    ///
    /// let step = sampler.step_delayed()?;
    /// assert!(step.accepted);
    /// assert_eq!(*sampler.chain_ref().state(), 0);
    /// # Ok::<(), DelayedStepError<Infallible>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`DelayedStepError`] when planning, numerical validation, or
    /// accepted-move commit fails.
    pub fn step_delayed(&mut self) -> Result<DelayedStep<P::Info>, DelayedStepError<P::Error>> {
        self.chain
            .step_delayed(self.target, &mut self.proposal, self.rng)
    }

    /// Run `steps` delayed-commit Metropolis-Hastings steps.
    ///
    /// Stops and returns the first error encountered.
    ///
    /// ```
    /// use core::convert::Infallible;
    /// use markov_chain_monte_carlo::prelude::delayed::*;
    /// use rand::{Rng, SeedableRng, rngs::StdRng};
    ///
    /// struct Flat;
    /// impl Target<i32> for Flat {
    ///     fn log_prob(&self, _: &i32) -> f64 { 0.0 }
    /// }
    ///
    /// struct Increment;
    /// impl DelayedProposal<i32> for Increment {
    ///     type Plan = i32;
    ///     type Info = i32;
    ///     type Error = Infallible;
    ///
    ///     fn propose_plan<R: Rng + ?Sized>(
    ///         &mut self,
    ///         _state: &i32,
    ///         _rng: &mut R,
    ///     ) -> Result<Option<i32>, Self::Error> {
    ///         Ok(Some(1))
    ///     }
    ///
    ///     fn proposed_log_prob<T: Target<i32>>(
    ///         &self,
    ///         state: &i32,
    ///         plan: &i32,
    ///         target: &T,
    ///     ) -> Result<f64, Self::Error> {
    ///         Ok(target.log_prob(&(*state + *plan)))
    ///     }
    ///
    ///     fn info(&self, plan: &i32) -> i32 { *plan }
    ///
    ///     fn commit<R: Rng + ?Sized>(
    ///         &mut self,
    ///         state: &mut i32,
    ///         plan: i32,
    ///         _rng: &mut R,
    ///     ) -> Result<(), Self::Error> {
    ///         *state += plan;
    ///         Ok(())
    ///     }
    /// }
    ///
    /// let target = Flat;
    /// let mut proposal = Increment;
    /// let mut rng = StdRng::seed_from_u64(42);
    /// let chain = Chain::new(0, &target)?;
    /// let mut sampler = Sampler::new(chain, &target, &mut proposal, &mut rng);
    ///
    /// sampler.run_delayed(3)?;
    /// assert_eq!(*sampler.chain_ref().state(), 3);
    /// # Ok::<(), DelayedStepError<Infallible>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`DelayedStepError`] on the first step that fails.
    pub fn run_delayed(&mut self, steps: usize) -> Result<(), DelayedStepError<P::Error>> {
        for _ in 0..steps {
            let _ = self.step_delayed()?;
        }
        Ok(())
    }

    /// Perform one delayed-commit step and observe the resulting chain state.
    ///
    /// Returns delayed-step telemetry together with the measurement.
    ///
    /// ```
    /// use core::convert::Infallible;
    /// use markov_chain_monte_carlo::prelude::delayed::*;
    /// use rand::{Rng, SeedableRng, rngs::StdRng};
    ///
    /// # struct Flat;
    /// # impl Target<i32> for Flat { fn log_prob(&self, _: &i32) -> f64 { 0.0 } }
    /// # struct Increment;
    /// # impl DelayedProposal<i32> for Increment {
    /// #     type Plan = i32;
    /// #     type Info = i32;
    /// #     type Error = Infallible;
    /// #     fn propose_plan<R: Rng + ?Sized>(&mut self, _: &i32, _: &mut R) -> Result<Option<i32>, Self::Error> { Ok(Some(1)) }
    /// #     fn proposed_log_prob<T: Target<i32>>(&self, s: &i32, p: &i32, t: &T) -> Result<f64, Self::Error> { Ok(t.log_prob(&(*s + *p))) }
    /// #     fn info(&self, plan: &i32) -> i32 { *plan }
    /// #     fn commit<R: Rng + ?Sized>(&mut self, state: &mut i32, plan: i32, _: &mut R) -> Result<(), Self::Error> { *state += plan; Ok(()) }
    /// # }
    /// let target = Flat;
    /// let mut proposal = Increment;
    /// let mut rng = StdRng::seed_from_u64(42);
    /// let chain = Chain::new(0, &target)?;
    /// let mut sampler = Sampler::new(chain, &target, &mut proposal, &mut rng);
    /// let mut coordinate = |state: &i32| *state;
    ///
    /// let (_step, sample) = sampler.step_delayed_observing(&mut coordinate)?;
    /// assert_eq!(sample, 1);
    /// # Ok::<(), DelayedStepError<Infallible>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`DelayedStepError`] if the step fails.  In that case the
    /// observable is not invoked.
    pub fn step_delayed_observing<O: Observable<S> + ?Sized>(
        &mut self,
        observable: &mut O,
    ) -> ObservedDelayedStepResult<P::Info, O::Output, P::Error> {
        let step = self.step_delayed()?;
        Ok((step, observable.observe(self.chain.state())))
    }

    /// Run delayed-commit steps and collect one observation after each step.
    ///
    /// ```
    /// use core::convert::Infallible;
    /// use markov_chain_monte_carlo::prelude::delayed::*;
    /// use rand::{Rng, SeedableRng, rngs::StdRng};
    ///
    /// struct Flat;
    /// impl Target<i32> for Flat {
    ///     fn log_prob(&self, _: &i32) -> f64 { 0.0 }
    /// }
    ///
    /// struct Increment;
    /// impl DelayedProposal<i32> for Increment {
    ///     type Plan = i32;
    ///     type Info = i32;
    ///     type Error = Infallible;
    ///
    ///     fn propose_plan<R: Rng + ?Sized>(
    ///         &mut self,
    ///         _state: &i32,
    ///         _rng: &mut R,
    ///     ) -> Result<Option<i32>, Self::Error> {
    ///         Ok(Some(1))
    ///     }
    ///
    ///     fn proposed_log_prob<T: Target<i32>>(
    ///         &self,
    ///         state: &i32,
    ///         plan: &i32,
    ///         target: &T,
    ///     ) -> Result<f64, Self::Error> {
    ///         Ok(target.log_prob(&(*state + *plan)))
    ///     }
    ///
    ///     fn info(&self, plan: &i32) -> i32 { *plan }
    ///
    ///     fn commit<R: Rng + ?Sized>(
    ///         &mut self,
    ///         state: &mut i32,
    ///         plan: i32,
    ///         _rng: &mut R,
    ///     ) -> Result<(), Self::Error> {
    ///         *state += plan;
    ///         Ok(())
    ///     }
    /// }
    ///
    /// let target = Flat;
    /// let mut proposal = Increment;
    /// let mut rng = StdRng::seed_from_u64(42);
    /// let chain = Chain::new(0, &target)?;
    /// let mut sampler = Sampler::new(chain, &target, &mut proposal, &mut rng);
    /// let mut coordinate = |state: &i32| *state;
    ///
    /// let samples = sampler.run_delayed_observing(3, &mut coordinate)?;
    /// assert_eq!(samples.as_slice(), &[1, 2, 3]);
    /// # Ok::<(), DelayedStepError<Infallible>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`DelayedStepError`] on the first step that fails.
    pub fn run_delayed_observing<O: Observable<S> + ?Sized>(
        &mut self,
        steps: usize,
        observable: &mut O,
    ) -> Result<SampleBuffer<O::Output>, DelayedStepError<P::Error>> {
        let mut samples = SampleBuffer::with_capacity(steps);
        for _ in 0..steps {
            let (_, sample) = self.step_delayed_observing(observable)?;
            samples.push(sample);
        }
        Ok(samples)
    }

    /// Perform one delayed-commit step and fallibly observe the resulting state.
    ///
    /// ```
    /// use core::convert::Infallible;
    /// use markov_chain_monte_carlo::prelude::delayed::*;
    /// use rand::{Rng, SeedableRng, rngs::StdRng};
    ///
    /// # struct Flat;
    /// # impl Target<i32> for Flat { fn log_prob(&self, _: &i32) -> f64 { 0.0 } }
    /// # struct Increment;
    /// # impl DelayedProposal<i32> for Increment {
    /// #     type Plan = i32;
    /// #     type Info = i32;
    /// #     type Error = Infallible;
    /// #     fn propose_plan<R: Rng + ?Sized>(&mut self, _: &i32, _: &mut R) -> Result<Option<i32>, Self::Error> { Ok(Some(1)) }
    /// #     fn proposed_log_prob<T: Target<i32>>(&self, s: &i32, p: &i32, t: &T) -> Result<f64, Self::Error> { Ok(t.log_prob(&(*s + *p))) }
    /// #     fn info(&self, plan: &i32) -> i32 { *plan }
    /// #     fn commit<R: Rng + ?Sized>(&mut self, state: &mut i32, plan: i32, _: &mut R) -> Result<(), Self::Error> { *state += plan; Ok(()) }
    /// # }
    /// let target = Flat;
    /// let mut proposal = Increment;
    /// let mut rng = StdRng::seed_from_u64(42);
    /// let chain = Chain::new(0, &target)
    ///     .map_err(DelayedStepError::Mcmc)
    ///     .map_err(ObservedStepError::Step)?;
    /// let mut sampler = Sampler::new(chain, &target, &mut proposal, &mut rng);
    /// let mut positive = |state: &i32| {
    ///     if *state > 0 { Ok(*state) } else { Err("not positive") }
    /// };
    ///
    /// let (_step, sample) = sampler.try_step_delayed_observing(&mut positive)?;
    /// assert_eq!(sample, 1);
    /// # Ok::<(), ObservedStepError<DelayedStepError<Infallible>, &'static str>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`ObservedStepError`] when either the delayed sampling step or
    /// the observable fails.
    pub fn try_step_delayed_observing<O: TryObservable<S> + ?Sized>(
        &mut self,
        observable: &mut O,
    ) -> TryObservedDelayedStepResult<P::Info, O::Output, P::Error, O::Error> {
        let step = self.step_delayed().map_err(ObservedStepError::Step)?;
        let sample = observable
            .try_observe(self.chain.state())
            .map_err(ObservedStepError::Observation)?;
        Ok((step, sample))
    }

    /// Run delayed-commit steps and collect fallible observations.
    ///
    /// ```
    /// use core::convert::Infallible;
    /// use markov_chain_monte_carlo::prelude::delayed::*;
    /// use rand::{Rng, SeedableRng, rngs::StdRng};
    ///
    /// # struct Flat;
    /// # impl Target<i32> for Flat { fn log_prob(&self, _: &i32) -> f64 { 0.0 } }
    /// # struct Increment;
    /// # impl DelayedProposal<i32> for Increment {
    /// #     type Plan = i32;
    /// #     type Info = i32;
    /// #     type Error = Infallible;
    /// #     fn propose_plan<R: Rng + ?Sized>(&mut self, _: &i32, _: &mut R) -> Result<Option<i32>, Self::Error> { Ok(Some(1)) }
    /// #     fn proposed_log_prob<T: Target<i32>>(&self, s: &i32, p: &i32, t: &T) -> Result<f64, Self::Error> { Ok(t.log_prob(&(*s + *p))) }
    /// #     fn info(&self, plan: &i32) -> i32 { *plan }
    /// #     fn commit<R: Rng + ?Sized>(&mut self, state: &mut i32, plan: i32, _: &mut R) -> Result<(), Self::Error> { *state += plan; Ok(()) }
    /// # }
    /// let target = Flat;
    /// let mut proposal = Increment;
    /// let mut rng = StdRng::seed_from_u64(42);
    /// let chain = Chain::new(0, &target)
    ///     .map_err(DelayedStepError::Mcmc)
    ///     .map_err(ObservedStepError::Step)?;
    /// let mut sampler = Sampler::new(chain, &target, &mut proposal, &mut rng);
    /// let mut positive = |state: &i32| {
    ///     if *state > 0 { Ok(*state) } else { Err("not positive") }
    /// };
    ///
    /// let samples = sampler.try_run_delayed_observing(2, &mut positive)?;
    /// assert_eq!(samples.as_slice(), &[1, 2]);
    /// # Ok::<(), ObservedStepError<DelayedStepError<Infallible>, &'static str>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`ObservedStepError`] on the first step or observation failure.
    pub fn try_run_delayed_observing<O: TryObservable<S> + ?Sized>(
        &mut self,
        steps: usize,
        observable: &mut O,
    ) -> TryObservedDelayedRunResult<O::Output, P::Error, O::Error> {
        let mut samples = SampleBuffer::with_capacity(steps);
        for _ in 0..steps {
            let (_, sample) = self.try_step_delayed_observing(observable)?;
            samples.push(sample);
        }
        Ok(samples)
    }
}

// --- Iterator (by-value proposal path only) ---

impl<S, T: Target<S>, P: Proposal<S>, R: Rng + ?Sized> Iterator for Sampler<'_, S, T, P, R> {
    type Item = Result<(), McmcError>;

    /// Perform one by-value step and yield the result.
    ///
    /// This is an infinite iterator — it always returns `Some`.
    /// Use `.take(n)` to limit the number of steps.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::by_value::*;
    /// use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};
    ///
    /// #[derive(Clone)]
    /// struct Val(f64);
    /// struct Flat;
    /// impl Target<Val> for Flat {
    ///     fn log_prob(&self, _: &Val) -> f64 { 0.0 }
    /// }
    /// struct Walk;
    /// impl Proposal<Val> for Walk {
    ///     fn propose<R: Rng + ?Sized>(&self, c: &Val, r: &mut R) -> Val {
    ///         Val(c.0 + r.random_range(-1.0..1.0))
    ///     }
    /// }
    ///
    /// let mut rng = StdRng::seed_from_u64(42);
    /// let chain = Chain::new(Val(0.0), &Flat)?;
    /// let mut sampler = Sampler::new(chain, &Flat, &Walk, &mut rng);
    ///
    /// // Take exactly 100 steps using iterator
    /// let errors: Vec<_> = sampler.by_ref().take(100).filter_map(Result::err).collect();
    /// assert!(errors.is_empty());
    /// assert_eq!(sampler.chain_ref().total_steps(), 100);
    /// # Ok::<(), McmcError>(())
    /// ```
    fn next(&mut self) -> Option<Self::Item> {
        Some(self.step())
    }
}

#[cfg(test)]
mod tests {
    use core::convert::Infallible;

    use super::*;
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    // --- Shared fixtures ---

    #[derive(Clone, Debug, PartialEq)]
    struct Scalar(f64);

    struct Normal;
    impl Target<Scalar> for Normal {
        fn log_prob(&self, s: &Scalar) -> f64 {
            -0.5 * s.0 * s.0
        }
    }

    struct Walk {
        width: f64,
    }
    impl Proposal<Scalar> for Walk {
        fn propose<R: Rng + ?Sized>(&self, c: &Scalar, rng: &mut R) -> Scalar {
            Scalar(c.0 + rng.random_range(-self.width..self.width))
        }
    }

    /// Non-Clone state for testing `step_mut` / `run_mut`.
    #[derive(Debug, PartialEq)]
    struct MutScalar(f64);

    impl Target<MutScalar> for Normal {
        fn log_prob(&self, s: &MutScalar) -> f64 {
            -0.5 * s.0 * s.0
        }
    }

    struct MutWalk {
        width: f64,
    }
    impl ProposalMut<MutScalar> for MutWalk {
        type Undo = f64;
        fn propose_mut<R: Rng + ?Sized>(&self, state: &mut MutScalar, rng: &mut R) -> Option<f64> {
            let old = state.0;
            state.0 += rng.random_range(-self.width..self.width);
            Some(old)
        }
        fn undo(&self, state: &mut MutScalar, old: f64) {
            state.0 = old;
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum ObservationFailure {
        Failed,
    }

    // --- Construction ---

    #[test]
    fn new_and_into_chain() {
        let mut rng = StdRng::seed_from_u64(42);
        let chain = Chain::new(Scalar(1.0), &Normal).unwrap();
        let sampler = Sampler::new(chain, &Normal, &Walk { width: 1.0 }, &mut rng);
        let chain = sampler.into_chain();
        assert_eq!(chain.state, Scalar(1.0));
    }

    // --- Debug ---

    #[test]
    fn debug_output_contains_chain() {
        let mut rng = StdRng::seed_from_u64(42);
        let chain = Chain::new(Scalar(1.0), &Normal).unwrap();
        let sampler = Sampler::new(chain, &Normal, &Walk { width: 1.0 }, &mut rng);
        let debug = format!("{sampler:?}");
        assert!(debug.contains("Sampler"));
        assert!(debug.contains("chain"));
    }

    // --- By-value: step and run ---

    #[test]
    fn step_advances_chain() {
        let mut rng = StdRng::seed_from_u64(42);
        let chain = Chain::new(Scalar(0.0), &Normal).unwrap();
        let mut sampler = Sampler::new(chain, &Normal, &Walk { width: 1.0 }, &mut rng);

        sampler.step().unwrap();
        assert_eq!(sampler.chain_ref().total_steps(), 1);
    }

    #[test]
    fn run_n_steps() {
        let mut rng = StdRng::seed_from_u64(42);
        let chain = Chain::new(Scalar(0.0), &Normal).unwrap();
        let mut sampler = Sampler::new(chain, &Normal, &Walk { width: 1.0 }, &mut rng);

        sampler.run(500).unwrap();
        assert_eq!(sampler.chain_ref().total_steps(), 500);
    }

    #[test]
    fn run_observing_collects() {
        let mut rng = StdRng::seed_from_u64(42);
        let chain = Chain::new(Scalar(0.0), &Normal).unwrap();
        let mut sampler = Sampler::new(chain, &Normal, &Walk { width: 1.0 }, &mut rng);
        let mut energy = |state: &Scalar| 0.5 * state.0 * state.0;

        let measurements = sampler.run_observing(25, &mut energy).unwrap();

        assert_eq!(measurements.len(), 25);
        assert_eq!(sampler.chain_ref().total_steps(), 25);
        assert!(measurements.iter().all(|value| *value >= 0.0));
    }

    #[test]
    fn try_run_observing_reports_error() {
        let mut rng = StdRng::seed_from_u64(42);
        let chain = Chain::new(Scalar(0.0), &Normal).unwrap();
        let mut sampler = Sampler::new(chain, &Normal, &Walk { width: 1.0 }, &mut rng);
        let mut calls = 0;
        let mut observable = |state: &Scalar| {
            calls += 1;
            if calls == 2 {
                Err(ObservationFailure::Failed)
            } else {
                Ok(state.0)
            }
        };

        let result = sampler.try_run_observing(3, &mut observable);

        assert!(matches!(
            result,
            Err(ObservedStepError::Observation(ObservationFailure::Failed))
        ));
        assert_eq!(sampler.chain_ref().total_steps(), 2);
    }

    #[test]
    fn try_step_observing_skips_error() {
        struct NanProposal;
        impl Proposal<Scalar> for NanProposal {
            fn propose<R: Rng + ?Sized>(&self, _current: &Scalar, _rng: &mut R) -> Scalar {
                Scalar(0.0)
            }

            fn log_q_ratio(&self, _current: &Scalar, _proposed: &Scalar) -> f64 {
                f64::NAN
            }
        }

        let mut rng = StdRng::seed_from_u64(42);
        let chain = Chain::new(Scalar(0.0), &Normal).unwrap();
        let mut sampler = Sampler::new(chain, &Normal, &NanProposal, &mut rng);
        let mut observed = false;
        let mut observable = |state: &Scalar| {
            observed = true;
            Ok::<f64, ObservationFailure>(state.0)
        };

        let result = sampler.try_step_observing(&mut observable);

        assert!(matches!(
            result,
            Err(ObservedStepError::Step(McmcError::NanLogQRatio))
        ));
        assert!(!observed);
    }

    // --- In-place: step_mut and run_mut ---

    #[test]
    fn step_mut_advances_chain() {
        let mut rng = StdRng::seed_from_u64(42);
        let chain = Chain::new(MutScalar(0.0), &Normal).unwrap();
        let mut sampler = Sampler::new(chain, &Normal, &MutWalk { width: 1.0 }, &mut rng);

        sampler.step_mut().unwrap();
        assert_eq!(sampler.chain_ref().total_steps(), 1);
    }

    #[test]
    fn run_mut_n_steps() {
        let mut rng = StdRng::seed_from_u64(42);
        let chain = Chain::new(MutScalar(0.0), &Normal).unwrap();
        let mut sampler = Sampler::new(chain, &Normal, &MutWalk { width: 1.0 }, &mut rng);

        sampler.run_mut(500).unwrap();
        assert_eq!(sampler.chain_ref().total_steps(), 500);
    }

    #[test]
    fn run_mut_observing_collects() {
        let mut rng = StdRng::seed_from_u64(42);
        let chain = Chain::new(MutScalar(0.0), &Normal).unwrap();
        let mut sampler = Sampler::new(chain, &Normal, &MutWalk { width: 1.0 }, &mut rng);
        let mut coordinate = |state: &MutScalar| state.0;

        let measurements = sampler.run_mut_observing(25, &mut coordinate).unwrap();

        assert_eq!(measurements.len(), 25);
        assert_eq!(sampler.chain_ref().total_steps(), 25);
    }

    #[test]
    fn try_run_mut_observing_errors() {
        let mut rng = StdRng::seed_from_u64(42);
        let chain = Chain::new(MutScalar(0.0), &Normal).unwrap();
        let mut sampler = Sampler::new(chain, &Normal, &MutWalk { width: 1.0 }, &mut rng);
        let mut observable = |_state: &MutScalar| Err::<f64, _>(ObservationFailure::Failed);

        let result = sampler.try_run_mut_observing(1, &mut observable);

        assert!(matches!(
            result,
            Err(ObservedStepError::Observation(ObservationFailure::Failed))
        ));
        assert_eq!(sampler.chain_ref().total_steps(), 1);
    }

    // --- Delayed commit: step_delayed and run_delayed ---

    struct DelayedToZero;
    impl DelayedProposal<MutScalar> for DelayedToZero {
        type Plan = f64;
        type Info = f64;
        type Error = Infallible;

        fn propose_plan<R: Rng + ?Sized>(
            &mut self,
            _state: &MutScalar,
            _rng: &mut R,
        ) -> Result<Option<f64>, Self::Error> {
            Ok(Some(0.0))
        }

        fn proposed_log_prob<T: Target<MutScalar>>(
            &self,
            _state: &MutScalar,
            plan: &f64,
            target: &T,
        ) -> Result<f64, Self::Error> {
            Ok(target.log_prob(&MutScalar(*plan)))
        }

        fn info(&self, plan: &f64) -> f64 {
            *plan
        }

        fn commit<R: Rng + ?Sized>(
            &mut self,
            state: &mut MutScalar,
            plan: f64,
            _rng: &mut R,
        ) -> Result<(), Self::Error> {
            state.0 = plan;
            Ok(())
        }
    }

    #[test]
    fn step_delayed_advances_chain() {
        let mut rng = StdRng::seed_from_u64(42);
        let chain = Chain::new(MutScalar(2.0), &Normal).unwrap();
        let mut proposal = DelayedToZero;
        let mut sampler = Sampler::new(chain, &Normal, &mut proposal, &mut rng);

        let step = sampler.step_delayed().unwrap();

        assert!(step.accepted);
        assert_eq!(sampler.chain_ref().state(), &MutScalar(0.0));
        assert_eq!(sampler.chain_ref().total_steps(), 1);
    }

    #[test]
    fn run_delayed_n_steps() {
        let mut rng = StdRng::seed_from_u64(42);
        let chain = Chain::new(MutScalar(2.0), &Normal).unwrap();
        let mut proposal = DelayedToZero;
        let mut sampler = Sampler::new(chain, &Normal, &mut proposal, &mut rng);

        sampler.run_delayed(10).unwrap();

        assert_eq!(sampler.chain_ref().state(), &MutScalar(0.0));
        assert_eq!(sampler.chain_ref().total_steps(), 10);
    }

    #[test]
    fn run_delayed_observing_collects() {
        let mut rng = StdRng::seed_from_u64(42);
        let chain = Chain::new(MutScalar(2.0), &Normal).unwrap();
        let mut proposal = DelayedToZero;
        let mut sampler = Sampler::new(chain, &Normal, &mut proposal, &mut rng);
        let mut coordinate = |state: &MutScalar| state.0;

        let measurements = sampler.run_delayed_observing(5, &mut coordinate).unwrap();

        assert_eq!(measurements.as_slice(), &[0.0; 5]);
        assert_eq!(sampler.chain_ref().total_steps(), 5);
    }

    #[test]
    fn try_run_delayed_observing_errors() {
        let mut rng = StdRng::seed_from_u64(42);
        let chain = Chain::new(MutScalar(2.0), &Normal).unwrap();
        let mut proposal = DelayedToZero;
        let mut sampler = Sampler::new(chain, &Normal, &mut proposal, &mut rng);
        let mut observable = |_state: &MutScalar| Err::<f64, _>(ObservationFailure::Failed);

        let result = sampler.try_run_delayed_observing(1, &mut observable);

        assert!(matches!(
            result,
            Err(ObservedStepError::Observation(ObservationFailure::Failed))
        ));
        assert_eq!(sampler.chain_ref().total_steps(), 1);
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum DelayedRunError {
        PlannedStop,
    }

    struct StopAfterFirstDelayed {
        calls: usize,
    }

    impl DelayedProposal<MutScalar> for StopAfterFirstDelayed {
        type Plan = f64;
        type Info = f64;
        type Error = DelayedRunError;

        fn propose_plan<R: Rng + ?Sized>(
            &mut self,
            _state: &MutScalar,
            _rng: &mut R,
        ) -> Result<Option<f64>, Self::Error> {
            self.calls += 1;
            if self.calls == 1 {
                Ok(Some(0.0))
            } else {
                Err(DelayedRunError::PlannedStop)
            }
        }

        fn proposed_log_prob<T: Target<MutScalar>>(
            &self,
            _state: &MutScalar,
            plan: &f64,
            target: &T,
        ) -> Result<f64, Self::Error> {
            Ok(target.log_prob(&MutScalar(*plan)))
        }

        fn info(&self, plan: &f64) -> f64 {
            *plan
        }

        fn commit<R: Rng + ?Sized>(
            &mut self,
            state: &mut MutScalar,
            plan: f64,
            _rng: &mut R,
        ) -> Result<(), Self::Error> {
            state.0 = plan;
            Ok(())
        }
    }

    #[test]
    fn run_delayed_stops_on_first_error() {
        let mut rng = StdRng::seed_from_u64(42);
        let chain = Chain::new(MutScalar(2.0), &Normal).unwrap();
        let mut proposal = StopAfterFirstDelayed { calls: 0 };
        let mut sampler = Sampler::new(chain, &Normal, &mut proposal, &mut rng);

        let result = sampler.run_delayed(3);

        assert!(matches!(
            result,
            Err(DelayedStepError::Plan(DelayedRunError::PlannedStop))
        ));
        assert_eq!(sampler.chain_ref().state(), &MutScalar(0.0));
        assert_eq!(sampler.chain_ref().total_steps(), 1);
    }

    // --- Iterator ---

    #[test]
    fn iterator_take_n() {
        let mut rng = StdRng::seed_from_u64(42);
        let chain = Chain::new(Scalar(0.0), &Normal).unwrap();
        let mut sampler = Sampler::new(chain, &Normal, &Walk { width: 1.0 }, &mut rng);

        // Use Iterator::take for burn-in
        for result in sampler.by_ref().take(200) {
            result.unwrap();
        }
        assert_eq!(sampler.chain_ref().total_steps(), 200);
    }

    #[test]
    fn iterator_is_infinite() {
        let mut rng = StdRng::seed_from_u64(42);
        let chain = Chain::new(Scalar(0.0), &Normal).unwrap();
        let mut sampler = Sampler::new(chain, &Normal, &Walk { width: 1.0 }, &mut rng);

        // next() always returns Some
        assert!(sampler.next().is_some());
        assert!(sampler.next().is_some());
        assert!(sampler.next().is_some());
    }

    // --- Equivalence: sampler produces same results as raw chain ---

    #[test]
    fn sampler_matches_raw_chain() {
        let proposal = Walk { width: 1.0 };
        let steps = 100;

        // Raw chain
        let mut chain = Chain::new(Scalar(0.0), &Normal).unwrap();
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..steps {
            chain.step(&Normal, &proposal, &mut rng).unwrap();
        }

        // Sampler
        let chain2 = Chain::new(Scalar(0.0), &Normal).unwrap();
        let mut rng2 = StdRng::seed_from_u64(42);
        let mut sampler = Sampler::new(chain2, &Normal, &proposal, &mut rng2);
        sampler.run(steps).unwrap();

        assert_eq!(chain.state, *sampler.chain_ref().state());
        assert_eq!(chain.accepted(), sampler.chain_ref().accepted());
        assert_eq!(chain.rejected(), sampler.chain_ref().rejected());
    }

    #[test]
    fn sampler_mut_matches_raw_chain() {
        let proposal = MutWalk { width: 1.0 };
        let steps = 100;

        // Raw chain
        let mut chain = Chain::new(MutScalar(0.0), &Normal).unwrap();
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..steps {
            chain.step_mut(&Normal, &proposal, &mut rng).unwrap();
        }

        // Sampler
        let chain2 = Chain::new(MutScalar(0.0), &Normal).unwrap();
        let mut rng2 = StdRng::seed_from_u64(42);
        let mut sampler = Sampler::new(chain2, &Normal, &proposal, &mut rng2);
        sampler.run_mut(steps).unwrap();

        assert_eq!(chain.state, *sampler.chain_ref().state());
        assert_eq!(chain.accepted(), sampler.chain_ref().accepted());
        assert_eq!(chain.rejected(), sampler.chain_ref().rejected());
    }
}
