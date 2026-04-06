//! Ergonomic sampling wrapper that bundles a chain with its components.

use std::fmt;

use rand::Rng;

use crate::{Chain, McmcError, Proposal, ProposalMut, Target};

/// Bundles a [`Chain`] with its target, proposal, and RNG for ergonomic
/// sampling.
///
/// `Sampler` owns the chain and borrows the target, proposal, and RNG.
/// It provides [`step`](Self::step) / [`step_mut`](Self::step_mut) for
/// single Metropolis–Hastings steps and [`run`](Self::run) /
/// [`run_mut`](Self::run_mut) for bulk sampling.
///
/// For the clone-based path (`P: Proposal<S>`), `Sampler` also implements
/// [`Iterator`], yielding `Result<(), McmcError>` on each step.
///
/// # Example
///
/// ```
/// use markov_chain_monte_carlo::prelude::*;
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
/// sampler.chain.reset_counters();
///
/// // Production sampling
/// sampler.run(10_000)?;
/// assert!(sampler.chain.acceptance_rate() > 0.0);
/// # Ok::<(), McmcError>(())
/// ```
#[must_use]
pub struct Sampler<'a, S, T, P, R: ?Sized> {
    /// The MCMC chain being sampled.
    pub chain: Chain<S>,
    target: &'a T,
    proposal: &'a P,
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
    pub const fn new(chain: Chain<S>, target: &'a T, proposal: &'a P, rng: &'a mut R) -> Self {
        Self {
            chain,
            target,
            proposal,
            rng,
        }
    }

    /// Consume the sampler and return the inner chain.
    pub fn into_chain(self) -> Chain<S> {
        self.chain
    }
}

// --- Clone-based stepping ---

impl<S, T, P, R> Sampler<'_, S, T, P, R>
where
    S: Clone,
    T: Target<S>,
    P: Proposal<S>,
    R: Rng + ?Sized,
{
    /// Perform a single clone-based Metropolis–Hastings step.
    ///
    /// Delegates to [`Chain::step`].
    ///
    /// # Errors
    ///
    /// Returns [`McmcError`] on NaN or +∞ log-probability or NaN log q-ratio.
    pub fn step(&mut self) -> Result<(), McmcError> {
        self.chain.step(self.target, self.proposal, self.rng)
    }

    /// Run `steps` clone-based Metropolis–Hastings steps.
    ///
    /// Stops and returns the first error encountered.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::*;
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
    /// assert_eq!(sampler.chain.total_steps(), 1000);
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
}

// --- In-place stepping ---

impl<S, T, P, R> Sampler<'_, S, T, P, R>
where
    T: Target<S>,
    P: ProposalMut<S>,
    R: Rng + ?Sized,
{
    /// Perform a single in-place Metropolis–Hastings step with rollback.
    ///
    /// Delegates to [`Chain::step_mut`].  Returns `Ok(true)` if accepted,
    /// `Ok(false)` if rejected.
    ///
    /// # Errors
    ///
    /// Returns [`McmcError`] on NaN or +∞ log-probability or NaN log q-ratio
    /// (state is rolled back before the error is returned).
    pub fn step_mut(&mut self) -> Result<bool, McmcError> {
        self.chain.step_mut(self.target, self.proposal, self.rng)
    }

    /// Run `steps` in-place Metropolis–Hastings steps.
    ///
    /// Stops and returns the first error encountered.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::*;
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
    /// assert_eq!(sampler.chain.total_steps(), 1000);
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
}

// --- Iterator (clone-based path only) ---

impl<S, T, P, R> Iterator for Sampler<'_, S, T, P, R>
where
    S: Clone,
    T: Target<S>,
    P: Proposal<S>,
    R: Rng + ?Sized,
{
    type Item = Result<(), McmcError>;

    /// Perform one clone-based step and yield the result.
    ///
    /// This is an infinite iterator — it always returns `Some`.
    /// Use `.take(n)` to limit the number of steps.
    fn next(&mut self) -> Option<Self::Item> {
        Some(self.step())
    }
}

#[cfg(test)]
mod tests {
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

    // --- Construction ---

    #[test]
    fn new_and_into_chain() {
        let mut rng = StdRng::seed_from_u64(42);
        let chain = Chain::new(Scalar(1.0), &Normal).unwrap();
        let sampler = Sampler::new(chain, &Normal, &Walk { width: 1.0 }, &mut rng);
        let chain = sampler.into_chain();
        assert_eq!(chain.state, Scalar(1.0));
    }

    // --- Clone-based: step and run ---

    #[test]
    fn step_advances_chain() {
        let mut rng = StdRng::seed_from_u64(42);
        let chain = Chain::new(Scalar(0.0), &Normal).unwrap();
        let mut sampler = Sampler::new(chain, &Normal, &Walk { width: 1.0 }, &mut rng);

        sampler.step().unwrap();
        assert_eq!(sampler.chain.total_steps(), 1);
    }

    #[test]
    fn run_n_steps() {
        let mut rng = StdRng::seed_from_u64(42);
        let chain = Chain::new(Scalar(0.0), &Normal).unwrap();
        let mut sampler = Sampler::new(chain, &Normal, &Walk { width: 1.0 }, &mut rng);

        sampler.run(500).unwrap();
        assert_eq!(sampler.chain.total_steps(), 500);
    }

    // --- In-place: step_mut and run_mut ---

    #[test]
    fn step_mut_advances_chain() {
        let mut rng = StdRng::seed_from_u64(42);
        let chain = Chain::new(MutScalar(0.0), &Normal).unwrap();
        let mut sampler = Sampler::new(chain, &Normal, &MutWalk { width: 1.0 }, &mut rng);

        sampler.step_mut().unwrap();
        assert_eq!(sampler.chain.total_steps(), 1);
    }

    #[test]
    fn run_mut_n_steps() {
        let mut rng = StdRng::seed_from_u64(42);
        let chain = Chain::new(MutScalar(0.0), &Normal).unwrap();
        let mut sampler = Sampler::new(chain, &Normal, &MutWalk { width: 1.0 }, &mut rng);

        sampler.run_mut(500).unwrap();
        assert_eq!(sampler.chain.total_steps(), 500);
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
        assert_eq!(sampler.chain.total_steps(), 200);
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

        assert_eq!(chain.state, sampler.chain.state);
        assert_eq!(chain.accepted(), sampler.chain.accepted());
        assert_eq!(chain.rejected(), sampler.chain.rejected());
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

        assert_eq!(chain.state, sampler.chain.state);
        assert_eq!(chain.accepted(), sampler.chain.accepted());
    }
}
