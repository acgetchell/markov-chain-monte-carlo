//! MCMC chain implementation.

use rand::{Rng, RngExt};

use crate::{McmcError, Proposal, ProposalMut, Target};

/// A single MCMC chain.
#[derive(Debug)]
#[must_use]
pub struct Chain<S> {
    /// Current state.
    pub(crate) state: S,
    /// Current log-probability of the state.
    log_prob: f64,
    /// Number of accepted moves.
    accepted: usize,
    /// Number of rejected moves.
    rejected: usize,
}

impl<S> Chain<S> {
    /// Create a new chain from an initial state.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::*;
    ///
    /// # struct T;
    /// # impl Target<f64> for T { fn log_prob(&self, x: &f64) -> f64 { -0.5 * x * x } }
    /// let chain = Chain::new(1.0_f64, &T)?;
    /// assert_eq!(chain.accepted(), 0);
    /// assert!((chain.log_prob() - (-0.5)).abs() < 1e-12);
    /// # Ok::<(), McmcError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`McmcError::NanInitialLogProb`] if the target's log-probability
    /// for the initial state is NaN, or [`McmcError::InfiniteInitialLogProb`]
    /// if it is +∞.
    pub fn new<T: Target<S>>(initial: S, target: &T) -> Result<Self, McmcError> {
        let log_prob = target.log_prob(&initial);
        if log_prob.is_nan() {
            return Err(McmcError::NanInitialLogProb);
        }
        if log_prob == f64::INFINITY {
            return Err(McmcError::InfiniteInitialLogProb);
        }
        Ok(Self {
            state: initial,
            log_prob,
            accepted: 0,
            rejected: 0,
        })
    }

    /// Perform a single Metropolis–Hastings step (clone-based).
    ///
    /// This method requires `S: Clone` because the proposal returns a new
    /// state by value.  For non-`Clone` state spaces, use [`step_mut`](Self::step_mut).
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
    /// let mut chain = Chain::new(S(0.0), &T)?;
    /// chain.step(&T, &P, &mut rng)?;
    /// assert_eq!(chain.total_steps(), 1);
    /// # Ok::<(), McmcError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`McmcError::NanProposedLogProb`] or
    /// [`McmcError::InfiniteProposedLogProb`] if the target's log-probability
    /// for the proposed state is NaN or +∞, [`McmcError::NanLogQRatio`] or
    /// [`McmcError::InfiniteLogQRatio`] if the proposal's log q-ratio is
    /// NaN or +∞.
    pub fn step<T, P, R>(&mut self, target: &T, proposal: &P, rng: &mut R) -> Result<(), McmcError>
    where
        S: Clone,
        T: Target<S>,
        P: Proposal<S>,
        R: Rng + ?Sized,
    {
        let proposed = proposal.propose(&self.state, rng);
        let log_prob_new = target.log_prob(&proposed);
        if log_prob_new.is_nan() {
            return Err(McmcError::NanProposedLogProb);
        }
        if log_prob_new == f64::INFINITY {
            return Err(McmcError::InfiniteProposedLogProb);
        }

        let log_q = proposal.log_q_ratio(&self.state, &proposed);
        if log_q.is_nan() {
            return Err(McmcError::NanLogQRatio);
        }
        if log_q == f64::INFINITY {
            return Err(McmcError::InfiniteLogQRatio);
        }

        let log_alpha = log_prob_new - self.log_prob + log_q;

        let accept = if log_alpha >= 0.0 {
            true
        } else {
            rng.random::<f64>() < log_alpha.exp()
        };

        if accept {
            self.state = proposed;
            self.log_prob = log_prob_new;
            self.accepted += 1;
        } else {
            self.rejected += 1;
        }
        Ok(())
    }

    /// Perform a single Metropolis–Hastings step (in-place with rollback).
    ///
    /// Unlike [`step`](Self::step), this method does not require `S: Clone`.
    /// The proposal mutates the state in place and returns an undo token;
    /// on rejection (or NaN error) the state is rolled back automatically.
    ///
    /// Returns `Ok(true)` if the move was accepted, `Ok(false)` if rejected
    /// (including when the proposal returns `None`).
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::*;
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
    /// let mut chain = Chain::new(S(0.0), &T)?;
    /// let accepted = chain.step_mut(&T, &P, &mut rng)?;
    /// assert_eq!(chain.total_steps(), 1);
    /// # Ok::<(), McmcError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`McmcError::NanProposedLogProb`],
    /// [`McmcError::InfiniteProposedLogProb`], [`McmcError::NanLogQRatio`],
    /// or [`McmcError::InfiniteLogQRatio`] after rolling back the state.
    pub fn step_mut<T, P, R>(
        &mut self,
        target: &T,
        proposal: &P,
        rng: &mut R,
    ) -> Result<bool, McmcError>
    where
        T: Target<S>,
        P: ProposalMut<S>,
        R: Rng + ?Sized,
    {
        let Some(token) = proposal.propose_mut(&mut self.state, rng) else {
            self.rejected += 1;
            return Ok(false);
        };

        let log_prob_new = target.log_prob(&self.state);
        if log_prob_new.is_nan() {
            proposal.undo(&mut self.state, token);
            return Err(McmcError::NanProposedLogProb);
        }
        if log_prob_new == f64::INFINITY {
            proposal.undo(&mut self.state, token);
            return Err(McmcError::InfiniteProposedLogProb);
        }

        let log_q = proposal.log_q_ratio(&self.state, &token);
        if log_q.is_nan() {
            proposal.undo(&mut self.state, token);
            return Err(McmcError::NanLogQRatio);
        }
        if log_q == f64::INFINITY {
            proposal.undo(&mut self.state, token);
            return Err(McmcError::InfiniteLogQRatio);
        }

        let log_alpha = log_prob_new - self.log_prob + log_q;

        let accept = if log_alpha >= 0.0 {
            true
        } else {
            rng.random::<f64>() < log_alpha.exp()
        };

        if accept {
            self.log_prob = log_prob_new;
            self.accepted += 1;
        } else {
            proposal.undo(&mut self.state, token);
            self.rejected += 1;
        }
        Ok(accept)
    }

    /// Shared reference to the current state.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::*;
    ///
    /// # struct T;
    /// # impl Target<f64> for T { fn log_prob(&self, x: &f64) -> f64 { -0.5 * x * x } }
    /// let chain = Chain::new(1.0_f64, &T)?;
    /// assert!((*chain.state() - 1.0).abs() < f64::EPSILON);
    /// # Ok::<(), McmcError>(())
    /// ```
    #[must_use]
    pub const fn state(&self) -> &S {
        &self.state
    }

    /// Replace the current state and recompute the cached log-probability.
    ///
    /// This is the safe way to externally mutate the chain state.  The
    /// cached `log_prob` is always kept in sync.
    ///
    /// # Errors
    ///
    /// Returns [`McmcError::NanInitialLogProb`] or
    /// [`McmcError::InfiniteInitialLogProb`] if the target's log-probability
    /// for `new_state` is NaN or +∞ (the chain is unchanged on error).
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::*;
    ///
    /// # struct T;
    /// # impl Target<f64> for T { fn log_prob(&self, x: &f64) -> f64 { -0.5 * x * x } }
    /// let mut chain = Chain::new(0.0_f64, &T)?;
    /// chain.replace_state(2.0, &T)?;
    /// assert!((*chain.state() - 2.0).abs() < f64::EPSILON);
    /// assert!((chain.log_prob() - (-2.0)).abs() < 1e-12);
    /// # Ok::<(), McmcError>(())
    /// ```
    pub fn replace_state<T: Target<S>>(
        &mut self,
        new_state: S,
        target: &T,
    ) -> Result<(), McmcError> {
        let lp = target.log_prob(&new_state);
        if lp.is_nan() {
            return Err(McmcError::NanInitialLogProb);
        }
        if lp == f64::INFINITY {
            return Err(McmcError::InfiniteInitialLogProb);
        }
        self.state = new_state;
        self.log_prob = lp;
        Ok(())
    }

    /// Consume the chain and return the state.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::*;
    ///
    /// # struct T;
    /// # impl Target<f64> for T { fn log_prob(&self, x: &f64) -> f64 { -0.5 * x * x } }
    /// let chain = Chain::new(3.0_f64, &T)?;
    /// let state = chain.into_state();
    /// assert!((state - 3.0).abs() < f64::EPSILON);
    /// # Ok::<(), McmcError>(())
    /// ```
    #[must_use]
    pub fn into_state(self) -> S {
        self.state
    }

    /// Current log-probability of the chain state.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::*;
    ///
    /// # struct T;
    /// # impl Target<f64> for T {
    /// #     fn log_prob(&self, x: &f64) -> f64 { -0.5 * x * x }
    /// # }
    /// let chain = Chain::new(1.0_f64, &T)?;
    /// assert!((chain.log_prob() - (-0.5)).abs() < 1e-12);
    /// # Ok::<(), McmcError>(())
    /// ```
    #[must_use]
    pub const fn log_prob(&self) -> f64 {
        self.log_prob
    }

    /// Number of accepted moves.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::*;
    ///
    /// # struct T;
    /// # impl Target<f64> for T { fn log_prob(&self, _: &f64) -> f64 { 0.0 } }
    /// let chain = Chain::new(0.0_f64, &T)?;
    /// assert_eq!(chain.accepted(), 0);
    /// # Ok::<(), McmcError>(())
    /// ```
    #[must_use]
    pub const fn accepted(&self) -> usize {
        self.accepted
    }

    /// Number of rejected moves.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::*;
    ///
    /// # struct T;
    /// # impl Target<f64> for T { fn log_prob(&self, _: &f64) -> f64 { 0.0 } }
    /// let chain = Chain::new(0.0_f64, &T)?;
    /// assert_eq!(chain.rejected(), 0);
    /// # Ok::<(), McmcError>(())
    /// ```
    #[must_use]
    pub const fn rejected(&self) -> usize {
        self.rejected
    }

    /// Total number of steps taken (`accepted + rejected`).
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
    /// let mut chain = Chain::new(S(0.0), &T)?;
    /// for _ in 0..100 {
    ///     chain.step(&T, &P, &mut rng)?;
    /// }
    /// assert_eq!(chain.total_steps(), 100);
    /// assert_eq!(chain.total_steps(), chain.accepted() + chain.rejected());
    /// # Ok::<(), McmcError>(())
    /// ```
    #[must_use]
    pub const fn total_steps(&self) -> usize {
        self.accepted + self.rejected
    }

    /// Acceptance rate of the chain.
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
    /// let mut chain = Chain::new(S(0.0), &T)?;
    /// assert_eq!(chain.acceptance_rate(), 0.0); // no steps yet
    ///
    /// for _ in 0..1000 {
    ///     chain.step(&T, &P, &mut rng)?;
    /// }
    /// assert!(chain.acceptance_rate() > 0.0);
    /// # Ok::<(), McmcError>(())
    /// ```
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "acceptance counts won't exceed 2^52"
    )]
    pub fn acceptance_rate(&self) -> f64 {
        let total = self.accepted + self.rejected;
        if total == 0 {
            0.0
        } else {
            self.accepted as f64 / total as f64
        }
    }

    /// Reset acceptance and rejection counters to zero.
    ///
    /// Useful after burn-in to measure the acceptance rate of the
    /// production phase only.
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
    /// let mut chain = Chain::new(S(0.0), &T)?;
    ///
    /// // Burn-in
    /// for _ in 0..1000 {
    ///     chain.step(&T, &P, &mut rng)?;
    /// }
    ///
    /// chain.reset_counters();
    /// assert_eq!(chain.total_steps(), 0);
    ///
    /// // Production — acceptance rate reflects only this phase
    /// for _ in 0..5000 {
    ///     chain.step(&T, &P, &mut rng)?;
    /// }
    /// assert!(chain.acceptance_rate() > 0.0);
    /// # Ok::<(), McmcError>(())
    /// ```
    pub const fn reset_counters(&mut self) {
        self.accepted = 0;
        self.rejected = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    // --- Test fixtures ---

    #[derive(Clone, Debug, PartialEq)]
    struct Scalar(f64);

    /// Target: standard normal log-density, log p(x) = -x²/2
    struct Normal;
    impl Target<Scalar> for Normal {
        fn log_prob(&self, state: &Scalar) -> f64 {
            -0.5 * state.0 * state.0
        }
    }

    /// Symmetric random-walk proposal: x' = x + U(-width, width)
    struct RandomWalk {
        width: f64,
    }
    impl Proposal<Scalar> for RandomWalk {
        fn propose<R: Rng + ?Sized>(&self, current: &Scalar, rng: &mut R) -> Scalar {
            let delta: f64 = rng.random_range(-self.width..self.width);
            Scalar(current.0 + delta)
        }
    }

    /// Deterministic proposal that always returns a fixed value.
    struct FixedProposal(f64);
    impl Proposal<Scalar> for FixedProposal {
        fn propose<R: Rng + ?Sized>(&self, _current: &Scalar, _rng: &mut R) -> Scalar {
            Scalar(self.0)
        }
    }

    // --- Chain::new ---

    #[test]
    fn new_initial_state() {
        let chain = Chain::new(Scalar(1.0), &Normal).unwrap();
        assert_eq!(chain.state, Scalar(1.0));
        assert_eq!(chain.accepted(), 0);
        assert_eq!(chain.rejected(), 0);
    }

    #[test]
    fn new_initial_log_prob() {
        let chain = Chain::new(Scalar(0.0), &Normal).unwrap();
        assert!(
            (chain.log_prob()).abs() < 1e-12,
            "log_prob at 0 should be 0.0"
        );

        let chain2 = Chain::new(Scalar(1.0), &Normal).unwrap();
        assert!(
            (chain2.log_prob() - (-0.5)).abs() < 1e-12,
            "log_prob at 1 should be -0.5"
        );
    }

    // --- acceptance_rate ---

    #[test]
    fn acceptance_rate_zero_steps() {
        let chain = Chain::new(Scalar(0.0), &Normal).unwrap();
        assert!((chain.acceptance_rate()).abs() < f64::EPSILON);
    }

    // --- MH acceptance logic ---

    #[test]
    fn step_accepts_uphill() {
        // From x=2.0 (log_prob=-2) to x=0.0 (log_prob=0): always accept
        let mut chain = Chain::new(Scalar(2.0), &Normal).unwrap();
        let proposal = FixedProposal(0.0);
        let mut rng = StdRng::seed_from_u64(42);

        chain.step(&Normal, &proposal, &mut rng).unwrap();

        assert_eq!(
            chain.state,
            Scalar(0.0),
            "Should accept move to higher probability"
        );
        assert_eq!(chain.accepted(), 1);
        assert_eq!(chain.rejected(), 0);
    }

    #[test]
    fn step_rejects_downhill() {
        // From x=0.0 (log_prob=0) to x=100.0 (log_prob=-5000): virtually always reject
        let mut chain = Chain::new(Scalar(0.0), &Normal).unwrap();
        let proposal = FixedProposal(100.0);
        let mut rng = StdRng::seed_from_u64(42);

        chain.step(&Normal, &proposal, &mut rng).unwrap();

        assert_eq!(
            chain.state,
            Scalar(0.0),
            "Should reject move to much lower probability"
        );
        assert_eq!(chain.accepted(), 0);
        assert_eq!(chain.rejected(), 1);
    }

    // --- Error handling ---

    #[test]
    fn new_rejects_nan_initial_log_prob() {
        struct NanTarget;
        impl Target<Scalar> for NanTarget {
            fn log_prob(&self, _state: &Scalar) -> f64 {
                f64::NAN
            }
        }
        let result = Chain::new(Scalar(0.0), &NanTarget);
        assert!(matches!(result, Err(McmcError::NanInitialLogProb)));
    }

    #[test]
    fn step_rejects_nan_proposed_log_prob() {
        struct NanAtOrigin;
        impl Target<Scalar> for NanAtOrigin {
            fn log_prob(&self, state: &Scalar) -> f64 {
                if state.0 == 0.0 {
                    f64::NAN
                } else {
                    -0.5 * state.0 * state.0
                }
            }
        }
        let mut chain = Chain::new(Scalar(1.0), &NanAtOrigin).unwrap();
        let proposal = FixedProposal(0.0);
        let mut rng = StdRng::seed_from_u64(42);

        let result = chain.step(&NanAtOrigin, &proposal, &mut rng);
        assert!(matches!(result, Err(McmcError::NanProposedLogProb)));
    }

    #[test]
    fn step_rejects_nan_log_q_ratio() {
        struct NanProposal;
        impl Proposal<Scalar> for NanProposal {
            fn propose<R: Rng + ?Sized>(&self, _current: &Scalar, _rng: &mut R) -> Scalar {
                Scalar(0.0)
            }
            fn log_q_ratio(&self, _current: &Scalar, _proposed: &Scalar) -> f64 {
                f64::NAN
            }
        }
        let mut chain = Chain::new(Scalar(1.0), &Normal).unwrap();
        let mut rng = StdRng::seed_from_u64(42);

        let result = chain.step(&Normal, &NanProposal, &mut rng);
        assert!(matches!(result, Err(McmcError::NanLogQRatio)));
    }

    #[test]
    fn new_rejects_inf_initial_log_prob() {
        struct InfTarget;
        impl Target<Scalar> for InfTarget {
            fn log_prob(&self, _state: &Scalar) -> f64 {
                f64::INFINITY
            }
        }
        let result = Chain::new(Scalar(0.0), &InfTarget);
        assert!(matches!(result, Err(McmcError::InfiniteInitialLogProb)));
    }

    #[test]
    fn step_rejects_inf_proposed_log_prob() {
        struct InfAtOrigin;
        impl Target<Scalar> for InfAtOrigin {
            fn log_prob(&self, state: &Scalar) -> f64 {
                if state.0 == 0.0 {
                    f64::INFINITY
                } else {
                    -0.5 * state.0 * state.0
                }
            }
        }
        let mut chain = Chain::new(Scalar(1.0), &InfAtOrigin).unwrap();
        let proposal = FixedProposal(0.0);
        let mut rng = StdRng::seed_from_u64(42);

        let result = chain.step(&InfAtOrigin, &proposal, &mut rng);
        assert!(matches!(result, Err(McmcError::InfiniteProposedLogProb)));
    }

    #[test]
    fn step_rejects_inf_log_q_ratio() {
        struct InfQProposal;
        impl Proposal<Scalar> for InfQProposal {
            fn propose<R: Rng + ?Sized>(&self, _current: &Scalar, _rng: &mut R) -> Scalar {
                Scalar(0.0)
            }
            fn log_q_ratio(&self, _current: &Scalar, _proposed: &Scalar) -> f64 {
                f64::INFINITY
            }
        }
        let mut chain = Chain::new(Scalar(1.0), &Normal).unwrap();
        let mut rng = StdRng::seed_from_u64(42);

        let result = chain.step(&Normal, &InfQProposal, &mut rng);
        assert!(matches!(result, Err(McmcError::InfiniteLogQRatio)));
    }

    #[test]
    fn step_mut_rolls_back_on_inf_log_q_ratio() {
        struct InfQMutProposal;
        impl ProposalMut<MutScalar> for InfQMutProposal {
            type Undo = f64;
            fn propose_mut<R: Rng + ?Sized>(
                &self,
                state: &mut MutScalar,
                _rng: &mut R,
            ) -> Option<f64> {
                let old = state.0;
                state.0 = 0.0;
                Some(old)
            }
            fn undo(&self, state: &mut MutScalar, old: f64) {
                state.0 = old;
            }
            fn log_q_ratio(&self, _state: &MutScalar, _token: &f64) -> f64 {
                f64::INFINITY
            }
        }
        let mut chain = Chain::new(MutScalar(1.0), &Normal).unwrap();
        let mut rng = StdRng::seed_from_u64(42);

        let result = chain.step_mut(&Normal, &InfQMutProposal, &mut rng);
        assert!(matches!(result, Err(McmcError::InfiniteLogQRatio)));
        assert_eq!(
            chain.state,
            MutScalar(1.0),
            "State should be rolled back after +inf log_q_ratio"
        );
    }

    #[test]
    fn step_mut_rolls_back_on_inf_log_prob() {
        struct InfAtOrigin;
        impl Target<MutScalar> for InfAtOrigin {
            fn log_prob(&self, state: &MutScalar) -> f64 {
                if state.0 == 0.0 {
                    f64::INFINITY
                } else {
                    -0.5 * state.0 * state.0
                }
            }
        }
        let mut chain = Chain::new(MutScalar(1.0), &InfAtOrigin).unwrap();
        let proposal = FixedMutProposal(0.0);
        let mut rng = StdRng::seed_from_u64(42);

        let result = chain.step_mut(&InfAtOrigin, &proposal, &mut rng);
        assert!(matches!(result, Err(McmcError::InfiniteProposedLogProb)));
        assert_eq!(
            chain.state,
            MutScalar(1.0),
            "State should be rolled back after +inf log_prob"
        );
    }

    // --- Seeded determinism ---

    #[test]
    fn step_deterministic() {
        let proposal = RandomWalk { width: 1.0 };
        let steps = 100;

        let mut chain1 = Chain::new(Scalar(0.0), &Normal).unwrap();
        let mut rng1 = StdRng::seed_from_u64(12345);
        for _ in 0..steps {
            chain1.step(&Normal, &proposal, &mut rng1).unwrap();
        }

        let mut chain2 = Chain::new(Scalar(0.0), &Normal).unwrap();
        let mut rng2 = StdRng::seed_from_u64(12345);
        for _ in 0..steps {
            chain2.step(&Normal, &proposal, &mut rng2).unwrap();
        }

        assert_eq!(
            chain1.state, chain2.state,
            "Same seed should produce identical final state"
        );
        assert_eq!(chain1.accepted(), chain2.accepted());
        assert_eq!(chain1.rejected(), chain2.rejected());
    }

    // --- Statistical sanity ---

    #[test]
    fn step_samples_near_normal_mode() {
        let proposal = RandomWalk { width: 1.0 };
        let mut chain = Chain::new(Scalar(5.0), &Normal).unwrap();
        let mut rng = StdRng::seed_from_u64(42);

        // Burn-in
        for _ in 0..1_000 {
            chain.step(&Normal, &proposal, &mut rng).unwrap();
        }

        // Collect samples
        let n = 10_000;
        let mut sum = 0.0;
        for _ in 0..n {
            chain.step(&Normal, &proposal, &mut rng).unwrap();
            sum += chain.state.0;
        }
        let mean = sum / f64::from(n);

        assert!(
            mean.abs() < 0.1,
            "Sample mean {mean} should be near 0 for standard normal"
        );

        let rate = chain.acceptance_rate();
        assert!(
            (0.1..0.95).contains(&rate),
            "Acceptance rate {rate} should be in a reasonable range"
        );
    }

    // --- log_q_ratio default ---

    #[test]
    fn symmetric_proposal_zero_log_q() {
        let proposal = RandomWalk { width: 1.0 };
        let ratio = proposal.log_q_ratio(&Scalar(0.0), &Scalar(1.0));
        assert!(
            ratio.abs() < f64::EPSILON,
            "Symmetric proposal should have log_q_ratio = 0"
        );
    }

    // =====================================================================
    // ProposalMut / step_mut tests
    // =====================================================================

    // --- Test fixtures for step_mut ---

    /// Non-Clone state for testing `ProposalMut`.
    #[derive(Debug, PartialEq)]
    struct MutScalar(f64);

    impl Target<MutScalar> for Normal {
        fn log_prob(&self, state: &MutScalar) -> f64 {
            -0.5 * state.0 * state.0
        }
    }

    /// Deterministic in-place proposal: set state to a fixed value.
    struct FixedMutProposal(f64);
    impl ProposalMut<MutScalar> for FixedMutProposal {
        type Undo = f64; // store old value
        fn propose_mut<R: Rng + ?Sized>(&self, state: &mut MutScalar, _rng: &mut R) -> Option<f64> {
            let old = state.0;
            state.0 = self.0;
            Some(old)
        }
        fn undo(&self, state: &mut MutScalar, old: f64) {
            state.0 = old;
        }
    }

    /// Proposal that always returns None (no valid move).
    struct NoMoveProposal;
    impl ProposalMut<MutScalar> for NoMoveProposal {
        type Undo = ();
        fn propose_mut<R: Rng + ?Sized>(&self, _state: &mut MutScalar, _rng: &mut R) -> Option<()> {
            None
        }
        fn undo(&self, _state: &mut MutScalar, _token: ()) {}
    }

    // --- step_mut acceptance ---

    #[test]
    fn step_mut_accepts_uphill() {
        // From x=2.0 (log_prob=-2) to x=0.0 (log_prob=0): always accept
        let mut chain = Chain::new(MutScalar(2.0), &Normal).unwrap();
        let proposal = FixedMutProposal(0.0);
        let mut rng = StdRng::seed_from_u64(42);

        let accepted = chain.step_mut(&Normal, &proposal, &mut rng).unwrap();

        assert!(accepted, "Should accept move to higher probability");
        assert_eq!(chain.state, MutScalar(0.0));
        assert_eq!(chain.accepted(), 1);
        assert_eq!(chain.rejected(), 0);
    }

    #[test]
    fn step_mut_rejects_downhill() {
        // From x=0.0 (log_prob=0) to x=100.0 (log_prob=-5000): virtually always reject
        let mut chain = Chain::new(MutScalar(0.0), &Normal).unwrap();
        let proposal = FixedMutProposal(100.0);
        let mut rng = StdRng::seed_from_u64(42);

        let accepted = chain.step_mut(&Normal, &proposal, &mut rng).unwrap();

        assert!(!accepted, "Should reject move to much lower probability");
        // State should be rolled back to original
        assert_eq!(
            chain.state,
            MutScalar(0.0),
            "State should be rolled back after rejection"
        );
        assert_eq!(chain.accepted(), 0);
        assert_eq!(chain.rejected(), 1);
    }

    // --- step_mut None proposal ---

    #[test]
    fn step_mut_returns_false_on_none_proposal() {
        let mut chain = Chain::new(MutScalar(1.0), &Normal).unwrap();
        let mut rng = StdRng::seed_from_u64(42);

        let accepted = chain.step_mut(&Normal, &NoMoveProposal, &mut rng).unwrap();

        assert!(!accepted, "Should return false when proposal returns None");
        assert_eq!(chain.state, MutScalar(1.0), "State should be unchanged");
        assert_eq!(chain.accepted(), 0);
        assert_eq!(chain.rejected(), 1);
    }

    // --- step_mut NaN rollback ---

    #[test]
    fn step_mut_rolls_back_on_nan_log_prob() {
        struct NanAtOrigin;
        impl Target<MutScalar> for NanAtOrigin {
            fn log_prob(&self, state: &MutScalar) -> f64 {
                if state.0 == 0.0 {
                    f64::NAN
                } else {
                    -0.5 * state.0 * state.0
                }
            }
        }
        let mut chain = Chain::new(MutScalar(1.0), &NanAtOrigin).unwrap();
        let proposal = FixedMutProposal(0.0);
        let mut rng = StdRng::seed_from_u64(42);

        let result = chain.step_mut(&NanAtOrigin, &proposal, &mut rng);
        assert!(matches!(result, Err(McmcError::NanProposedLogProb)));
        assert_eq!(
            chain.state,
            MutScalar(1.0),
            "State should be rolled back after NaN log_prob"
        );
    }

    #[test]
    fn step_mut_rolls_back_on_nan_log_q_ratio() {
        struct NanQProposal;
        impl ProposalMut<MutScalar> for NanQProposal {
            type Undo = f64;
            fn propose_mut<R: Rng + ?Sized>(
                &self,
                state: &mut MutScalar,
                _rng: &mut R,
            ) -> Option<f64> {
                let old = state.0;
                state.0 = 0.0;
                Some(old)
            }
            fn undo(&self, state: &mut MutScalar, old: f64) {
                state.0 = old;
            }
            fn log_q_ratio(&self, _state: &MutScalar, _token: &f64) -> f64 {
                f64::NAN
            }
        }
        let mut chain = Chain::new(MutScalar(1.0), &Normal).unwrap();
        let mut rng = StdRng::seed_from_u64(42);

        let result = chain.step_mut(&Normal, &NanQProposal, &mut rng);
        assert!(matches!(result, Err(McmcError::NanLogQRatio)));
        assert_eq!(
            chain.state,
            MutScalar(1.0),
            "State should be rolled back after NaN log_q_ratio"
        );
    }

    // --- step_mut seeded determinism ---

    #[test]
    fn step_mut_deterministic() {
        /// Random-walk `ProposalMut` for `MutScalar`.
        struct MutRandomWalk {
            width: f64,
        }
        impl ProposalMut<MutScalar> for MutRandomWalk {
            type Undo = f64;
            fn propose_mut<R: Rng + ?Sized>(
                &self,
                state: &mut MutScalar,
                rng: &mut R,
            ) -> Option<f64> {
                let old = state.0;
                let delta: f64 = rng.random_range(-self.width..self.width);
                state.0 += delta;
                Some(old)
            }
            fn undo(&self, state: &mut MutScalar, old: f64) {
                state.0 = old;
            }
        }

        let proposal = MutRandomWalk { width: 1.0 };
        let steps = 100;

        let mut chain1 = Chain::new(MutScalar(0.0), &Normal).unwrap();
        let mut rng1 = StdRng::seed_from_u64(12345);
        for _ in 0..steps {
            chain1.step_mut(&Normal, &proposal, &mut rng1).unwrap();
        }

        let mut chain2 = Chain::new(MutScalar(0.0), &Normal).unwrap();
        let mut rng2 = StdRng::seed_from_u64(12345);
        for _ in 0..steps {
            chain2.step_mut(&Normal, &proposal, &mut rng2).unwrap();
        }

        assert_eq!(
            chain1.state, chain2.state,
            "Same seed should produce identical final state"
        );
        assert_eq!(chain1.accepted(), chain2.accepted());
        assert_eq!(chain1.rejected(), chain2.rejected());
    }

    // --- step_mut statistical sanity ---

    #[test]
    fn step_mut_samples_near_normal_mode() {
        struct MutRandomWalk {
            width: f64,
        }
        impl ProposalMut<MutScalar> for MutRandomWalk {
            type Undo = f64;
            fn propose_mut<R: Rng + ?Sized>(
                &self,
                state: &mut MutScalar,
                rng: &mut R,
            ) -> Option<f64> {
                let old = state.0;
                state.0 += rng.random_range(-self.width..self.width);
                Some(old)
            }
            fn undo(&self, state: &mut MutScalar, old: f64) {
                state.0 = old;
            }
        }

        let proposal = MutRandomWalk { width: 1.0 };
        let mut chain = Chain::new(MutScalar(5.0), &Normal).unwrap();
        let mut rng = StdRng::seed_from_u64(42);

        // Burn-in
        for _ in 0..1_000 {
            chain.step_mut(&Normal, &proposal, &mut rng).unwrap();
        }

        // Collect samples
        let n = 10_000;
        let mut sum = 0.0;
        for _ in 0..n {
            chain.step_mut(&Normal, &proposal, &mut rng).unwrap();
            sum += chain.state.0;
        }
        let mean = sum / f64::from(n);

        assert!(
            mean.abs() < 0.1,
            "Sample mean {mean} should be near 0 for standard normal"
        );

        let rate = chain.acceptance_rate();
        assert!(
            (0.1..0.95).contains(&rate),
            "Acceptance rate {rate} should be in a reasonable range"
        );
    }

    // --- step_mut non-Clone state ---

    #[test]
    fn step_mut_non_clone_state() {
        // MutScalar intentionally does not derive Clone.
        // This test verifies the ProposalMut path compiles and works.
        let mut chain = Chain::new(MutScalar(5.0), &Normal).unwrap();
        let proposal = FixedMutProposal(0.0);
        let mut rng = StdRng::seed_from_u64(42);

        // Should accept (moving to mode)
        let accepted = chain.step_mut(&Normal, &proposal, &mut rng).unwrap();
        assert!(accepted);
        assert_eq!(chain.state, MutScalar(0.0));
    }

    // --- ProposalMut log_q_ratio default ---

    #[test]
    fn symmetric_proposal_mut_zero_log_q() {
        let proposal = FixedMutProposal(0.0);
        let ratio = proposal.log_q_ratio(&MutScalar(1.0), &2.0);
        assert!(
            ratio.abs() < f64::EPSILON,
            "Default ProposalMut log_q_ratio should be 0"
        );
    }

    // --- state accessors ---

    #[test]
    fn replace_state_updates_state_and_log_prob() {
        let mut chain = Chain::new(Scalar(1.0), &Normal).unwrap();
        chain.replace_state(Scalar(2.0), &Normal).unwrap();
        assert_eq!(chain.state, Scalar(2.0));
        assert!(
            (chain.log_prob() - (-2.0)).abs() < 1e-12,
            "log_prob should be recomputed after replace_state"
        );
    }

    #[test]
    fn replace_state_rejects_nan() {
        struct NanTarget;
        impl Target<Scalar> for NanTarget {
            fn log_prob(&self, _: &Scalar) -> f64 {
                f64::NAN
            }
        }
        let mut chain = Chain::new(Scalar(1.0), &Normal).unwrap();
        let result = chain.replace_state(Scalar(0.0), &NanTarget);
        assert!(matches!(result, Err(McmcError::NanInitialLogProb)));
        // State should be unchanged on error
        assert_eq!(chain.state, Scalar(1.0));
    }

    #[test]
    fn replace_state_rejects_inf() {
        struct InfTarget;
        impl Target<Scalar> for InfTarget {
            fn log_prob(&self, _: &Scalar) -> f64 {
                f64::INFINITY
            }
        }
        let mut chain = Chain::new(Scalar(1.0), &Normal).unwrap();
        let result = chain.replace_state(Scalar(0.0), &InfTarget);
        assert!(matches!(result, Err(McmcError::InfiniteInitialLogProb)));
        assert_eq!(chain.state, Scalar(1.0));
    }

    #[test]
    fn into_state_returns_state() {
        let chain = Chain::new(Scalar(3.0), &Normal).unwrap();
        let state = chain.into_state();
        assert_eq!(state, Scalar(3.0));
    }

    // --- reset_counters and total_steps ---

    #[test]
    fn reset_counters_zeros_counts() {
        let mut chain = Chain::new(Scalar(0.0), &Normal).unwrap();
        let proposal = RandomWalk { width: 1.0 };
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..100 {
            chain.step(&Normal, &proposal, &mut rng).unwrap();
        }
        assert!(chain.total_steps() > 0);

        chain.reset_counters();
        assert_eq!(chain.accepted(), 0);
        assert_eq!(chain.rejected(), 0);
        assert_eq!(chain.total_steps(), 0);
        assert!((chain.acceptance_rate()).abs() < f64::EPSILON);
    }

    #[test]
    fn total_steps_equals_accepted_plus_rejected() {
        let mut chain = Chain::new(Scalar(0.0), &Normal).unwrap();
        let proposal = RandomWalk { width: 1.0 };
        let mut rng = StdRng::seed_from_u64(42);

        let steps = 50;
        for _ in 0..steps {
            chain.step(&Normal, &proposal, &mut rng).unwrap();
        }
        assert_eq!(chain.total_steps(), steps);
        assert_eq!(chain.total_steps(), chain.accepted() + chain.rejected());
    }

    // --- Asymmetric proposal tests ---

    /// Deterministic proposal with a configurable log q-ratio.
    struct AsymmetricCloneProposal {
        target_value: f64,
        log_q: f64,
    }
    impl Proposal<Scalar> for AsymmetricCloneProposal {
        fn propose<R: Rng + ?Sized>(&self, _current: &Scalar, _rng: &mut R) -> Scalar {
            Scalar(self.target_value)
        }
        fn log_q_ratio(&self, _current: &Scalar, _proposed: &Scalar) -> f64 {
            self.log_q
        }
    }

    /// In-place version of `AsymmetricCloneProposal`.
    struct AsymmetricMutProposal {
        target_value: f64,
        log_q: f64,
    }
    impl ProposalMut<MutScalar> for AsymmetricMutProposal {
        type Undo = f64;
        fn propose_mut<R: Rng + ?Sized>(&self, state: &mut MutScalar, _rng: &mut R) -> Option<f64> {
            let old = state.0;
            state.0 = self.target_value;
            Some(old)
        }
        fn undo(&self, state: &mut MutScalar, old: f64) {
            state.0 = old;
        }
        fn log_q_ratio(&self, _state: &MutScalar, _token: &f64) -> f64 {
            self.log_q
        }
    }

    #[test]
    fn positive_log_q_ratio_promotes_acceptance() {
        // From x=0 (log_prob=0) to x=1 (log_prob=-0.5).
        // Without asymmetry: log_alpha = -0.5, might reject.
        // With large positive log_q: log_alpha = -0.5 + 100 = 99.5, always accept.
        let mut chain = Chain::new(Scalar(0.0), &Normal).unwrap();
        let proposal = AsymmetricCloneProposal {
            target_value: 1.0,
            log_q: 100.0,
        };
        let mut rng = StdRng::seed_from_u64(42);

        chain.step(&Normal, &proposal, &mut rng).unwrap();
        assert_eq!(
            chain.state,
            Scalar(1.0),
            "Large positive log_q should force acceptance"
        );
    }

    #[test]
    fn negative_log_q_ratio_promotes_rejection() {
        // From x=2 (log_prob=-2) to x=0 (log_prob=0).
        // Without asymmetry: log_alpha = 2.0, always accept.
        // With large negative log_q: log_alpha = 2 - 100 = -98, always reject.
        let mut chain = Chain::new(Scalar(2.0), &Normal).unwrap();
        let proposal = AsymmetricCloneProposal {
            target_value: 0.0,
            log_q: -100.0,
        };
        let mut rng = StdRng::seed_from_u64(42);

        chain.step(&Normal, &proposal, &mut rng).unwrap();
        assert_eq!(
            chain.state,
            Scalar(2.0),
            "Large negative log_q should force rejection"
        );
    }

    #[test]
    fn step_mut_respects_asymmetric_log_q_ratio() {
        // Acceptance via step_mut with positive log_q
        let mut chain = Chain::new(MutScalar(0.0), &Normal).unwrap();
        let proposal = AsymmetricMutProposal {
            target_value: 1.0,
            log_q: 100.0,
        };
        let mut rng = StdRng::seed_from_u64(42);

        let accepted = chain.step_mut(&Normal, &proposal, &mut rng).unwrap();
        assert!(accepted, "Large positive log_q should force acceptance");
        assert_eq!(chain.state, MutScalar(1.0));

        // Rejection via step_mut with negative log_q
        let mut chain2 = Chain::new(MutScalar(2.0), &Normal).unwrap();
        let proposal2 = AsymmetricMutProposal {
            target_value: 0.0,
            log_q: -100.0,
        };
        let mut rng2 = StdRng::seed_from_u64(42);

        let accepted2 = chain2.step_mut(&Normal, &proposal2, &mut rng2).unwrap();
        assert!(!accepted2, "Large negative log_q should force rejection");
        assert_eq!(chain2.state, MutScalar(2.0));
    }

    // --- Edge case: -inf log-probability ---

    #[test]
    fn step_rejects_neg_inf() {
        struct NegInfAt(f64);
        impl Target<Scalar> for NegInfAt {
            fn log_prob(&self, state: &Scalar) -> f64 {
                if (state.0 - self.0).abs() < f64::EPSILON {
                    f64::NEG_INFINITY
                } else {
                    -0.5 * state.0 * state.0
                }
            }
        }
        let target = NegInfAt(0.0);
        let mut chain = Chain::new(Scalar(1.0), &target).unwrap();
        let proposal = FixedProposal(0.0);
        let mut rng = StdRng::seed_from_u64(42);

        chain.step(&target, &proposal, &mut rng).unwrap();
        // exp(-inf - (-0.5)) = exp(-inf) = 0 → always reject
        assert_eq!(
            chain.state,
            Scalar(1.0),
            "Should reject move to -inf log_prob"
        );
    }

    #[test]
    fn step_escapes_neg_inf() {
        struct FiniteAtOrigin;
        impl Target<Scalar> for FiniteAtOrigin {
            fn log_prob(&self, state: &Scalar) -> f64 {
                if state.0.abs() < f64::EPSILON {
                    0.0
                } else {
                    f64::NEG_INFINITY
                }
            }
        }
        let target = FiniteAtOrigin;
        let mut chain = Chain::new(Scalar(1.0), &target).unwrap();
        assert!(chain.log_prob().is_infinite() && chain.log_prob().is_sign_negative());

        let proposal = FixedProposal(0.0);
        let mut rng = StdRng::seed_from_u64(42);

        chain.step(&target, &proposal, &mut rng).unwrap();
        // log_alpha = 0 - (-inf) = inf → always accept
        assert_eq!(
            chain.state,
            Scalar(0.0),
            "Should accept escape from -inf log_prob"
        );
    }

    #[test]
    fn step_rejects_both_neg_inf() {
        struct AlwaysNegInf;
        impl Target<Scalar> for AlwaysNegInf {
            fn log_prob(&self, _state: &Scalar) -> f64 {
                f64::NEG_INFINITY
            }
        }
        let mut chain = Chain::new(Scalar(0.0), &AlwaysNegInf).unwrap();
        let proposal = FixedProposal(1.0);
        let mut rng = StdRng::seed_from_u64(42);

        chain.step(&AlwaysNegInf, &proposal, &mut rng).unwrap();
        // log_alpha = (-inf) - (-inf) = NaN → NaN comparisons false → reject
        assert_eq!(
            chain.state,
            Scalar(0.0),
            "Should reject when both states have -inf log_prob"
        );
        assert_eq!(chain.rejected(), 1);
    }
}
