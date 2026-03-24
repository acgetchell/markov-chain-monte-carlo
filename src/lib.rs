//! Markov Chain Monte Carlo (MCMC) framework.
//!
//! 🚧 **Pre-release (0.0.x)** — This crate is under active development and
//! not yet ready for production use. APIs may change without notice.
//!
//! This crate aims to provide a composable, zero-cost abstraction for MCMC
//! methods over arbitrary state spaces, including discrete and combinatorial
//! systems (e.g., triangulations).
//!
//! # Example
//!
//! Sample from a standard normal distribution using Metropolis–Hastings:
//!
//! ```
//! use markov_chain_monte_carlo::prelude::*;
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
//! use markov_chain_monte_carlo::prelude::*;
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

use rand::{Rng, RngExt};
use std::fmt;

/// Errors that can occur during MCMC operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum McmcError {
    /// Target returned NaN log-probability for the initial state.
    NanInitialLogProb,
    /// Target returned NaN log-probability for a proposed state.
    NanProposedLogProb,
    /// Proposal returned NaN log q-ratio.
    NanLogQRatio,
}

impl fmt::Display for McmcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NanInitialLogProb => {
                write!(
                    f,
                    "target returned NaN log-probability for the initial state"
                )
            }
            Self::NanProposedLogProb => {
                write!(
                    f,
                    "target returned NaN log-probability for a proposed state"
                )
            }
            Self::NanLogQRatio => write!(f, "proposal returned NaN log q-ratio"),
        }
    }
}

impl std::error::Error for McmcError {}

/// Convenience re-exports for common usage.
///
/// ```
/// use markov_chain_monte_carlo::prelude::*;
/// ```
pub mod prelude {
    pub use crate::{Chain, McmcError, Proposal, ProposalMut, Target};
}

/// Target distribution
pub trait Target<S> {
    /// Compute log-probability (or negative energy/action).
    fn log_prob(&self, state: &S) -> f64;
}

/// Proposal distribution for generating new states.
///
/// This trait uses a clone-based model: [`propose`](Proposal::propose) returns
/// a new state by value.  For state spaces where cloning is expensive (e.g.,
/// triangulations, large graphs), see [`ProposalMut`] which mutates in place
/// and supports cheap rollback.
pub trait Proposal<S: Clone> {
    /// Propose a new state from the current one.
    fn propose<R: Rng + ?Sized>(&self, current: &S, rng: &mut R) -> S;

    /// Log proposal ratio:
    /// log(q(current | proposed) / q(proposed | current))
    ///
    /// Defaults to 0 for symmetric proposals.
    fn log_q_ratio(&self, _current: &S, _proposed: &S) -> f64 {
        0.0
    }
}

/// In-place proposal distribution with rollback.
///
/// Unlike [`Proposal`], which clones the state for each proposal,
/// `ProposalMut` mutates the state in place and returns an undo token
/// that can reverse the mutation on rejection.  This is the natural
/// model for combinatorial state spaces (e.g., triangulations, graphs)
/// where moves are invertible and cloning is expensive.
///
/// # Associated Types
///
/// * [`Undo`](ProposalMut::Undo) — a small token that captures
///   exactly what is needed to reverse a move.
pub trait ProposalMut<S> {
    /// Token that records how to reverse a proposed move.
    type Undo;

    /// Mutate `state` in place, returning `Some(undo_token)` on success
    /// or `None` if no valid move could be found.
    fn propose_mut<R: Rng + ?Sized>(&self, state: &mut S, rng: &mut R) -> Option<Self::Undo>;

    /// Reverse a previously applied move using its undo token.
    fn undo(&self, state: &mut S, token: Self::Undo);

    /// Log proposal ratio for the in-place move.
    ///
    /// Defaults to 0 for symmetric proposals.
    fn log_q_ratio(&self, _state: &S, _token: &Self::Undo) -> f64 {
        0.0
    }
}

/// A single MCMC chain.
#[derive(Debug)]
pub struct Chain<S> {
    /// Current state
    pub state: S,
    /// Current log probability
    pub log_prob: f64,
    /// Accepted moves
    pub accepted: usize,
    /// Rejected moves
    pub rejected: usize,
}

impl<S> Chain<S> {
    /// Create a new chain from an initial state.
    ///
    /// # Errors
    ///
    /// Returns [`McmcError::NanInitialLogProb`] if the target's log-probability
    /// for the initial state is NaN.
    pub fn new<T: Target<S>>(initial: S, target: &T) -> Result<Self, McmcError> {
        let log_prob = target.log_prob(&initial);
        if log_prob.is_nan() {
            return Err(McmcError::NanInitialLogProb);
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
    /// # Errors
    ///
    /// Returns [`McmcError::NanProposedLogProb`] if the target's log-probability
    /// for the proposed state is NaN, or [`McmcError::NanLogQRatio`] if the
    /// proposal's log q-ratio is NaN.
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

        let log_q = proposal.log_q_ratio(&self.state, &proposed);
        if log_q.is_nan() {
            return Err(McmcError::NanLogQRatio);
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
    /// # Errors
    ///
    /// Returns [`McmcError::NanProposedLogProb`] or [`McmcError::NanLogQRatio`]
    /// after rolling back the state.
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

        let log_q = proposal.log_q_ratio(&self.state, &token);
        if log_q.is_nan() {
            proposal.undo(&mut self.state, token);
            return Err(McmcError::NanLogQRatio);
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

    /// Acceptance rate of the chain.
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

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
    fn new_chain_has_correct_initial_state() {
        let chain = Chain::new(Scalar(1.0), &Normal).unwrap();
        assert_eq!(chain.state, Scalar(1.0));
        assert_eq!(chain.accepted, 0);
        assert_eq!(chain.rejected, 0);
    }

    #[test]
    fn new_chain_computes_initial_log_prob() {
        let chain = Chain::new(Scalar(0.0), &Normal).unwrap();
        assert!(
            (chain.log_prob).abs() < 1e-12,
            "log_prob at 0 should be 0.0"
        );

        let chain2 = Chain::new(Scalar(1.0), &Normal).unwrap();
        assert!(
            (chain2.log_prob - (-0.5)).abs() < 1e-12,
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
    fn step_accepts_move_to_higher_probability() {
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
        assert_eq!(chain.accepted, 1);
        assert_eq!(chain.rejected, 0);
    }

    #[test]
    fn step_rejects_move_to_much_lower_probability() {
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
        assert_eq!(chain.accepted, 0);
        assert_eq!(chain.rejected, 1);
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

    // --- Seeded determinism ---

    #[test]
    fn same_seed_produces_identical_chains() {
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
        assert_eq!(chain1.accepted, chain2.accepted);
        assert_eq!(chain1.rejected, chain2.rejected);
    }

    // --- Statistical sanity ---

    #[test]
    fn chain_samples_near_mode_of_normal() {
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
    fn symmetric_proposal_has_zero_log_q_ratio() {
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
    fn step_mut_accepts_move_to_higher_probability() {
        // From x=2.0 (log_prob=-2) to x=0.0 (log_prob=0): always accept
        let mut chain = Chain::new(MutScalar(2.0), &Normal).unwrap();
        let proposal = FixedMutProposal(0.0);
        let mut rng = StdRng::seed_from_u64(42);

        let accepted = chain.step_mut(&Normal, &proposal, &mut rng).unwrap();

        assert!(accepted, "Should accept move to higher probability");
        assert_eq!(chain.state, MutScalar(0.0));
        assert_eq!(chain.accepted, 1);
        assert_eq!(chain.rejected, 0);
    }

    #[test]
    fn step_mut_rejects_move_to_much_lower_probability() {
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
        assert_eq!(chain.accepted, 0);
        assert_eq!(chain.rejected, 1);
    }

    // --- step_mut None proposal ---

    #[test]
    fn step_mut_returns_false_on_none_proposal() {
        let mut chain = Chain::new(MutScalar(1.0), &Normal).unwrap();
        let mut rng = StdRng::seed_from_u64(42);

        let accepted = chain.step_mut(&Normal, &NoMoveProposal, &mut rng).unwrap();

        assert!(!accepted, "Should return false when proposal returns None");
        assert_eq!(chain.state, MutScalar(1.0), "State should be unchanged");
        assert_eq!(chain.accepted, 0);
        assert_eq!(chain.rejected, 1);
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
    fn step_mut_same_seed_produces_identical_chains() {
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
        assert_eq!(chain1.accepted, chain2.accepted);
        assert_eq!(chain1.rejected, chain2.rejected);
    }

    // --- step_mut statistical sanity ---

    #[test]
    fn step_mut_samples_near_mode_of_normal() {
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
    fn step_mut_works_with_non_clone_state() {
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
    fn symmetric_proposal_mut_has_zero_log_q_ratio() {
        let proposal = FixedMutProposal(0.0);
        let ratio = proposal.log_q_ratio(&MutScalar(1.0), &2.0);
        assert!(
            ratio.abs() < f64::EPSILON,
            "Default ProposalMut log_q_ratio should be 0"
        );
    }
}
