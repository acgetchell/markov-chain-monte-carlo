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
//! impl State for Scalar {}
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
    pub use crate::{Chain, McmcError, Proposal, State, Target};
}

/// Marker trait for MCMC state types.
pub trait State: Clone {}

/// Target distribution (or log-probability / negative action).
pub trait Target<S> {
    /// Compute log-probability (or negative energy/action).
    fn log_prob(&self, state: &S) -> f64;
}

/// Proposal distribution for generating new states.
pub trait Proposal<S> {
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

impl<S: State> Chain<S> {
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

    /// Perform a single Metropolis–Hastings step.
    ///
    /// # Errors
    ///
    /// Returns [`McmcError::NanProposedLogProb`] if the target's log-probability
    /// for the proposed state is NaN, or [`McmcError::NanLogQRatio`] if the
    /// proposal's log q-ratio is NaN.
    pub fn step<T, P, R>(&mut self, target: &T, proposal: &P, rng: &mut R) -> Result<(), McmcError>
    where
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
    impl State for Scalar {}

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
}
