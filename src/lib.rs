//! Markov Chain Monte Carlo (MCMC) framework.
//!
//! 🚧 **Pre-release (0.0.x)** — This crate is under active development and
//! not yet ready for production use. APIs may change without notice.
//!
//! This crate aims to provide a composable, zero-cost abstraction for MCMC
//! methods over arbitrary state spaces, including discrete and combinatorial
//! systems (e.g., triangulations).

use rand::{Rng, RngExt};

/// Marker trait for MCMC state types.
pub trait State: Clone {}

/// Target distribution (or log-probability / negative action).
pub trait Target<S: State> {
    /// Compute log-probability (or negative energy/action).
    fn log_prob(&self, state: &S) -> f64;
}

/// Proposal distribution for generating new states.
pub trait Proposal<S: State> {
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
pub struct Chain<S: State> {
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
    pub fn new<T: Target<S>>(initial: S, target: &T) -> Self {
        let log_prob = target.log_prob(&initial);
        Self {
            state: initial,
            log_prob,
            accepted: 0,
            rejected: 0,
        }
    }

    /// Perform a single Metropolis–Hastings step.
    pub fn step<T, P, R>(&mut self, target: &T, proposal: &P, rng: &mut R)
    where
        T: Target<S>,
        P: Proposal<S>,
        R: Rng + ?Sized,
    {
        let proposed = proposal.propose(&self.state, rng);
        let log_prob_new = target.log_prob(&proposed);

        let log_q = proposal.log_q_ratio(&self.state, &proposed);
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
        let chain = Chain::new(Scalar(1.0), &Normal);
        assert_eq!(chain.state, Scalar(1.0));
        assert_eq!(chain.accepted, 0);
        assert_eq!(chain.rejected, 0);
    }

    #[test]
    fn new_chain_computes_initial_log_prob() {
        let chain = Chain::new(Scalar(0.0), &Normal);
        assert!(
            (chain.log_prob).abs() < 1e-12,
            "log_prob at 0 should be 0.0"
        );

        let chain2 = Chain::new(Scalar(1.0), &Normal);
        assert!(
            (chain2.log_prob - (-0.5)).abs() < 1e-12,
            "log_prob at 1 should be -0.5"
        );
    }

    // --- acceptance_rate ---

    #[test]
    fn acceptance_rate_zero_steps() {
        let chain = Chain::new(Scalar(0.0), &Normal);
        assert!((chain.acceptance_rate()).abs() < f64::EPSILON);
    }

    // --- MH acceptance logic ---

    #[test]
    fn step_accepts_move_to_higher_probability() {
        // From x=2.0 (log_prob=-2) to x=0.0 (log_prob=0): always accept
        let mut chain = Chain::new(Scalar(2.0), &Normal);
        let proposal = FixedProposal(0.0);
        let mut rng = StdRng::seed_from_u64(42);

        chain.step(&Normal, &proposal, &mut rng);

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
        let mut chain = Chain::new(Scalar(0.0), &Normal);
        let proposal = FixedProposal(100.0);
        let mut rng = StdRng::seed_from_u64(42);

        chain.step(&Normal, &proposal, &mut rng);

        assert_eq!(
            chain.state,
            Scalar(0.0),
            "Should reject move to much lower probability"
        );
        assert_eq!(chain.accepted, 0);
        assert_eq!(chain.rejected, 1);
    }

    // --- Seeded determinism ---

    #[test]
    fn same_seed_produces_identical_chains() {
        let proposal = RandomWalk { width: 1.0 };
        let steps = 100;

        let mut chain1 = Chain::new(Scalar(0.0), &Normal);
        let mut rng1 = StdRng::seed_from_u64(12345);
        for _ in 0..steps {
            chain1.step(&Normal, &proposal, &mut rng1);
        }

        let mut chain2 = Chain::new(Scalar(0.0), &Normal);
        let mut rng2 = StdRng::seed_from_u64(12345);
        for _ in 0..steps {
            chain2.step(&Normal, &proposal, &mut rng2);
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
        let mut chain = Chain::new(Scalar(5.0), &Normal);
        let mut rng = StdRng::seed_from_u64(42);

        // Burn-in
        for _ in 0..1_000 {
            chain.step(&Normal, &proposal, &mut rng);
        }

        // Collect samples
        let n = 10_000;
        let mut sum = 0.0;
        for _ in 0..n {
            chain.step(&Normal, &proposal, &mut rng);
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
