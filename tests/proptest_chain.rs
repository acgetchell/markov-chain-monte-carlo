//! Property-based tests for [`Chain`] invariants.
//!
//! These tests verify mathematical properties of Metropolis–Hastings that must
//! hold for *all* inputs, not just specific test cases.

use markov_chain_monte_carlo::prelude::*;
use proptest::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng};

// ---------------------------------------------------------------------------
// Shared fixtures
// ---------------------------------------------------------------------------

/// Clone-able scalar state (used by both `step` and `step_mut` paths).
#[derive(Clone, Debug, PartialEq)]
struct Scalar(f64);

/// Standard normal target: log p(x) = −x²/2.
struct Normal;
impl Target<Scalar> for Normal {
    fn log_prob(&self, state: &Scalar) -> f64 {
        -0.5 * state.0 * state.0
    }
}

/// Clone-based random walk proposal.
struct CloneWalk {
    width: f64,
}
impl Proposal<Scalar> for CloneWalk {
    fn propose<R: Rng + ?Sized>(&self, current: &Scalar, rng: &mut R) -> Scalar {
        Scalar(current.0 + rng.random_range(-self.width..self.width))
    }
}

/// In-place random walk proposal (equivalent to `CloneWalk`).
struct MutWalk {
    width: f64,
}
impl ProposalMut<Scalar> for MutWalk {
    type Undo = f64;
    fn propose_mut<R: Rng + ?Sized>(&self, state: &mut Scalar, rng: &mut R) -> Option<f64> {
        let old = state.0;
        state.0 += rng.random_range(-self.width..self.width);
        Some(old)
    }
    fn undo(&self, state: &mut Scalar, old: f64) {
        state.0 = old;
    }
}

// ---------------------------------------------------------------------------
// Properties
// ---------------------------------------------------------------------------

proptest! {
    /// After any number of steps, `chain.log_prob` must equal the target
    /// evaluated at the current state.  This catches bugs where `log_prob`
    /// is not updated on acceptance or is corrupted during rollback.
    #[test]
    fn log_prob_consistent_after_step_mut(
        initial in -10.0f64..10.0,
        width in 0.1f64..5.0,
        steps in 1u32..500,
        seed in any::<u64>(),
    ) {
        let mut chain = Chain::new(Scalar(initial), &Normal).unwrap();
        let proposal = MutWalk { width };
        let mut rng = StdRng::seed_from_u64(seed);

        for _ in 0..steps {
            chain.step_mut(&Normal, &proposal, &mut rng).unwrap();
        }

        let expected = Normal.log_prob(&chain.state);
        prop_assert!(
            (chain.log_prob - expected).abs() < 1e-12,
            "log_prob {:.15} != target {:.15} after {} steps",
            chain.log_prob, expected, steps,
        );
    }

    /// Same property for the clone-based `step`.
    #[test]
    fn log_prob_consistent_after_step(
        initial in -10.0f64..10.0,
        width in 0.1f64..5.0,
        steps in 1u32..500,
        seed in any::<u64>(),
    ) {
        let mut chain = Chain::new(Scalar(initial), &Normal).unwrap();
        let proposal = CloneWalk { width };
        let mut rng = StdRng::seed_from_u64(seed);

        for _ in 0..steps {
            chain.step(&Normal, &proposal, &mut rng).unwrap();
        }

        let expected = Normal.log_prob(&chain.state);
        prop_assert!(
            (chain.log_prob - expected).abs() < 1e-12,
            "log_prob {:.15} != target {:.15} after {} steps",
            chain.log_prob, expected, steps,
        );
    }

    /// `step` and `step_mut` must produce identical results when given the
    /// same seed.  `CloneWalk` and `MutWalk` draw the same random delta,
    /// so acceptance decisions must agree exactly.
    #[test]
    fn step_and_step_mut_are_equivalent(
        initial in -10.0f64..10.0,
        width in 0.1f64..5.0,
        steps in 1u32..200,
        seed in any::<u64>(),
    ) {
        let clone_proposal = CloneWalk { width };
        let mut_proposal = MutWalk { width };

        let mut chain_clone = Chain::new(Scalar(initial), &Normal).unwrap();
        let mut rng_clone = StdRng::seed_from_u64(seed);

        let mut chain_mut = Chain::new(Scalar(initial), &Normal).unwrap();
        let mut rng_mut = StdRng::seed_from_u64(seed);

        for _ in 0..steps {
            chain_clone.step(&Normal, &clone_proposal, &mut rng_clone).unwrap();
            chain_mut.step_mut(&Normal, &mut_proposal, &mut rng_mut).unwrap();
        }

        prop_assert_eq!(
            chain_clone.state, chain_mut.state,
            "Final states diverged after {} steps", steps,
        );
        prop_assert!(
            (chain_clone.log_prob - chain_mut.log_prob).abs() < 1e-12,
            "log_prob diverged: clone={:.15}, mut={:.15}",
            chain_clone.log_prob, chain_mut.log_prob,
        );
        prop_assert_eq!(chain_clone.accepted, chain_mut.accepted);
        prop_assert_eq!(chain_clone.rejected, chain_mut.rejected);
    }

    /// accepted + rejected must always equal the number of steps taken.
    #[test]
    fn counts_invariant_step_mut(
        initial in -10.0f64..10.0,
        width in 0.1f64..5.0,
        steps in 1u32..500,
        seed in any::<u64>(),
    ) {
        let mut chain = Chain::new(Scalar(initial), &Normal).unwrap();
        let proposal = MutWalk { width };
        let mut rng = StdRng::seed_from_u64(seed);

        for _ in 0..steps {
            chain.step_mut(&Normal, &proposal, &mut rng).unwrap();
        }

        prop_assert_eq!(
            chain.accepted + chain.rejected,
            steps as usize,
            "accepted ({}) + rejected ({}) != steps ({})",
            chain.accepted, chain.rejected, steps,
        );
    }

    /// Same counts invariant for the clone-based `step`.
    #[test]
    fn counts_invariant_step(
        initial in -10.0f64..10.0,
        width in 0.1f64..5.0,
        steps in 1u32..500,
        seed in any::<u64>(),
    ) {
        let mut chain = Chain::new(Scalar(initial), &Normal).unwrap();
        let proposal = CloneWalk { width };
        let mut rng = StdRng::seed_from_u64(seed);

        for _ in 0..steps {
            chain.step(&Normal, &proposal, &mut rng).unwrap();
        }

        prop_assert_eq!(
            chain.accepted + chain.rejected,
            steps as usize,
            "accepted ({}) + rejected ({}) != steps ({})",
            chain.accepted, chain.rejected, steps,
        );
    }
}
