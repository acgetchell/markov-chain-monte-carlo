//! Property-based tests for [`Chain`] invariants.
//!
//! These tests verify mathematical properties of Metropolis–Hastings that must
//! hold for *all* inputs, not just specific test cases.

use core::convert::Infallible;

use approx::relative_eq;
use markov_chain_monte_carlo::prelude::by_value::Proposal;
use markov_chain_monte_carlo::prelude::delayed::DelayedProposal;
use markov_chain_monte_carlo::prelude::in_place::ProposalMut;
use markov_chain_monte_carlo::prelude::{Chain, Sampler, Target};
use proptest::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng};

// ---------------------------------------------------------------------------
// Shared fixtures
// ---------------------------------------------------------------------------

/// Clone-able scalar state (used by both `step` and `step_mut` paths).
#[derive(Clone, Copy, Debug, PartialEq)]
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
    type Info = f64;
    fn propose_mut<R: Rng + ?Sized>(&mut self, state: &mut Scalar, rng: &mut R) -> Option<f64> {
        let old = state.0;
        state.0 += rng.random_range(-self.width..self.width);
        Some(old)
    }
    fn info(&self, state: &Scalar, _old: &f64) -> f64 {
        state.0
    }
    fn undo(&mut self, state: &mut Scalar, old: f64) {
        state.0 = old;
    }
}

/// Accept-before-mutation random walk proposal equivalent to `CloneWalk`.
struct DelayedWalk {
    width: f64,
}

impl DelayedProposal<Scalar> for DelayedWalk {
    type Plan = f64;
    type Info = f64;
    type Error = Infallible;

    fn propose_plan<R: Rng + ?Sized>(
        &mut self,
        _state: &Scalar,
        rng: &mut R,
    ) -> Result<Option<f64>, Self::Error> {
        Ok(Some(rng.random_range(-self.width..self.width)))
    }

    fn proposed_log_prob<T: Target<Scalar>>(
        &self,
        state: &Scalar,
        plan: &f64,
        target: &T,
    ) -> Result<f64, Self::Error> {
        Ok(target.log_prob(&Scalar(state.0 + *plan)))
    }

    fn info(&self, plan: &f64) -> f64 {
        *plan
    }

    fn commit<R: Rng + ?Sized>(
        &mut self,
        state: &mut Scalar,
        plan: f64,
        _rng: &mut R,
    ) -> Result<(), Self::Error> {
        state.0 += plan;
        Ok(())
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
    fn step_mut_preserves_log_prob(
        initial in -10.0f64..10.0,
        width in 0.1f64..5.0,
        steps in 1u32..500,
        seed in any::<u64>(),
    ) {
        let mut chain = Chain::new(Scalar(initial), &Normal).unwrap();
        let mut proposal = MutWalk { width };
        let mut rng = StdRng::seed_from_u64(seed);

        for _ in 0..steps {
            let _ = chain.step_mut(&Normal, &mut proposal, &mut rng).unwrap();
        }

        let expected = Normal.log_prob(chain.state());
        prop_assert!(
            relative_eq!(chain.log_prob(), expected, epsilon = 1e-12),
            "log_prob {:.15} != target {:.15} after {} steps",
            chain.log_prob(), expected, steps,
        );
    }

    /// Same property for the by-value `step`.
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

        let expected = Normal.log_prob(chain.state());
        prop_assert!(
            relative_eq!(chain.log_prob(), expected, epsilon = 1e-12),
            "log_prob {:.15} != target {:.15} after {} steps",
            chain.log_prob(), expected, steps,
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
        let mut mut_proposal = MutWalk { width };

        let mut chain_clone = Chain::new(Scalar(initial), &Normal).unwrap();
        let mut rng_clone = StdRng::seed_from_u64(seed);

        let mut chain_mut = Chain::new(Scalar(initial), &Normal).unwrap();
        let mut rng_mut = StdRng::seed_from_u64(seed);

        for _ in 0..steps {
            chain_clone.step(&Normal, &clone_proposal, &mut rng_clone).unwrap();
            let _ = chain_mut
                .step_mut(&Normal, &mut mut_proposal, &mut rng_mut)
                .unwrap();
        }

        prop_assert_eq!(
            chain_clone.state(), chain_mut.state(),
            "Final states diverged after {} steps", steps,
        );
        prop_assert!(
            relative_eq!(chain_clone.log_prob(), chain_mut.log_prob(), epsilon = 1e-12),
            "log_prob diverged: clone={:.15}, mut={:.15}",
            chain_clone.log_prob(), chain_mut.log_prob(),
        );
        prop_assert_eq!(chain_clone.accepted(), chain_mut.accepted());
        prop_assert_eq!(chain_clone.rejected(), chain_mut.rejected());
    }

    /// `step` and `step_delayed` must produce identical results when the
    /// delayed plan describes the same proposed state as the by-value path.
    #[test]
    fn step_and_step_delayed_are_equivalent(
        initial in -10.0f64..10.0,
        width in 0.1f64..5.0,
        steps in 1u32..200,
        seed in any::<u64>(),
    ) {
        let clone_proposal = CloneWalk { width };
        let mut delayed_proposal = DelayedWalk { width };

        let mut chain_clone = Chain::new(Scalar(initial), &Normal).unwrap();
        let mut rng_clone = StdRng::seed_from_u64(seed);

        let mut chain_delayed = Chain::new(Scalar(initial), &Normal).unwrap();
        let mut rng_delayed = StdRng::seed_from_u64(seed);

        for _ in 0..steps {
            chain_clone.step(&Normal, &clone_proposal, &mut rng_clone).unwrap();
            let _ = chain_delayed
                .step_delayed(&Normal, &mut delayed_proposal, &mut rng_delayed)
                .unwrap();
        }

        prop_assert_eq!(
            chain_clone.state(), chain_delayed.state(),
            "Final states diverged after {} steps", steps,
        );
        prop_assert!(
            relative_eq!(chain_clone.log_prob(), chain_delayed.log_prob(), epsilon = 1e-12),
            "log_prob diverged: clone={:.15}, delayed={:.15}",
            chain_clone.log_prob(), chain_delayed.log_prob(),
        );
        prop_assert_eq!(chain_clone.accepted(), chain_delayed.accepted());
        prop_assert_eq!(chain_clone.rejected(), chain_delayed.rejected());
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
        let mut proposal = MutWalk { width };
        let mut rng = StdRng::seed_from_u64(seed);

        for _ in 0..steps {
            let _ = chain.step_mut(&Normal, &mut proposal, &mut rng).unwrap();
        }

        prop_assert_eq!(
            chain.accepted() + chain.rejected(),
            steps as usize,
            "accepted ({}) + rejected ({}) != steps ({})",
            chain.accepted(), chain.rejected(), steps,
        );
    }

    /// Same counts invariant for the by-value `step`.
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
            chain.accepted() + chain.rejected(),
            steps as usize,
            "accepted ({}) + rejected ({}) != steps ({})",
            chain.accepted(), chain.rejected(), steps,
        );
    }

    /// `Sampler::run` must produce identical results to a raw `Chain` loop
    /// with the same seed.
    #[test]
    fn sampler_run_matches_raw_chain(
        initial in -10.0f64..10.0,
        width in 0.1f64..5.0,
        steps in 1u32..500,
        seed in any::<u64>(),
    ) {
        let proposal = CloneWalk { width };

        // Raw chain
        let mut chain = Chain::new(Scalar(initial), &Normal).unwrap();
        let mut rng = StdRng::seed_from_u64(seed);
        for _ in 0..steps {
            chain.step(&Normal, &proposal, &mut rng).unwrap();
        }

        // Sampler
        let chain2 = Chain::new(Scalar(initial), &Normal).unwrap();
        let mut rng2 = StdRng::seed_from_u64(seed);
        let mut sampler = Sampler::new(chain2, &Normal, &proposal, &mut rng2).unwrap();
        sampler.run(steps as usize).unwrap();

        prop_assert_eq!(chain.state(), sampler.chain_ref().state());
        prop_assert_eq!(chain.accepted(), sampler.chain_ref().accepted());
        prop_assert_eq!(chain.rejected(), sampler.chain_ref().rejected());
    }

    /// Chunked by-value sampler runs must preserve RNG state, counters, and
    /// checkpoint-compatible continuation state exactly like one-shot runs.
    #[test]
    fn sampler_run_chunk_matches_one_shot(
        initial in -10.0f64..10.0,
        width in 0.1f64..5.0,
        steps in 1u32..500,
        split in 0u32..500,
        seed in any::<u64>(),
    ) {
        let proposal = CloneWalk { width };
        let steps = steps as usize;
        let first = split as usize % (steps + 1);
        let second = steps - first;

        let one_shot_chain = Chain::new(Scalar(initial), &Normal).unwrap();
        let mut one_shot_rng = StdRng::seed_from_u64(seed);
        let mut one_shot =
            Sampler::new(one_shot_chain, &Normal, &proposal, &mut one_shot_rng).unwrap();
        one_shot.run(steps).unwrap();

        let chunked_chain = Chain::new(Scalar(initial), &Normal).unwrap();
        let mut chunked_rng = StdRng::seed_from_u64(seed);
        let mut chunked =
            Sampler::new(chunked_chain, &Normal, &proposal, &mut chunked_rng).unwrap();

        let first_total_steps = {
            let continuation = chunked.run_chunk(first).unwrap();
            continuation.total_steps()
        };
        prop_assert_eq!(first_total_steps, first);

        let continuation = chunked.run_chunk(second).unwrap();
        prop_assert_eq!(one_shot.chain_ref().state(), *continuation.state());
        prop_assert_eq!(one_shot.chain_ref().accepted(), continuation.accepted());
        prop_assert_eq!(one_shot.chain_ref().rejected(), continuation.rejected());
        prop_assert_eq!(one_shot.chain_ref().total_steps(), continuation.total_steps());
        prop_assert!(
            relative_eq!(
                one_shot.chain_ref().log_prob(),
                chunked.chain_ref().log_prob(),
                epsilon = 1e-12,
            ),
            "cached log_prob diverged: one-shot={:.15}, chunked={:.15}",
            one_shot.chain_ref().log_prob(),
            chunked.chain_ref().log_prob(),
        );
    }

    /// `Sampler::run_mut` must produce identical results to a raw `Chain`
    /// loop with the same seed.
    #[test]
    fn sampler_run_mut_matches_raw_chain(
        initial in -10.0f64..10.0,
        width in 0.1f64..5.0,
        steps in 1u32..500,
        seed in any::<u64>(),
    ) {
        let mut proposal = MutWalk { width };

        // Raw chain
        let mut chain = Chain::new(Scalar(initial), &Normal).unwrap();
        let mut rng = StdRng::seed_from_u64(seed);
        for _ in 0..steps {
            let _ = chain.step_mut(&Normal, &mut proposal, &mut rng).unwrap();
        }

        // Sampler
        let chain2 = Chain::new(Scalar(initial), &Normal).unwrap();
        let mut rng2 = StdRng::seed_from_u64(seed);
        let mut sampler = Sampler::new(chain2, &Normal, MutWalk { width }, &mut rng2).unwrap();
        sampler.run_mut(steps as usize).unwrap();

        prop_assert_eq!(chain.state(), sampler.chain_ref().state());
        prop_assert_eq!(chain.accepted(), sampler.chain_ref().accepted());
        prop_assert_eq!(chain.rejected(), sampler.chain_ref().rejected());
    }

    /// Chunked in-place sampler runs must match one-shot runs with the same
    /// seed and expose continuation counters after each chunk.
    #[test]
    fn sampler_run_mut_chunk_matches_one_shot(
        initial in -10.0f64..10.0,
        width in 0.1f64..5.0,
        steps in 1u32..500,
        split in 0u32..500,
        seed in any::<u64>(),
    ) {
        let steps = steps as usize;
        let first = split as usize % (steps + 1);
        let second = steps - first;

        let one_shot_chain = Chain::new(Scalar(initial), &Normal).unwrap();
        let mut one_shot_rng = StdRng::seed_from_u64(seed);
        let mut one_shot = Sampler::new(
            one_shot_chain,
            &Normal,
            MutWalk { width },
            &mut one_shot_rng,
        )
        .unwrap();
        one_shot.run_mut(steps).unwrap();

        let chunked_chain = Chain::new(Scalar(initial), &Normal).unwrap();
        let mut chunked_rng = StdRng::seed_from_u64(seed);
        let mut chunked = Sampler::new(
            chunked_chain,
            &Normal,
            MutWalk { width },
            &mut chunked_rng,
        )
        .unwrap();

        let first_total_steps = {
            let continuation = chunked.run_mut_chunk(first).unwrap();
            continuation.total_steps()
        };
        prop_assert_eq!(first_total_steps, first);

        let continuation = chunked.run_mut_chunk(second).unwrap();
        prop_assert_eq!(one_shot.chain_ref().state(), *continuation.state());
        prop_assert_eq!(one_shot.chain_ref().accepted(), continuation.accepted());
        prop_assert_eq!(one_shot.chain_ref().rejected(), continuation.rejected());
        prop_assert_eq!(one_shot.chain_ref().total_steps(), continuation.total_steps());
        prop_assert!(
            relative_eq!(
                one_shot.chain_ref().log_prob(),
                chunked.chain_ref().log_prob(),
                epsilon = 1e-12,
            ),
            "cached log_prob diverged: one-shot={:.15}, chunked={:.15}",
            one_shot.chain_ref().log_prob(),
            chunked.chain_ref().log_prob(),
        );
    }

    /// Chunked delayed sampler runs must match one-shot delayed runs with the
    /// same seed and preserve delayed proposal telemetry counters.
    #[test]
    fn sampler_run_delayed_chunk_matches_one_shot(
        initial in -10.0f64..10.0,
        width in 0.1f64..5.0,
        steps in 1u32..200,
        split in 0u32..200,
        seed in any::<u64>(),
    ) {
        let steps = steps as usize;
        let first = split as usize % (steps + 1);
        let second = steps - first;

        let one_shot_chain = Chain::new(Scalar(initial), &Normal).unwrap();
        let mut one_shot_rng = StdRng::seed_from_u64(seed);
        let mut one_shot_proposal = DelayedWalk { width };
        let mut one_shot = Sampler::new(
            one_shot_chain,
            &Normal,
            &mut one_shot_proposal,
            &mut one_shot_rng,
        ).unwrap();
        one_shot.run_delayed(steps).unwrap();

        let chunked_chain = Chain::new(Scalar(initial), &Normal).unwrap();
        let mut chunked_rng = StdRng::seed_from_u64(seed);
        let mut chunked_proposal = DelayedWalk { width };
        let mut chunked = Sampler::new(
            chunked_chain,
            &Normal,
            &mut chunked_proposal,
            &mut chunked_rng,
        ).unwrap();

        let first_total_steps = {
            let continuation = chunked.run_delayed_chunk(first).unwrap();
            continuation.total_steps()
        };
        prop_assert_eq!(first_total_steps, first);

        let continuation = chunked.run_delayed_chunk(second).unwrap();
        prop_assert_eq!(one_shot.chain_ref().state(), *continuation.state());
        prop_assert_eq!(one_shot.chain_ref().accepted(), continuation.accepted());
        prop_assert_eq!(one_shot.chain_ref().rejected(), continuation.rejected());
        prop_assert_eq!(one_shot.chain_ref().total_steps(), continuation.total_steps());
        prop_assert!(
            relative_eq!(
                one_shot.chain_ref().log_prob(),
                chunked.chain_ref().log_prob(),
                epsilon = 1e-12,
            ),
            "cached log_prob diverged: one-shot={:.15}, chunked={:.15}",
            one_shot.chain_ref().log_prob(),
            chunked.chain_ref().log_prob(),
        );
    }
}

#[test]
fn run_chunk_allows_next_chunk_size_from_current_state() {
    #[derive(Clone, Debug, PartialEq)]
    struct Counter(i32);

    struct Flat;
    impl Target<Counter> for Flat {
        fn log_prob(&self, _: &Counter) -> f64 {
            0.0
        }
    }

    struct Increment;
    impl Proposal<Counter> for Increment {
        fn propose<R: Rng + ?Sized>(&self, current: &Counter, _: &mut R) -> Counter {
            Counter(current.0 + 1)
        }
    }

    let mut one_shot_rng = StdRng::seed_from_u64(42);
    let one_shot_chain = Chain::new(Counter(0), &Flat).unwrap();
    let mut one_shot = Sampler::new(one_shot_chain, &Flat, &Increment, &mut one_shot_rng).unwrap();
    one_shot.run(5).unwrap();

    let mut chunked_rng = StdRng::seed_from_u64(42);
    let chunked_chain = Chain::new(Counter(0), &Flat).unwrap();
    let mut chunked = Sampler::new(chunked_chain, &Flat, &Increment, &mut chunked_rng).unwrap();

    let next_chunk_size = {
        let continuation = chunked.run_chunk(2).unwrap();
        assert_eq!(continuation.state().0, 2);
        usize::try_from(continuation.state().0 + 1).unwrap()
    };

    let continuation = chunked.run_chunk(next_chunk_size).unwrap();

    assert_eq!(one_shot.chain_ref().state(), *continuation.state());
    assert_eq!(one_shot.chain_ref().accepted(), continuation.accepted());
    assert_eq!(one_shot.chain_ref().rejected(), continuation.rejected());
    assert_eq!(
        one_shot.chain_ref().total_steps(),
        continuation.total_steps()
    );
    assert!(relative_eq!(
        one_shot.chain_ref().log_prob(),
        chunked.chain_ref().log_prob(),
        epsilon = 1e-12,
    ));
}
