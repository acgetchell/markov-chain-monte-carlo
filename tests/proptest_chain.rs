//! Property-based tests for [`Chain`] invariants.
//!
//! These tests exercise cache, transition, and continuation invariants over
//! generated finite states and reproducible seeded proposal sequences.

use core::convert::Infallible;

use approx::relative_eq;
use markov_chain_monte_carlo::prelude::by_value::Proposal;
use markov_chain_monte_carlo::prelude::delayed::DelayedProposal;
use markov_chain_monte_carlo::prelude::in_place::ProposalMut;
use markov_chain_monte_carlo::prelude::{Chain, Sampler, Target, ThinningInterval};
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

    fn proposed_log_prob<T: Target<Scalar> + ?Sized>(
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
    /// Compare actual retained states with complete raw trajectories, so an
    /// off-by-one sampling phase or duplicated output cannot pass by length alone.
    #[test]
    fn thinned_workflows_retain_the_expected_post_step_states(
        initial in -10.0f64..10.0,
        width in 0.1f64..5.0,
        steps in 0usize..50,
        interval in 1usize..10,
        seed in any::<u64>(),
    ) {
        let proposal = CloneWalk { width };
        let mut reference = Chain::new(Scalar(initial), &Normal).unwrap();
        let mut reference_rng = StdRng::seed_from_u64(seed);
        let mut trajectory = Vec::new();
        for _ in 0..steps {
            let _ = reference.step(&Normal, &proposal, &mut reference_rng).unwrap();
            trajectory.push(*reference.state());
        }
        let expected: Vec<_> = trajectory
            .chunks_exact(interval)
            .map(|block| block[interval - 1])
            .collect();
        let thin = ThinningInterval::new(interval).unwrap();

        let mut by_value_rng = StdRng::seed_from_u64(seed);
        let mut by_value = Sampler::from_state(
            Scalar(initial), &Normal, &proposal, &mut by_value_rng,
        ).unwrap();
        let states = by_value.run_with_thinning(steps, thin).unwrap();
        prop_assert_eq!(states.as_slice(), expected.as_slice());
        prop_assert_eq!(by_value.chain_ref().state(), reference.state());
        prop_assert_eq!(by_value.chain_ref().total_steps(), steps);

        let mut in_place_rng = StdRng::seed_from_u64(seed);
        let mut in_place = Sampler::from_state(
            Scalar(initial), &Normal, MutWalk { width }, &mut in_place_rng,
        ).unwrap();
        let states = in_place.run_mut_with_thinning(steps, thin).unwrap();
        prop_assert_eq!(states.as_slice(), expected.as_slice());
        prop_assert_eq!(in_place.chain_ref().state(), reference.state());
        prop_assert_eq!(in_place.chain_ref().total_steps(), steps);

        let mut delayed_rng = StdRng::seed_from_u64(seed);
        let mut delayed = Sampler::from_state(
            Scalar(initial), &Normal, DelayedWalk { width }, &mut delayed_rng,
        ).unwrap();
        let states = delayed.run_delayed_with_thinning(steps, thin).unwrap();
        prop_assert_eq!(states.as_slice(), expected.as_slice());
        prop_assert_eq!(delayed.chain_ref().state(), reference.state());
        prop_assert_eq!(delayed.chain_ref().total_steps(), steps);
    }

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

        for step in 0..steps {
            let _ = chain.step_mut(&Normal, &mut proposal, &mut rng).unwrap();
            let expected = Normal.log_prob(chain.state());
            prop_assert_eq!(
                chain.log_prob().to_bits(), expected.to_bits(),
                "cache mismatch at step {} with seed {}", step, seed,
            );
        }
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

        for step in 0..steps {
            let _ = chain.step(&Normal, &proposal, &mut rng).unwrap();
            let expected = Normal.log_prob(chain.state());
            prop_assert_eq!(
                chain.log_prob().to_bits(), expected.to_bits(),
                "cache mismatch at step {} with seed {}", step, seed,
            );
        }
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

        for step in 0..steps {
            let by_value_step = chain_clone
                .step(&Normal, &clone_proposal, &mut rng_clone)
                .unwrap();
            let in_place_step = chain_mut
                .step_mut(&Normal, &mut mut_proposal, &mut rng_mut)
                .unwrap();
            prop_assert_eq!(
                by_value_step.outcome(), in_place_step.outcome(),
                "outcomes diverged at step {} with seed {}", step, seed,
            );
            prop_assert_eq!(chain_clone.state(), chain_mut.state());
            prop_assert_eq!(by_value_step.log_alpha(), in_place_step.log_alpha());
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

        for step in 0..steps {
            let by_value_step = chain_clone
                .step(&Normal, &clone_proposal, &mut rng_clone)
                .unwrap();
            let delayed_step = chain_delayed
                .step_delayed(&Normal, &mut delayed_proposal, &mut rng_delayed)
                .unwrap();
            prop_assert_eq!(
                by_value_step.outcome(), delayed_step.outcome(),
                "outcomes diverged at step {} with seed {}", step, seed,
            );
            prop_assert_eq!(chain_clone.state(), chain_delayed.state());
            prop_assert_eq!(by_value_step.log_alpha(), delayed_step.log_alpha());
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
            let _ = chain.step(&Normal, &proposal, &mut rng).unwrap();
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
            let _ = chain.step(&Normal, &proposal, &mut rng).unwrap();
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
        let _ = one_shot.into_chain();
        let _ = chunked.into_chain();
        prop_assert_eq!(one_shot_rng.random::<u64>(), chunked_rng.random::<u64>());
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
        let _ = one_shot.into_chain();
        let _ = chunked.into_chain();
        prop_assert_eq!(one_shot_rng.random::<u64>(), chunked_rng.random::<u64>());
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
        let _ = one_shot.into_chain();
        let _ = chunked.into_chain();
        prop_assert_eq!(one_shot_rng.random::<u64>(), chunked_rng.random::<u64>());
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

    struct Toggle;
    impl Proposal<Counter> for Toggle {
        fn propose<R: Rng + ?Sized>(&self, current: &Counter, _: &mut R) -> Counter {
            Counter(1 - current.0)
        }
    }

    let mut one_shot_rng = StdRng::seed_from_u64(42);
    let one_shot_chain = Chain::new(Counter(2), &Flat).unwrap();
    let mut one_shot = Sampler::new(one_shot_chain, &Flat, &Toggle, &mut one_shot_rng).unwrap();
    one_shot.run(5).unwrap();

    let mut chunked_rng = StdRng::seed_from_u64(42);
    let chunked_chain = Chain::new(Counter(2), &Flat).unwrap();
    let mut chunked = Sampler::new(chunked_chain, &Flat, &Toggle, &mut chunked_rng).unwrap();

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
