//! Adaptive warmup contracts and independent target-distribution checks.

use core::cell::Cell;

use approx::assert_relative_eq;
use markov_chain_monte_carlo::prelude::by_value::Proposal;
use markov_chain_monte_carlo::prelude::delayed::{DelayedProposal, DelayedStepError};
use markov_chain_monte_carlo::prelude::in_place::ProposalMut;
use markov_chain_monte_carlo::prelude::{
    AdaptiveScale, AdaptiveScaleError, Chain, McmcError, OnlineStats, Sampler, Target,
    TunableProposal,
};
use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};

struct Flat;

impl Target<bool> for Flat {
    fn log_prob(&self, _: &bool) -> f64 {
        0.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fault {
    None,
    Plan,
    Score,
    Ratio,
    Commit,
}

struct Probe {
    scale: f64,
    ratio: f64,
    absent: bool,
    fault: Fault,
    transition_scale: Cell<f64>,
    metadata_calls: Cell<usize>,
}

impl Probe {
    const fn new(ratio: f64) -> Self {
        Self {
            scale: 99.0,
            ratio,
            absent: false,
            fault: Fault::None,
            transition_scale: Cell::new(99.0),
            metadata_calls: Cell::new(0),
        }
    }

    fn check_scale(&self) {
        assert_eq!(self.scale.to_bits(), self.transition_scale.get().to_bits());
    }

    fn record_metadata(&self) {
        self.metadata_calls.set(self.metadata_calls.get() + 1);
    }
}

impl TunableProposal for Probe {
    fn set_scale(&mut self, scale: f64) {
        self.scale = scale;
    }
}

impl Proposal<bool> for Probe {
    fn propose<R: Rng + ?Sized>(&self, current: &bool, _: &mut R) -> bool {
        self.transition_scale.set(self.scale);
        !current
    }

    fn log_q_ratio(&self, _: &bool, _: &bool) -> f64 {
        self.check_scale();
        self.ratio
    }
}

impl ProposalMut<bool> for Probe {
    type Undo = bool;
    type Info = ();

    fn propose_mut<R: Rng + ?Sized>(&mut self, state: &mut bool, _: &mut R) -> Option<bool> {
        self.transition_scale.set(self.scale);
        if self.absent {
            return None;
        }
        let old = *state;
        *state = !old;
        Some(old)
    }

    fn info(&self, _: &bool, _: &bool) {
        self.record_metadata();
    }

    fn no_proposal_info(&mut self) -> Option<Self::Info> {
        self.record_metadata();
        Some(())
    }

    fn undo(&mut self, state: &mut bool, old: bool) {
        self.check_scale();
        *state = old;
    }

    fn log_q_ratio(&self, _: &bool, _: &bool) -> f64 {
        self.check_scale();
        self.ratio
    }
}

impl DelayedProposal<bool> for Probe {
    type Plan = bool;
    type Info = ();
    type Error = Fault;

    fn propose_plan<R: Rng + ?Sized>(
        &mut self,
        state: &bool,
        _: &mut R,
    ) -> Result<Option<bool>, Fault> {
        self.transition_scale.set(self.scale);
        if self.fault == Fault::Plan {
            return Err(Fault::Plan);
        }
        Ok((!self.absent).then_some(!state))
    }

    fn proposed_log_prob<T: Target<bool> + ?Sized>(
        &self,
        _: &bool,
        plan: &bool,
        target: &T,
    ) -> Result<f64, Fault> {
        self.check_scale();
        if self.fault == Fault::Score {
            return Err(Fault::Score);
        }
        Ok(target.log_prob(plan))
    }

    fn log_q_ratio(&self, _: &bool, _: &bool) -> Result<f64, Fault> {
        self.check_scale();
        if self.fault == Fault::Ratio {
            return Err(Fault::Ratio);
        }
        Ok(self.ratio)
    }

    fn info(&self, _: &bool) {
        self.record_metadata();
    }

    fn no_plan_info(&mut self) -> Option<Self::Info> {
        self.record_metadata();
        Some(())
    }

    fn commit<R: Rng + ?Sized>(
        &mut self,
        state: &mut bool,
        plan: bool,
        _: &mut R,
    ) -> Result<(), Fault> {
        self.check_scale();
        if self.fault == Fault::Commit {
            return Err(Fault::Commit);
        }
        *state = plan;
        Ok(())
    }
}

#[test]
fn invalid_configuration_is_rejected_at_construction() {
    for target in [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        -0.1,
        0.0,
        1.0,
        1.1,
    ] {
        assert!(matches!(
            AdaptiveScale::new(1.0, target, 0.1..=10.0),
            Err(AdaptiveScaleError::InvalidTargetAcceptance { .. })
        ));
    }
    for (min, max) in [
        (0.0, 1.0),
        (-1.0, 1.0),
        (2.0, 1.0),
        (1.0, 0.0),
        (f64::NAN, 1.0),
        (1.0, f64::NAN),
        (1.0, f64::INFINITY),
        (f64::NEG_INFINITY, 1.0),
    ] {
        assert!(matches!(
            AdaptiveScale::new(1.0, 0.5, min..=max),
            Err(AdaptiveScaleError::InvalidBounds { .. })
        ));
    }
    for scale in [
        0.0,
        -1.0,
        0.01,
        11.0,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ] {
        assert!(matches!(
            AdaptiveScale::new(scale, 0.5, 0.1..=10.0),
            Err(AdaptiveScaleError::InvalidInitialScale { .. })
        ));
    }
}

#[test]
fn first_update_matches_closed_form_and_rejection_restores_state() {
    for (ratio, expected, accepted) in [
        (0.0, 0.5_f64.exp(), true),
        (f64::NEG_INFINITY, (-0.5_f64).exp(), false),
    ] {
        for workflow in 0..3 {
            let mut rng = StdRng::seed_from_u64(7);
            let mut tuning = AdaptiveScale::new(1.0, 0.5, 0.1..=10.0).unwrap();
            let mut proposal = Probe::new(ratio);
            let chain = Chain::new(false, &Flat).unwrap();
            let mut sampler = Sampler::new(chain, &Flat, &mut proposal, &mut rng).unwrap();
            match workflow {
                0 => sampler.warm_up(1, &mut tuning).unwrap(),
                1 => sampler.warm_up_mut(1, &mut tuning).unwrap(),
                _ => sampler.warm_up_delayed(1, &mut tuning).unwrap(),
            }
            assert_eq!(*sampler.chain_ref().state(), accepted);
            assert_eq!(sampler.chain_ref().accepted(), usize::from(accepted));
            assert_eq!(sampler.chain_ref().total_steps(), 1);
            assert_relative_eq!(tuning.scale(), expected, epsilon = 1e-14);
            assert_eq!(tuning.completed_steps(), 1);
            assert_eq!(
                sampler.proposal_ref().scale.to_bits(),
                tuning.scale().to_bits()
            );
        }
    }
}

#[test]
fn learning_rate_decreases_without_restarting_between_calls() {
    let mut rng = StdRng::seed_from_u64(7);
    let mut tuning = AdaptiveScale::new(1.0, 0.5, 0.1..=10.0).unwrap();
    let chain = Chain::new(false, &Flat).unwrap();
    let mut sampler = Sampler::new(chain, &Flat, Probe::new(0.0), &mut rng).unwrap();
    sampler.warm_up(1, &mut tuning).unwrap();
    sampler.proposal_mut().ratio = f64::NEG_INFINITY;
    sampler.warm_up(1, &mut tuning).unwrap();
    // n=1: log width = 1/2; n=2: subtract (1/2) * 2^(-3/5).
    let expected = (-0.5_f64).mul_add(0.659_753_955_386_447_1, 0.5).exp();
    assert_relative_eq!(tuning.scale(), expected, epsilon = 1e-14);
    assert_eq!(tuning.completed_steps(), 2);
}

#[test]
fn no_proposal_self_loops_reduce_scale_and_count_as_warmup() {
    for delayed in [false, true] {
        let mut rng = StdRng::seed_from_u64(7);
        let mut tuning = AdaptiveScale::new(1.0, 0.5, 0.1..=10.0).unwrap();
        let mut proposal = Probe::new(0.0);
        proposal.absent = true;
        let chain = Chain::new(false, &Flat).unwrap();
        let mut sampler = Sampler::new(chain, &Flat, proposal, &mut rng).unwrap();
        if delayed {
            sampler.warm_up_delayed(1, &mut tuning).unwrap();
        } else {
            sampler.warm_up_mut(1, &mut tuning).unwrap();
        }
        assert!(!sampler.chain_ref().state());
        assert_eq!(sampler.chain_ref().rejected(), 1);
        assert_eq!(tuning.completed_steps(), 1);
        assert_relative_eq!(tuning.scale(), (-0.5_f64).exp(), epsilon = 1e-14);
    }
}

#[test]
fn numerical_failures_preserve_tuning_and_completed_work() {
    for workflow in 0..3 {
        let mut rng = StdRng::seed_from_u64(7);
        let mut tuning = AdaptiveScale::new(1.0, 0.5, 0.1..=10.0).unwrap();
        let chain = Chain::new(false, &Flat).unwrap();
        let mut sampler = Sampler::new(chain, &Flat, Probe::new(0.0), &mut rng).unwrap();
        sampler.warm_up(1, &mut tuning).unwrap();
        let before = tuning.clone();
        sampler.proposal_mut().ratio = f64::NAN;
        match workflow {
            0 => assert_eq!(
                sampler.warm_up(10, &mut tuning),
                Err(McmcError::NanLogQRatio)
            ),
            1 => assert_eq!(
                sampler.warm_up_mut(10, &mut tuning),
                Err(McmcError::NanLogQRatio)
            ),
            _ => assert!(matches!(
                sampler.warm_up_delayed(10, &mut tuning),
                Err(DelayedStepError::Mcmc(McmcError::NanLogQRatio))
            )),
        }
        assert_eq!(tuning, before);
        assert!(*sampler.chain_ref().state());
        assert_eq!(sampler.chain_ref().total_steps(), 1);
        assert_eq!(sampler.chain_ref().log_prob().to_bits(), 0.0_f64.to_bits());
    }
}

#[test]
fn delayed_failures_do_not_adapt_or_count_a_transition() {
    for fault in [Fault::Plan, Fault::Score, Fault::Ratio, Fault::Commit] {
        let mut rng = StdRng::seed_from_u64(7);
        let mut tuning = AdaptiveScale::new(1.0, 0.5, 0.1..=10.0).unwrap();
        let before = tuning.clone();
        let mut proposal = Probe::new(0.0);
        proposal.fault = fault;
        let chain = Chain::new(false, &Flat).unwrap();
        let mut sampler = Sampler::new(chain, &Flat, proposal, &mut rng).unwrap();
        let expected = match fault {
            Fault::Plan => DelayedStepError::Plan(fault),
            Fault::Score => DelayedStepError::ProposedLogProb(fault),
            Fault::Ratio => DelayedStepError::LogQRatio(fault),
            Fault::Commit => DelayedStepError::Commit(fault),
            Fault::None => unreachable!("this test supplies a failure"),
        };
        assert_eq!(sampler.warm_up_delayed(10, &mut tuning), Err(expected));
        assert_eq!(tuning, before);
        assert!(!sampler.chain_ref().state());
        assert_eq!(sampler.chain_ref().total_steps(), 0);
    }
}

#[test]
fn zero_steps_are_a_no_op_and_production_keeps_the_final_scale() {
    let mut rng = StdRng::seed_from_u64(7);
    let mut tuning = AdaptiveScale::new(1.0, 0.5, 0.1..=10.0).unwrap();
    let chain = Chain::new(false, &Flat).unwrap();
    let mut sampler = Sampler::new(chain, &Flat, Probe::new(0.0), &mut rng).unwrap();
    sampler.warm_up(0, &mut tuning).unwrap();
    sampler.warm_up_mut(0, &mut tuning).unwrap();
    sampler.warm_up_delayed(0, &mut tuning).unwrap();
    assert_eq!(sampler.proposal_ref().scale.to_bits(), 99.0_f64.to_bits());
    assert_eq!(tuning.completed_steps(), 0);
    sampler.warm_up(10, &mut tuning).unwrap();
    let frozen = tuning.clone();
    sampler.reset_counters();
    sampler.run(5).unwrap();
    sampler.run_mut(5).unwrap();
    sampler.run_delayed(5).unwrap();
    assert_eq!(tuning, frozen);
    assert_eq!(
        sampler.proposal_ref().scale.to_bits(),
        frozen.scale().to_bits()
    );
    assert_eq!(sampler.chain_ref().total_steps(), 15);
}

#[test]
fn scale_bounds_hold_at_float_extremes_and_under_sustained_updates() {
    for (min, max) in [
        (f64::from_bits(1), f64::MAX),
        (f64::from_bits(1), f64::from_bits(1)),
        (f64::MAX, f64::MAX),
        (0.99, 1.01),
    ] {
        for (initial, ratio) in [(min, f64::NEG_INFINITY), (max, 0.0)] {
            let mut tuning = AdaptiveScale::new(initial, 0.5, min..=max).unwrap();
            let mut rng = StdRng::seed_from_u64(7);
            let chain = Chain::new(false, &Flat).unwrap();
            let mut sampler = Sampler::new(chain, &Flat, Probe::new(ratio), &mut rng).unwrap();
            sampler.warm_up(1_000, &mut tuning).unwrap();
            assert!(tuning.scale().is_finite());
            assert!(tuning.scale() > 0.0);
            assert!(tuning.bounds().contains(&tuning.scale()));
            assert_relative_eq!(tuning.target_acceptance(), 0.5);
        }
    }
}

#[test]
fn warmup_omits_metadata_and_preserves_transition_outcomes() {
    let run = |warmup, delayed, ratio, absent| {
        let mut rng = StdRng::seed_from_u64(10);
        let mut proposal = Probe::new(ratio);
        proposal.absent = absent;
        proposal.set_scale(1.0);
        let chain = Chain::new(false, &Flat).unwrap();
        let mut sampler = Sampler::new(chain, &Flat, proposal, &mut rng).unwrap();
        // Fixed bounds isolate the stepping path from changes to the kernel.
        let mut tuning = AdaptiveScale::new(1.0, 0.5, 1.0..=1.0).unwrap();
        match (warmup, delayed) {
            (true, true) => sampler.warm_up_delayed(100, &mut tuning).unwrap(),
            (true, false) => sampler.warm_up_mut(100, &mut tuning).unwrap(),
            (false, _) => {
                for _ in 0..100 {
                    if delayed {
                        let _ = sampler.step_delayed().unwrap();
                    } else {
                        let _ = sampler.step_mut().unwrap();
                    }
                }
            }
        }
        assert_eq!(tuning.completed_steps(), if warmup { 100 } else { 0 });
        let metadata_calls = sampler.proposal_ref().metadata_calls.get();
        let checkpoint = sampler.into_checkpoint();
        (checkpoint, metadata_calls, rng.random::<u64>())
    };
    for delayed in [false, true] {
        // Deterministic acceptance, rejection, absence, and mixed decisions.
        for (ratio, absent) in [
            (0.0, false),
            (f64::NEG_INFINITY, false),
            (0.0, true),
            (-1.0, false),
        ] {
            let (warmup_checkpoint, warmup_metadata, warmup_rng) =
                run(true, delayed, ratio, absent);
            let (step_checkpoint, step_metadata, step_rng) = run(false, delayed, ratio, absent);
            assert_eq!(warmup_metadata, 0);
            assert_eq!(step_metadata, 100);
            assert_eq!(warmup_checkpoint, step_checkpoint);
            assert_eq!(warmup_rng, step_rng);
        }
    }
}

struct Normal;

impl Target<f64> for Normal {
    fn log_prob(&self, x: &f64) -> f64 {
        -0.5 * x * x
    }
}

struct Walk(f64);

impl Proposal<f64> for Walk {
    fn propose<R: Rng + ?Sized>(&self, current: &f64, rng: &mut R) -> f64 {
        current + self.0 * rng.random_range(-1.0..1.0)
    }
}

impl TunableProposal for Walk {
    fn set_scale(&mut self, scale: f64) {
        self.0 = scale;
    }
}

#[test]
fn chunking_and_counter_resets_preserve_the_schedule_and_rng_stream() {
    let run = |chunked: bool| {
        let mut rng = StdRng::seed_from_u64(15);
        let mut tuning = AdaptiveScale::new(0.1, 0.44, 0.001..=100.0).unwrap();
        let chain = Chain::new(0.0, &Normal).unwrap();
        let mut sampler = Sampler::new(chain, &Normal, Walk(99.0), &mut rng).unwrap();
        if chunked {
            sampler.warm_up(37, &mut tuning).unwrap();
            sampler.reset_counters();
            sampler.warm_up(0, &mut tuning).unwrap();
            sampler.warm_up(963, &mut tuning).unwrap();
        } else {
            sampler.warm_up(1_000, &mut tuning).unwrap();
        }
        sampler.reset_counters();
        sampler.run(1_000).unwrap();
        (sampler.into_checkpoint(), tuning, rng.random::<u64>())
    };
    assert_eq!(run(false), run(true));
}

#[test]
fn tuned_normal_walk_recovers_analytical_moments_from_poor_initial_widths() {
    let mut means = OnlineStats::new();
    let mut second_moments = OnlineStats::new();
    for (group, initial) in [(0, 0.01), (1, 1.0), (2, 100.0)] {
        for seed in 0..8 {
            let mut rng = StdRng::seed_from_u64(100 + 8 * group + seed);
            let mut tuning = AdaptiveScale::new(initial, 0.44, 0.001..=1_000.0).unwrap();
            let chain = Chain::new(0.0, &Normal).unwrap();
            let mut sampler = Sampler::new(chain, &Normal, Walk(initial), &mut rng).unwrap();
            sampler.warm_up(10_000, &mut tuning).unwrap();
            assert!(
                (2.0..8.0).contains(&tuning.scale()),
                "{initial}: {tuning:?}"
            );
            // Allow settling under the frozen kernel before measuring moments.
            sampler.run(1_000).unwrap();
            sampler.reset_counters();
            let mut positions = OnlineStats::new();
            let mut squares = OnlineStats::new();
            for _ in 0..20_000 {
                let _ = sampler.step().unwrap();
                let x = *sampler.chain_ref().state();
                positions.try_push(x).unwrap();
                squares.try_push(x * x).unwrap();
            }
            assert!((sampler.chain_ref().acceptance_rate() - 0.44).abs() < 0.1);
            means.try_push(positions.mean().unwrap()).unwrap();
            second_moments.try_push(squares.mean().unwrap()).unwrap();
        }
    }
    // Independent chain means retain within-chain correlation effects. Eight
    // empirical standard errors is a conservative stochastic regression bound,
    // not a convergence proof. Analytical N(0,1) moments are 0 and 1.
    assert!(means.mean().unwrap().abs() < 8.0 * means.standard_error().unwrap());
    assert!(
        (second_moments.mean().unwrap() - 1.0).abs()
            < 8.0 * second_moments.standard_error().unwrap()
    );
}
