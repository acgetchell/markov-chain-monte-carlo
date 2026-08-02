//! Validate simple proposal distributions with detailed-balance helpers.
//!
//! Demonstrates by-value, in-place, delayed, and batch checks.
//!
//! Run with: `cargo run --example detailed_balance`

use core::convert::Infallible;

use markov_chain_monte_carlo::prelude::testing::*;
use rand::{Rng, SeedableRng, rngs::StdRng};

struct TwoStateTarget;
impl Target<bool> for TwoStateTarget {
    fn log_prob(&self, state: &bool) -> f64 {
        if *state { -2.0 } else { 0.0 }
    }
}

struct Flip;
impl Proposal<bool> for Flip {
    fn propose<R: Rng + ?Sized>(&self, current: &bool, _: &mut R) -> bool {
        !current
    }
}

struct FlipMut;
impl ProposalMut<bool> for FlipMut {
    type Undo = bool;
    type Info = bool;

    fn propose_mut<R: Rng + ?Sized>(&mut self, state: &mut bool, _: &mut R) -> Option<bool> {
        let previous = *state;
        *state = !*state;
        Some(previous)
    }

    fn info(&self, state: &bool, _token: &bool) -> bool {
        *state
    }

    fn undo(&mut self, state: &mut bool, token: bool) {
        *state = token;
    }
}

struct DelayedFlip;
impl DelayedProposal<bool> for DelayedFlip {
    type Plan = bool;
    type Info = bool;
    type Error = Infallible;

    fn propose_plan<R: Rng + ?Sized>(
        &mut self,
        state: &bool,
        _: &mut R,
    ) -> Result<Option<bool>, Infallible> {
        Ok(Some(!*state))
    }

    fn proposed_log_prob<T: Target<bool>>(
        &self,
        _: &bool,
        plan: &bool,
        target: &T,
    ) -> Result<f64, Infallible> {
        Ok(target.log_prob(plan))
    }

    fn info(&self, plan: &bool) -> bool {
        *plan
    }

    fn commit<R: Rng + ?Sized>(
        &mut self,
        state: &mut bool,
        plan: bool,
        _: &mut R,
    ) -> Result<(), Infallible> {
        *state = plan;
        Ok(())
    }
}

fn main() -> Result<(), DetailedBalanceError> {
    let target = TwoStateTarget;
    let config = DetailedBalanceConfig::new(256, 1e-10, 1)?;

    let mut rng = StdRng::seed_from_u64(42);
    let by_value = verify_detailed_balance(&false, &true, &target, &Flip, &mut rng, config)?;

    let mut rng = StdRng::seed_from_u64(42);
    let mut in_place_proposal = FlipMut;
    let in_place = verify_detailed_balance_mut(
        &false,
        &true,
        &target,
        &mut in_place_proposal,
        &mut rng,
        config,
    )?;

    let pairs = [(false, true), (true, false)];
    let mut rng = StdRng::seed_from_u64(42);
    let batch = verify_detailed_balance_many(
        pairs.iter().map(|(current, proposed)| (current, proposed)),
        &target,
        &Flip,
        &mut rng,
        config,
    );

    let forward = |plan: &bool| *plan;
    let reverse = |plan: &bool| !*plan;
    let delayed_transitions = [DetailedBalanceDelayedTransition::new(
        &false, &true, &forward, &reverse,
    )];
    let mut rng = StdRng::seed_from_u64(42);
    let mut delayed_proposal = DelayedFlip;
    let delayed = verify_detailed_balance_delayed_many(
        delayed_transitions,
        &target,
        &mut delayed_proposal,
        &mut rng,
        config,
    );

    assert!(batch.is_success());
    assert!(delayed.is_success());

    println!(
        "by-value residual: {:+.3e} (se {:.3e})",
        by_value.log_balance_residual, by_value.log_balance_standard_error
    );
    println!(
        "in-place residual: {:+.3e} (se {:.3e})",
        in_place.log_balance_residual, in_place.log_balance_standard_error
    );
    println!(
        "batch checks: {} by-value, {} delayed",
        batch.reports.len(),
        delayed.reports.len()
    );
    println!("Detailed balance checks passed");

    Ok(())
}
