//! Record delayed-step telemetry across resumable chunks.
//!
//! Demonstrates [`Sampler::run_delayed_chunk_observing`], which composes the
//! resumable continuation of [`Sampler::run_delayed_chunk`] with per-step
//! [`DelayedStep`] telemetry. For every step the observer receives the selected
//! move family ([`DelayedStep::info`]), the rejection reason
//! ([`DelayedStep::rejection_reason`]), and the post-step chain state, while the
//! sampler keeps ownership of the Metropolis-Hastings accept/reject draw and the
//! chain counters. Each chunk returns a [`ChainCheckpoint`] used to resume the
//! next chunk.
//!
//! This mirrors how a downstream crate (for example, CDT physics) can keep
//! domain-specific statistics outside the generic sampler.
//!
//! Run with: `cargo run --example delayed_chunked_telemetry`

use core::convert::Infallible;

use markov_chain_monte_carlo::prelude::delayed::*;
use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng};

// --- Move family recorded as per-step telemetry ---

#[derive(Clone, Copy, Debug)]
enum Move {
    Up,
    Down,
}

// --- Target: mild bell centered on zero over an integer lattice ---

struct Centered;
impl Target<i32> for Centered {
    fn log_prob(&self, state: &i32) -> f64 {
        let coordinate = f64::from(*state);
        -0.5 * 0.1 * coordinate * coordinate
    }
}

// --- Delayed proposal: a bounded +/-1 walk that reports its move family ---
//
// When the drawn direction would leave the bounded region there is no valid
// site, so `propose_plan` returns `Ok(None)`. The chosen family is still
// reported through `no_plan_info`, so no-site self-loops carry telemetry too.

const LATTICE_BOUND: u32 = 5;

struct BoundedWalk {
    last_family: Option<Move>,
}

impl DelayedProposal<i32> for BoundedWalk {
    type Plan = i32;
    type Info = Move;
    type Error = Infallible;

    fn propose_plan<R: Rng + ?Sized>(
        &mut self,
        state: &i32,
        rng: &mut R,
    ) -> Result<Option<i32>, Infallible> {
        let up = rng.random_range(0..2) == 0;
        let (family, delta) = if up { (Move::Up, 1) } else { (Move::Down, -1) };
        self.last_family = Some(family);

        if (state + delta).unsigned_abs() > LATTICE_BOUND {
            // No valid site in this direction; telemetry still records the family.
            Ok(None)
        } else {
            Ok(Some(delta))
        }
    }

    fn no_plan_info(&mut self) -> Option<Move> {
        self.last_family.take()
    }

    fn proposed_log_prob<T: Target<i32>>(
        &self,
        state: &i32,
        plan: &i32,
        target: &T,
    ) -> Result<f64, Infallible> {
        Ok(target.log_prob(&(state + plan)))
    }

    fn info(&self, plan: &i32) -> Move {
        if *plan > 0 { Move::Up } else { Move::Down }
    }

    fn commit<R: Rng + ?Sized>(
        &mut self,
        state: &mut i32,
        plan: i32,
        _: &mut R,
    ) -> Result<(), Infallible> {
        *state += plan;
        Ok(())
    }
}

fn main() -> Result<(), DelayedStepError<Infallible>> {
    let seed = 42;
    let chunk_len = 200;
    let chunks = 3;

    let mut rng = StdRng::seed_from_u64(seed);
    let target = Centered;
    let mut proposal = BoundedWalk { last_family: None };
    let chain = Chain::new(0, &target).map_err(DelayedStepError::Mcmc)?;
    let mut sampler =
        Sampler::new(chain, &target, &mut proposal, &mut rng).map_err(DelayedStepError::Mcmc)?;

    println!("Delayed chunked telemetry (seed={seed})");

    // Domain-specific statistics kept outside the generic sampler.
    let mut up = 0_u64;
    let mut down = 0_u64;
    let mut accepted = 0_u64;
    let mut rejected = 0_u64;
    let mut rejected_mh = 0_u64;
    let mut no_site = 0_u64;
    let mut state_sum = 0.0_f64;
    let mut observed = 0.0_f64;

    for chunk in 1..=chunks {
        let continuation = sampler.run_delayed_chunk_observing(chunk_len, |step, state| {
            // Selected family is available for every step, including no-site loops.
            if let Some(family) = step.info {
                match family {
                    Move::Up => up += 1,
                    Move::Down => down += 1,
                }
            }

            // The sampler owns the accept/reject draw; we only classify the
            // recorded outcome. `rejection_reason` is non-exhaustive, so the
            // breakdown keeps a wildcard arm for forward compatibility.
            if step.outcome.is_accepted() {
                accepted += 1;
            } else {
                rejected += 1;
                match step.rejection_reason() {
                    Some(StepRejectionReason::RejectedProposal) => rejected_mh += 1,
                    Some(StepRejectionReason::NoProposal) => no_site += 1,
                    _ => {}
                }
            }

            // Post-step measurement read from the state the sampler still owns.
            state_sum += f64::from(*state);
            observed += 1.0;
        })?;

        println!(
            "  chunk {chunk}: total steps {}, state {}",
            continuation.total_steps(),
            **continuation.state()
        );
    }

    // Every step carries a family and is classified as accepted or rejected.
    assert_eq!(up + down, accepted + rejected);

    let mean_state = if observed > 0.0 {
        state_sum / observed
    } else {
        0.0
    };

    println!("\nPer-step telemetry across {} steps:", accepted + rejected);
    println!("  Up proposals:       {up}");
    println!("  Down proposals:     {down}");
    println!("  Accepted:           {accepted}");
    println!("  MH rejections:      {rejected_mh}");
    println!("  No-site self-loops: {no_site}");
    println!("  Mean state:         {mean_state:+.4}");
    println!("Delayed chunked telemetry complete");

    Ok(())
}
