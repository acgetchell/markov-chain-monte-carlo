//! Add a target-bias term to Metropolis-Hastings acceptance.
//!
//! Demonstrates [`AdditiveTarget`] for composing a model log weight with an
//! auxiliary bias term.  This is the same place an externally supplied
//! regularizer would enter: as part of the target log weight, not as an ad hoc
//! rejection filter.  The proposal remains symmetric, so the Hastings
//! correction is still supplied by the proposal API and defaults to zero here.
//!
//! Run with: `cargo run --example additive_target_bias`

use markov_chain_monte_carlo::prelude::by_value::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// --- Target components over a two-state model ---

struct FlatModel;
impl Target<bool> for FlatModel {
    fn log_prob(&self, _: &bool) -> f64 {
        0.0
    }
}

struct BiasTowardTrue {
    true_log_weight: f64,
}
impl Target<bool> for BiasTowardTrue {
    fn log_prob(&self, state: &bool) -> f64 {
        if *state { self.true_log_weight } else { 0.0 }
    }
}

// --- Proposal: symmetric flip between the two states ---

struct Flip;
impl Proposal<bool> for Flip {
    fn propose<R: Rng + ?Sized>(&self, current: &bool, _: &mut R) -> bool {
        !*current
    }
}

fn main() -> Result<(), McmcError> {
    let seed = 42;
    let mut rng = StdRng::seed_from_u64(seed);

    let target = AdditiveTarget::new(
        FlatModel,
        BiasTowardTrue {
            true_log_weight: 3.0_f64.ln(),
        },
    );
    let proposal = Flip;
    let chain = Chain::new(false, &target)?;
    let mut sampler = Sampler::new(chain, &target, &proposal, &mut rng)?;

    let false_weight = target.log_prob(&false).exp();
    let true_weight = target.log_prob(&true).exp();
    let expected_true_fraction = true_weight / (false_weight + true_weight);

    println!("AdditiveTarget bias example (seed={seed})");
    println!("  model log weight: flat");
    println!("  bias log weight for true: ln(3)");
    println!("  proposal log_q_ratio: symmetric default 0");
    println!("  expected P(true): {expected_true_fraction:.3}");

    sampler.run(1_000)?;
    sampler.reset_counters();

    let n_samples: u32 = 20_000;
    let mut true_count = 0_u32;
    for _ in 0..n_samples {
        let _ = sampler.step()?;
        true_count += u32::from(*sampler.chain_ref().state());
    }

    let observed_true_fraction = f64::from(true_count) / f64::from(n_samples);

    println!("\nResults ({n_samples} samples):");
    println!("  observed P(true): {observed_true_fraction:.3}");
    println!(
        "  acceptance rate: {:.1}%",
        sampler.chain_ref().acceptance_rate() * 100.0
    );

    Ok(())
}
