//! Sample from a 1D standard normal distribution using Metropolis–Hastings.
//!
//! Demonstrates [`Sampler`] with `run` for burn-in and `step` for
//! per-sample collection.
//!
//! Run with: `cargo run --example normal_1d`

use markov_chain_monte_carlo::prelude::by_value::*;
use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng};

// --- State ---

#[derive(Clone, Debug)]
struct Scalar(f64);

// --- Target: N(0,1) ---

struct StandardNormal;
impl Target<Scalar> for StandardNormal {
    fn log_prob(&self, state: &Scalar) -> f64 {
        -0.5 * state.0 * state.0
    }
}

// --- Proposal: symmetric random walk ---

struct RandomWalk {
    width: f64,
}
impl Proposal<Scalar> for RandomWalk {
    fn propose<R: Rng + ?Sized>(&self, current: &Scalar, rng: &mut R) -> Scalar {
        let delta: f64 = rng.random_range(-self.width..self.width);
        Scalar(current.0 + delta)
    }
}

fn main() -> Result<(), McmcError> {
    let seed = 42;
    let mut rng = StdRng::seed_from_u64(seed);

    let target = StandardNormal;
    let proposal = RandomWalk { width: 1.0 };
    let chain = Chain::new(Scalar(5.0), &target)?;
    let mut sampler = Sampler::new(chain, &target, &proposal, &mut rng)?;

    println!("Sampling N(0,1) with Metropolis–Hastings (seed={seed})");
    println!("Initial state: x = {:.3}", sampler.chain_ref().state().0);

    // Burn-in
    let burn_in = 1_000;
    sampler.run(burn_in)?;
    println!(
        "After {burn_in} burn-in steps: x = {:.3}",
        sampler.chain_ref().state().0
    );

    // Reset counters so acceptance rate reflects production only
    sampler.reset_counters();

    // Collect samples
    let n_samples = 10_000;
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    for _ in 0..n_samples {
        sampler.step()?;
        sum += sampler.chain_ref().state().0;
        sum_sq += sampler.chain_ref().state().0 * sampler.chain_ref().state().0;
    }

    let mean = sum / f64::from(n_samples);
    let variance = sum_sq / f64::from(n_samples) - mean * mean;

    println!("\nResults ({n_samples} samples):");
    println!("  Sample mean:     {mean:+.4} (expected: 0.0)");
    println!("  Sample variance: {variance:.4} (expected: 1.0)");
    println!(
        "  Acceptance rate: {:.1}%",
        sampler.chain_ref().acceptance_rate() * 100.0
    );
    Ok(())
}
