//! Sample from a 1D standard normal distribution using Metropolis–Hastings.
//!
//! Demonstrates [`Sampler`] with `run` for burn-in and [`OnlineStats`] for
//! streaming measurements. This is a mechanics example: the target is known,
//! the proposal is symmetric, and the printed summary is a quick sanity check
//! rather than a convergence proof.
//!
//! Run with: `cargo run --example normal_1d`

use markov_chain_monte_carlo::prelude::by_value::*;
use markov_chain_monte_carlo::prelude::{OnlineStats, StatisticsError};
use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng};

// --- State ---

struct Scalar(f64);

// --- Target: N(0,1), up to an additive log-normalization constant ---

struct StandardNormal;
impl Target<Scalar> for StandardNormal {
    fn log_prob(&self, state: &Scalar) -> f64 {
        -0.5 * state.0 * state.0
    }
}

// --- Proposal: symmetric random walk, so log q(x | y) - log q(y | x) = 0 ---

struct RandomWalk {
    width: f64,
}
impl Proposal<Scalar> for RandomWalk {
    fn propose<R: Rng + ?Sized>(&self, current: &Scalar, rng: &mut R) -> Scalar {
        let delta: f64 = rng.random_range(-self.width..self.width);
        Scalar(current.0 + delta)
    }
}

fn main() -> ObservedIntoRunResult<McmcError, StatisticsError> {
    let seed = 42;
    let mut rng = StdRng::seed_from_u64(seed);

    let target = StandardNormal;
    let proposal = RandomWalk { width: 1.0 };
    let mut sampler = Sampler::from_state(Scalar(5.0), &target, &proposal, &mut rng)
        .map_err(ObservedStreamError::Step)?;

    println!("Sampling N(0,1) with Metropolis–Hastings (seed={seed})");
    println!("Initial state: x = {:.3}", sampler.chain_ref().state().0);

    // Burn-in
    let burn_in = 1_000;
    sampler.run(burn_in).map_err(ObservedStreamError::Step)?;
    println!(
        "After {burn_in} burn-in steps: x = {:.3}",
        sampler.chain_ref().state().0
    );

    // Reset counters so acceptance rate reflects production only
    sampler.reset_counters();

    // Collect samples
    let n_samples = 10_000;
    let mut stats = OnlineStats::new();
    sampler.run_observing_into(n_samples, &mut |state: &Scalar| state.0, &mut stats)?;

    println!("\nResults ({n_samples} samples):");
    if let (Some(mean), Some(variance)) = (stats.mean(), stats.sample_variance()) {
        println!("  Sample mean:     {mean:+.4} (expected: 0.0)");
        println!("  Sample variance: {variance:.4} (expected: 1.0)");
    }
    println!(
        "  Acceptance rate: {:.1}%",
        sampler.chain_ref().acceptance_rate() * 100.0
    );
    Ok(())
}
