//! Iterator-based sampling from a 1D standard normal distribution.
//!
//! Demonstrates [`Sampler`]'s `Iterator` implementation for idiomatic Rust
//! composability with `.take(n)`, `.by_ref()`, and standard iterator adaptors.
//!
//! Run with: `cargo run --example iterator_sampling`

use markov_chain_monte_carlo::prelude::by_value::*;
use markov_chain_monte_carlo::prelude::{OnlineStats, StatisticsError};
use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng};

// --- State ---

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
        Scalar(current.0 + rng.random_range(-self.width..self.width))
    }
}

fn main() -> ObservedIntoRunResult<McmcError, StatisticsError> {
    let seed = 42;
    let mut rng = StdRng::seed_from_u64(seed);

    let target = StandardNormal;
    let proposal = RandomWalk { width: 1.0 };
    let mut sampler = Sampler::from_state(Scalar(5.0), &target, &proposal, &mut rng)
        .map_err(ObservedStreamError::Step)?;

    println!("Iterator-based sampling of N(0,1) (seed={seed})");

    // Burn-in using iterator: take exactly 1000 steps, stop on first error.
    // This is the primary use case for the Iterator impl — fire-and-forget
    // stepping where you don't need to inspect state between steps.
    let burn_in = 1_000;
    sampler
        .by_ref()
        .take(burn_in)
        .try_for_each(|result| result.map(|_step| ()))
        .map_err(ObservedStreamError::Step)?;
    println!("Burn-in complete ({burn_in} steps via iterator)");

    // Reset counters so acceptance rate reflects production only
    sampler.reset_counters();

    // Collect samples: use step() when you need to read state between steps.
    // The iterator yields step telemetry. Here we also observe the resulting state.
    let n_samples = 10_000;
    let mut stats = OnlineStats::new();
    for _ in 0..n_samples {
        let _ = sampler.step().map_err(ObservedStreamError::Step)?;
        stats
            .try_push(sampler.chain_ref().state().0)
            .map_err(ObservedStreamError::Accumulation)?;
    }

    println!("\nResults ({n_samples} samples via step()):");
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
