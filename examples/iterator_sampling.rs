//! Iterator-based sampling from a 1D standard normal distribution.
//!
//! Demonstrates [`Sampler`]'s `Iterator` implementation for idiomatic Rust
//! composability with `.take(n)`, `.by_ref()`, and standard iterator adaptors.
//!
//! Run with: `cargo run --example iterator_sampling`

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
        Scalar(current.0 + rng.random_range(-self.width..self.width))
    }
}

fn main() -> Result<(), McmcError> {
    let seed = 42;
    let mut rng = StdRng::seed_from_u64(seed);

    let target = StandardNormal;
    let proposal = RandomWalk { width: 1.0 };
    let chain = Chain::new(Scalar(5.0), &target)?;
    let mut sampler = Sampler::new(chain, &target, &proposal, &mut rng)?;

    println!("Iterator-based sampling of N(0,1) (seed={seed})");

    // Burn-in using iterator: take exactly 1000 steps, stop on first error.
    // This is the primary use case for the Iterator impl — fire-and-forget
    // stepping where you don't need to inspect state between steps.
    let burn_in = 1_000;
    sampler
        .by_ref()
        .take(burn_in)
        .try_for_each(|result| result)?;
    println!("Burn-in complete ({burn_in} steps via iterator)");

    // Reset counters so acceptance rate reflects production only
    sampler.reset_counters();

    // Collect samples: use step() when you need to read state between steps.
    // The iterator yields Result<(), McmcError> (not the state), so collecting
    // samples requires interleaving step + state access.
    let n_samples: u32 = 10_000;
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    for _ in 0..n_samples {
        sampler.step()?;
        let x = sampler.chain_ref().state().0;
        sum += x;
        sum_sq += x * x;
    }

    let mean = sum / f64::from(n_samples);
    let variance = sum_sq / f64::from(n_samples) - mean * mean;

    println!("\nResults ({n_samples} samples via iterator):");
    println!("  Sample mean:     {mean:+.4} (expected: 0.0)");
    println!("  Sample variance: {variance:.4} (expected: 1.0)");
    println!(
        "  Acceptance rate: {:.1}%",
        sampler.chain_ref().acceptance_rate() * 100.0
    );

    Ok(())
}
