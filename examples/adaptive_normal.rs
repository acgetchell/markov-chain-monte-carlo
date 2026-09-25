//! Tune a symmetric random-walk width, then sample with a fixed proposal.
//!
//! Run with: `cargo run --release --example adaptive_normal`

use core::fmt;
use std::error::Error;

use markov_chain_monte_carlo::prelude::by_value::*;
use markov_chain_monte_carlo::prelude::{AdaptiveScale, AdaptiveScaleError, TunableProposal};
use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};

struct Normal;

impl Target<f64> for Normal {
    fn log_prob(&self, x: &f64) -> f64 {
        -0.5 * x * x
    }
}

struct Walk {
    width: f64,
}

impl Proposal<f64> for Walk {
    fn propose<R: Rng + ?Sized>(&self, x: &f64, rng: &mut R) -> f64 {
        x + self.width * rng.random_range(-1.0..1.0)
    }
}

impl TunableProposal for Walk {
    fn set_scale(&mut self, scale: f64) {
        self.width = scale;
    }
}

#[derive(Debug)]
enum ExampleError {
    Adaptation(AdaptiveScaleError),
    Sampling(McmcError),
}

impl fmt::Display for ExampleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Adaptation(error) => error.fmt(f),
            Self::Sampling(error) => error.fmt(f),
        }
    }
}

impl Error for ExampleError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Adaptation(error) => Some(error),
            Self::Sampling(error) => Some(error),
        }
    }
}

fn main() -> Result<(), ExampleError> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut tuning =
        AdaptiveScale::new(0.01, 0.44, 0.001..=100.0).map_err(ExampleError::Adaptation)?;
    let chain = Chain::new(5.0, &Normal).map_err(ExampleError::Sampling)?;
    let mut sampler = Sampler::new(chain, &Normal, Walk { width: 0.01 }, &mut rng)
        .map_err(ExampleError::Sampling)?;

    println!(
        "Adaptive normal sampling: initial width = {}",
        tuning.scale()
    );
    // These draws are discarded. Reusing tuning preserves its learning schedule.
    sampler
        .warm_up(5_000, &mut tuning)
        .map_err(ExampleError::Sampling)?;
    sampler
        .warm_up(5_000, &mut tuning)
        .map_err(ExampleError::Sampling)?;
    println!("Frozen proposal width = {:.4}", tuning.scale());

    // Allow additional settling under the fixed kernel, then measure production.
    sampler.run(1_000).map_err(ExampleError::Sampling)?;
    sampler.reset_counters();
    let count = 50_000;
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    for _ in 0..count {
        let _ = sampler.step().map_err(ExampleError::Sampling)?;
        let x = *sampler.chain_ref().state();
        sum += x;
        sum_sq = x.mul_add(x, sum_sq);
    }
    let mean = sum / f64::from(count);
    println!("Sample mean: {mean:.4} (analytical: 0)");
    println!(
        "Second moment: {:.4} (analytical: 1)",
        sum_sq / f64::from(count)
    );
    println!(
        "Acceptance rate: {:.3} (tuning target: 0.44)",
        sampler.chain_ref().acceptance_rate()
    );
    println!("Tuning and these moment checks do not establish convergence.");
    Ok(())
}
