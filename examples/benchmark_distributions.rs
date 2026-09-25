//! Seeded reference-target sampling with analytical moments and scalar mean ESS.
//!
//! Run with: `cargo run --release --features benchmarks --example benchmark_distributions`
//! Fixed random walks deliberately expose difficult geometry and missed modes.
//! Output is illustrative; neither moment agreement nor a finite ESS proves convergence.

use std::time::Instant;

use markov_chain_monte_carlo::prelude::by_value::*;
use markov_chain_monte_carlo::{Autocorrelation, BenchmarkTarget};
use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng};

struct RandomWalk {
    widths: [f64; 2],
}

impl Proposal<[f64; 2]> for RandomWalk {
    fn propose<R: Rng + ?Sized>(&self, current: &[f64; 2], rng: &mut R) -> [f64; 2] {
        std::array::from_fn(|i| current[i] + rng.random_range(-self.widths[i]..self.widths[i]))
    }
}

fn main() -> Result<(), McmcError> {
    const WARMUP: u32 = 5_000;
    const DRAWS: u16 = 20_000;
    const MAX_LAG: usize = 511;
    println!("Benchmark distributions: 2D fixed reference targets");
    println!("warmup={WARMUP}, draws={DRAWS}, recording interval=1, max lag={MAX_LAG}");
    println!("Timing includes production transitions and recording; excludes warmup and analysis.");
    println!("Moment errors and ESS are illustrative, not a convergence certificate.");

    for (target, start, widths, seed) in [
        (BenchmarkTarget::Rosenbrock, [1.0, 1.0], [0.5, 0.5], 20),
        (BenchmarkTarget::NealsFunnel, [0.0, 0.0], [1.0, 1.0], 21),
        (
            BenchmarkTarget::GaussianMixture,
            [-5.0, 0.0],
            [1.0, 1.0],
            22,
        ),
        (BenchmarkTarget::Banana, [0.0, -3.0], [3.0, 1.0], 23),
    ] {
        let mut rng = StdRng::seed_from_u64(seed);
        let proposal = RandomWalk { widths };
        let mut chain = Chain::new(start, &target)?;
        for _ in 0..WARMUP {
            let _ = chain.step(&target, &proposal, &mut rng)?;
        }
        chain.reset_counters();
        let mut samples: [Vec<f64>; 2] =
            std::array::from_fn(|_| Vec::with_capacity(usize::from(DRAWS)));
        let started = Instant::now();
        for _ in 0..DRAWS {
            let _ = chain.step(&target, &proposal, &mut rng)?;
            for (column, value) in samples.iter_mut().zip(chain.state()) {
                column.push(*value);
            }
        }
        let elapsed = started.elapsed();
        println!(
            "\n{target:?}: seed={seed}, start={start:?}, uniform proposal half-widths={widths:?}"
        );
        println!("Acceptance rate: {:.3}", chain.acceptance_rate());
        println!("Production seconds: {:.6}", elapsed.as_secs_f64());
        let expected_mean = target.mean();
        let expected_covariance = target.covariance();
        let means: [f64; 2] =
            std::array::from_fn(|i| samples[i].iter().sum::<f64>() / f64::from(DRAWS));
        let cross_covariance = samples[0]
            .iter()
            .zip(&samples[1])
            .map(|(x, y)| (x - means[0]) * (y - means[1]))
            .sum::<f64>()
            / f64::from(DRAWS - 1);
        println!(
            "Cross covariance: {cross_covariance:.5}, expected={:.5}, error={:+.5}",
            expected_covariance[0][1],
            cross_covariance - expected_covariance[0][1]
        );
        for (i, column) in samples.iter().enumerate() {
            let mean = means[i];
            let variance =
                column.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / f64::from(DRAWS - 1);
            println!(
                "coordinate {i}: mean={mean:.5}, expected={:.5}, error={:+.5}; variance={variance:.5}, expected={:.5}",
                expected_mean[i],
                mean - expected_mean[i],
                expected_covariance[i][i]
            );
            // Preserve diagnostic failures, including an insufficient lag window.
            match Autocorrelation::estimate(column, MAX_LAG).and_then(|acf| acf.integrated_time()) {
                Ok(time) => {
                    println!("  mean ESS: {:.1}", time.effective_sample_size());
                    match time.effective_sample_size_per_second(elapsed) {
                        Ok(rate) => println!("  mean ESS/second: {rate:.1}"),
                        Err(error) => println!("  mean ESS/second unavailable: {error}"),
                    }
                }
                Err(error) => println!("  mean ESS unavailable: {error}"),
            }
        }
    }
    Ok(())
}
