//! ACF, mean ESS, and classical split R-hat for four standard-normal chains.
//!
//! Run with: `cargo run --release --example diagnostics`
//! Chains run sequentially with distinct seeds and dispersed starts. Diagnostics
//! need comparable traces, not parallel execution. No optional feature is needed.

use markov_chain_monte_carlo::prelude::by_value::*;
use markov_chain_monte_carlo::{Autocorrelation, SplitRhat};
use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};

struct StandardNormal;

impl Target<f64> for StandardNormal {
    fn log_prob(&self, state: &f64) -> f64 {
        -0.5 * state * state
    }
}

struct RandomWalk;

impl Proposal<f64> for RandomWalk {
    fn propose<R: Rng + ?Sized>(&self, current: &f64, rng: &mut R) -> f64 {
        current + rng.random_range(-2.0..2.0)
    }
}

fn main() -> Result<(), McmcError> {
    const WARMUP: usize = 2_000;
    const DRAWS: usize = 10_000;
    const MAX_LAG: usize = 511;

    println!("Scalar diagnostics: four sequential N(0,1) chains");
    println!("warmup={WARMUP}, draws per chain={DRAWS}, recording interval=1, max lag={MAX_LAG}");
    println!("Observable: position; fixed uniform random-walk half-width=2.");
    let mut traces = Vec::with_capacity(4);
    for (id, (start, seed)) in [(-6.0, 40), (-2.0, 41), (2.0, 42), (6.0, 43)]
        .into_iter()
        .enumerate()
    {
        let target = StandardNormal;
        let proposal = RandomWalk;
        let mut rng = StdRng::seed_from_u64(seed);
        let chain = Chain::new(start, &target)?;
        let mut sampler = Sampler::new(chain, &target, &proposal, &mut rng)?;
        sampler.run(WARMUP)?;
        sampler.reset_counters();
        let mut draws = Vec::with_capacity(DRAWS);
        for _ in 0..DRAWS {
            let _ = sampler.step()?;
            // Record every post-step state, including repeated rejected states.
            draws.push(*sampler.chain_ref().state());
        }
        println!(
            "chain {id}: seed={seed}, start={start:+.1}, acceptance={:.3}",
            sampler.chain_ref().acceptance_rate()
        );
        match Autocorrelation::estimate(&draws, MAX_LAG) {
            Ok(acf) => {
                println!("chain {id}: ACF[1]={:.4}", acf.values()[1]);
                match acf.integrated_time() {
                    Ok(time) => println!(
                        "chain {id}: mean ESS={:.1}, integrated time={:.3}, window={}",
                        time.effective_sample_size(),
                        time.estimate(),
                        time.window()
                    ),
                    Err(error) => println!("chain {id}: mean ESS unavailable: {error}"),
                }
            }
            Err(error) => println!("chain {id}: ACF and mean ESS unavailable: {error}"),
        }
        traces.push(draws);
    }

    // Borrow separate chains: concatenating them would corrupt ACF/ESS ordering.
    let chains: Vec<&[f64]> = traces.iter().map(Vec::as_slice).collect();
    match SplitRhat::estimate(&chains) {
        Ok(rhat) => println!("Classical split R-hat: {:.4}", rhat.value()),
        Err(error) => println!("Classical split R-hat unavailable: {error}"),
    }
    println!(
        "Mean ESS is per chain; R-hat uses raw moments, without rank normalization or folding."
    );
    println!("These estimates do not certify convergence or exploration of all modes.");
    Ok(())
}
