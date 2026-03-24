//! 1-D Ising model sampled with `ProposalMut` (in-place mutation + rollback).
//!
//! Demonstrates the zero-copy MCMC API on a discrete, non-Clone state space.
//!
//! Run with: `cargo run --example ising_1d`

use markov_chain_monte_carlo::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng};

// --- State: a chain of ±1 spins (intentionally not Clone) ---

/// A one-dimensional chain of Ising spins.
struct SpinChain {
    spins: Vec<i8>,
}

impl SpinChain {
    /// Create a uniform +1 spin chain of the given length.
    fn all_up(n: usize) -> Self {
        Self { spins: vec![1; n] }
    }

    /// Magnetization per spin: m = (1/N) Σ `s_i`.
    fn magnetization(&self) -> f64 {
        let sum: i32 = self.spins.iter().map(|&s| i32::from(s)).sum();
        #[expect(
            clippy::cast_precision_loss,
            reason = "spin chain length won't exceed 2^52"
        )]
        let n = self.spins.len() as f64;
        f64::from(sum) / n
    }
}

// --- Target: nearest-neighbour Ising energy at inverse temperature β ---

/// Nearest-neighbour Ising Hamiltonian: H = −J Σ `s_i` · `s_{i+1}`.
///
/// `log_prob = −β H = β J Σ s_i · s_{i+1}`.
struct Ising {
    /// Coupling constant (positive = ferromagnetic).
    coupling: f64,
    /// Inverse temperature.
    beta: f64,
}

impl Target<SpinChain> for Ising {
    fn log_prob(&self, state: &SpinChain) -> f64 {
        let s = &state.spins;
        let interaction: f64 = s
            .windows(2)
            .map(|w| f64::from(w[0]) * f64::from(w[1]))
            .sum();
        self.beta * self.coupling * interaction
    }
}

// --- Proposal: flip one random spin ---

/// Single-site spin flip.  Undo token is the flipped site index.
struct SpinFlip;

impl ProposalMut<SpinChain> for SpinFlip {
    type Undo = usize;

    fn propose_mut<R: Rng + ?Sized>(&self, state: &mut SpinChain, rng: &mut R) -> Option<usize> {
        let idx = rng.random_range(0..state.spins.len());
        state.spins[idx] *= -1;
        Some(idx)
    }

    fn undo(&self, state: &mut SpinChain, idx: usize) {
        state.spins[idx] *= -1; // flipping twice = identity
    }
}

fn main() -> Result<(), McmcError> {
    let seed = 42;
    let mut rng = StdRng::seed_from_u64(seed);

    let n_spins = 50;
    let beta = 0.5; // moderate temperature
    let coupling = 1.0;

    let target = Ising { coupling, beta };
    let proposal = SpinFlip;
    let state = SpinChain::all_up(n_spins);
    let mut chain = Chain::new(state, &target)?;

    println!("1-D Ising model ({n_spins} spins, β={beta}, J={coupling}, seed={seed})");
    println!("Initial magnetization: {:.3}", chain.state.magnetization());

    // Burn-in
    let burn_in = 5_000;
    for _ in 0..burn_in {
        chain.step_mut(&target, &proposal, &mut rng)?;
    }
    println!(
        "After {burn_in} burn-in steps: m = {:.3}",
        chain.state.magnetization()
    );

    // Collect samples
    let n_samples: i32 = 20_000;
    let mut mag_sum = 0.0;
    let mut mag_sq_sum = 0.0;
    for _ in 0..n_samples {
        chain.step_mut(&target, &proposal, &mut rng)?;
        let m = chain.state.magnetization();
        mag_sum += m;
        mag_sq_sum += m * m;
    }

    let mean_mag = mag_sum / f64::from(n_samples);
    let mean_mag_sq = mag_sq_sum / f64::from(n_samples);
    let susceptibility = f64::from(n_samples) * (mean_mag_sq - mean_mag * mean_mag);

    println!("\nResults ({n_samples} samples):");
    println!("  <m>:             {mean_mag:+.4}");
    println!("  <m²>:            {mean_mag_sq:.4}");
    println!("  susceptibility:  {susceptibility:.2}");
    println!("  acceptance rate: {:.1}%", chain.acceptance_rate() * 100.0);

    Ok(())
}
