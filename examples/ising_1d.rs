//! 1-D Ising model sampled with `ProposalMut` (in-place mutation + rollback).
//!
//! Demonstrates [`Sampler`] with `run_mut` for burn-in, `step_mut` for
//! per-sample collection on a discrete, non-Clone state space, and
//! [`TraceRecorder`] for CSV trace export.  The example is intentionally small:
//! it shows the sampler contract for a familiar statistical-physics model,
//! not a finite-size scaling study.
//!
//! Run with: `just example ising_1d`

use std::error::Error;
use std::fmt;
use std::fs::{self, File};
use std::io;

use markov_chain_monte_carlo::prelude::in_place::*;
use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng};

/// Errors from the Ising trace example.
#[derive(Debug)]
enum ExampleError {
    /// MCMC transition or initialization failed.
    Mcmc(McmcError),
    /// Trace construction failed.
    Trace(TraceError),
    /// CSV export failed.
    Io(io::Error),
}

impl fmt::Display for ExampleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mcmc(err) => write!(f, "{err}"),
            Self::Trace(err) => write!(f, "{err}"),
            Self::Io(err) => write!(f, "{err}"),
        }
    }
}

impl Error for ExampleError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Mcmc(err) => Some(err),
            Self::Trace(err) => Some(err),
            Self::Io(err) => Some(err),
        }
    }
}

impl From<McmcError> for ExampleError {
    fn from(err: McmcError) -> Self {
        Self::Mcmc(err)
    }
}

impl From<TraceError> for ExampleError {
    fn from(err: TraceError) -> Self {
        Self::Trace(err)
    }
}

impl From<io::Error> for ExampleError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

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

impl Ising {
    /// Energy of the nearest-neighbour spin chain.
    fn energy(&self, state: &SpinChain) -> f64 {
        let interaction: f64 = state
            .spins
            .windows(2)
            .map(|w| f64::from(w[0]) * f64::from(w[1]))
            .sum();
        -self.coupling * interaction
    }
}

impl Target<SpinChain> for Ising {
    fn log_prob(&self, state: &SpinChain) -> f64 {
        -self.beta * self.energy(state)
    }
}

// --- Proposal: flip one random spin, symmetric over sites for non-empty chains ---

/// Single-site spin flip.  Undo token is the flipped site index.
struct SpinFlip;

impl ProposalMut<SpinChain> for SpinFlip {
    type Undo = usize;

    fn propose_mut<R: Rng + ?Sized>(&self, state: &mut SpinChain, rng: &mut R) -> Option<usize> {
        if state.spins.is_empty() {
            return None;
        }
        let idx = rng.random_range(0..state.spins.len());
        state.spins[idx] *= -1;
        Some(idx)
    }

    fn undo(&self, state: &mut SpinChain, idx: usize) {
        state.spins[idx] *= -1; // flipping twice = identity
    }
}

fn main() -> Result<(), ExampleError> {
    let seed = 42;
    let mut rng = StdRng::seed_from_u64(seed);

    let n_spins = 50_usize;
    #[expect(
        clippy::cast_precision_loss,
        reason = "example spin count is small enough to represent exactly"
    )]
    let n_spins_f64 = n_spins as f64;
    let beta = 0.5; // moderate temperature
    let coupling = 1.0;

    let target = Ising { coupling, beta };
    let proposal = SpinFlip;
    let chain = Chain::new(SpinChain::all_up(n_spins), &target)?;
    let mut sampler = Sampler::new(chain, &target, &proposal, &mut rng)?;

    println!("1-D Ising model ({n_spins} spins, β={beta}, J={coupling}, seed={seed})");
    println!(
        "Initial magnetization: {:.3}",
        sampler.chain_ref().state().magnetization()
    );

    // Burn-in
    let burn_in = 5_000;
    sampler.run_mut(burn_in)?;
    println!(
        "After {burn_in} burn-in steps: m = {:.3}",
        sampler.chain_ref().state().magnetization()
    );

    // Reset counters so acceptance rate reflects production only
    sampler.reset_counters();

    // Collect samples and export a reusable trace for downstream diagnostics.
    let n_samples: u32 = 20_000;
    let mut mag_sum = 0.0;
    let mut mag_sq_sum = 0.0;
    let mut trace = TraceRecorder::new(ChainId::new(0), ["energy", "magnetization"])?;
    for _ in 0..n_samples {
        let accepted = sampler.step_mut()?;
        let chain = sampler.chain_ref();
        let state = chain.state();
        let energy = target.energy(state);
        let m = state.magnetization();
        trace.record(
            chain,
            TraceStepOutcome::from_proposal_acceptance(accepted),
            [energy, m],
        )?;
        mag_sum += m;
        mag_sq_sum += m * m;
    }
    fs::create_dir_all("target")?;
    let trace_path = "target/ising_1d_trace.csv";
    trace.into_trace().write_csv(File::create(trace_path)?)?;

    let mean_mag = mag_sum / f64::from(n_samples);
    let mean_mag_sq = mag_sq_sum / f64::from(n_samples);
    let susceptibility = beta * n_spins_f64 * (mean_mag_sq - mean_mag * mean_mag);

    println!("\nResults ({n_samples} samples):");
    println!("  <m>:             {mean_mag:+.4}");
    println!("  <m²>:            {mean_mag_sq:.4}");
    println!("  susceptibility:  {susceptibility:.2}");
    println!(
        "  acceptance rate: {:.1}%",
        sampler.chain_ref().acceptance_rate() * 100.0
    );
    println!("  trace CSV:        {trace_path}");

    Ok(())
}
