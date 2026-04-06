//! Markov Chain Monte Carlo (MCMC) framework.
//!
//! 🚧 **Pre-release (0.x)** — This crate is under active development and
//! not yet ready for production use. APIs may change without notice.
//!
//! This crate aims to provide a composable, zero-cost abstraction for MCMC
//! methods over arbitrary state spaces, including discrete and combinatorial
//! systems (e.g., triangulations).
//!
//! # Example
//!
//! Sample from a standard normal distribution using Metropolis–Hastings:
//!
//! ```
//! use markov_chain_monte_carlo::prelude::*;
//! use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};
//!
//! #[derive(Clone)]
//! struct Scalar(f64);
//!
//! struct Normal;
//! impl Target<Scalar> for Normal {
//!     fn log_prob(&self, state: &Scalar) -> f64 {
//!         -0.5 * state.0 * state.0
//!     }
//! }
//!
//! struct RandomWalk { width: f64 }
//! impl Proposal<Scalar> for RandomWalk {
//!     fn propose<R: Rng + ?Sized>(&self, current: &Scalar, rng: &mut R) -> Scalar {
//!         let delta: f64 = rng.random_range(-self.width..self.width);
//!         Scalar(current.0 + delta)
//!     }
//! }
//!
//! fn main() -> Result<(), McmcError> {
//!     let mut rng = StdRng::seed_from_u64(42);
//!     let mut chain = Chain::new(Scalar(0.0), &Normal)?;
//!     let proposal = RandomWalk { width: 1.0 };
//!
//!     for _ in 0..1000 {
//!         chain.step(&Normal, &proposal, &mut rng)?;
//!     }
//!
//!     assert!(chain.acceptance_rate() > 0.2);
//!     Ok(())
//! }
//! ```
//!
//! # In-place mutation with rollback
//!
//! For state spaces where cloning is expensive, use [`ProposalMut`] with
//! [`Chain::step_mut`].  The proposal mutates the state in place and returns
//! a small undo token for rollback on rejection:
//!
//! ```
//! use markov_chain_monte_carlo::prelude::*;
//! use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};
//!
//! /// A lattice of spins (not Clone — only mutated in place).
//! struct SpinChain { spins: Vec<i8> }
//!
//! /// Energy = −Σ s_i · s_{i+1}  (1-D Ising, no field).
//! struct Ising;
//! impl Target<SpinChain> for Ising {
//!     fn log_prob(&self, state: &SpinChain) -> f64 {
//!         let s = &state.spins;
//!         let energy: f64 = s.windows(2)
//!             .map(|w| -f64::from(w[0]) * f64::from(w[1]))
//!             .sum();
//!         -energy  // log_prob = −E  (T = 1)
//!     }
//! }
//!
//! /// Flip one random spin; undo token is the site index.
//! struct SpinFlip;
//! impl ProposalMut<SpinChain> for SpinFlip {
//!     type Undo = usize;
//!     fn propose_mut<R: Rng + ?Sized>(&self, state: &mut SpinChain, rng: &mut R) -> Option<usize> {
//!         if state.spins.is_empty() { return None; }
//!         let idx = rng.random_range(0..state.spins.len());
//!         state.spins[idx] *= -1;
//!         Some(idx)
//!     }
//!     fn undo(&self, state: &mut SpinChain, idx: usize) {
//!         state.spins[idx] *= -1;  // flipping twice = identity
//!     }
//! }
//!
//! fn main() -> Result<(), McmcError> {
//!     let mut rng = StdRng::seed_from_u64(42);
//!     let state = SpinChain { spins: vec![1; 20] };
//!     let mut chain = Chain::new(state, &Ising)?;
//!
//!     for _ in 0..1000 {
//!         chain.step_mut(&Ising, &SpinFlip, &mut rng)?;
//!     }
//!
//!     assert!(chain.acceptance_rate() > 0.0);
//!     Ok(())
//! }
//! ```
//!
//! # Ergonomic sampling with [`Sampler`]
//!
//! [`Sampler`] bundles a chain with its target, proposal, and RNG so you
//! don't have to pass them on every step:
//!
//! ```
//! use markov_chain_monte_carlo::prelude::*;
//! use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};
//!
//! # #[derive(Clone)] struct Scalar(f64);
//! # struct Normal;
//! # impl Target<Scalar> for Normal {
//! #     fn log_prob(&self, s: &Scalar) -> f64 { -0.5 * s.0 * s.0 }
//! # }
//! # struct Walk;
//! # impl Proposal<Scalar> for Walk {
//! #     fn propose<R: Rng + ?Sized>(&self, c: &Scalar, r: &mut R) -> Scalar {
//! #         Scalar(c.0 + r.random_range(-1.0..1.0))
//! #     }
//! # }
//! let mut rng = StdRng::seed_from_u64(42);
//! let chain = Chain::new(Scalar(0.0), &Normal)?;
//! let mut sampler = Sampler::new(chain, &Normal, &Walk, &mut rng);
//!
//! // Burn-in
//! sampler.run(1000)?;
//! sampler.chain.reset_counters();
//!
//! // Production
//! sampler.run(10_000)?;
//! assert!(sampler.chain.acceptance_rate() > 0.0);
//! # Ok::<(), McmcError>(())
//! ```

mod chain;
mod error;
mod sampler;
mod traits;

pub use chain::Chain;
pub use error::McmcError;
pub use sampler::Sampler;
pub use traits::{Proposal, ProposalMut, Target};

/// Convenience re-exports for common usage.
///
/// ```
/// use markov_chain_monte_carlo::prelude::*;
/// ```
pub mod prelude {
    pub use crate::{Chain, McmcError, Proposal, ProposalMut, Sampler, Target};
}
