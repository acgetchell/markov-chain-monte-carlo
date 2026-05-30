// ruleid: mcmc.rust.no-unwrap-expect-in-doctests
/// let value = Some(1_u32).unwrap();

// ruleid: mcmc.rust.no-unwrap-expect-in-doctests
//! let value = Ok::<u32, &'static str>(1).expect("doctest should not panic");

// ruleid: mcmc.rust.no-unwrap-expect-in-doctests
///     .unwrap();

// ruleid: mcmc.rust.no-unwrap-expect-in-doctests
//!     .expect("doctest should not panic");

// ruleid: mcmc.rust.no-unwrap-expect-in-doctests
/// let value = parse_value()?.unwrap();

// ruleid: mcmc.rust.no-unwrap-expect-in-doctests
/** let value = Some(1_u32).unwrap(); */

// ruleid: mcmc.rust.no-unwrap-expect-in-doctests
/*! let value = Ok::<u32, &'static str>(1).expect("doctest should not panic"); */

// ok: mcmc.rust.no-unwrap-expect-in-doctests
/// # fn main() -> Result<(), markov_chain_monte_carlo::McmcError> { Ok(()) }

// ok: mcmc.rust.no-unwrap-expect-in-doctests
/// Do not use `.unwrap()` in public examples.

// ok: mcmc.rust.no-unwrap-expect-in-doctests
/// Prefer `?` to `.expect("message")` in public examples.

// ok: mcmc.rust.no-unwrap-expect-in-doctests
/** Avoid `.unwrap()` in block doctests; prefer `?`. */
