use std::error::Error;

fn production_diagnostics() {
    // ruleid: mcmc.rust.no-stdio-diagnostics-in-src
    println!("debug");
    // ruleid: mcmc.rust.no-stdio-diagnostics-in-src
    eprintln!("debug");
}

fn nonfinite_defaults(value: Option<f64>) -> f64 {
    // ruleid: mcmc.rust.no-nonfinite-unwrap-defaults
    value.unwrap_or(f64::NAN)
}

fn panics(value: Option<u8>) -> u8 {
    // ruleid: mcmc.rust.no-production-unwrap-panic
    value.unwrap()
}

fn expects(value: Option<u8>) -> u8 {
    // ruleid: mcmc.rust.no-production-unwrap-panic
    value.expect("present")
}

fn raw_panic() {
    // ruleid: mcmc.rust.no-production-unwrap-panic
    panic!("boom");
}

pub struct InvariantType;

impl InvariantType {
    // ruleid: mcmc.rust.no-public-unchecked-apis
    pub fn new_unchecked() -> Self {
        Self
    }

    // ruleid: mcmc.rust.no-public-unit-validators
    pub fn validate_positive(value: usize) -> Result<(), ()> {
        if value == 0 { Err(()) } else { Ok(()) }
    }
}

pub struct OnlineStats;
pub struct BinningAnalysis;

impl OnlineStats {
    // ruleid: mcmc.rust.no-infallible-statistics-f64-ingestion
    pub fn push(&mut self, sample: f64) {
        let _ = sample;
    }
}

// ruleid: mcmc.rust.no-infallible-statistics-f64-ingestion
impl Extend<f64> for BinningAnalysis {
    fn extend<T: IntoIterator<Item = f64>>(&mut self, iter: T) {
        for sample in iter {
            let _ = sample;
        }
    }
}

// ruleid: mcmc.rust.no-infallible-statistics-f64-ingestion
impl FromIterator<f64> for OnlineStats {
    fn from_iter<T: IntoIterator<Item = f64>>(iter: T) -> Self {
        for sample in iter {
            let _ = sample;
        }
        Self
    }
}

pub struct DetailedBalanceConfig {
    // ruleid: mcmc.rust.invariant-fields-use-refined-types
    samples: usize,
    // ruleid: mcmc.rust.invariant-fields-use-refined-types
    min_hits: usize,
}

pub struct DiscreteProposalRatio {
    // ruleid: mcmc.rust.invariant-fields-use-refined-types
    forward_site_count: usize,
}

// ruleid: mcmc.rust.no-box-dyn-error-in-src
fn erased_error() -> Result<(), Box<dyn Error>> {
    Ok(())
}

// ruleid: mcmc.rust.public-error-enums-non-exhaustive
pub enum MissingNonExhaustiveError {
    InvalidInput,
}

#[derive(Debug)]
// ruleid: mcmc.rust.public-error-enums-non-exhaustive
pub enum DerivedMissingNonExhaustiveError {
    InvalidInput,
}

// ok: mcmc.rust.public-error-enums-non-exhaustive
#[non_exhaustive]
pub enum ExtensibleError {
    InvalidInput,
}

// ruleid: mcmc.rust.no-clippy-allow-lints
#[allow(clippy::cast_precision_loss)]
fn clippy_allow(value: usize) -> f64 {
    value as f64
}

// ruleid: mcmc.rust.expect-requires-reason
#[expect(clippy::cast_precision_loss)]
fn clippy_expect_without_reason(value: usize) -> f64 {
    value as f64
}

#[cfg(test)]
mod tests {
    #[test]
    fn unwrap_is_ok_in_tests() {
        // ok: mcmc.rust.no-production-unwrap-panic
        Some(1).unwrap();
    }
}
