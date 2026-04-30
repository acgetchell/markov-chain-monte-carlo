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

// ruleid: mcmc.rust.no-box-dyn-error-in-src
fn erased_error() -> Result<(), Box<dyn Error>> {
    Ok(())
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
