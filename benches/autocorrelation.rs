//! Focused Criterion workloads for scalar autocorrelation diagnostics.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use markov_chain_monte_carlo::Autocorrelation;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// Generate a fixed scalar AR(1) workload outside the timed operations.
fn samples(count: usize) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(73);
    let mut state = 0.0;
    let mut values = Vec::with_capacity(count);
    for index in 0..count + 2_000 {
        state = 0.95_f64.mul_add(state, rng.random_range(-1.0..1.0));
        if index >= 2_000 {
            values.push(state);
        }
    }
    values
}

/// Check bounded benchmark fixtures through expanded, unscaled raw moments.
///
/// This reference does not reuse the production centering, scaling or Kahan
/// reduction. Its subtraction is well conditioned for these near-zero-mean
/// fixtures; it is not a general-purpose ACF implementation.
#[expect(
    clippy::cast_precision_loss,
    reason = "fixture counts are at most 100,000"
)]
fn assert_reference_acf(samples: &[f64], acf: &Autocorrelation) {
    let count = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / count;
    let squares = samples.iter().map(|value| value * value).sum::<f64>();
    let mean_term = count * mean * mean;
    assert!(
        mean_term <= squares / 2.0,
        "raw-moment reference is ill conditioned"
    );
    let variance_sum = squares - mean_term;
    assert!(variance_sum.is_finite() && variance_sum > 0.0);
    // Conservative roundoff allowance for these fixtures and naive sums,
    // not statistical uncertainty or a bound promised by the public API.
    let tolerance = 16.0 * count * f64::EPSILON;
    for (lag, &actual) in acf.values().iter().enumerate() {
        let left = &samples[..samples.len() - lag];
        let right = &samples[lag..];
        let products = left.iter().zip(right).map(|(a, b)| a * b).sum::<f64>();
        let endpoints = left.iter().chain(right).sum::<f64>();
        let covariance = mean.mul_add(-endpoints, products);
        let covariance = (left.len() as f64 * mean).mul_add(mean, covariance);
        let expected = covariance / variance_sum;
        assert!(
            (actual - expected).abs() <= tolerance,
            "ACF benchmark: N={}, lag={lag}, actual={actual}, reference={expected}",
            samples.len()
        );
    }
}

fn bench_estimate(c: &mut Criterion) {
    let mut group = c.benchmark_group("autocorrelation/estimate");
    for (count, max_lag) in [
        (4_096, 0),
        (4_096, 1),
        (4_096, 32),
        (20_000, 2_000),
        (100_000, 400),
    ] {
        let values = samples(count);
        // Preflight the public path once. Timed calls include input validation,
        // workspace/result allocation and destruction, but no fixture setup.
        match Autocorrelation::estimate(&values, max_lag) {
            Ok(acf) => {
                assert_eq!(acf.sample_count(), count);
                assert_eq!(acf.values().len(), max_lag + 1);
                assert!(acf.values().iter().all(|value| value.is_finite()));
                assert_reference_acf(&values, &acf);
            }
            Err(error) => panic!("ACF benchmark fixture: {error}"),
        }
        group.bench_with_input(
            BenchmarkId::new(format!("samples_{count}"), max_lag),
            &max_lag,
            |b, &lag| {
                b.iter(|| Autocorrelation::estimate(black_box(&values), black_box(lag)));
            },
        );
    }
    group.finish();
}

fn bench_integrated_time(c: &mut Criterion) {
    let values = samples(20_000);
    let acf = match Autocorrelation::estimate(&values, 2_000) {
        Ok(acf) => acf,
        Err(error) => panic!("integrated-time benchmark ACF: {error}"),
    };
    assert_reference_acf(&values, &acf);
    match acf.integrated_time() {
        Ok(time) => assert!(time.estimate() > 0.0 && time.estimate().is_finite()),
        Err(error) => panic!("integrated-time benchmark fixture: {error}"),
    }
    // Reuse the same immutable ACF; this isolates the allocation-free window scan.
    c.bench_function(
        "autocorrelation/integrated_time/samples_20000_lag_2000",
        |b| {
            b.iter(|| black_box(&acf).integrated_time());
        },
    );
}

criterion_group!(benches, bench_estimate, bench_integrated_time);
criterion_main!(benches);
