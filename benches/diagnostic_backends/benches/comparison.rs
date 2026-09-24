//! Same-process comparisons; correctness preflights run before any measurements.

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use diagnostic_backend_comparison::{ar1, arima_acf, assert_acf, reference_acf};
use ferromorphic::bayes::integrated_autocorrelation_time_within;
use markov_chain_monte_carlo::Autocorrelation;

fn compare(c: &mut Criterion) {
    let mut group = c.benchmark_group("acf");
    for (n, lag) in [
        (4_096, 0),
        (4_096, 1),
        (4_096, 32),
        (20_000, 2_000),
        (100_000, 400),
    ] {
        let samples = ar1(n, 0.95, 73);
        let expected = reference_acf(&samples, lag);
        let tolerance = 16.0 * n as f64 * f64::EPSILON;
        assert_acf(
            Autocorrelation::estimate(&samples, lag)
                .unwrap_or_else(|error| panic!("diagnostic fixture failed: {error:?}"))
                .values(),
            &expected,
            tolerance,
        );
        assert_acf(
            &arima_acf(&samples, lag)
                .unwrap_or_else(|error| panic!("diagnostic fixture failed: {error:?}")),
            &expected,
            tolerance,
        );
        assert_acf(
            &arima::acf::acf(&samples, Some(lag), false)
                .unwrap_or_else(|error| panic!("diagnostic fixture failed: {error:?}")),
            &expected,
            tolerance,
        );
        let fixture = format!("n{n}_lag{lag}");
        group.bench_function(BenchmarkId::new("native", &fixture), |b| {
            b.iter(|| Autocorrelation::estimate(black_box(&samples), black_box(lag)));
        });
        group.bench_function(BenchmarkId::new("arima_adapter", &fixture), |b| {
            b.iter(|| arima_acf(black_box(&samples), black_box(lag)));
        });
        // A lower-cost reference, not a contract-compatible replacement.
        group.bench_function(BenchmarkId::new("arima_raw", &fixture), |b| {
            b.iter(|| arima::acf::acf(black_box(&samples), Some(black_box(lag)), false));
        });
    }
    group.finish();

    // IPS and IMS are different estimators. These are separately labelled
    // end-to-end workloads, never a claim of equivalent numerical results.
    let mut group = c.benchmark_group("time_from_samples");
    for (n, lag, phi) in [
        (20_000, 2_000, 0.0),
        (20_000, 2_000, 0.95),
        (100_000, 400, -0.5),
    ] {
        let samples = ar1(n, phi, 73);
        let acf = Autocorrelation::estimate(&samples, lag)
            .unwrap_or_else(|error| panic!("diagnostic fixture failed: {error:?}"));
        let ims = acf
            .integrated_time()
            .unwrap_or_else(|error| panic!("diagnostic fixture failed: {error:?}"));
        let ips = integrated_autocorrelation_time_within(&samples, lag)
            .unwrap_or_else(|error| panic!("diagnostic fixture failed: {error:?}"));
        assert!(!ips.floored && !ips.truncated_at_cap);
        // Analytic stationary AR(1) value; broad deterministic regression
        // allowance for these fixtures, not a confidence interval.
        let analytic = (1.0 + phi) / (1.0 - phi);
        assert!((ims.estimate() / analytic - 1.0).abs() < 0.35);
        assert!((ips.value / analytic - 1.0).abs() < 0.35);
        let fixture = format!("n{n}_lag{lag}_phi{phi}");
        eprintln!(
            "{fixture}: IMS={} window={}, IPS={} window={}",
            ims.estimate(),
            ims.window(),
            ips.value,
            ips.lags
        );
        group.bench_function(BenchmarkId::new("native_ims", &fixture), |b| {
            b.iter(|| {
                Autocorrelation::estimate(black_box(&samples), black_box(lag))
                    .and_then(|acf| acf.integrated_time())
            });
        });
        group.bench_function(BenchmarkId::new("ferromorphic_ips", &fixture), |b| {
            b.iter(|| integrated_autocorrelation_time_within(black_box(&samples), black_box(lag)));
        });
    }
    group.finish();
    let samples = ar1(20_000, 0.95, 73);
    let acf = Autocorrelation::estimate(&samples, 2_000)
        .unwrap_or_else(|error| panic!("diagnostic fixture failed: {error:?}"));
    c.bench_function("time_from_existing_acf/native_ims", |b| {
        b.iter(|| black_box(&acf).integrated_time());
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(30).warm_up_time(Duration::from_secs(1)).measurement_time(Duration::from_secs(3));
    targets = compare
}
criterion_main!(benches);
