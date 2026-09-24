//! Independent arithmetic and stochastic checks for scalar diagnostics.

use approx::assert_relative_eq;
use markov_chain_monte_carlo::prelude::{
    Autocorrelation, AutocorrelationError, ChainId, Trace, TraceError, TraceRecord,
    TraceStepOutcome,
};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

#[test]
fn acf_matches_hand_calculated_biased_covariances() {
    let acf = Autocorrelation::estimate(&[1.0, 2.0, 3.0, 4.0], 3).unwrap();
    assert_eq!(acf.values().len(), 4);
    assert_eq!(acf.sample_count(), 4);
    // Mean 2.5; covariance numerators 5, 1.25, -1.5, -2.25.
    for (&actual, expected) in acf.values().iter().zip([1.0, 0.25, -0.3, -0.45]) {
        assert_relative_eq!(actual, expected, epsilon = 1e-14);
    }
    let time = acf.integrated_time().unwrap();
    assert_relative_eq!(time.estimate(), 1.5, epsilon = 1e-14);
    assert_eq!(time.window(), 1);
    assert_eq!(time.sample_count(), 4);
}

#[test]
fn short_constant_and_nonfinite_inputs_are_explicit_errors() {
    for samples in [&[][..], &[1.0][..]] {
        assert!(
            matches!(Autocorrelation::estimate(samples, 0), Err(AutocorrelationError::InsufficientSamples { count, .. }) if count == samples.len())
        );
    }
    for samples in [
        [0.0, -0.0],
        [f64::MAX, f64::MAX],
        [f64::MIN_POSITIVE, f64::MIN_POSITIVE],
    ] {
        assert_eq!(
            Autocorrelation::estimate(&samples, 0),
            Err(AutocorrelationError::ConstantTrace)
        );
    }
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matches!(
            Autocorrelation::estimate(&[0.0, invalid], 0),
            Err(AutocorrelationError::NonFiniteSample { index: 1, .. })
        ));
    }
    for max_lag in [2, usize::MAX] {
        assert!(matches!(
            Autocorrelation::estimate(&[0.0, 1.0], max_lag),
            Err(AutocorrelationError::InvalidMaxLag {
                max_lag: actual,
                sample_count: 2,
                ..
            }) if actual == max_lag
        ));
    }
}

#[test]
fn error_precedence_and_first_nonfinite_index_are_stable() {
    assert!(matches!(
        Autocorrelation::estimate(&[], usize::MAX),
        Err(AutocorrelationError::InsufficientSamples { count: 0, .. })
    ));
    assert!(matches!(
        Autocorrelation::estimate(&[f64::NAN], usize::MAX),
        Err(AutocorrelationError::InsufficientSamples { count: 1, .. })
    ));
    assert!(matches!(
        Autocorrelation::estimate(&[f64::NAN, f64::INFINITY], 2),
        Err(AutocorrelationError::InvalidMaxLag {
            max_lag: 2,
            sample_count: 2,
            ..
        })
    ));
    for (samples, expected_index) in [
        ([f64::NAN, f64::NAN, f64::NAN], 0),
        ([0.0, f64::INFINITY, f64::NAN], 1),
        ([0.0, 1.0, f64::NEG_INFINITY], 2),
    ] {
        let result = Autocorrelation::estimate(&samples, 0);
        assert!(
            matches!(result, Err(AutocorrelationError::NonFiniteSample { index, .. }) if index == expected_index),
            "samples={samples:?}, expected index={expected_index}, got {result:?}"
        );
    }
}

#[test]
fn lag_budget_does_not_silently_truncate_integrated_time() {
    for max_lag in 0..=2 {
        let acf = Autocorrelation::estimate(&[1.0, 2.0, 3.0, 4.0], max_lag).unwrap();
        assert_eq!(acf.values().len(), max_lag + 1);
        assert!(
            matches!(acf.integrated_time(), Err(AutocorrelationError::TruncationNotFound { max_lag: actual, .. }) if actual == max_lag)
        );
    }
    let short = Autocorrelation::estimate(&[0.0, 1.0], 1).unwrap();
    assert_eq!(short.values().len(), 2);
    assert_eq!(short.sample_count(), 2);
    assert_relative_eq!(short.values()[0], 1.0);
    assert_relative_eq!(short.values()[1], -0.5);
    assert!(matches!(
        short.integrated_time(),
        Err(AutocorrelationError::TruncationNotFound { max_lag: 1, .. })
    ));
}

#[test]
fn anticorrelation_is_not_floored_to_one() {
    let acf = Autocorrelation::estimate(&[3.0, -1.0, -1.0, -1.0], 3).unwrap();
    assert_relative_eq!(
        acf.integrated_time().unwrap().estimate(),
        5.0 / 6.0,
        epsilon = 1e-14
    );
    let degenerate = Autocorrelation::estimate(&[1.0, -1.0, 0.0, 0.0], 3).unwrap();
    assert_eq!(
        degenerate.integrated_time(),
        Err(AutocorrelationError::NonPositiveTime)
    );
    // Exact covariance numerators are [170, -96, 43, -48, 16]; the
    // first pair gives tau = -11/85 and the next pair stops the sequence.
    let negative = Autocorrelation::estimate(&[-2.0, 1.0, -2.0, -1.0, -2.0], 4).unwrap();
    assert_eq!(
        negative.integrated_time(),
        Err(AutocorrelationError::NonPositiveTime)
    );
}

#[test]
fn scaling_offsets_and_extreme_values_preserve_acf() {
    // Seven nonzero lags exercise both a complete batch and the scalar remainder.
    let samples = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let reference = Autocorrelation::estimate(&samples, 7).unwrap();
    let ulp = f64::from_bits(1);
    let large = 1e300_f64;
    let adjacent = large.next_up() - large;
    for transformed in [
        samples.map(|value| value * ulp),
        samples.map(|value| adjacent.mul_add(value, large)),
        samples.map(|value| (7.0 - value) * 1e300),
    ] {
        let acf = Autocorrelation::estimate(&transformed, 7).unwrap();
        assert_eq!(acf.values().len(), reference.values().len());
        for (&actual, &expected) in acf.values().iter().zip(reference.values()) {
            assert_relative_eq!(actual, expected, epsilon = 1e-14);
        }
    }
    let extremes = Autocorrelation::estimate(
        &[
            -f64::MAX,
            f64::MAX,
            -f64::MAX,
            f64::MAX,
            -f64::MAX,
            f64::MAX,
            -f64::MAX,
            f64::MAX,
        ],
        7,
    )
    .unwrap();
    assert_eq!(extremes.values().len(), 8);
    // Alternating +/-MAX has rho[k] = (-1)^k * (8-k)/8 with biased covariance.
    for (&actual, expected) in extremes
        .values()
        .iter()
        .zip([1.0, -0.875, 0.75, -0.625, 0.5, -0.375, 0.25, -0.125])
    {
        assert_relative_eq!(actual, expected, epsilon = 1e-14);
    }
}

/// Exact ACF for a nearly constant trace with one representable outlier.
#[test]
fn isolated_one_ulp_outlier_matches_exact_covariances() {
    // Seven equal values and a final distinct value have rho[k] = -k/56
    // for k > 0, independently of their offset and separation. The first
    // retained pair gives tau = 27/28 and the next pair is negative.
    for (background, outlier) in [(f64::MAX, f64::MAX.next_down()), (0.0, f64::from_bits(1))] {
        let mut samples = [background; 8];
        samples[7] = outlier;
        let acf = Autocorrelation::estimate(&samples, 7).unwrap();
        assert_eq!(acf.values().len(), 8);
        assert_relative_eq!(acf.values()[0], 1.0);
        for (&actual, lag) in acf.values()[1..].iter().zip(1..=7) {
            assert_relative_eq!(actual, -f64::from(lag) / 56.0, epsilon = 1e-14);
        }
        let time = acf.integrated_time().unwrap();
        assert_relative_eq!(time.estimate(), 27.0 / 28.0, epsilon = 1e-14);
        assert_eq!(time.window(), 1);
        assert_eq!(time.sample_count(), 8);
    }
}

/// Gaussian AR(1) after a 2,000-step warm-up, using Box-Muller innovations.
fn ar1(coefficient: f64, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut state = 0.0;
    let mut samples = Vec::with_capacity(100_000);
    for index in 0..102_000 {
        let radius = (-2.0 * (1.0 - rng.random::<f64>()).ln()).sqrt();
        let innovation = radius * (std::f64::consts::TAU * rng.random::<f64>()).cos();
        state = coefficient.mul_add(state, innovation);
        if index >= 2_000 {
            samples.push(state);
        }
    }
    samples
}

#[test]
fn independent_and_correlated_traces_match_analytic_ar1_behavior() {
    // Fixed seeds make regressions reproducible. At N=100,000, the generous
    // tolerances cover sampling error (including ~5,000 correlation times at
    // phi=0.9). Multiple seeds avoid relying on one favorable realization.
    // These are regression tolerances, not calibrated confidence intervals;
    // the exact arithmetic tests independently check numerical correctness.
    for (coefficient, seed) in [0.0, 0.9, -0.5]
        .into_iter()
        .flat_map(|coefficient| [73, 74, 75].map(|seed| (coefficient, seed)))
    {
        let samples = ar1(coefficient, seed);
        let acf = Autocorrelation::estimate(&samples, 400)
            .unwrap_or_else(|error| panic!("phi={coefficient}, seed={seed}: {error:?}"));
        assert_eq!(acf.sample_count(), 100_000);
        assert_eq!(acf.values().len(), 401);
        for lag in 1_i32..=5 {
            let actual = acf.values()[usize::try_from(lag).unwrap()];
            let expected = coefficient.powi(lag);
            assert!(
                (actual - expected).abs() <= 0.025,
                "phi={coefficient}, seed={seed}, lag={lag}: ACF={actual}, expected {expected}"
            );
        }
        let expected_time = (1.0 + coefficient) / (1.0 - coefficient);
        let time = acf
            .integrated_time()
            .unwrap_or_else(|error| panic!("phi={coefficient}, seed={seed}: {error:?}"));
        assert!(
            (time.estimate() - expected_time).abs() <= 0.2 * expected_time,
            "phi={coefficient}, seed={seed}: time={time:?}, expected tau={expected_time}"
        );
    }
}

#[test]
fn observable_selection_preserves_chains_and_self_loops() {
    let mut trace = Trace::new(["magnetization", "energy"]).unwrap();
    for (index, value) in [3.0, -1.0, -1.0, -1.0].into_iter().enumerate() {
        for (chain, energy) in [(0, value), (1, 100.0)] {
            trace
                .push(TraceRecord::new(
                    ChainId::new(chain),
                    index + 1,
                    match index {
                        0 | 1 => TraceStepOutcome::accepted(),
                        2 => TraceStepOutcome::rejected_proposal(),
                        _ => TraceStepOutcome::no_proposal(),
                    },
                    0.0,
                    vec![0.0, energy],
                ))
                .unwrap();
        }
    }
    let energy: Vec<_> = trace
        .observable_values(ChainId::new(0), "energy")
        .unwrap()
        .copied()
        .collect();
    assert_eq!(energy, [3.0, -1.0, -1.0, -1.0]);
    let other_chain: Vec<_> = trace
        .observable_values(ChainId::new(1), "energy")
        .unwrap()
        .copied()
        .collect();
    assert_eq!(other_chain, [100.0; 4]);
    let acf = Autocorrelation::estimate(&energy, 3).unwrap();
    assert_eq!(acf.sample_count(), 4);
    assert_relative_eq!(
        acf.integrated_time().unwrap().estimate(),
        5.0 / 6.0,
        epsilon = 1e-14
    );
}

#[test]
fn named_observable_lookup_is_exact_and_distinguishes_missing_from_empty() {
    let trace = Trace::new(["energy"]).unwrap();
    for requested in ["missing", "Energy", ""] {
        assert!(matches!(
            trace.observable_values(ChainId::new(0), requested),
            Err(TraceError::UnknownObservable { name, .. }) if name == requested
        ));
    }
    assert_eq!(
        trace
            .observable_values(ChainId::new(0), "energy")
            .unwrap()
            .count(),
        0
    );
    let mut populated = trace;
    populated
        .push(TraceRecord::new(
            ChainId::new(0),
            1,
            TraceStepOutcome::accepted(),
            0.0,
            vec![2.0],
        ))
        .unwrap();
    for chain_id in [ChainId::new(0), ChainId::new(9)] {
        assert!(matches!(
            populated.observable_values(chain_id, "missing"),
            Err(TraceError::UnknownObservable { name, .. }) if name == "missing"
        ));
    }
    assert_eq!(
        populated
            .observable_values(ChainId::new(9), "energy")
            .unwrap()
            .count(),
        0
    );
    let no_columns = Trace::new(std::iter::empty::<String>()).unwrap();
    assert!(
        matches!(no_columns.observable_values(ChainId::new(0), "energy"),
        Err(TraceError::UnknownObservable { name, .. }) if name == "energy")
    );
}

#[test]
fn observable_iterator_borrows_storage_without_retaining_query_name() {
    let mut trace = Trace::new(["magnetization", "energy"]).unwrap();
    trace
        .push(TraceRecord::new(
            ChainId::new(0),
            1,
            TraceStepOutcome::accepted(),
            0.0,
            vec![0.5, -2.0],
        ))
        .unwrap();
    let mut values = {
        let temporary_name = String::from("energy");
        trace
            .observable_values(ChainId::new(0), &temporary_name)
            .unwrap()
    };
    let selected = values.next().unwrap();
    assert!(std::ptr::eq(
        selected,
        &raw const trace.records()[0].observable_values()[1]
    ));
    assert_relative_eq!(*selected, -2.0);
    assert!(values.next().is_none());
}

#[test]
fn observable_selection_stays_aligned_after_rejected_writes_and_merge() {
    let chain_id = ChainId::new(0);
    let mut trace = Trace::new(["magnetization", "energy"]).unwrap();
    trace
        .push(TraceRecord::new(
            chain_id,
            10,
            TraceStepOutcome::accepted(),
            -1.0,
            vec![0.5, 2.0],
        ))
        .unwrap();
    let before = trace.clone();
    assert!(matches!(
        trace.push(TraceRecord::new(
            chain_id,
            11,
            TraceStepOutcome::no_proposal(),
            -1.0,
            vec![0.5],
        )),
        Err(TraceError::ObservableCountMismatch {
            expected: 2,
            actual: 1,
            ..
        })
    ));
    assert_eq!(trace, before);

    // Matching widths do not suffice: reversed headers must reject the entire
    // merge, retaining both row metadata and named-column meaning.
    let mut reversed = Trace::new(["energy", "magnetization"]).unwrap();
    reversed
        .push(TraceRecord::new(
            chain_id,
            11,
            TraceStepOutcome::accepted(),
            -2.0,
            vec![4.0, 0.25],
        ))
        .unwrap();
    assert!(matches!(
        trace.extend(reversed),
        Err(TraceError::ObservableNamesMismatch { .. })
    ));
    assert_eq!(trace, before);

    let mut continuation = Trace::new(["magnetization", "energy"]).unwrap();
    for (chain, step, outcome, log_prob, values) in [
        (
            chain_id,
            11,
            TraceStepOutcome::no_proposal(),
            -1.0,
            [0.5, 2.0],
        ),
        (
            ChainId::new(1),
            1,
            TraceStepOutcome::accepted(),
            -9.0,
            [1.0, 18.0],
        ),
        (
            chain_id,
            12,
            TraceStepOutcome::accepted(),
            -2.0,
            [0.25, 4.0],
        ),
    ] {
        continuation
            .push(TraceRecord::new(
                chain,
                step,
                outcome,
                log_prob,
                values.to_vec(),
            ))
            .unwrap();
    }
    let mut expected_records = before.records().to_vec();
    expected_records.extend_from_slice(continuation.records());
    trace.extend(continuation).unwrap();
    trace
        .extend(Trace::new(["magnetization", "energy"]).unwrap())
        .unwrap();
    assert_eq!(trace.records(), expected_records);
    assert_eq!(trace.observable_names(), ["magnetization", "energy"]);
    for (name, expected) in [
        ("energy", [2.0, 2.0, 4.0]),
        ("magnetization", [0.5, 0.5, 0.25]),
    ] {
        let values: Vec<_> = trace
            .observable_values(chain_id, name)
            .unwrap()
            .copied()
            .collect();
        assert_eq!(values, expected);
    }
    assert_relative_eq!(trace.acceptance_rate(chain_id), 2.0 / 3.0);
}

#[test]
fn observable_selection_preserves_order_and_nonfinite_values() {
    let mut trace = Trace::new(["energy"]).unwrap();
    for (chain, step, value) in [
        (0, 20, 2.0),
        (1, 1, 99.0),
        (0, 10, f64::NAN),
        (0, 10, f64::INFINITY),
    ] {
        trace
            .push(TraceRecord::new(
                ChainId::new(chain),
                step,
                TraceStepOutcome::accepted(),
                0.0,
                vec![value],
            ))
            .unwrap();
    }
    let energy: Vec<_> = trace
        .observable_values(ChainId::new(0), "energy")
        .unwrap()
        .copied()
        .collect();
    assert_eq!(energy.len(), 3);
    assert_relative_eq!(energy[0], 2.0);
    assert!(energy[1].is_nan());
    assert_eq!(energy[2].to_bits(), f64::INFINITY.to_bits());
    assert!(matches!(
        Autocorrelation::estimate(&energy, 1),
        Err(AutocorrelationError::NonFiniteSample { index: 1, .. })
    ));
}
