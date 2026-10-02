//! Public multi-chain ESS/MCSE contracts and independent `ArviZ` 0.22.0 fixtures.

use std::{error::Error, time::Duration};

use approx::assert_relative_eq;
use markov_chain_monte_carlo::{
    DiagnosticTiming, DiagnosticTimingError, EssEstimate, EssEstimator, MeanMcse, MonteCarloError,
    QuantileMcse, SplitRhatError, TailEss,
};
use serde_json::Value;

fn error_name(error: MonteCarloError) -> &'static str {
    match error {
        MonteCarloError::ConstantSamples => "ConstantSamples",
        MonteCarloError::NoWithinChainVariation => "NoWithinChainVariation",
        MonteCarloError::DegenerateIndicator => "DegenerateIndicator",
        MonteCarloError::CollapsedQuantileInterval => "CollapsedQuantileInterval",
        _ => panic!("unexpected fixture error: {error:?}"),
    }
}

fn check_reference(case: &Value, metric: &str, actual: Result<f64, MonteCarloError>) {
    let name = case["name"].as_str().unwrap();
    if let Some(expected) = case["unavailable"][metric].as_str() {
        assert_eq!(
            error_name(actual.unwrap_err()),
            expected,
            "{name}: {metric}"
        );
    } else {
        let actual = actual.unwrap_or_else(|error| panic!("{name}: {metric}: {error:?}"));
        let expected = case["reference"][metric].as_f64().unwrap();
        assert!(actual.is_finite());
        assert!(
            (actual - expected).abs() <= 2e-11_f64.mul_add(expected.abs(), 2e-13),
            "{name} {metric}: actual={actual:.17e}, expected={expected:.17e}"
        );
    }
}

#[test]
fn pinned_reference_corpus_and_input_immutability() {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/ess.json")).unwrap();
    assert_eq!(fixture["reference"]["version"], "0.22.0");
    assert_eq!(fixture["reference"]["relative_tolerance"], 2e-11);
    assert_eq!(fixture["reference"]["absolute_tolerance"], 2e-13);
    let cases = fixture["cases"].as_array().unwrap();
    let names: Vec<_> = cases
        .iter()
        .map(|case| case["name"].as_str().unwrap())
        .collect();
    assert_eq!(
        names,
        [
            "independent",
            "positive_correlation",
            "antithetic",
            "location_disagreement",
            "scale_disagreement",
            "heavy_tails",
            "counts",
            "binary",
            "odd_lengths",
            "constant",
            "stuck_halves",
            "minimum_length",
            "one_constant_half"
        ]
    );
    for case in cases {
        let chains: Vec<Vec<f64>> = serde_json::from_value(case["chains"].clone()).unwrap();
        let before: Vec<_> = chains.iter().flatten().map(|x| x.to_bits()).collect();
        let borrowed: Vec<_> = chains.iter().map(Vec::as_slice).collect();
        for (metric, estimator) in [
            ("mean_ess", EssEstimator::Mean),
            ("bulk_ess", EssEstimator::Bulk),
            ("q05_ess", EssEstimator::Quantile(0.05)),
            ("q50_ess", EssEstimator::Quantile(0.5)),
            ("q95_ess", EssEstimator::Quantile(0.95)),
        ] {
            let result = EssEstimate::estimate(&borrowed, estimator);
            check_reference(case, metric, result.map(EssEstimate::value));
            if let Ok(ess) = result {
                assert_eq!(ess.estimator(), estimator);
                assert_eq!(ess.chain_count(), chains.len());
                assert_eq!(ess.samples_per_chain(), chains[0].len());
                assert_eq!(ess.samples_per_split_chain(), chains[0].len() / 2);
                assert_eq!(ess.sample_count(), chains.len() * 2 * (chains[0].len() / 2));
                assert_eq!(ess.original_sample_count(), chains.len() * chains[0].len());
                assert!(ess.relative() > 0.0);
                let count = f64::from(u32::try_from(ess.sample_count()).unwrap());
                assert!(ess.relative() <= count.log10() + 1e-12);
            }
        }
        check_reference(
            case,
            "mean_mcse",
            MeanMcse::estimate(&borrowed).map(MeanMcse::value),
        );
        for (label, p) in [("q05", 0.05), ("q50", 0.5), ("q95", 0.95)] {
            let result = QuantileMcse::estimate(&borrowed, p);
            check_reference(
                case,
                &format!("{label}_mcse"),
                result.map(QuantileMcse::value),
            );
            if let Ok(mcse) = result {
                check_reference(case, &format!("{label}_value"), Ok(mcse.quantile()));
                assert_eq!(
                    mcse.effective_sample_size().estimator(),
                    EssEstimator::Quantile(p)
                );
                assert!(mcse.interval()[0] < mcse.interval()[1]);
            }
        }
        let tail = TailEss::estimate(&borrowed).unwrap();
        check_reference(case, "q05_ess", tail.lower().map(EssEstimate::value));
        check_reference(case, "q95_ess", tail.upper().map(EssEstimate::value));
        assert_eq!(
            tail.sample_count(),
            chains.len() * 2 * (chains[0].len() / 2)
        );
        if tail.lower().is_err() || tail.upper().is_err() {
            assert!(tail.minimum().is_none());
            assert_eq!(tail.value(), None);
            assert_eq!(tail.relative(), None);
        } else {
            check_reference(case, "tail_ess", Ok(tail.value().unwrap()));
            assert_eq!(tail.relative(), Some(tail.minimum().unwrap().relative()));
        }
        assert_eq!(
            before,
            chains
                .iter()
                .flatten()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn chain_order_and_time_reversal_match_reference_corpus() {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/ess.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let chains: Vec<Vec<f64>> = serde_json::from_value(case["chains"].clone()).unwrap();
        for (reverse_chains, reverse_time) in [(true, false), (false, true), (true, true)] {
            let mut reordered = chains.clone();
            if reverse_chains {
                reordered.reverse();
            }
            if reverse_time {
                for chain in reordered.iter_mut().step_by(2) {
                    chain.reverse();
                }
            }
            let borrowed: Vec<_> = reordered.iter().map(Vec::as_slice).collect();
            for (metric, estimator) in [
                ("mean_ess", EssEstimator::Mean),
                ("bulk_ess", EssEstimator::Bulk),
                ("q05_ess", EssEstimator::Quantile(0.05)),
                ("q50_ess", EssEstimator::Quantile(0.5)),
                ("q95_ess", EssEstimator::Quantile(0.95)),
            ] {
                check_reference(
                    case,
                    metric,
                    EssEstimate::estimate(&borrowed, estimator).map(EssEstimate::value),
                );
            }
            check_reference(
                case,
                "mean_mcse",
                MeanMcse::estimate(&borrowed).map(MeanMcse::value),
            );
            for (label, probability) in [("q05", 0.05), ("q50", 0.5), ("q95", 0.95)] {
                let mcse = QuantileMcse::estimate(&borrowed, probability);
                check_reference(
                    case,
                    &format!("{label}_mcse"),
                    mcse.map(QuantileMcse::value),
                );
                if let Ok(mcse) = mcse {
                    check_reference(case, &format!("{label}_value"), Ok(mcse.quantile()));
                }
            }
        }
    }
}

#[test]
fn median_indicator_zero_pair_has_only_the_two_reference_rounding_limits() {
    // CI counterexample: the exact median is -1. With eight five-draw halves,
    // indicator W=9/40 and V+=13/50 give rho[1]=1/10, rho[2]=-3/260,
    // rho[3]=3/260. Their second pair is exactly zero. Floating reductions may
    // retain or discard it under ArviZ's >= 0 rule; at this finite-lag boundary
    // tau is respectively 309/260 or 6/5. Tight reordering invariance is false,
    // but each result must agree with one of these independently derived limits.
    let chains = [
        [0., 1., -2., -2., 0., -3., -2., -2., 0., -2.],
        [-3., -2., 0., 0., 0., 0., 1., 0., -2., -2.],
        [0., 1., 0., 0., 0., -2., -1., 0., -1., -1.],
        [0., 1., -1., 0., -1., -2., -1., -1., 0., -1.],
    ];
    let borrowed: Vec<_> = chains.iter().map(<[f64; 10]>::as_slice).collect();
    let reordered: Vec<Vec<_>> = chains
        .iter()
        .rev()
        .enumerate()
        .map(|(index, chain)| {
            let mut values = chain.to_vec();
            if index % 2 == 0 {
                values.reverse();
            }
            values
        })
        .collect();
    let reordered: Vec<_> = reordered.iter().map(Vec::as_slice).collect();
    for draws in [&borrowed, &reordered] {
        let ess = EssEstimate::estimate(draws, EssEstimator::Quantile(0.5)).unwrap();
        assert!(!ess.is_regularized());
        assert!(
            [10400.0 / 309.0, 100.0 / 3.0]
                .into_iter()
                .any(|limit| (ess.value() - limit).abs() < 2e-12),
            "unexpected zero-pair ESS: {}",
            ess.value()
        );
    }
}

#[test]
fn between_chain_disagreement_and_antithetic_regularization_are_visible() {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/ess.json")).unwrap();
    let cases = fixture["cases"].as_array().unwrap();
    let independent = cases[0]["reference"]["mean_ess"].as_f64().unwrap();
    let separated: Vec<Vec<f64>> = serde_json::from_value(cases[3]["chains"].clone()).unwrap();
    let borrowed: Vec<_> = separated.iter().map(Vec::as_slice).collect();
    let mean = EssEstimate::estimate(&borrowed, EssEstimator::Mean).unwrap();
    assert!(mean.value() < independent / 10.0);
    // Exact alternating halves give a negative estimated time; the declared
    // reference bound sets ESS=S*log10(S), with a visible regularization flag.
    let alternating: Vec<_> = (0..32).map(|i| f64::from(i % 2)).collect();
    let ess = EssEstimate::estimate(&[&alternating, &alternating], EssEstimator::Mean).unwrap();
    assert!(ess.is_regularized());
    assert_relative_eq!(ess.value(), 64.0 * 64.0_f64.log10(), epsilon = 1e-12);
    assert!(ess.relative() > 1.0);
}

#[test]
fn inputs_probabilities_and_degenerate_results_have_typed_outcomes() {
    let valid = [0.0, 1.0, 2.0, 3.0];
    for result in [
        EssEstimate::estimate(&[&valid], EssEstimator::Mean).map(drop),
        MeanMcse::estimate(&[&valid]).map(drop),
        QuantileMcse::estimate(&[&valid], 0.5).map(drop),
        TailEss::estimate(&[&valid]).map(drop),
    ] {
        let error = result.unwrap_err();
        assert!(matches!(
            error,
            MonteCarloError::Input(SplitRhatError::InsufficientChains { count: 1, .. })
        ));
        let cause = error.source().unwrap();
        assert!(matches!(
            cause.downcast_ref::<SplitRhatError>(),
            Some(SplitRhatError::InsufficientChains { count: 1, .. })
        ));
        assert_eq!(error.to_string(), cause.to_string());
        assert!(cause.source().is_none());
    }
    for probability in [
        0.0,
        -0.0,
        1.0,
        -0.1,
        1.1,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ] {
        // Probability errors precede malformed chains, including empty input.
        assert_eq!(
            EssEstimate::estimate(&[], EssEstimator::Quantile(probability)),
            Err(MonteCarloError::InvalidProbability)
        );
        assert_eq!(
            QuantileMcse::estimate(&[], probability),
            Err(MonteCarloError::InvalidProbability)
        );
    }
    assert!(matches!(
        EssEstimate::estimate(&[&valid, &[0.0; 3]], EssEstimator::Bulk),
        Err(MonteCarloError::Input(
            SplitRhatError::InsufficientSamples {
                chain_index: 1,
                count: 3,
                ..
            }
        ))
    ));
    assert!(matches!(
        MeanMcse::estimate(&[&valid, &[0.0; 5]]),
        Err(MonteCarloError::Input(SplitRhatError::UnequalLengths {
            chain_index: 1,
            expected: 4,
            actual: 5,
            ..
        }))
    ));
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let omitted = [0.0, 1.0, bad, 2.0, 3.0];
        assert!(matches!(
            TailEss::estimate(&[&omitted, &omitted]),
            Err(MonteCarloError::Input(SplitRhatError::NonFiniteSample {
                chain_index: 0,
                sample_index: 2,
                ..
            }))
        ));
    }
    let zeros = [-0.0, 0.0, -0.0, 0.0];
    let unavailable = EssEstimate::estimate(&[&zeros, &zeros], EssEstimator::Mean).unwrap_err();
    assert_eq!(unavailable, MonteCarloError::ConstantSamples);
    assert!(unavailable.source().is_none());
    let stuck = [0.0, 0.0, 1.0, 1.0];
    assert_eq!(
        EssEstimate::estimate(&[&stuck, &stuck], EssEstimator::Mean),
        Err(MonteCarloError::NoWithinChainVariation)
    );
    let binary = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    assert_eq!(
        QuantileMcse::estimate(&[&binary, &binary], 0.05),
        Err(MonteCarloError::CollapsedQuantileInterval)
    );
    let tail = TailEss::estimate(&[&binary, &binary]).unwrap();
    assert!(tail.lower().is_ok());
    assert_eq!(tail.upper(), Err(MonteCarloError::DegenerateIndicator));
    assert_eq!(tail.value(), None);
}

#[test]
fn timing_requires_present_positive_matching_workload() {
    let values = [0., 1., 3., 2., 1., 0., 2., 3.];
    let ess = EssEstimate::estimate(&[&values, &values], EssEstimator::Bulk).unwrap();
    assert_eq!(
        ess.per_second(None),
        Err(DiagnosticTimingError::MissingTiming)
    );
    assert_eq!(
        DiagnosticTiming::try_new(Duration::ZERO, 2, 8),
        Err(DiagnosticTimingError::ZeroDuration)
    );
    assert_eq!(
        DiagnosticTiming::try_new(Duration::from_secs(1), 0, 8),
        Err(DiagnosticTimingError::InvalidCounts)
    );
    assert_eq!(
        DiagnosticTiming::try_new(Duration::from_secs(1), 2, 0),
        Err(DiagnosticTimingError::InvalidCounts)
    );
    let full_run = DiagnosticTiming::try_new(Duration::from_secs(2), 2, 16).unwrap();
    assert!(matches!(
        ess.per_second(Some(&full_run)),
        Err(DiagnosticTimingError::CountMismatch {
            expected_chains: 2,
            expected_samples_per_chain: 8,
            actual_chains: 2,
            actual_samples_per_chain: 16,
            ..
        })
    ));
    let timing = DiagnosticTiming::try_new(Duration::from_secs(2), 2, 8).unwrap();
    assert_eq!(timing.elapsed(), Duration::from_secs(2));
    assert_eq!(timing.chain_count(), 2);
    assert_eq!(timing.samples_per_chain(), 8);
    assert_relative_eq!(
        ess.per_second(Some(&timing)).unwrap(),
        ess.value() / 2.0,
        epsilon = 1e-14
    );
    let unavailable = TailEss::estimate(&[&[1.0; 8], &[1.0; 8]]).unwrap();
    assert_eq!(
        unavailable.per_second(None),
        Err(DiagnosticTimingError::MissingTiming)
    );
    assert!(matches!(
        unavailable.per_second(Some(&full_run)),
        Err(DiagnosticTimingError::CountMismatch { .. })
    ));
    assert_eq!(unavailable.per_second(Some(&timing)), Ok(None));
}

#[test]
fn odd_length_ess_rates_require_timing_of_all_original_draws() {
    let a = [0., 4., 2., 6., 3.5, 1., 5., 3., 7.];
    let b = [8., 12., 10., 14., 11.5, 9., 13., 11., 15.];
    let ess = EssEstimate::estimate(&[&a, &b], EssEstimator::Bulk).unwrap();
    let tail = TailEss::estimate(&[&a, &b]).unwrap();
    assert_eq!(tail.chain_count(), 2);
    assert_eq!(tail.samples_per_chain(), 9);
    assert_eq!(tail.samples_per_split_chain(), 4);
    assert_eq!(tail.original_sample_count(), 18);
    assert_eq!(tail.sample_count(), 16);

    // Timing must match both original dimensions, even though splitting omits
    // one draw per chain. Fractional seconds must contribute to the rate.
    let elapsed = Duration::from_millis(2500);
    let timing = DiagnosticTiming::try_new(elapsed, 2, 9).unwrap();
    assert_relative_eq!(
        ess.per_second(Some(&timing)).unwrap(),
        ess.value() / 2.5,
        epsilon = 1e-14
    );
    assert_relative_eq!(
        tail.per_second(Some(&timing)).unwrap().unwrap(),
        tail.value().unwrap() / 2.5,
        epsilon = 1e-14
    );
    for (chains, samples) in [(1, 9), (2, 8)] {
        let mismatch = DiagnosticTiming::try_new(elapsed, chains, samples).unwrap();
        for error in [
            ess.per_second(Some(&mismatch)).unwrap_err(),
            tail.per_second(Some(&mismatch)).unwrap_err(),
        ] {
            assert!(matches!(
                error,
                DiagnosticTimingError::CountMismatch {
                    expected_chains: 2,
                    expected_samples_per_chain: 9,
                    actual_chains,
                    actual_samples_per_chain,
                    ..
                } if actual_chains == chains && actual_samples_per_chain == samples
            ));
        }
    }
}

#[test]
fn quantile_mcse_stays_finite_when_the_interval_width_overflows() {
    for (lower, upper, expected_mcse, expected_median) in [
        (-f64::MAX, f64::MAX, f64::MAX, 0.0),
        (-f64::MAX, f64::MAX * 0.5, f64::MAX * 0.75, -f64::MAX * 0.25),
    ] {
        let alternating = [lower, upper, lower, upper, lower, upper, lower, upper];
        let mcse = QuantileMcse::estimate(&[&alternating, &alternating], 0.5).unwrap();
        assert_eq!(
            mcse.interval().map(f64::to_bits),
            [lower, upper].map(f64::to_bits)
        );
        assert!((upper - lower).is_infinite());
        // These two-point samples put both uncertainty bounds at the endpoints;
        // their half-width and median remain representable in original units.
        assert_relative_eq!(
            mcse.value(),
            expected_mcse,
            max_relative = 2.0 * f64::EPSILON
        );
        assert_relative_eq!(
            mcse.quantile(),
            expected_median,
            max_relative = 2.0 * f64::EPSILON
        );
    }
}

#[test]
fn finite_range_and_original_units_survive_power_of_two_rescaling() {
    let a = [-1.0, 0.5, 0.75, 1.5, -0.5, 0.75, 1.5, 0.5];
    let b = [1.5, 0.75, 0.5, -1.0, 0.75, 0.5, -0.5, 1.5];
    let mean = EssEstimate::estimate(&[&a, &b], EssEstimator::Mean).unwrap();
    let mcse = MeanMcse::estimate(&[&a, &b]).unwrap();
    let quantile = QuantileMcse::estimate(&[&a, &b], 0.5).unwrap();
    for scale in [f64::from_bits(0x7fe0_0000_0000_0000), 2.0_f64.powi(-900)] {
        let scaled_a = a.map(|x| x * scale);
        let scaled_b = b.map(|x| x * scale);
        assert_relative_eq!(
            EssEstimate::estimate(&[&scaled_a, &scaled_b], EssEstimator::Mean)
                .unwrap()
                .value(),
            mean.value(),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            MeanMcse::estimate(&[&scaled_a, &scaled_b]).unwrap().value() / scale,
            mcse.value(),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            QuantileMcse::estimate(&[&scaled_a, &scaled_b], 0.5)
                .unwrap()
                .value()
                / scale,
            quantile.value(),
            epsilon = 1e-12
        );
    }
    let tiny = [0., 1., 2., 3., 0., 1., 2., 3.].map(|x| x * f64::from_bits(8));
    assert!(
        EssEstimate::estimate(&[&tiny, &tiny], EssEstimator::Mean)
            .unwrap()
            .value()
            .is_finite()
    );
    assert!(MeanMcse::estimate(&[&tiny, &tiny]).unwrap().value() > 0.0);
    let small = [0.0, 1e-200, 2e-200, 3e-200, 0.0, 1e-200, 2e-200, 3e-200];
    let large = [1e200, 2e200, 3e200, 4e200, 1e200, 2e200, 3e200, 4e200];
    assert!(matches!(
        EssEstimate::estimate(&[&small, &large], EssEstimator::Mean),
        Err(MonteCarloError::UnresolvedVariance {
            chain_index: 0,
            half: 0,
            ..
        })
    ));
}

#[test]
fn odd_middle_draws_affect_original_variance_and_cutoffs_but_not_raw_ess() {
    let a = [0., 1., 3., 2., 100., 1., 0., 2., 3.];
    let b = [2., 0., 1., 3., -100., 0., 2., 3., 1.];
    let mut changed_a = a;
    let mut changed_b = b;
    changed_a[4] = 1.0;
    changed_b[4] = 2.0;
    let first = EssEstimate::estimate(&[&a, &b], EssEstimator::Mean).unwrap();
    let changed = EssEstimate::estimate(&[&changed_a, &changed_b], EssEstimator::Mean).unwrap();
    assert_eq!(first, changed);
    assert_eq!(first.sample_count(), 16);
    assert_eq!(first.original_sample_count(), 18);
    assert!(
        MeanMcse::estimate(&[&a, &b]).unwrap().value()
            > 10.0
                * MeanMcse::estimate(&[&changed_a, &changed_b])
                    .unwrap()
                    .value()
    );
    // Extreme middle draws move original-pool quantiles beyond retained data.
    // A split-first cutoff would incorrectly return a computable indicator ESS.
    assert_eq!(
        EssEstimate::estimate(&[&a, &b], EssEstimator::Quantile(0.01)),
        Err(MonteCarloError::DegenerateIndicator)
    );
    assert!(EssEstimate::estimate(&[&changed_a, &changed_b], EssEstimator::Quantile(0.01)).is_ok());
}

#[test]
fn unresolved_aggregate_variance_and_final_mcse_underflow_are_not_structural_constancy() {
    // One half has representable scaled variance at the subnormal floor;
    // averaging with constant halves loses it. The original pool is varying.
    let tiny_half = [0., 4.5e-162, 4.5e-162, 0., 0., 0., 0., 0.];
    let other = [1.; 8];
    assert_eq!(
        EssEstimate::estimate(&[&tiny_half, &other], EssEstimator::Mean),
        Err(MonteCarloError::NumericalFailure)
    );
    let least = f64::from_bits(1);
    let alternating = [0., least, 0., least, 0., least, 0., least];
    assert!(EssEstimate::estimate(&[&alternating, &alternating], EssEstimator::Mean).is_ok());
    assert_eq!(
        MeanMcse::estimate(&[&alternating, &alternating]),
        Err(MonteCarloError::NumericalFailure)
    );
    assert_eq!(
        QuantileMcse::estimate(&[&alternating, &alternating], 0.5),
        Err(MonteCarloError::NumericalFailure)
    );
}
