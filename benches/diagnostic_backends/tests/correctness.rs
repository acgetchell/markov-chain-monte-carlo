//! Independent arithmetic, adapter boundaries, and incompatible-contract evidence.

use diagnostic_backend_comparison::{ar1, arima_acf, assert_acf, exact_covariances, exact_time};
use ferromorphic::bayes::{BayesError, integrated_autocorrelation_time_within};
use markov_chain_monte_carlo::{Autocorrelation, AutocorrelationError};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

#[test]
fn ips_and_ims_can_differ_even_when_both_find_a_cutoff() {
    let mut rng = StdRng::seed_from_u64(73);
    let mut differences = 0;
    for _ in 0..1_000 {
        let raw: [i16; 16] = std::array::from_fn(|_| rng.random_range(-16..=17));
        let (numerators, denominator) = exact_covariances(&raw);
        let (ips, ips_window, capped) = exact_time(&numerators, denominator, 8, false);
        let (ims, ims_window, _) = exact_time(&numerators, denominator, 8, true);
        if capped || ips <= 0 || ims <= 0 || ips == ims {
            continue;
        }
        if numerators[..10]
            .as_chunks::<2>()
            .0
            .iter()
            .any(|p| p[0] + p[1] == 0)
        {
            continue;
        }
        let samples = raw.map(f64::from);
        let external = integrated_autocorrelation_time_within(&samples, 8)
            .unwrap_or_else(|error| panic!("diagnostic fixture failed: {error:?}"));
        let native = Autocorrelation::estimate(&samples, 8)
            .unwrap_or_else(|error| panic!("diagnostic fixture failed: {error:?}"))
            .integrated_time()
            .unwrap_or_else(|error| panic!("diagnostic fixture failed: {error:?}"));
        assert!(!external.floored && !external.truncated_at_cap);
        assert_eq!(external.lags, ips_window);
        assert_eq!(native.window(), ims_window);
        assert!((external.value - ips as f64 / denominator as f64).abs() < 1e-13);
        assert!((native.estimate() - ims as f64 / denominator as f64).abs() < 1e-13);
        differences += 1;
        if differences == 1 {
            eprintln!(
                "uncapped IPS/IMS counterexample: {raw:?}; IPS={ips}/{denominator}, IMS={ims}/{denominator}; window={ips_window}"
            );
        }
    }
    assert!(differences > 0);
    eprintln!(
        "{differences}/1000 fixed-seed integer fixtures have positive, uncapped but different IPS and IMS estimates"
    );
}

#[test]
fn exhaustive_eight_sample_integer_oracle() {
    let mut checked = 0;
    let mut boundary_cases = 0;
    let mut method_differences = 0;
    for mut code in 0..3_usize.pow(8) {
        let raw: [i16; 8] = std::array::from_fn(|_| {
            let x = i16::try_from(code % 3)
                .unwrap_or_else(|error| panic!("diagnostic fixture failed: {error:?}"))
                - 1;
            code /= 3;
            x
        });
        if raw.iter().all(|x| *x == raw[0]) {
            continue;
        }
        let samples = raw.map(f64::from);
        let (numerators, denominator) = exact_covariances(&raw);
        let expected: Vec<_> = numerators
            .iter()
            .map(|&x| x as f64 / denominator as f64)
            .collect();
        let native = Autocorrelation::estimate(&samples, 7)
            .unwrap_or_else(|error| panic!("diagnostic fixture failed: {error:?}"));
        for lag in [0, 1, 4, 7] {
            assert_acf(
                &arima_acf(&samples, lag)
                    .unwrap_or_else(|error| panic!("diagnostic fixture failed: {error:?}")),
                &expected[..=lag],
                1e-13,
            );
            assert_acf(
                &arima::acf::acf(&samples, Some(lag), false)
                    .unwrap_or_else(|error| panic!("diagnostic fixture failed: {error:?}")),
                &expected[..=lag],
                1e-13,
            );
        }
        assert_acf(native.values(), &expected, 1e-13);
        checked += 1;
        // Exact zero signs are roundoff-sensitive by the public contract.
        // Keep their ACF coverage above, separate them from strict sign checks.
        if numerators
            .as_chunks::<2>()
            .0
            .iter()
            .any(|p| p[0] + p[1] == 0)
        {
            boundary_cases += 1;
            continue;
        }
        let (ims, window, capped) = exact_time(&numerators, denominator, 7, true);
        match native.integrated_time() {
            Err(AutocorrelationError::TruncationNotFound { .. }) => assert!(capped),
            Err(AutocorrelationError::NonPositiveTime) => assert!(!capped && ims <= 0),
            Ok(time) => {
                assert!(!capped && ims > 0);
                assert_eq!(time.window(), window);
                assert!((time.estimate() - ims as f64 / denominator as f64).abs() < 1e-13);
            }
            other => panic!("unexpected native result: {other:?}"),
        }
        let actual = integrated_autocorrelation_time_within(&samples, 7)
            .unwrap_or_else(|error| panic!("diagnostic fixture failed: {error:?}"));
        let (ips, window, capped) = exact_time(&numerators, denominator, 4, false);
        assert_eq!(actual.lags, window);
        assert_eq!(actual.truncated_at_cap, capped);
        // An exactly zero final result is likewise a rounded decision boundary.
        if ips != 0 {
            assert_eq!(actual.floored, ips < 0);
            let expected = if ips < 0 {
                1.0
            } else {
                ips as f64 / denominator as f64
            };
            assert!((actual.value - expected).abs() < 1e-13);
        }
        let (same_cap_ims, _, _) = exact_time(&numerators, denominator, 4, true);
        if ips > 0 && same_cap_ims > 0 && ips != same_cap_ims {
            method_differences += 1;
            if method_differences == 1 {
                eprintln!(
                    "IPS/IMS counterexample: {raw:?}, IPS={ips}/{denominator}, IMS={same_cap_ims}/{denominator}"
                );
            }
        }
    }
    assert_eq!(checked, 6_558);
    eprintln!(
        "{checked} exact ACF fixtures; {boundary_cases} rounded-sign boundary fixtures; {method_differences} IPS/IMS differences at identical cap"
    );
}

#[test]
fn normalization_is_required_for_external_acf() {
    for (background, outlier) in [(f64::MAX, f64::MAX.next_down()), (0.0, f64::from_bits(1))] {
        let mut samples = [background; 8];
        samples[7] = outlier;
        let expected: Vec<_> = (0..8)
            .map(|k| if k == 0 { 1.0 } else { -f64::from(k) / 56.0 })
            .collect();
        assert_acf(
            &arima_acf(&samples, 7)
                .unwrap_or_else(|error| panic!("diagnostic fixture failed: {error:?}")),
            &expected,
            1e-13,
        );
        assert_acf(
            Autocorrelation::estimate(&samples, 7)
                .unwrap_or_else(|error| panic!("diagnostic fixture failed: {error:?}"))
                .values(),
            &expected,
            1e-13,
        );
        let raw = arima::acf::acf(&samples, Some(7), false)
            .unwrap_or_else(|error| panic!("diagnostic fixture failed: {error:?}"));
        assert!(raw[1..].iter().any(|x| !x.is_finite()));
        let external = integrated_autocorrelation_time_within(&samples, 7);
        if background == 0.0 {
            assert!(matches!(
                external,
                Err(BayesError::NoVariation { n: 8, value: 0.0 })
            ));
        } else {
            let value =
                external.unwrap_or_else(|error| panic!("diagnostic fixture failed: {error:?}"));
            assert!(value.floored && value.truncated_at_cap);
            assert_eq!(value.value, 1.0);
        }
        eprintln!(
            "extreme fixture background={background}, raw arima={raw:?}, ferromorphic={:?}",
            integrated_autocorrelation_time_within(&samples, 7)
        );
    }
    let samples = [
        -f64::MAX,
        f64::MAX,
        -f64::MAX,
        f64::MAX,
        -f64::MAX,
        f64::MAX,
        -f64::MAX,
        f64::MAX,
    ];
    assert_acf(
        &arima_acf(&samples, 7)
            .unwrap_or_else(|error| panic!("diagnostic fixture failed: {error:?}")),
        &[1.0, -0.875, 0.75, -0.625, 0.5, -0.375, 0.25, -0.125],
        1e-13,
    );
}

#[test]
fn adapter_rejects_invalid_input_before_calling_arima() {
    for (samples, lag, expected) in [
        (&[][..], 0, "insufficient samples: 0"),
        (&[1.0][..], 0, "insufficient samples: 1"),
        (&[1.0, 2.0][..], 2, "invalid max_lag 2 for 2 samples"),
        (&[0.0, f64::NAN][..], 1, "non-finite sample at 1"),
        (&[0.0, f64::INFINITY][..], 1, "non-finite sample at 1"),
        (&[1.0, 1.0][..], 1, "constant trace"),
    ] {
        assert_eq!(arima_acf(samples, lag).unwrap_err(), expected);
    }
}

#[test]
fn ferromorphic_has_a_different_domain_and_failure_contract() {
    assert!(matches!(
        integrated_autocorrelation_time_within(&[1.0, 2.0, 3.0, 4.0], 3),
        Err(BayesError::TooShort { got: 4, want: 8 })
    ));
    let alternating = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
    let external = integrated_autocorrelation_time_within(&alternating, 7)
        .unwrap_or_else(|error| panic!("diagnostic fixture failed: {error:?}"));
    assert!(external.floored && external.truncated_at_cap);
    assert_eq!(external.value, 1.0);
    assert_eq!(external.lags, 3);
    assert!(matches!(
        Autocorrelation::estimate(&alternating, 7)
            .unwrap_or_else(|error| panic!("diagnostic fixture failed: {error:?}"))
            .integrated_time(),
        Err(AutocorrelationError::TruncationNotFound { .. })
    ));
}

#[test]
fn both_estimators_track_ar1_closed_forms_on_fixed_seeds() {
    for phi in [0.0, 0.9, -0.5] {
        for seed in [73, 74, 75] {
            let samples = ar1(100_000, phi, seed);
            let acf = Autocorrelation::estimate(&samples, 400)
                .unwrap_or_else(|error| panic!("diagnostic fixture failed: {error:?}"));
            let ims = acf
                .integrated_time()
                .unwrap_or_else(|error| panic!("diagnostic fixture failed: {error:?}"));
            let ips = integrated_autocorrelation_time_within(&samples, 400)
                .unwrap_or_else(|error| panic!("diagnostic fixture failed: {error:?}"));
            let expected = (1.0 + phi) / (1.0 - phi);
            assert!(!ips.truncated_at_cap && !ips.floored);
            // Deterministic regression allowances, not confidence intervals.
            assert!((ims.estimate() / expected - 1.0).abs() < 0.2);
            assert!((ips.value / expected - 1.0).abs() < 0.2);
            eprintln!(
                "phi={phi}, seed={seed}: IMS={}, IPS={}, analytic={expected}",
                ims.estimate(),
                ips.value
            );
        }
    }
}
