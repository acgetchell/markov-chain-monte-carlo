//! Independent arithmetic, numerical boundaries, and stochastic diagnostic checks.

use std::time::Duration;

use approx::assert_relative_eq;
use markov_chain_monte_carlo::{
    Autocorrelation, AutocorrelationError, EssRateError, SplitRhat, SplitRhatError,
};
use rand::{RngExt, SeedableRng, rngs::StdRng};

#[test]
fn ess_and_rate_have_sample_and_wall_clock_units() {
    let time = Autocorrelation::estimate(&[1.0, 2.0, 3.0, 4.0], 3)
        .unwrap()
        .integrated_time()
        .unwrap();
    // Exact covariances give rho[1]=1/4 and tau=3/2, hence ESS=8/3.
    assert_relative_eq!(time.effective_sample_size(), 8.0 / 3.0, epsilon = 1e-14);
    for seconds in [0.5, 2.0, 100.0] {
        assert_relative_eq!(
            time.effective_sample_size_per_second(Duration::from_secs_f64(seconds))
                .unwrap(),
            (8.0 / 3.0) / seconds,
            epsilon = 1e-14
        );
    }
    assert_eq!(
        time.effective_sample_size_per_second(Duration::ZERO),
        Err(EssRateError)
    );
    // Relative error matters at both extremes: an absolute tolerance could
    // accept zero or an arbitrary positive value for the very small rate.
    for (duration, expected) in [
        (Duration::from_nanos(1), 8e9 / 3.0),
        (Duration::MAX, (8.0 / 3.0) * 2.0_f64.powi(-64)),
    ] {
        let rate = time.effective_sample_size_per_second(duration).unwrap();
        assert_relative_eq!(rate, expected, epsilon = 0.0, max_relative = 1e-14);
    }
    let anticorrelated = Autocorrelation::estimate(&[3.0, -1.0, -1.0, -1.0], 3)
        .unwrap()
        .integrated_time()
        .unwrap();
    assert_relative_eq!(
        anticorrelated.effective_sample_size(),
        24.0 / 5.0,
        epsilon = 1e-14
    );
    assert!(matches!(
        Autocorrelation::estimate(&[1.0; 4], 3),
        Err(AutocorrelationError::ConstantTrace)
    ));
    assert!(matches!(
        Autocorrelation::estimate(&[0.0, 1.0], 1)
            .unwrap()
            .integrated_time(),
        Err(AutocorrelationError::TruncationNotFound { .. })
    ));
}

#[test]
fn split_rhat_matches_exact_between_and_within_variance() {
    let left = [0.0, 2.0, 0.0, 2.0];
    let right = [2.0, 4.0, 2.0, 4.0];
    // Half means [1,1,3,3], unbiased variance 4/3, W=2, n=2.
    // var+ = W/2 + 4/3 = 7/3, so Rhat^2=7/6.
    let estimate = SplitRhat::estimate(&[&left, &right]).unwrap();
    assert_relative_eq!(estimate.value(), (7.0_f64 / 6.0).sqrt(), epsilon = 1e-14);
    assert_eq!(estimate.chain_count(), 2);
    assert_eq!(estimate.samples_per_chain(), 4);
    assert_eq!(estimate.samples_per_split_chain(), 2);
    let same_means = SplitRhat::estimate(&[&left, &left]).unwrap();
    assert_relative_eq!(same_means.value(), 0.5_f64.sqrt(), epsilon = 1e-14);
    // Odd middle draws are discarded, even when they would dominate moments.
    let odd_left = [0.0, 2.0, f64::MAX, 0.0, 2.0];
    let odd_right = [2.0, 4.0, -f64::MAX, 2.0, 4.0];
    let odd = SplitRhat::estimate(&[&odd_left, &odd_right]).unwrap();
    assert_eq!(odd.value().to_bits(), estimate.value().to_bits());
    assert_eq!(odd.samples_per_chain(), 5);
    assert_eq!(odd.samples_per_split_chain(), 2);
}

#[test]
fn invalid_rhat_inputs_are_typed_and_located() {
    let valid = [0.0, 1.0, 2.0, 3.0];
    for chains in [vec![], vec![valid.as_slice()]] {
        assert!(
            matches!(SplitRhat::estimate(&chains), Err(SplitRhatError::InsufficientChains { count, .. }) if count == chains.len())
        );
    }
    for count in 0..4 {
        assert!(
            matches!(SplitRhat::estimate(&[&valid, &valid[..count]]), Err(SplitRhatError::InsufficientSamples { chain_index: 1, count: actual, .. }) if actual == count)
        );
    }
    assert!(matches!(
        SplitRhat::estimate(&[&valid, &[1.0; 5]]),
        Err(SplitRhatError::UnequalLengths {
            chain_index: 1,
            expected: 4,
            actual: 5,
            ..
        })
    ));
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for index in 0..5 {
            let mut bad = [0.0, 1.0, 2.0, 3.0, 4.0];
            bad[index] = invalid;
            assert!(
                matches!(SplitRhat::estimate(&[&[0.0, 1.0, 2.0, 3.0, 4.0], &bad]), Err(SplitRhatError::NonFiniteSample { chain_index: 1, sample_index, .. }) if sample_index == index)
            );
        }
    }
    for (bad, expected_half) in [([1.0; 4], 0), ([0.0, 1.0, 2.0, 2.0], 1)] {
        assert!(
            matches!(SplitRhat::estimate(&[&valid, &bad]), Err(SplitRhatError::ConstantSplitChain { chain_index: 1, half, .. }) if half == expected_half)
        );
    }
    for pair in [
        [&[1.0; 4][..], &[1.0; 4][..]],
        [&[1.0; 4][..], &[2.0; 4][..]],
    ] {
        assert!(matches!(
            SplitRhat::estimate(&pair),
            Err(SplitRhatError::ConstantSplitChain { .. })
        ));
    }
}

#[test]
fn split_rhat_validates_shapes_before_values_and_halves() {
    let valid = [0.0, 1.0, 2.0, 3.0];
    let nonfinite = [0.0, f64::NAN, f64::INFINITY, 3.0];

    let result = SplitRhat::estimate(&[&[f64::NAN]]);
    assert!(
        matches!(
            result,
            Err(SplitRhatError::InsufficientChains { count: 1, .. })
        ),
        "chain count must precede length and finiteness: {result:?}"
    );
    let result = SplitRhat::estimate(&[&[f64::NAN; 3], &valid]);
    assert!(
        matches!(
            result,
            Err(SplitRhatError::InsufficientSamples {
                chain_index: 0,
                count: 3,
                ..
            })
        ),
        "the first chain's minimum length must be checked: {result:?}"
    );
    let result = SplitRhat::estimate(&[&nonfinite, &[0.0; 5], &[]]);
    assert!(
        matches!(
            result,
            Err(SplitRhatError::UnequalLengths {
                chain_index: 1,
                expected: 4,
                actual: 5,
                ..
            })
        ),
        "lengths are checked in order, before any values: {result:?}"
    );
    let result = SplitRhat::estimate(&[&[0.0; 4], &nonfinite]);
    assert!(
        matches!(
            result,
            Err(SplitRhatError::NonFiniteSample {
                chain_index: 1,
                sample_index: 1,
                ..
            })
        ),
        "all values must be checked before detecting constant halves: {result:?}"
    );
    let result = SplitRhat::estimate(&[&valid, &[0.0, -0.0, 1.0, 2.0]]);
    assert!(
        matches!(
            result,
            Err(SplitRhatError::ConstantSplitChain {
                chain_index: 1,
                half: 0,
                ..
            })
        ),
        "signed zeros must not count as within-half variation: {result:?}"
    );
}

#[test]
fn split_rhat_handles_offsets_extremes_and_unresolved_variance() {
    let left = [0.0, 2.0, 0.0, 2.0];
    let right = [2.0, 4.0, 2.0, 4.0];
    let expected = (7.0_f64 / 6.0).sqrt();
    for scale in [f64::from_bits(1), 1e-200, 1e200, f64::MAX / 4.0] {
        let a = left.map(|x| x * scale);
        let b = right.map(|x| x * scale);
        assert_relative_eq!(
            SplitRhat::estimate(&[&a, &b]).unwrap().value(),
            expected,
            epsilon = 1e-14
        );
    }
    for offset in [-1e16, 1e16] {
        let a = left.map(|x| x + offset);
        let b = right.map(|x| x + offset);
        assert_relative_eq!(
            SplitRhat::estimate(&[&a, &b]).unwrap().value(),
            expected,
            epsilon = 1e-14
        );
    }
    let extreme = [0.0, f64::MAX, -f64::MAX, 0.0];
    let reference = [0.0, 1.0, -1.0, 0.0];
    assert_relative_eq!(
        SplitRhat::estimate(&[&extreme, &extreme]).unwrap().value(),
        SplitRhat::estimate(&[&reference, &reference])
            .unwrap()
            .value(),
        epsilon = 1e-14
    );
    let tiny = [0.0, f64::MIN_POSITIVE, 0.0, f64::MIN_POSITIVE];
    let huge = [0.0, f64::MAX, 0.0, f64::MAX];
    let mixed = [0.0, f64::MAX, 0.0, f64::MIN_POSITIVE];
    for (chains, expected_chain, expected_half) in [
        ([&tiny[..], &huge[..]], 0, 0),
        ([&huge[..], &tiny[..]], 1, 0),
        ([&mixed[..], &huge[..]], 0, 1),
        ([&huge[..], &mixed[..]], 1, 1),
    ] {
        let error = SplitRhat::estimate(&chains).unwrap_err();
        assert!(
            matches!(error, SplitRhatError::UnresolvedVariance { chain_index, half, .. }
                if chain_index == expected_chain && half == expected_half),
            "expected unresolved variance in chain {expected_chain}, half {expected_half}: {error:?}"
        );
        let message = error.to_string();
        assert!(
            message.contains(&format!("chain {expected_chain} half {expected_half}")),
            "the message must locate the unresolved variance: {message}"
        );
    }
}

#[test]
fn rhat_detects_separated_locations_and_within_chain_drift() {
    let low = [0.0, 1.0, 0.0, 1.0];
    let high = [10.0, 11.0, 10.0, 11.0];
    assert!(SplitRhat::estimate(&[&low, &high]).unwrap().value() > 8.0);
    // Both original chains have identical moments, so unsplit R-hat would
    // miss their opposing trends. Splitting exposes their different halves.
    let rising = [0.0, 1.0, 10.0, 11.0];
    let falling = [11.0, 10.0, 1.0, 0.0];
    assert!(SplitRhat::estimate(&[&rising, &falling]).unwrap().value() > 8.0);
}

#[test]
fn independent_and_correlated_samples_follow_analytic_ess() {
    const COUNT: u32 = 30_000;
    let mut independent_chains = Vec::new();
    for seed in [123, 456, 789, 1011] {
        let mut rng = StdRng::seed_from_u64(seed);
        let independent: Vec<f64> = (0..COUNT).map(|_| rng.random_range(-1.0..1.0)).collect();
        let correlated: Vec<_> = independent
            .iter()
            .scan(0.0, |state, &noise| {
                *state = 0.8_f64.mul_add(*state, noise);
                Some(*state)
            })
            .collect();
        // AR(1) with bounded independent innovations has rho[k]=0.8^k and
        // true tau=9. Discard 1000 transitions to remove initialization.
        for (samples, true_time) in [(&independent[1000..], 1.0), (&correlated[1000..], 9.0)] {
            let time = Autocorrelation::estimate(samples, 400)
                .unwrap()
                .integrated_time()
                .unwrap();
            let expected = f64::from(COUNT - 1000) / true_time;
            // Broad 25% estimator tolerance across fixed distinct seeds;
            // exact fixtures above test arithmetic independently of this noise.
            assert!(
                (time.effective_sample_size() / expected - 1.0).abs() < 0.25,
                "seed={seed}, tau={true_time}, time={time:?}"
            );
        }
        independent_chains.push(independent);
    }
    let chains: Vec<_> = independent_chains.iter().map(Vec::as_slice).collect();
    assert!((SplitRhat::estimate(&chains).unwrap().value() - 1.0).abs() < 0.005);
}
