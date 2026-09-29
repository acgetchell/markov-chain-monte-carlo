//! Rank-normalized component: independent references and deterministic regimes.

use std::f64::consts::PI;

use approx::assert_relative_eq;
use markov_chain_monte_carlo::{RankNormalizedSplitRhat, SplitRhat, SplitRhatError};

fn estimate(chains: &[Vec<f64>]) -> RankNormalizedSplitRhat {
    let borrowed: Vec<_> = chains.iter().map(Vec::as_slice).collect();
    RankNormalizedSplitRhat::estimate(&borrowed)
        .unwrap_or_else(|error| panic!("rank normalization failed for {chains:?}: {error:?}"))
}

#[test]
fn arviz_fixtures_match_component_not_maximum() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/rank_normalized_rhat.json")).unwrap();
    let tolerance = fixture["reference"]["relative_tolerance"].as_f64().unwrap();
    assert!(tolerance > 0.0 && tolerance <= 5e-13);
    let cases = fixture["cases"].as_array().unwrap();
    assert!(
        !cases.is_empty(),
        "the independent oracle must exercise cases"
    );
    for case in cases {
        let chains: Vec<Vec<f64>> = case["chains"]
            .as_array()
            .unwrap()
            .iter()
            .map(|chain| {
                chain
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|draw| draw.as_f64().unwrap())
                    .collect()
            })
            .collect();
        let actual = estimate(&chains);
        let expected = case["expected"].as_f64().unwrap();
        assert!(
            (actual.value() / expected - 1.0).abs() <= tolerance,
            "{}: {actual:?}, expected {expected}",
            case["name"]
        );
        assert_eq!(actual.chain_count(), chains.len());
        assert_eq!(actual.samples_per_chain(), chains[0].len());
        assert_eq!(actual.samples_per_split_chain(), chains[0].len() / 2);
        if let Some(combined) = case["combined_reference"].as_f64() {
            assert!(combined > 1.9);
            assert!(
                actual.value() < 1.0,
                "folding must remain a separate component"
            );
        }
    }
}

#[test]
fn two_level_ranks_match_exact_variance_ratio() {
    // Pooled ranks are 2.5 and 6.5, hence the normal scores are -z and z.
    // Each half has mean zero and unbiased variance 2*z^2. B=0, n=2,
    // so the result is sqrt(1/2), independent of inverse-normal approximation.
    let a = [-7.0, 100.0, 100.0, -7.0];
    let b = [100.0, -7.0, -7.0, 100.0];
    let rhat = RankNormalizedSplitRhat::estimate(&[&a, &b]).unwrap();
    assert_relative_eq!(rhat.value(), 0.5_f64.sqrt(), epsilon = 1e-14);
}

#[test]
fn ranks_preserve_strict_monotone_transforms_and_chain_identity() {
    let chains = vec![
        vec![-4.0_f64, -1.0, 0.0, 0.25, 1.0, 6.0],
        vec![2.0, -3.0, 8.0, 0.5, -0.5, 3.0],
        vec![1.0, 2.0, -2.0, 9.0, 4.0, 0.0],
    ];
    let expected = estimate(&chains).value();
    // Exp and negation preserve represented distinctions and ties here;
    // squaring would not be monotone over this domain and is deliberately absent.
    for transform in [f64::exp, |x: f64| -x] {
        let transformed: Vec<Vec<_>> = chains
            .iter()
            .map(|chain| chain.iter().copied().map(transform).collect())
            .collect();
        assert_relative_eq!(estimate(&transformed).value(), expected, epsilon = 1e-14);
    }
    let mut reversed = chains;
    reversed.reverse();
    for chain in &mut reversed {
        chain.reverse();
    }
    assert_relative_eq!(estimate(&reversed).value(), expected, epsilon = 1e-14);

    let separated = [&[0.0, 1.0, 0.0, 1.0][..], &[10.0, 11.0, 10.0, 11.0]];
    let drifting = [&[0.0, 1.0, 10.0, 11.0][..], &[10.0, 11.0, 0.0, 1.0]];
    let location = RankNormalizedSplitRhat::estimate(&separated).unwrap();
    let drift = RankNormalizedSplitRhat::estimate(&drifting).unwrap();
    assert!(location.value() > 1.5);
    assert_relative_eq!(location.value(), drift.value(), epsilon = 1e-14);
}

#[test]
fn extremes_signed_zero_and_omitted_draws_preserve_borrowed_inputs() {
    let tiny = f64::from_bits(1);
    let a = [f64::MAX, tiny, f64::MIN, -0.0, tiny];
    let b = [-f64::MAX, tiny, f64::MAX, 0.0, -tiny];
    let before = [a.map(f64::to_bits), b.map(f64::to_bits)];
    // Ordered finite values [-MAX,-tiny,0,tiny,MAX] map to [-2,-1,0,1,2].
    let reference = [&[2.0, 1.0, 0.0, 1.0][..], &[-2.0, 1.0, 0.0, -1.0]];
    let expected = RankNormalizedSplitRhat::estimate(&reference).unwrap();
    let actual = RankNormalizedSplitRhat::estimate(&[&a, &b]).unwrap();
    assert_eq!(actual.value().to_bits(), expected.value().to_bits());
    assert_eq!(actual.samples_per_chain(), 5);
    assert_eq!(actual.samples_per_split_chain(), 2);
    assert_eq!([a.map(f64::to_bits), b.map(f64::to_bits)], before);

    // Raw moments lose the tiny half's variance under common scaling. Ranks
    // preserve its ordering without subtracting or squaring the original values.
    let small = [0.0, tiny, 0.0, tiny];
    let large = [0.0, f64::MAX, 0.0, f64::MAX];
    let error = SplitRhat::estimate(&[&small, &large]).unwrap_err();
    assert!(
        matches!(
            error,
            SplitRhatError::UnresolvedVariance {
                chain_index: 0,
                half: 0,
                ..
            }
        ),
        "{error:?}"
    );
    let ranked = RankNormalizedSplitRhat::estimate(&[&small, &large]).unwrap();
    // ArviZ 0.22.0, rhat(method="z_scale"), for the order-equivalent
    // [[0,1,0,1], [0,2,0,2]]; SciPy 1.16.2 and NumPy 2.2.6.
    assert_relative_eq!(ranked.value(), 0.743_027_833_460_545_2, epsilon = 5e-13);
}

#[test]
fn input_errors_report_original_indices() {
    let good = [0.0, 1.0, 2.0, 3.0];
    for chains in [vec![], vec![good.as_slice()]] {
        let error = RankNormalizedSplitRhat::estimate(&chains).unwrap_err();
        assert!(
            matches!(error, SplitRhatError::InsufficientChains { count, .. } if count == chains.len()),
            "{error:?}"
        );
    }
    for chain_index in 0..3 {
        for count in 0..4 {
            let mut chains = [good.as_slice(); 3];
            chains[chain_index] = &good[..count];
            let error = RankNormalizedSplitRhat::estimate(&chains).unwrap_err();
            assert!(
                matches!(error, SplitRhatError::InsufficientSamples { chain_index: actual_chain, count: actual_count, .. }
                    if actual_chain == chain_index && actual_count == count),
                "chain={chain_index}, count={count}: {error:?}"
            );
        }
    }
    let error = RankNormalizedSplitRhat::estimate(&[&[f64::NAN; 4], &[0.0; 5]]).unwrap_err();
    assert!(
        matches!(
            error,
            SplitRhatError::UnequalLengths {
                chain_index: 1,
                expected: 4,
                actual: 5,
                ..
            }
        ),
        "{error:?}"
    );
    for chain_index in 0..3 {
        for nonfinite in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for sample_index in 0..5 {
                let mut bad = [0.0, 1.0, 999.0, 2.0, 3.0];
                bad[sample_index] = nonfinite;
                let mut chains = [&[1.0; 5][..]; 3];
                chains[chain_index] = &bad;
                let error = RankNormalizedSplitRhat::estimate(&chains).unwrap_err();
                assert!(
                    matches!(error, SplitRhatError::NonFiniteSample { chain_index: actual_chain, sample_index: actual_sample, .. }
                        if actual_chain == chain_index && actual_sample == sample_index),
                    "chain={chain_index}, sample={sample_index}, value={nonfinite}: {error:?}"
                );
            }
        }
        for (bad, half) in [([-0.0, 0.0, 1.0, 2.0], 0), ([1.0, 2.0, -0.0, 0.0], 1)] {
            let mut chains = [good.as_slice(); 3];
            chains[chain_index] = &bad;
            let error = RankNormalizedSplitRhat::estimate(&chains).unwrap_err();
            assert!(
                matches!(error, SplitRhatError::ConstantSplitChain { chain_index: actual_chain, half: actual_half, .. }
                    if actual_chain == chain_index && actual_half == half),
                "chain={chain_index}, half={half}: {error:?}"
            );
        }
    }
    for chains in [[&[1.0; 4][..], &[1.0; 4]], [&[1.0; 4][..], &[2.0; 4]]] {
        let error = RankNormalizedSplitRhat::estimate(&chains).unwrap_err();
        assert!(
            matches!(
                error,
                SplitRhatError::ConstantSplitChain {
                    chain_index: 0,
                    half: 0,
                    ..
                }
            ),
            "{error:?}"
        );
    }
}

#[test]
fn cauchy_location_shift_exposes_raw_moment_weakness() {
    // A fixed midpoint quantile grid of Cauchy(0,1), not random simulation or
    // evidence of converged sampling. Its symmetric tails dominate raw variance.
    let half: Vec<_> = (0..512_u32)
        .map(|i| (PI * ((f64::from(i) + 0.5) / 512.0 - 0.5)).tan())
        .collect();
    let chain: Vec<_> = half.iter().chain(half.iter().rev()).copied().collect();
    let shifted: Vec<_> = chain.iter().map(|x| x + 4.0).collect();
    let control = RankNormalizedSplitRhat::estimate(&[&chain, &chain]).unwrap();
    // Every control half has the same distribution and exactly zero mean
    // normal score, hence B=0 and Rhat=sqrt((n-1)/n).
    assert_relative_eq!(control.value(), (511.0_f64 / 512.0).sqrt(), epsilon = 1e-14);
    let raw = SplitRhat::estimate(&[&chain, &shifted]).unwrap();
    let ranked = RankNormalizedSplitRhat::estimate(&[&chain, &shifted]).unwrap();
    // Deliberately wide separation checks on deterministic inputs; these are
    // demonstrations of sensitivity, not general-purpose decision thresholds.
    assert!(raw.value() < 1.01, "{raw:?}");
    assert!(ranked.value() > 1.1, "{ranked:?}");
}
