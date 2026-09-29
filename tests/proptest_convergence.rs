//! Exact integer moments and metamorphic oracles for both split R-hat estimators.

use approx::assert_relative_eq;
use markov_chain_monte_carlo::{RankNormalizedSplitRhat, SplitRhat};
use proptest::prelude::*;

/// Generate equal-length chains with independently guaranteed variation in
/// every half. No production result participates in admitting an input.
fn comparable_chains() -> impl Strategy<Value = Vec<Vec<i16>>> {
    (2usize..=5, 2usize..=8)
        .prop_flat_map(|(chains, half_length)| {
            prop::collection::vec(prop::collection::vec(-16i16..=16, half_length), 2 * chains)
        })
        .prop_map(|mut halves| {
            for half in &mut halves {
                half[1] = half[0] + 1;
            }
            halves
                .as_chunks::<2>()
                .0
                .iter()
                .map(|[first, last]| first.iter().chain(last).copied().collect())
                .collect()
        })
}

/// Expand the variance ratio using integer raw moments, without the floating
/// centering, scaling, or compensated sums used by production. With 4..=10
/// halves of 2..=8 integers in -16..=17, every intermediate fits in `i32`.
fn exact_split_rhat(chains: &[Vec<i16>]) -> f64 {
    let half_length = chains[0].len() / 2;
    let n = i32::try_from(half_length).unwrap();
    let m = i32::try_from(2 * chains.len()).unwrap();
    let mut total = 0;
    let mut squared_sums = 0;
    let mut within_numerator = 0;
    for half in chains
        .iter()
        .flat_map(|chain| chain.chunks_exact(half_length))
    {
        let sum: i32 = half.iter().map(|&value| i32::from(value)).sum();
        let squares: i32 = half.iter().map(|&value| i32::from(value).pow(2)).sum();
        let variation = n * squares - sum.pow(2);
        assert!(variation > 0, "the generator guarantees nonconstant halves");
        total += sum;
        squared_sums += sum.pow(2);
        within_numerator += variation;
    }
    // A=sum(n*Q-S^2), D=m*sum(S^2)-sum(S)^2 for half sums S and
    // squared sums Q. Rhat^2 = (n-1)*((m-1)*A+D)/(n*(m-1)*A).
    let between_numerator = m * squared_sums - total.pow(2);
    let numerator = (n - 1) * ((m - 1) * within_numerator + between_numerator);
    let denominator = n * (m - 1) * within_numerator;
    (f64::from(numerator) / f64::from(denominator)).sqrt()
}

#[test]
fn unequal_half_variances_match_hand_calculated_ratio() {
    let raw = vec![
        vec![0, 1, 2, 0, 1, 2],
        vec![2, 4, 6, 2, 4, 6],
        vec![4, 7, 10, 4, 7, 10],
    ];
    // Six half means [1,1,4,4,7,7] give B/n=36/5. Their variances
    // [1,1,4,4,9,9] give W=14/3, so Rhat^2=2/3+54/35=232/105.
    let expected = (232.0_f64 / 105.0).sqrt();
    assert_relative_eq!(exact_split_rhat(&raw), expected, epsilon = 1e-14);
    let samples: Vec<Vec<_>> = raw
        .iter()
        .map(|chain| chain.iter().map(|&value| f64::from(value)).collect())
        .collect();
    let slices: Vec<_> = samples.iter().map(Vec::as_slice).collect();
    let estimate = SplitRhat::estimate(&slices).unwrap();
    assert_relative_eq!(estimate.value(), expected, epsilon = 1e-14);
    assert_eq!(estimate.chain_count(), 3);
    assert_eq!(estimate.samples_per_chain(), 6);
    assert_eq!(estimate.samples_per_split_chain(), 3);
}

proptest! {
    #[test]
    fn rank_normalized_binary_chains_match_exact_integer_moments(raw in comparable_chains()) {
        // Adjacent integers in each half map to distinct binary values. Thus
        // every admitted half varies, without consulting the production result.
        let binary: Vec<Vec<_>> = raw.iter().map(|chain| {
            chain.iter().map(|value| value.rem_euclid(2)).collect()
        }).collect();
        // Pooled normal scores have exactly two distinct levels. They are an
        // affine transform of these binary inputs, so the integer-moment oracle
        // remains exact without reimplementing ranks or inverse-normal scores.
        let expected = exact_split_rhat(&binary);
        let samples: Vec<Vec<_>> = binary.iter().map(|chain| {
            chain.iter().map(|&value| f64::from(value)).collect()
        }).collect();
        let slices: Vec<_> = samples.iter().map(Vec::as_slice).collect();
        let actual = RankNormalizedSplitRhat::estimate(&slices).map_err(|error| {
            TestCaseError::fail(format!("binary={binary:?}, unexpected error={error:?}"))
        })?;
        prop_assert!(
            (actual.value() / expected - 1.0).abs() <= 1e-12,
            "binary={:?}, actual={}, exact={}", binary, actual.value(), expected,
        );
    }

    #[test]
    fn rank_normalized_chains_preserve_order_ties_and_omitted_middles(raw in comparable_chains()) {
        let samples: Vec<Vec<_>> = raw.iter().map(|chain| {
            chain.iter().map(|&value| f64::from(value)).collect()
        }).collect();
        let slices: Vec<_> = samples.iter().map(Vec::as_slice).collect();
        let expected = RankNormalizedSplitRhat::estimate(&slices).map_err(|error| {
            TestCaseError::fail(format!("original raw={raw:?}, unexpected error={error:?}"))
        })?.value();
        // Cubing these small integers is exactly representable and strictly
        // increasing. Negation reverses order while retaining every tie.
        let cubic: Vec<Vec<_>> = samples.iter().map(|chain| {
            chain.iter().map(|value| value.powi(3)).collect()
        }).collect();
        let decreasing: Vec<Vec<_>> = cubic.iter().map(|chain| {
            chain.iter().map(|value| -value).collect()
        }).collect();
        let reordered: Vec<Vec<_>> = samples.iter().rev().enumerate().map(|(index, chain)| {
            let mut chain = chain.clone();
            // Reverse only some chains so preservation cannot depend on a
            // single reversal of the flattened pooled sequence.
            if index % 2 == 0 {
                chain.reverse();
            }
            chain
        }).collect();
        let odd: Vec<Vec<_>> = samples.iter().enumerate().map(|(index, chain)| {
            let mut chain = chain.clone();
            chain.insert(chain.len() / 2, if index % 2 == 0 { f64::MAX } else { -f64::MAX });
            chain
        }).collect();
        for (label, input) in [
            ("increasing cubic", &cubic),
            ("decreasing cubic", &decreasing),
            ("reordered chains and selected time reversals", &reordered),
            ("odd middle omitted", &odd),
        ] {
            let slices: Vec<_> = input.iter().map(Vec::as_slice).collect();
            let actual = RankNormalizedSplitRhat::estimate(&slices).map_err(|error| {
                TestCaseError::fail(format!("{label}: raw={raw:?}, unexpected error={error:?}"))
            })?;
            prop_assert_eq!(actual.chain_count(), raw.len());
            prop_assert_eq!(actual.samples_per_chain(), input[0].len());
            prop_assert_eq!(actual.samples_per_split_chain(), raw[0].len() / 2);
            prop_assert!(
                (actual.value() / expected - 1.0).abs() <= 1e-12,
                "{}: raw={:?}, actual={}, expected={}", label, raw, actual.value(), expected,
            );
        }
    }

    #[test]
    fn split_rhat_matches_integer_moments_and_invariances(
        raw in comparable_chains(),
        exponent in -900i32..=900,
        offset in -128i16..=128,
    ) {
        let expected = exact_split_rhat(&raw);
        let samples: Vec<Vec<_>> = raw.iter()
            .map(|chain| chain.iter().map(|&value| f64::from(value)).collect())
            .collect();
        let scale = -2.0_f64.powi(exponent);
        let affine: Vec<Vec<_>> = raw.iter()
            .map(|chain| chain.iter().map(|&value| f64::from(value + offset) * scale).collect())
            .collect();
        let reversed: Vec<Vec<_>> = samples.iter().rev()
            .map(|chain| chain.iter().rev().copied().collect()).collect();
        let odd: Vec<Vec<_>> = samples.iter().enumerate().map(|(index, chain)| {
            let mut extended = chain.clone();
            // Omitted draws must not affect moments or their numerical scale.
            extended.insert(chain.len() / 2, if index % 2 == 0 { f64::MAX } else { -f64::MAX });
            extended
        }).collect();

        for (label, input) in [
            ("original", &samples),
            ("affine", &affine),
            ("reordered chains and reversed time", &reversed),
            ("odd middle omitted", &odd),
        ] {
            let slices: Vec<_> = input.iter().map(Vec::as_slice).collect();
            let estimate = SplitRhat::estimate(&slices).map_err(|error| {
                TestCaseError::fail(format!(
                    "{label}: raw={raw:?}, exponent={exponent}, offset={offset}, \
                     unexpected error={error:?}"
                ))
            })?;
            prop_assert_eq!(estimate.chain_count(), raw.len());
            prop_assert_eq!(estimate.samples_per_chain(), input[0].len());
            prop_assert_eq!(estimate.samples_per_split_chain(), raw[0].len() / 2);
            prop_assert!(
                (estimate.value() / expected - 1.0).abs() <= 1e-12,
                "{}: raw={:?}, exponent={}, offset={}, actual={}, exact={}",
                label, raw, exponent, offset, estimate.value(), expected,
            );
        }
    }
}
