//! Independent folded/combined references and explicit unavailable diagnostics.

use approx::assert_relative_eq;
use markov_chain_monte_carlo::prelude::{
    CombinedRhat, FoldedRankNormalizedSplitRhat, RankNormalizedSplitRhat, SplitRhatError,
};

/// Preserve the fixture case and component in numerical failure diagnostics.
fn assert_reference(actual: f64, expected: f64, tolerance: f64, name: &str, component: &str) {
    assert!(
        (actual / expected - 1.0).abs() <= tolerance,
        "{name} {component}: actual={actual}, reference={expected}, tolerance={tolerance}"
    );
}

/// Compare rejection with raw-input expectations, never another production path.
fn assert_invalid_input(chains: &[&[f64]], expected: fn(SplitRhatError) -> bool, message: &str) {
    for (estimator, result) in [
        (
            "ranked",
            RankNormalizedSplitRhat::estimate(chains).map(drop),
        ),
        (
            "folded",
            FoldedRankNormalizedSplitRhat::estimate(chains).map(drop),
        ),
        ("combined", CombinedRhat::estimate(chains).map(drop)),
    ] {
        let error = result.expect_err("invalid raw inputs must be rejected");
        assert!(
            expected(error),
            "{estimator}: chains={chains:?}, error={error:?}"
        );
        assert_eq!(error.to_string(), message, "{estimator}");
    }
}

/// Keep every independently referenced scientific regime in the fixture corpus.
fn assert_fixture_cases(cases: &[serde_json::Value]) {
    let mut names: Vec<_> = cases
        .iter()
        .map(|case| case["name"].as_str().unwrap())
        .collect();
    names.sort_unstable();
    assert_eq!(
        names,
        [
            "binary_folded_degeneracy",
            "discrete_counts",
            "heavy_tails",
            "location",
            "odd_median",
            "odd_pooled_count",
            "one_folded_half_constant",
            "scale_only",
            "well_mixed_control"
        ]
    );
}

#[test]
fn posterior_fixtures_and_input_immutability() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/combined_rhat.json")).unwrap();
    assert_eq!(fixture["reference"]["version"], "1.7.0");
    let tolerance = fixture["reference"]["relative_tolerance"].as_f64().unwrap();
    assert!(tolerance > 0.0 && tolerance <= 5e-13);
    let cases = fixture["cases"].as_array().unwrap();
    assert_fixture_cases(cases);
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let chains: Vec<Vec<f64>> = case["chains"]
            .as_array()
            .unwrap()
            .iter()
            .map(|chain| {
                chain
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_f64().unwrap())
                    .collect()
            })
            .collect();
        let before: Vec<_> = chains.iter().flatten().map(|x| x.to_bits()).collect();
        let borrowed: Vec<_> = chains.iter().map(Vec::as_slice).collect();
        let report = CombinedRhat::estimate(&borrowed)
            .unwrap_or_else(|error| panic!("{name}: input rejected: {error:?}"));
        let location = report
            .rank_normalized()
            .unwrap_or_else(|error| panic!("{name}: location unavailable: {error:?}"));
        assert_reference(
            location.value(),
            case["location"].as_f64().unwrap(),
            tolerance,
            name,
            "location",
        );
        assert_eq!(
            location,
            RankNormalizedSplitRhat::estimate(&borrowed).unwrap()
        );
        assert_eq!(report.chain_count(), chains.len());
        assert_eq!(report.samples_per_chain(), chains[0].len());
        assert_eq!(report.samples_per_split_chain(), chains[0].len() / 2);
        assert_eq!(
            report.folded(),
            FoldedRankNormalizedSplitRhat::estimate(&borrowed),
            "{name}"
        );
        if case["unavailable"].as_bool().unwrap_or(false) {
            assert!(matches!(
                report.folded(),
                Err(SplitRhatError::ConstantSplitChain {
                    chain_index: 0,
                    half: 0,
                    ..
                })
            ));
            assert_eq!(report.value(), None);
        } else {
            let folded = report
                .folded()
                .unwrap_or_else(|error| panic!("{name}: folded unavailable: {error:?}"));
            assert_reference(
                folded.value(),
                case["folded"].as_f64().unwrap(),
                tolerance,
                name,
                "folded",
            );
            assert_reference(
                report.value().expect("both components succeeded"),
                case["combined"].as_f64().unwrap(),
                tolerance,
                name,
                "combined",
            );
            assert_eq!(folded.chain_count(), report.chain_count());
            assert_eq!(folded.samples_per_chain(), report.samples_per_chain());
            assert_eq!(
                folded.samples_per_split_chain(),
                report.samples_per_split_chain()
            );
            if case["name"] == "scale_only" {
                assert!(location.value() < 1.0);
                assert!(folded.value() > 1.9);
            }
        }
        let after: Vec<_> = chains.iter().flatten().map(|x| x.to_bits()).collect();
        assert_eq!(before, after);
    }
}

#[test]
fn binary_observables_can_have_both_components_available() {
    // The pooled median is zero, so folding preserves the binary observations.
    // All half means agree and n=4: both variance ratios equal sqrt(3/4).
    let a = [-0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0];
    let b = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -0.0];
    let report = CombinedRhat::estimate(&[&a, &b]).unwrap();
    assert_relative_eq!(
        report.rank_normalized().unwrap().value(),
        0.75_f64.sqrt(),
        epsilon = 1e-14
    );
    assert_relative_eq!(
        report.folded().unwrap().value(),
        0.75_f64.sqrt(),
        epsilon = 1e-14
    );
    assert_relative_eq!(report.value().unwrap(), 0.75_f64.sqrt(), epsilon = 1e-14);
}

#[test]
fn invalid_inputs_are_separate_from_unavailable_components() {
    let good = [0.0, 1.0, 2.0, 3.0];
    assert_invalid_input(
        &[],
        |error| matches!(error, SplitRhatError::InsufficientChains { count: 0, .. }),
        "split R-hat needs at least two original chains, got 0",
    );
    assert_invalid_input(
        &[&good],
        |error| matches!(error, SplitRhatError::InsufficientChains { count: 1, .. }),
        "split R-hat needs at least two original chains, got 1",
    );
    assert_invalid_input(
        &[&good, &[0.0; 3]],
        |error| {
            matches!(
                error,
                SplitRhatError::InsufficientSamples {
                    chain_index: 1,
                    count: 3,
                    ..
                }
            )
        },
        "chain 1 needs at least four samples, got 3",
    );
    assert_invalid_input(
        &[&good, &[0.0; 5]],
        |error| {
            matches!(
                error,
                SplitRhatError::UnequalLengths {
                    chain_index: 1,
                    expected: 4,
                    actual: 5,
                    ..
                }
            )
        },
        "chain 1 has 5 samples; expected 4",
    );
    // Shape errors precede a nonfinite draw in an earlier chain.
    assert_invalid_input(
        &[&[f64::NAN; 4], &[0.0; 3]],
        |error| {
            matches!(
                error,
                SplitRhatError::InsufficientSamples {
                    chain_index: 1,
                    count: 3,
                    ..
                }
            )
        },
        "chain 1 needs at least four samples, got 3",
    );
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let middle = [0.0, 1.0, bad, 2.0, 3.0];
        assert_invalid_input(
            &[&[0.0; 5], &middle],
            |error| {
                matches!(
                    error,
                    SplitRhatError::NonFiniteSample {
                        chain_index: 1,
                        sample_index: 2,
                        ..
                    }
                )
            },
            "chain 1 sample 2 is not finite",
        );
    }
    let constant = CombinedRhat::estimate(&[&[1.0; 4], &[1.0; 4]]).unwrap();
    assert!(matches!(
        constant.rank_normalized(),
        Err(SplitRhatError::ConstantSplitChain {
            chain_index: 0,
            half: 0,
            ..
        })
    ));
    assert!(matches!(
        constant.folded(),
        Err(SplitRhatError::ConstantSplitChain {
            chain_index: 0,
            half: 0,
            ..
        })
    ));
    assert_eq!(constant.value(), None);
    assert_eq!(constant.samples_per_chain(), 4);
}

#[test]
fn overflow_safe_median_and_deviations_preserve_scale_equivalence() {
    // Exact power-of-two scaling preserves deviations and ties. Median minus
    // the negative extreme exceeds MAX, requiring common halving.
    let a = [-1.75, 0.5, 0.75, 1.5, -0.5, 0.75, 1.5, 0.5];
    let b = [1.5, 0.75, 0.5, -1.75, 0.75, 0.5, -0.5, 1.5];
    let scale = f64::from_bits(0x7fe0_0000_0000_0000);
    let large_a = a.map(|x| x * scale);
    let large_b = b.map(|x| x * scale);
    let expected = CombinedRhat::estimate(&[&a, &b]).unwrap();
    let actual = CombinedRhat::estimate(&[&large_a, &large_b]).unwrap();
    assert_relative_eq!(
        actual.value().unwrap(),
        expected.value().unwrap(),
        epsilon = 1e-14
    );
    assert_relative_eq!(
        actual.folded().unwrap().value(),
        expected.folded().unwrap().value(),
        epsilon = 1e-14
    );
    let near_max = [0.25, 0.5, 0.75, 1.0, 0.5, 0.75, 1.0, 0.25].map(|x| x * f64::MAX);
    // Every half is a permutation of the same multiset: B=0, n=4.
    let near_max = CombinedRhat::estimate(&[&near_max, &near_max]).unwrap();
    assert_relative_eq!(
        near_max.rank_normalized().unwrap().value(),
        0.75_f64.sqrt(),
        epsilon = 1e-14
    );
    assert_relative_eq!(
        near_max.folded().unwrap().value(),
        0.75_f64.sqrt(),
        epsilon = 1e-14
    );
}

#[test]
fn subnormal_folding_and_unresolved_rescaling_are_explicit() {
    let a = [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0];
    let small = a.map(|x| x * f64::from_bits(2));
    let actual = CombinedRhat::estimate(&[&small, &small]).unwrap();
    // Identical half multisets give B=0, so both components equal sqrt(3/4).
    assert_relative_eq!(
        actual.rank_normalized().unwrap().value(),
        0.75_f64.sqrt(),
        epsilon = 1e-14
    );
    assert_relative_eq!(
        actual.folded().unwrap().value(),
        0.75_f64.sqrt(),
        epsilon = 1e-14
    );
    let mixed = [
        -f64::MAX,
        f64::from_bits(1),
        f64::MAX,
        f64::MAX,
        0.5 * f64::MAX,
        f64::MAX,
        0.75 * f64::MAX,
        f64::MAX,
    ];
    let report = CombinedRhat::estimate(&[&mixed, &mixed]).unwrap();
    assert!(report.rank_normalized().is_ok());
    assert_eq!(
        report.folded().unwrap_err(),
        SplitRhatError::UnresolvedFolding
    );
    assert_eq!(report.value(), None);
    assert_eq!(
        FoldedRankNormalizedSplitRhat::estimate(&[&mixed, &mixed]),
        report.folded()
    );
    assert!(
        report
            .folded()
            .unwrap_err()
            .to_string()
            .contains("subnormal")
    );
}

#[test]
fn unavailable_components_preserve_every_original_chain_and_half_index() {
    for chain_index in 0..3 {
        for half in 0..2 {
            for folded_only in [false, true] {
                let mut chains = vec![vec![-2.0, 2.0, -3.0, 3.0, -2.0, 2.0, -3.0, 3.0]; 3];
                let replacement = if folded_only {
                    [-1.0, 1.0, -1.0, 1.0]
                } else {
                    [0.0; 4]
                };
                chains[chain_index][half * 4..(half + 1) * 4].copy_from_slice(&replacement);
                let borrowed: Vec<_> = chains.iter().map(Vec::as_slice).collect();
                let report = CombinedRhat::estimate(&borrowed).unwrap();
                let error = report.folded().unwrap_err();
                assert!(
                    matches!(error, SplitRhatError::ConstantSplitChain {
                    chain_index: actual_chain, half: actual_half, ..
                } if actual_chain == chain_index && actual_half == half),
                    "chain={chain_index}, half={half}, folded_only={folded_only}: {error:?}"
                );
                assert_eq!(
                    FoldedRankNormalizedSplitRhat::estimate(&borrowed).unwrap_err(),
                    error
                );
                assert_eq!(
                    error.to_string(),
                    format!("chain {chain_index} half {half} is constant")
                );
                if folded_only {
                    // Each half is symmetric around zero: ranked half means agree.
                    assert_relative_eq!(
                        report.rank_normalized().unwrap().value(),
                        0.75_f64.sqrt(),
                        epsilon = 1e-14
                    );
                } else {
                    assert_eq!(report.rank_normalized().unwrap_err(), error);
                }
                assert_eq!(report.value(), None);
                assert_eq!(report.chain_count(), 3);
                assert_eq!(report.samples_per_chain(), 8);
                assert_eq!(report.samples_per_split_chain(), 4);
            }
        }
    }
}

#[test]
fn odd_middle_draws_affect_folding_but_not_location() {
    let a = [-3.0, 0.0, 100.0, 1.0, 4.0];
    let b = [-2.0, 1.0, 200.0, 3.0, 8.0];
    let mut changed_a = a;
    let mut changed_b = b;
    changed_a[2] = -100.0;
    changed_b[2] = -200.0;
    let original = CombinedRhat::estimate(&[&a, &b]).unwrap();
    let changed = CombinedRhat::estimate(&[&changed_a, &changed_b]).unwrap();
    assert_eq!(original.rank_normalized(), changed.rank_normalized());
    assert!((original.folded().unwrap().value() - changed.folded().unwrap().value()).abs() > 0.01);
}
