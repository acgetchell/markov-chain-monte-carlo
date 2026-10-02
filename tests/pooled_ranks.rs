//! Plot ranks preserve original chain/draw identity and share diagnostic tie rules.

use markov_chain_monte_carlo::{
    EssEstimate, EssEstimator, MeanMcse, MonteCarloError, PooledRankError, PooledRanks,
    QuantileMcse, TailEss,
};
use serde_json::Value;

fn check_estimate(prefix: &Value, name: &str, actual: Result<f64, MonteCarloError>) {
    if let Some(reason) = prefix["unavailable"][name].as_str() {
        assert_eq!(format!("{:?}", actual.unwrap_err()), reason);
    } else {
        let expected = prefix["estimates"][name].as_f64().unwrap();
        let value = actual.unwrap();
        assert!(value.is_finite());
        assert!(
            (value - expected).abs() <= 2e-11_f64.mul_add(expected.abs(), 2e-13),
            "{name}: {value} != {expected}"
        );
    }
}

#[test]
fn reranked_prefixes_match_pinned_scipy_and_arviz_references() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/diagnostic_plots.json")).unwrap();
    let inputs: Value = serde_json::from_str(include_str!("fixtures/ess.json")).unwrap();
    assert_eq!(fixture["reference"]["scipy"], "1.16.2");
    assert_eq!(fixture["reference"]["arviz"], "0.22.0");
    assert_eq!(fixture["reference"]["relative_tolerance"], 2e-11);
    assert_eq!(fixture["reference"]["absolute_tolerance"], 2e-13);
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(
        cases
            .iter()
            .map(|case| case["name"].as_str().unwrap())
            .collect::<Vec<_>>(),
        [
            "independent",
            "positive_correlation",
            "location_disagreement",
            "scale_disagreement",
            "counts",
        ]
    );
    for case in cases {
        let input = inputs["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|input| input["name"] == case["name"])
            .unwrap();
        let draws: Vec<Vec<f64>> = serde_json::from_value(input["chains"].clone()).unwrap();
        let prefixes = case["prefixes"].as_array().unwrap();
        assert_eq!(
            prefixes
                .iter()
                .map(|prefix| prefix["draws_per_chain"].as_u64().unwrap())
                .collect::<Vec<_>>(),
            [7, 16, 31]
        );
        for prefix in prefixes {
            let length = usize::try_from(prefix["draws_per_chain"].as_u64().unwrap()).unwrap();
            let chains: Vec<_> = draws.iter().map(|draws| &draws[..length]).collect();
            let ranks = PooledRanks::from_chains(&chains).unwrap();
            let expected: Vec<Vec<f64>> = serde_json::from_value(prefix["ranks"].clone()).unwrap();
            for (index, expected) in expected.iter().enumerate() {
                assert_eq!(ranks.chain(index).unwrap(), expected);
            }
            check_estimate(
                prefix,
                "mean_ess",
                EssEstimate::estimate(&chains, EssEstimator::Mean).map(EssEstimate::value),
            );
            check_estimate(
                prefix,
                "bulk_ess",
                EssEstimate::estimate(&chains, EssEstimator::Bulk).map(EssEstimate::value),
            );
            check_estimate(
                prefix,
                "mean_mcse",
                MeanMcse::estimate(&chains).map(MeanMcse::value),
            );
            for (label, probability) in [("q05", 0.05), ("q50", 0.5), ("q95", 0.95)] {
                check_estimate(
                    prefix,
                    &format!("{label}_ess"),
                    EssEstimate::estimate(&chains, EssEstimator::Quantile(probability))
                        .map(EssEstimate::value),
                );
                let mcse = QuantileMcse::estimate(&chains, probability);
                check_estimate(
                    prefix,
                    &format!("{label}_mcse"),
                    mcse.map(QuantileMcse::value),
                );
                if let Ok(mcse) = mcse {
                    check_estimate(prefix, &format!("{label}_value"), Ok(mcse.quantile()));
                }
            }
            let tail = TailEss::estimate(&chains).unwrap();
            if tail.lower().is_ok() && tail.upper().is_ok() {
                check_estimate(prefix, "tail_ess", Ok(tail.value().unwrap()));
            } else {
                assert_eq!(tail.value(), None);
            }
        }
    }
}

#[test]
fn ranks_preserve_order_ties_signed_zeros_and_input_bits() {
    let first = [3., -0., 1., 3., 0.];
    let second = [2., 1., -1.];
    let before = [first.as_slice(), second.as_slice()].map(|chain| {
        chain
            .iter()
            .map(|value| f64::to_bits(*value))
            .collect::<Vec<_>>()
    });
    let ranks = PooledRanks::from_chains(&[&first, &second]).unwrap();
    assert_eq!(ranks.chain_count(), 2);
    assert_eq!(ranks.sample_count(), 8);
    assert_eq!(ranks.chain(0).unwrap(), [7.5, 2.5, 4.5, 7.5, 2.5]);
    assert_eq!(ranks.chain(1).unwrap(), [6., 4.5, 1.]);
    assert_eq!(ranks.chain(2), None);
    assert_eq!(ranks.chain(usize::MAX), None);
    assert_eq!(
        before,
        [first.as_slice(), second.as_slice()].map(|chain| chain
            .iter()
            .map(|value| f64::to_bits(*value))
            .collect::<Vec<_>>())
    );
}

#[test]
fn original_odd_middle_draws_are_ranked_and_prefixes_are_reranked() {
    let first = [0., 1., 100., 2., 3.];
    let second = [4., 5., -100., 6., 7.];
    let full = PooledRanks::from_chains(&[&first, &second]).unwrap();
    assert_eq!(full.sample_count(), 10);
    assert_eq!(full.chain(0).unwrap(), [2., 3., 10., 4., 5.]);
    assert_eq!(full.chain(1).unwrap(), [6., 7., 1., 8., 9.]);
    let prefix = PooledRanks::from_chains(&[&first[..2], &second[..2]]).unwrap();
    assert_eq!(prefix.chain(0).unwrap(), [1., 2.]);
    assert_eq!(prefix.chain(1).unwrap(), [3., 4.]);
    assert_ne!(prefix.chain(0).unwrap(), &full.chain(0).unwrap()[..2]);
}

#[test]
fn constants_single_draws_and_extreme_finite_values_remain_plot_data() {
    let constant = PooledRanks::from_chains(&[&[7.; 3], &[7.; 2]]).unwrap();
    assert_eq!(constant.chain(0).unwrap(), [3.; 3]);
    assert_eq!(constant.chain(1).unwrap(), [3.; 2]);
    let singleton = PooledRanks::from_chains(&[&[f64::MAX]]).unwrap();
    assert_eq!(singleton.chain(0).unwrap(), [1.]);
    let extremes =
        PooledRanks::from_chains(&[&[f64::MAX, -f64::MAX, f64::from_bits(1), 0.]]).unwrap();
    assert_eq!(extremes.chain(0).unwrap(), [4., 1., 3., 2.]);
}

#[test]
fn malformed_plot_inputs_have_original_chain_and_draw_locations() {
    let missing = PooledRanks::from_chains(&[]).unwrap_err();
    assert_eq!(missing, PooledRankError::NoChains);
    assert_eq!(
        missing.to_string(),
        "pooled ranks require at least one chain"
    );

    // Shape errors take precedence over nonfinite values, and identify the
    // first empty original chain even when more than one is malformed.
    let empty = PooledRanks::from_chains(&[&[f64::NAN], &[], &[]]).unwrap_err();
    assert!(matches!(
        empty,
        PooledRankError::EmptyChain { chain_index: 1, .. }
    ));
    assert_eq!(empty.to_string(), "chain 1 has no draws to rank");

    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        // Distinct chain/draw indices detect swapped coordinates. Later bad
        // samples must not replace the first failure in original input order.
        let error = PooledRanks::from_chains(&[
            &[20., 10.],
            &[4., 3., 2., bad, f64::NEG_INFINITY],
            &[f64::NAN],
        ])
        .unwrap_err();
        assert!(matches!(
            error,
            PooledRankError::NonFiniteSample {
                chain_index: 1,
                sample_index: 3,
                ..
            }
        ));
        assert_eq!(error.to_string(), "chain 1 sample 3 is not finite");
    }
}

#[test]
fn root_and_prelude_types_match_and_ranks_outlive_input_storage() {
    use markov_chain_monte_carlo::prelude;

    let ranks: prelude::PooledRanks = {
        let values = vec![3., 1., 2.];
        PooledRanks::from_chains(&[&values]).unwrap()
    };
    assert_eq!(ranks.chain(0).unwrap(), [3., 1., 2.]);
    let error: prelude::PooledRankError = PooledRankError::NoChains;
    assert_eq!(PooledRanks::from_chains(&[]), Err(error));
}
