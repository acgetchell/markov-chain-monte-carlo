//! Metamorphic ESS/MCSE contracts, independent of autocovariance implementation.

use markov_chain_monte_carlo::{
    EssEstimate, EssEstimator, MeanMcse, MonteCarloError, QuantileMcse,
};
use proptest::prelude::*;

fn varying_chains() -> impl Strategy<Value = Vec<Vec<f64>>> {
    (2usize..=4, 2usize..=16)
        .prop_flat_map(|(chains, half)| {
            prop::collection::vec(prop::collection::vec(-16i16..=16, half), 2 * chains)
        })
        .prop_map(|mut halves| {
            // Admit data by construction, rather than filtering on an estimator.
            for half in &mut halves {
                half[1] = half[0] + 1;
            }
            halves
                .as_chunks::<2>()
                .0
                .iter()
                .map(|[a, b]| a.iter().chain(b).map(|&x| f64::from(x)).collect())
                .collect()
        })
}

proptest! {
    #[test]
    fn affine_units_and_chain_time_reordering_preserve_information(chains in varying_chains()) {
        let borrowed: Vec<_> = chains.iter().map(Vec::as_slice).collect();
        // These transforms are exact for the small integer inputs. Reverse
        // selected chains, preserving cadence without flattening boundaries.
        let transformed: Vec<Vec<_>> = chains.iter().rev().enumerate().map(|(index, chain)| {
            let mut values: Vec<_> = chain.iter().map(|&x| 16.0_f64.mul_add(x, 512.0)).collect();
            if index % 2 == 0 { values.reverse(); }
            values
        }).collect();
        let transformed: Vec<_> = transformed.iter().map(Vec::as_slice).collect();
        for method in [EssEstimator::Mean, EssEstimator::Bulk, EssEstimator::Quantile(0.5)] {
            let original = EssEstimate::estimate(&borrowed, method);
            let changed = EssEstimate::estimate(&transformed, method);
            match (original, changed) {
                (Ok(a), Ok(b)) => prop_assert!((a.value() / b.value() - 1.0).abs() < 2e-10),
                (Err(a), Err(b)) => {
                    prop_assert!(matches!(method, EssEstimator::Quantile(_)),
                        "varying halves must support {method:?}: {a:?}, {b:?}");
                    prop_assert!(matches!(a,
                        MonteCarloError::DegenerateIndicator | MonteCarloError::NoWithinChainVariation),
                        "unexpected quantile ESS error: {a:?}");
                    prop_assert_eq!(a, b);
                }
                (a, b) => prop_assert!(false, "availability changed: {:?}, {:?}", a, b),
            }
        }
        let a = MeanMcse::estimate(&borrowed).unwrap();
        let b = MeanMcse::estimate(&transformed).unwrap();
        prop_assert!((b.value() / (16.0 * a.value()) - 1.0).abs() < 2e-10);
        for p in [0.05, 0.5, 0.95] {
            match (QuantileMcse::estimate(&borrowed, p), QuantileMcse::estimate(&transformed, p)) {
                (Ok(a), Ok(b)) => {
                    prop_assert!((b.quantile() - 16.0_f64.mul_add(a.quantile(), 512.0)).abs() < 2e-12);
                    // Bounds are selected integer observations; their affine
                    // units and half-differences must be exactly representable.
                    prop_assert_eq!(b.value().to_bits(), (16.0 * a.value()).to_bits());
                    prop_assert_eq!(b.interval().map(f64::to_bits),
                        a.interval().map(|x| 16.0_f64.mul_add(x, 512.0).to_bits()));
                }
                (Err(a), Err(b)) => {
                    prop_assert!(matches!(a,
                        MonteCarloError::DegenerateIndicator | MonteCarloError::NoWithinChainVariation
                            | MonteCarloError::CollapsedQuantileInterval),
                        "unexpected quantile MCSE error at {p}: {a:?}");
                    prop_assert_eq!(a, b);
                }
                (a, b) => prop_assert!(false, "MCSE availability changed at {}: {:?}, {:?}", p, a, b),
            }
        }
    }

    #[test]
    fn bulk_ess_preserves_ties_under_nonlinear_monotone_transforms(chains in varying_chains()) {
        let borrowed: Vec<_> = chains.iter().map(Vec::as_slice).collect();
        let cubic: Vec<Vec<_>> = chains.iter().map(|chain| {
            chain.iter().map(|x| x.powi(3)).collect()
        }).collect();
        let cubic: Vec<_> = cubic.iter().map(Vec::as_slice).collect();
        let original = EssEstimate::estimate(&borrowed, EssEstimator::Bulk).unwrap();
        let changed = EssEstimate::estimate(&cubic, EssEstimator::Bulk).unwrap();
        prop_assert_eq!(original, changed);
    }
}
