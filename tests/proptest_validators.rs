//! Property tests for public validator constructors.

use markov_chain_monte_carlo::prelude::testing::{
    DetailedBalanceConfig, DetailedBalanceError, DiscreteProposalRatio, DiscreteProposalRatioError,
};
use markov_chain_monte_carlo::prelude::{BinningAnalysis, OnlineStats, StatisticsError};
use proptest::prelude::*;

proptest! {
    #[test]
    fn discrete_proposal_ratio_accepts_only_valid_constructor_inputs(
        forward_weight in any::<f64>(),
        forward_site_count in any::<usize>(),
        reverse_weight in any::<f64>(),
        reverse_site_count in any::<usize>(),
    ) {
        let ratio = DiscreteProposalRatio::new(
            forward_weight,
            forward_site_count,
            reverse_weight,
            reverse_site_count,
        );
        let should_accept = forward_weight.is_finite()
            && forward_weight > 0.0
            && reverse_weight.is_finite()
            && reverse_weight >= 0.0
            && forward_site_count > 0;

        prop_assert_eq!(ratio.is_ok(), should_accept);

        if let Ok(ratio) = ratio {
            let log_q_ratio = ratio.log_q_ratio();
            if reverse_weight == 0.0 || reverse_site_count == 0 {
                prop_assert!(log_q_ratio.is_infinite());
                prop_assert!(log_q_ratio.is_sign_negative());
            } else {
                prop_assert!(log_q_ratio.is_finite());
            }
        }
    }

    #[test]
    fn detailed_balance_config_accepts_only_valid_constructor_inputs(
        samples in any::<usize>(),
        tolerance in any::<f64>(),
        min_hits in any::<usize>(),
    ) {
        let config = DetailedBalanceConfig::new(samples, tolerance, min_hits);
        let should_accept = samples > 0
            && tolerance.is_finite()
            && tolerance >= 0.0
            && min_hits > 0
            && min_hits <= samples;

        prop_assert_eq!(config.is_ok(), should_accept);

        if let Ok(config) = config {
            prop_assert_eq!(config.samples(), samples);
            prop_assert_eq!(config.tolerance().to_bits(), tolerance.to_bits());
            prop_assert_eq!(config.min_hits(), min_hits);
        }
    }

    #[test]
    fn online_stats_accepts_only_finite_single_samples(sample in any::<f64>()) {
        let mut stats = OnlineStats::new();
        let result = stats.try_push(sample);

        if sample.is_finite() {
            prop_assert_eq!(result, Ok(()));
            prop_assert_eq!(stats.count(), 1);
            prop_assert_eq!(stats.mean(), Some(sample));
        } else {
            let expected = if sample.is_nan() {
                StatisticsError::NanSample
            } else {
                StatisticsError::InfiniteSample
            };
            prop_assert_eq!(result, Err(expected));
            prop_assert!(stats.is_empty());
        }
    }

    #[test]
    fn binning_analysis_accepts_only_finite_single_samples(sample in any::<f64>()) {
        let mut bins = BinningAnalysis::new();
        let result = bins.try_push(sample);

        if sample.is_finite() {
            prop_assert_eq!(result, Ok(()));
            prop_assert_eq!(bins.count(), 1);
            prop_assert_eq!(bins.mean(), Some(sample));
            prop_assert_eq!(bins.estimates().count(), 1);
        } else {
            let expected = if sample.is_nan() {
                StatisticsError::NanSample
            } else {
                StatisticsError::InfiniteSample
            };
            prop_assert_eq!(result, Err(expected));
            prop_assert!(bins.is_empty());
        }
    }
}

#[test]
fn validators_reject_antagonistic_float_values() {
    for value in [f64::NEG_INFINITY, -0.0, f64::NAN, f64::INFINITY] {
        let ratio = DiscreteProposalRatio::new(value, 1, 1.0, 1);
        assert!(matches!(
            ratio,
            Err(DiscreteProposalRatioError::InvalidForwardWeight { .. })
        ));
    }

    for value in [f64::NEG_INFINITY, -1.0, f64::NAN, f64::INFINITY] {
        let config = DetailedBalanceConfig::new(1, value, 1);
        assert!(matches!(
            config,
            Err(DetailedBalanceError::InvalidTolerance { .. })
        ));
    }
}
