//! Property tests for public validator constructors.

use markov_chain_monte_carlo::{
    DetailedBalanceConfig, DetailedBalanceError, DiscreteProposalRatio, DiscreteProposalRatioError,
};
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
