//! Independent analytical, histogram, and seeded continuous-proposal checks.

use std::{convert::Infallible, error::Error};

use approx::assert_relative_eq;
use markov_chain_monte_carlo::{
    Proposal, ProposalBinsError, ProposalDensityError, verify_proposal_bins,
    verify_proposal_density,
};
use rand::{Rng, RngExt, SeedableRng, TryRng, distr::Open01, rngs::StdRng};

#[test]
fn drifted_normal_density_detects_sign_errors_and_omitted_corrections() {
    // For y-x = delta and N(x+drift, sigma^2), expansion of the two
    // squared terms gives the independent ratio -2*drift*delta/sigma^2.
    let (delta, drift, sigma) = (1.25_f64, 0.5_f64, 2.0_f64);
    let normalizer = -(sigma * (2.0 * std::f64::consts::PI).sqrt()).ln();
    let forward = (-0.5_f64).mul_add(((delta - drift) / sigma).powi(2), normalizer);
    let reverse = (-0.5_f64).mul_add(((-delta - drift) / sigma).powi(2), normalizer);
    let expected = -2.0 * drift * delta / sigma.powi(2);
    let report = verify_proposal_density(forward, reverse, expected, 1e-14).unwrap();
    assert_relative_eq!(report.expected_log_ratio(), expected, epsilon = 1e-14);
    assert_relative_eq!(report.residual(), 0.0, epsilon = 1e-14);
    assert_eq!(report.forward_log_density().to_bits(), forward.to_bits());
    assert_eq!(report.reverse_log_density().to_bits(), reverse.to_bits());
    assert_eq!(report.reported_log_ratio().to_bits(), expected.to_bits());
    for wrong in [-expected, 0.0] {
        assert!(matches!(
            verify_proposal_density(forward, reverse, wrong, 1e-14),
            Err(ProposalDensityError::Violation { .. })
        ));
    }
    let reversed = verify_proposal_density(reverse, forward, -expected, 1e-14).unwrap();
    assert_relative_eq!(reversed.expected_log_ratio(), -expected, epsilon = 1e-14);
}

#[test]
fn state_dependent_normalizers_must_be_in_the_ratio() {
    // Uniform q(y|x) on [x-s(x), x+s(x)], s(0)=1 and s(0.5)=2.
    // Both endpoints are in support; densities are 1/2 and 1/4.
    let forward = -2.0_f64.ln();
    let reverse = -4.0_f64.ln();
    let report = verify_proposal_density(forward, reverse, -2.0_f64.ln(), 1e-14).unwrap();
    assert_relative_eq!(report.residual(), 0.0, epsilon = 1e-14);
    assert!(matches!(
        verify_proposal_density(forward, reverse, 0.0, 1e-14),
        Err(ProposalDensityError::Violation { .. })
    ));
    // Densities greater than one and a shared log-normalization offset are valid.
    let shifted =
        verify_proposal_density(forward + 10.0, reverse + 10.0, -2.0_f64.ln(), 1e-14).unwrap();
    assert!(shifted.forward_log_density() > 0.0);
    assert_relative_eq!(
        shifted.expected_log_ratio(),
        report.expected_log_ratio(),
        epsilon = 1e-14
    );
}

#[test]
fn reverse_support_is_checked_without_exponentiating_densities() {
    let report =
        verify_proposal_density(-1000.0, f64::NEG_INFINITY, f64::NEG_INFINITY, 0.0).unwrap();
    assert_eq!(
        report.expected_log_ratio().to_bits(),
        f64::NEG_INFINITY.to_bits()
    );
    assert_eq!(report.residual().to_bits(), 0.0_f64.to_bits());
    for (reverse, reported, residual) in [
        (f64::NEG_INFINITY, 0.0, f64::INFINITY),
        (-1000.0, f64::NEG_INFINITY, f64::NEG_INFINITY),
    ] {
        let Err(ProposalDensityError::Violation { report, .. }) =
            verify_proposal_density(-1000.0, reverse, reported, f64::MAX)
        else {
            panic!("support mismatch must fail even with the largest finite tolerance");
        };
        assert_eq!(report.residual().to_bits(), residual.to_bits());
    }
    let finite = verify_proposal_density(-1000.0, -1001.0, -1.0, 0.0).unwrap();
    assert_eq!(finite.residual().to_bits(), 0.0_f64.to_bits());
}

#[test]
fn density_tolerance_is_absolute_and_inclusive() {
    // Dyadic inputs make the residual exact. Vary its sign and the ratio's
    // scale so a one-sided or relative-tolerance comparison cannot pass.
    for expected in [0.0, -1.0, 1.0, -1024.0, 1024.0] {
        for residual in [-0.25_f64, 0.25] {
            let reported = expected + residual;
            let report =
                verify_proposal_density(0.0, expected, reported, 0.25).unwrap_or_else(|error| {
                    panic!(
                        "inclusive bound failed for ratio {expected}, residual {residual}: {error}"
                    )
                });
            assert_eq!(report.residual().to_bits(), residual.to_bits());
            assert_eq!(report.tolerance().to_bits(), 0.25_f64.to_bits());
            let result = verify_proposal_density(0.0, expected, reported, 0.25_f64.next_down());
            let Err(ProposalDensityError::Violation { report, .. }) = result else {
                panic!("ratio {expected}, residual {residual} must exceed the bound: {result:?}");
            };
            assert_eq!(report.residual().to_bits(), residual.to_bits());
            assert_eq!(report.tolerance().to_bits(), 0.25_f64.next_down().to_bits());
        }
    }
}

#[test]
fn density_reports_retain_tolerances() {
    // A downstream collector can keep reports without a parallel tolerance
    // array, even when it retains both successful and failed comparisons.
    let reports = [0.125, 0.25, 0.5].map(|tolerance| {
        match verify_proposal_density(-2.0, -1.0, 1.25, tolerance) {
            Ok(report) | Err(ProposalDensityError::Violation { report, .. }) => report,
            Err(error) => panic!("valid density comparison failed: {error}"),
        }
    });
    let outcomes = reports
        .each_ref()
        .map(|report| report.residual().abs() <= report.tolerance());
    assert_eq!(outcomes, [false, true, true]);
}

#[test]
fn invalid_density_inputs_preserve_context() {
    for tolerance in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let result = verify_proposal_density(0.0, 0.0, 0.0, tolerance);
        assert!(
            matches!(result,
                Err(ProposalDensityError::InvalidTolerance { tolerance: actual, .. })
                    if actual.to_bits() == tolerance.to_bits()
            ),
            "tolerance {tolerance:?}: {result:?}"
        );
    }
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let result = verify_proposal_density(invalid, 0.0, 0.0, 0.0);
        assert!(
            matches!(result,
                Err(ProposalDensityError::InvalidForwardDensity { log_density, .. })
                    if log_density.to_bits() == invalid.to_bits()
            ),
            "forward density {invalid:?}: {result:?}"
        );
    }
    for invalid in [f64::NAN, f64::INFINITY] {
        let result = verify_proposal_density(0.0, invalid, 0.0, 0.0);
        assert!(
            matches!(result,
                Err(ProposalDensityError::InvalidReverseDensity { log_density, .. })
                    if log_density.to_bits() == invalid.to_bits()
            ),
            "reverse density {invalid:?}: {result:?}"
        );
        let result = verify_proposal_density(0.0, 0.0, invalid, 0.0);
        assert!(
            matches!(result,
                Err(ProposalDensityError::InvalidLogRatio { log_ratio, .. })
                    if log_ratio.to_bits() == invalid.to_bits()
            ),
            "log ratio {invalid:?}: {result:?}"
        );
    }
}

#[test]
fn density_overflows_preserve_distinct_context() {
    for (forward, reverse, reported) in [
        (-f64::MAX, f64::MAX, 0.0),
        (f64::MAX, -f64::MAX, f64::NEG_INFINITY),
    ] {
        let result = verify_proposal_density(forward, reverse, reported, 0.0);
        let Err(ProposalDensityError::ExpectedLogRatioOverflow {
            forward_log_density,
            reverse_log_density,
            ..
        }) = result
        else {
            panic!("expected-ratio overflow must retain its operands: {result:?}");
        };
        assert_eq!(forward_log_density.to_bits(), forward.to_bits());
        assert_eq!(reverse_log_density.to_bits(), reverse.to_bits());
    }
    for expected in [-f64::MAX, f64::MAX] {
        let reported = -expected;
        let result = verify_proposal_density(0.0, expected, reported, 0.0);
        let Err(ProposalDensityError::ResidualOverflow {
            expected_log_ratio,
            reported_log_ratio,
            ..
        }) = result
        else {
            panic!("residual overflow must retain its operands: {result:?}");
        };
        assert_eq!(expected_log_ratio.to_bits(), expected.to_bits());
        assert_eq!(reported_log_ratio.to_bits(), reported.to_bits());
    }
}

#[test]
fn histogram_report_matches_hand_calculated_probabilities_and_bound() {
    let report = verify_proposal_bins(&[260, 740], &[0.25, 0.75], 0.01).unwrap();
    assert_eq!(report.samples(), 1000);
    assert_eq!(report.bins(), 2);
    assert_eq!(report.worst_bin(), 0);
    assert_eq!(report.false_positive_rate().to_bits(), 0.01_f64.to_bits());
    assert_relative_eq!(report.max_residual(), 0.01, epsilon = 1e-15);
    // Independent literal: sqrt(ln(400)/2000).
    assert_relative_eq!(
        report.tolerance(),
        0.054_733_283_051_119_734,
        epsilon = 1e-15
    );
    let scaled = verify_proposal_bins(&[1000, 3000], &[0.25, 0.75], 0.01).unwrap();
    assert_relative_eq!(
        scaled.tolerance(),
        report.tolerance() / 2.0,
        epsilon = 1e-15
    );
    let Err(ProposalBinsError::Violation { report, .. }) =
        verify_proposal_bins(&[500, 500], &[0.25, 0.75], 0.01)
    else {
        panic!("incorrect bin masses must fail");
    };
    assert_eq!(report.max_residual().to_bits(), 0.25_f64.to_bits());
    assert_eq!(report.worst_bin(), 0);
}

#[test]
fn histogram_reports_choose_first_worst_bin() {
    // Counts and probabilities are exact dyadic fractions. The largest
    // discrepancy can be negative, positive, or tied away from bin zero.
    for (counts, probabilities, worst_bin, residual) in [
        ([256, 256, 512], [0.125, 0.125, 0.75], 2, 0.25_f64),
        ([256, 512, 256], [0.375, 0.25, 0.375], 1, 0.25),
        ([256, 384, 384], [0.25, 0.25, 0.5], 1, 0.125),
    ] {
        let result = verify_proposal_bins(&counts, &probabilities, 0.01);
        let Err(ProposalBinsError::Violation { report, .. }) = result else {
            panic!("counts {counts:?}, probabilities {probabilities:?} must fail: {result:?}");
        };
        assert_eq!(report.samples(), 1024);
        assert_eq!(report.bins(), 3);
        assert_eq!(report.worst_bin(), worst_bin, "counts {counts:?}");
        assert_eq!(report.max_residual().to_bits(), residual.to_bits());
        assert_eq!(report.false_positive_rate().to_bits(), 0.01_f64.to_bits());
        // sqrt(ln(600)/2048), independently evaluated with 70-digit Decimal
        // arithmetic. Three bins also detect a hard-coded two-bin bound.
        assert_relative_eq!(
            report.tolerance(),
            0.055_888_288_649_868_396,
            epsilon = 1e-15
        );
    }
}

#[test]
fn histogram_probability_roundoff_is_allowed_without_renormalizing() {
    // 2^-42 is within the documented 1e-12 allowance on either side of one;
    // 2^-39 is outside it. These dyadic sums and residuals are exact in f64.
    for offset in [-2.0_f64.powi(-42), 2.0_f64.powi(-42)] {
        let report = verify_proposal_bins(&[50, 50], &[0.5, 0.5 + offset], 0.01).unwrap();
        assert_eq!(report.worst_bin(), 1);
        assert_eq!(report.max_residual().to_bits(), offset.abs().to_bits());
    }
    for offset in [-2.0_f64.powi(-39), 2.0_f64.powi(-39)] {
        let result = verify_proposal_bins(&[50, 50], &[0.5, 0.5 + offset], 0.01);
        let Err(ProposalBinsError::InvalidProbabilitySum { sum, .. }) = result else {
            panic!("probability sum outside the allowance must fail: {result:?}");
        };
        assert_eq!(sum.to_bits(), (1.0 + offset).to_bits());
    }
}

#[test]
fn histogram_bound_limits_binomial_false_rejections() {
    // Exhaust all outcomes of 16 fair binary draws. Binomial multiplicities
    // provide an exact oracle independent of the concentration-bound formula.
    let mut multiplicity = 1_u32;
    let mut rejected_outcomes = 0_u32;
    for successes in 0_u32..=16 {
        let counts = [successes as usize, (16 - successes) as usize];
        match verify_proposal_bins(&counts, &[0.5, 0.5], 0.2) {
            Ok(_) => {}
            Err(ProposalBinsError::Violation { .. }) => rejected_outcomes += multiplicity,
            Err(other) => panic!("valid binomial experiment failed: {other}"),
        }
        if successes < 16 {
            multiplicity = multiplicity * (16 - successes) / (successes + 1);
        }
    }
    assert!(rejected_outcomes > 0);
    assert!(f64::from(rejected_outcomes) / 65_536.0 <= 0.2);
}

#[test]
fn impossible_bins_fail_even_below_the_statistical_tolerance() {
    assert!(matches!(
        verify_proposal_bins(&[4999, 5000, 1], &[0.5, 0.5, 0.0], 1e-6),
        Err(ProposalBinsError::SupportViolation {
            bin: 2,
            count: 1,
            ..
        })
    ));
    // An impossible observation also takes precedence over a sample budget
    // whose statistical tolerance would otherwise be uninformative.
    assert!(matches!(
        verify_proposal_bins(&[1, 0], &[0.0, 1.0], 0.01),
        Err(ProposalBinsError::SupportViolation {
            bin: 0,
            count: 1,
            ..
        })
    ));
    let rare = verify_proposal_bins(&[0, 10_000], &[1e-8, 1.0 - 1e-8], 1e-6).unwrap();
    assert_eq!(rare.samples(), 10_000);
    assert_relative_eq!(rare.max_residual(), 1e-8, epsilon = 1e-16);
    let impossible = verify_proposal_bins(&[0, 10_000], &[0.0, 1.0], 1e-6).unwrap();
    assert_eq!(impossible.samples(), 10_000);
    assert_eq!(impossible.max_residual().to_bits(), 0.0_f64.to_bits());
}

#[test]
fn histogram_shape_and_probability_errors_preserve_inputs() {
    for (counts, probabilities) in [
        (&[][..], &[][..]),
        (&[10][..], &[1.0][..]),
        (&[10, 10][..], &[1.0][..]),
        (&[10, 10][..], &[0.25, 0.25, 0.5][..]),
    ] {
        let result = verify_proposal_bins(counts, probabilities, 0.01);
        assert!(
            matches!(result,
                Err(ProposalBinsError::InvalidBinCount { counts: actual_counts, probabilities: actual_probabilities, .. })
                    if actual_counts == counts.len() && actual_probabilities == probabilities.len()
            ),
            "counts {counts:?}, probabilities {probabilities:?}: {result:?}"
        );
    }
    for rate in [
        0.0,
        -0.1,
        1.0,
        1.1,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ] {
        let result = verify_proposal_bins(&[50, 50], &[0.5, 0.5], rate);
        assert!(
            matches!(result,
                Err(ProposalBinsError::InvalidFalsePositiveRate { rate: actual, .. })
                    if actual.to_bits() == rate.to_bits()
            ),
            "rate {rate:?}: {result:?}"
        );
    }
    for probability in [-0.1, 1.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for invalid_bin in 0..2 {
            let mut probabilities = [0.5; 2];
            probabilities[invalid_bin] = probability;
            let result = verify_proposal_bins(&[50, 50], &probabilities, 0.01);
            assert!(
                matches!(result,
                    Err(ProposalBinsError::InvalidProbability { bin, probability: actual, .. })
                        if bin == invalid_bin && actual.to_bits() == probability.to_bits()
                ),
                "probability {probability:?} in bin {invalid_bin}: {result:?}"
            );
        }
    }
    for (probabilities, expected_sum) in [
        ([0.0, 0.0], 0.0_f64),
        ([0.25, 0.5], 0.75),
        ([0.75, 0.5], 1.25),
    ] {
        let result = verify_proposal_bins(&[50, 50], &probabilities, 0.01);
        assert!(
            matches!(result,
                Err(ProposalBinsError::InvalidProbabilitySum { sum, .. })
                    if sum.to_bits() == expected_sum.to_bits()
            ),
            "probabilities {probabilities:?}: {result:?}"
        );
    }
}

#[test]
fn histogram_count_errors_preserve_context() {
    assert_eq!(
        verify_proposal_bins(&[0, 0], &[0.5, 0.5], 0.01),
        Err(ProposalBinsError::NoSamples)
    );
    // The mathematical total exceeds the floating-point limit on 64-bit
    // platforms too, but integer overflow has priority and locates the bin.
    let result = verify_proposal_bins(&[2, usize::MAX - 2, 1], &[0.25, 0.5, 0.25], 0.01);
    let Err(ProposalBinsError::SampleCountOverflow {
        bin,
        partial_sum,
        count,
        ..
    }) = result
    else {
        panic!("overflow must locate the failing count addition: {result:?}");
    };
    assert_eq!(bin, 2);
    assert_eq!(partial_sum, usize::MAX);
    assert_eq!(count, 1);
    if let Ok(limit) = usize::try_from(1_u64 << 53) {
        let result = verify_proposal_bins(&[limit, 1], &[0.5, 0.5], 0.01);
        let Err(ProposalBinsError::SampleCountTooLarge {
            samples,
            max_samples,
            ..
        }) = result
        else {
            panic!("precision-limit error must retain the total and limit: {result:?}");
        };
        assert_eq!(samples, limit + 1);
        assert_eq!(max_samples, limit);
        let report = verify_proposal_bins(&[limit / 2, limit / 2], &[0.5, 0.5], 0.01).unwrap();
        assert_eq!(report.samples(), limit);
        assert_eq!(report.max_residual().to_bits(), 0.0_f64.to_bits());
    }
}

#[test]
fn invalid_inputs_precede_arithmetic_failures() {
    // Computing the expected ratio would overflow, but the reported ratio
    // must first satisfy its input contract.
    assert!(matches!(
        verify_proposal_density(-f64::MAX, f64::MAX, f64::NAN, 0.0),
        Err(ProposalDensityError::InvalidLogRatio { log_ratio, .. }) if log_ratio.is_nan()
    ));
    // Count overflow must not hide an invalid probability.
    assert!(matches!(
        verify_proposal_bins(&[usize::MAX, 1], &[0.5, -0.5], 0.01),
        Err(ProposalBinsError::InvalidProbability { bin: 1, probability, .. })
            if probability.to_bits() == (-0.5_f64).to_bits()
    ));
}

#[test]
fn histogram_bound_extremes_preserve_context() {
    let result = verify_proposal_bins(&[1, 0], &[0.5, 0.5], 0.01);
    let Err(ProposalBinsError::UninformativeBound {
        samples, tolerance, ..
    }) = result
    else {
        panic!("one sample must produce an uninformative bound: {result:?}");
    };
    assert_eq!(samples, 1);
    // Independent high-precision value for sqrt(ln(400)/2).
    assert_relative_eq!(tolerance, 1.730_818_382_602_285_4, epsilon = 1e-15);
    // Extremely small alpha remains finite through log-space evaluation.
    let report = verify_proposal_bins(&[5000, 5000], &[0.5, 0.5], f64::from_bits(1)).unwrap();
    // alpha=2^-1074 gives sqrt(1076*ln(2)/20000); clamping alpha to the
    // smallest normal float must not silently weaken this requested bound.
    assert_relative_eq!(
        report.tolerance(),
        0.193_109_601_817_530_17,
        epsilon = 1e-15
    );
    assert_eq!(report.false_positive_rate().to_bits(), 1);
}

struct Increasing;

impl Proposal<f64> for Increasing {
    fn propose<R: Rng + ?Sized>(&self, _: &f64, rng: &mut R) -> f64 {
        rng.sample::<f64, _>(Open01).sqrt()
    }

    fn log_q_ratio(&self, current: &f64, proposed: &f64) -> f64 {
        current.ln() - proposed.ln()
    }
}

fn quarter_bin(sample: f64) -> usize {
    if !(sample > 0.0 && sample < 1.0) {
        4
    } else if sample < 0.25 {
        0
    } else if sample < 0.5 {
        1
    } else if sample < 0.75 {
        2
    } else {
        3
    }
}

/// Force the smallest or largest uniform variate to exercise support boundaries.
struct ConstantByteRng(u8);

impl TryRng for ConstantByteRng {
    type Error = Infallible;

    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        Ok(u32::from_le_bytes([self.0; 4]))
    }

    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        Ok(u64::from_le_bytes([self.0; 8]))
    }

    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Self::Error> {
        dst.fill(self.0);
        Ok(())
    }
}

#[test]
fn proposal_preserves_open_support_at_rng_extremes() {
    for byte in [0, u8::MAX] {
        let y = Increasing.propose(&0.5, &mut ConstantByteRng(byte));
        assert!(y > 0.0 && y < 1.0, "endpoint escaped open support: {y}");
        let ratio = Increasing.log_q_ratio(&0.5, &y);
        assert!(ratio.is_finite());
        let report = verify_proposal_density((2.0 * y).ln(), 0.0, ratio, 1e-12).unwrap();
        assert!(report.residual().abs() <= 1e-12);
    }
    // The impossible-outcome bin must also catch exact support boundaries.
    for invalid in [0.0, 1.0, f64::NAN, f64::NEG_INFINITY, f64::INFINITY] {
        assert_eq!(quarter_bin(invalid), 4);
    }
}

#[test]
fn seeded_density_and_generator_checks() {
    // CDF(y)=y^2: independent quarter-bin masses, including an impossible tail.
    let probabilities = [1.0 / 16.0, 3.0 / 16.0, 5.0 / 16.0, 7.0 / 16.0, 0.0];
    for seed in [7, 42, 2026] {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut counts = [0; 5];
        let mut wrong_counts = [0; 5];
        for _ in 0..20_000 {
            let y = Increasing.propose(&0.5, &mut rng);
            counts[quarter_bin(y)] += 1;
            // A generator accidentally using U instead of sqrt(U).
            wrong_counts[quarter_bin(rng.sample::<f64, _>(Open01))] += 1;
        }
        let report = verify_proposal_bins(&counts, &probabilities, 1e-6)
            .unwrap_or_else(|error| panic!("seed {seed}, counts {counts:?}: {error}"));
        assert_eq!(report.samples(), 20_000);
        let result = verify_proposal_bins(&wrong_counts, &probabilities, 1e-6);
        assert!(
            matches!(result, Err(ProposalBinsError::Violation { .. })),
            "seed {seed}, wrong counts {wrong_counts:?}: {result:?}"
        );
    }
    for x in [0.125_f64, 0.5, 0.875] {
        for y in [0.125_f64, 0.5, 0.875] {
            let report = verify_proposal_density(
                (2.0 * y).ln(),
                (2.0 * x).ln(),
                Increasing.log_q_ratio(&x, &y),
                1e-14,
            )
            .unwrap();
            assert_relative_eq!(report.expected_log_ratio(), (x / y).ln(), epsilon = 1e-14);
        }
    }
}

#[test]
fn diagnostic_errors_expose_actionable_context() {
    for reported in [-1.0_f64, 1.0] {
        let density = verify_proposal_density(0.0, 0.0, reported, 0.125).unwrap_err();
        let message = density.to_string();
        assert!(message.contains("absolute proposal log-ratio residual 1"));
        assert!(message.contains("exceeds tolerance 0.125"));
        assert!(message.contains(&format!("reported {reported}")));
        assert!(message.contains("expected 0"));
        assert!(density.source().is_none());
    }
    let ratio = verify_proposal_density(-f64::MAX, f64::MAX, 0.0, 0.0).unwrap_err();
    let message = ratio.to_string();
    assert!(message.contains("expected log ratio overflowed"));
    assert!(message.contains(&format!("forward log density {}", -f64::MAX)));
    assert!(message.contains(&format!("reverse log density {}", f64::MAX)));
    let residual = verify_proposal_density(0.0, f64::MAX, -f64::MAX, 0.0).unwrap_err();
    let message = residual.to_string();
    assert!(message.contains("residual overflowed"));
    assert!(message.contains(&format!("expected {}", f64::MAX)));
    assert!(message.contains(&format!("reported {}", -f64::MAX)));
    let count = verify_proposal_bins(&[usize::MAX, 1], &[0.5, 0.5], 0.01).unwrap_err();
    let message = count.to_string();
    assert!(message.contains("overflows usize at bin 1"));
    assert!(message.contains(&format!("{} + 1", usize::MAX)));
    let bins = verify_proposal_bins(&[0, 100], &[0.5, 0.5], 0.01).unwrap_err();
    assert!(bins.to_string().contains("bin 0 probability residual 0.5"));
    assert!(bins.source().is_none());
}
