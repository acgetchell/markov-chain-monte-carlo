//! Test-facing checks for continuous proposal densities and sampled bin masses.

use std::{error::Error, fmt};

use crate::numerics::{compensated_sum, count_as_f64};

/// Evidence from comparing a reported Hastings ratio with independent densities.
///
/// Obtain this report from [`verify_proposal_density`]. A failed comparison
/// also carries a report in [`ProposalDensityError::Violation`], so possession
/// of a report alone does not mean the check passed. Densities, ratios, and
/// residuals use natural-log units. The configured [`Self::tolerance`] travels
/// with the report so retained results keep their original decision threshold.
///
/// # Examples
///
/// Inspect the expected asymmetric correction and its discrepancy:
///
/// ```
/// use markov_chain_monte_carlo::prelude::testing::{
///     ProposalDensityError, verify_proposal_density,
/// };
///
/// let report = verify_proposal_density(-1.0, -2.0, -1.0, 1e-12)?;
/// assert!(report.expected_log_ratio() < 0.0);
/// assert!(report.residual().abs() <= report.tolerance());
/// # Ok::<(), ProposalDensityError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct ProposalDensityReport {
    forward_log_density: f64,
    reverse_log_density: f64,
    reported_log_ratio: f64,
    expected_log_ratio: f64,
    residual: f64,
    tolerance: f64,
}

impl ProposalDensityReport {
    /// Supplied `log q(proposed | current)`.
    #[must_use]
    pub const fn forward_log_density(&self) -> f64 {
        self.forward_log_density
    }

    /// Supplied `log q(current | proposed)`; may be negative infinity.
    #[must_use]
    pub const fn reverse_log_density(&self) -> f64 {
        self.reverse_log_density
    }

    /// Natural-log ratio reported for this concrete move; may be negative infinity.
    #[must_use]
    pub const fn reported_log_ratio(&self) -> f64 {
        self.reported_log_ratio
    }

    /// Independent reverse-minus-forward log density; may be negative infinity.
    #[must_use]
    pub const fn expected_log_ratio(&self) -> f64 {
        self.expected_log_ratio
    }

    /// Reported minus expected ratio, or zero when both are negative infinity.
    ///
    /// A mismatch in reverse support gives a signed infinite residual.
    #[must_use]
    pub const fn residual(&self) -> f64 {
        self.residual
    }

    /// Configured absolute tolerance in natural-log units, finite and nonnegative.
    ///
    /// This is the threshold used for the original verification, including
    /// reports retained in [`ProposalDensityError::Violation`].
    #[must_use]
    pub const fn tolerance(&self) -> f64 {
        self.tolerance
    }
}

/// Invalid inputs, unresolved arithmetic, or a proposal-density mismatch.
///
/// Returned by [`verify_proposal_density`]. [`Self::Violation`] includes a
/// [`ProposalDensityReport`] for inspecting a failed comparison; other variants
/// indicate that the inputs or arithmetic could not support that comparison.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum ProposalDensityError {
    /// The absolute log-ratio tolerance must be finite and nonnegative.
    #[non_exhaustive]
    InvalidTolerance {
        /// Supplied tolerance.
        tolerance: f64,
    },
    /// A successful forward move must have a finite log density.
    #[non_exhaustive]
    InvalidForwardDensity {
        /// Supplied forward log density.
        log_density: f64,
    },
    /// Reverse log density must be finite or negative infinity (zero support).
    #[non_exhaustive]
    InvalidReverseDensity {
        /// Supplied reverse log density.
        log_density: f64,
    },
    /// A reported log ratio must be finite or negative infinity.
    #[non_exhaustive]
    InvalidLogRatio {
        /// Supplied proposal log ratio.
        log_ratio: f64,
    },
    /// Subtracting finite forward density from finite reverse density overflowed.
    #[non_exhaustive]
    ExpectedLogRatioOverflow {
        /// Supplied finite forward log density.
        forward_log_density: f64,
        /// Supplied finite reverse log density.
        reverse_log_density: f64,
    },
    /// The expected and reported ratios are finite, but their difference overflowed.
    #[non_exhaustive]
    ResidualOverflow {
        /// Finite reverse-minus-forward log density.
        expected_log_ratio: f64,
        /// Supplied finite proposal log ratio.
        reported_log_ratio: f64,
    },
    /// Independent densities disagree with the reported ratio.
    #[non_exhaustive]
    Violation {
        /// Evaluated densities, ratios, residual, and requested tolerance.
        report: ProposalDensityReport,
    },
}

impl fmt::Display for ProposalDensityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTolerance { tolerance } => {
                write!(
                    f,
                    "density tolerance must be finite and nonnegative, got {tolerance}"
                )
            }
            Self::InvalidForwardDensity { log_density } => {
                write!(f, "forward log density must be finite, got {log_density}")
            }
            Self::InvalidReverseDensity { log_density } => write!(
                f,
                "reverse log density must be finite or -infinity, got {log_density}"
            ),
            Self::InvalidLogRatio { log_ratio } => write!(
                f,
                "proposal log ratio must be finite or -infinity, got {log_ratio}"
            ),
            Self::ExpectedLogRatioOverflow {
                forward_log_density,
                reverse_log_density,
            } => write!(
                f,
                "expected log ratio overflowed subtracting forward log density {forward_log_density} from reverse log density {reverse_log_density}"
            ),
            Self::ResidualOverflow {
                expected_log_ratio,
                reported_log_ratio,
            } => write!(
                f,
                "log-ratio residual overflowed subtracting expected {expected_log_ratio} from reported {reported_log_ratio}"
            ),
            Self::Violation { report } => write!(
                f,
                "absolute proposal log-ratio residual {} exceeds tolerance {} (reported {}, expected {})",
                report.residual.abs(),
                report.tolerance,
                report.reported_log_ratio,
                report.expected_log_ratio
            ),
        }
    }
}

impl Error for ProposalDensityError {}

/// Check a concrete proposal's Hastings ratio against independent log densities.
///
/// Supply `log q(y | x)`, `log q(x | y)`, and the proposal's reported
/// `log_q_ratio` for the same move `x -> y`. The expected ratio is reverse minus
/// forward. Densities must use the same reference measure, including any
/// Jacobians and state-dependent normalizers. Positive log densities are valid.
/// A zero reverse density is represented by negative infinity; a zero forward
/// density cannot describe a successful forward move and is rejected.
///
/// This deterministic check needs neither endpoint equality nor sampling, and
/// accepts ratios from [`crate::Proposal::log_q_ratio`],
/// [`crate::ProposalMut::log_q_ratio`], or
/// [`crate::DelayedProposal::log_q_ratio`]. Callers own evaluation of the endpoints and
/// rollback of hypothetical in-place moves, including on diagnostic failure.
/// Use an independently derived density: reusing the production ratio formula
/// is circular evidence. This does not test the generator or prove balance,
/// normalization, convergence, or mixing. Pair it with [`verify_proposal_bins`].
///
/// # Errors
///
/// Returns:
///
/// - [`ProposalDensityError::InvalidTolerance`] if `tolerance` is negative,
///   NaN, or infinite. Tolerance is absolute in natural-log units.
/// - [`ProposalDensityError::InvalidForwardDensity`] if `forward_log_density`
///   is not finite, including negative infinity for an impossible forward move.
/// - [`ProposalDensityError::InvalidReverseDensity`] if `reverse_log_density`
///   is NaN or positive infinity. Negative infinity represents zero reverse density.
/// - [`ProposalDensityError::InvalidLogRatio`] if `reported_log_ratio` is NaN
///   or positive infinity. A finite value or negative infinity is accepted.
/// - [`ProposalDensityError::ExpectedLogRatioOverflow`] if subtracting finite
///   forward log density from finite reverse log density overflows.
/// - [`ProposalDensityError::ResidualOverflow`] if the expected and reported
///   ratios are finite but computing `reported - expected` overflows.
/// - [`ProposalDensityError::Violation`] if `abs(reported - expected)` exceeds
///   `tolerance`, including any reverse-support mismatch. Matching negative
///   infinities have residual zero and pass.
///
/// Inputs are validated in the order above before arithmetic. Ratio overflow
/// takes precedence over residual overflow; neither produces a partial report.
///
/// # Examples
///
/// An independence proposal with density `q(y | x) = 2*y` on `(0, 1)`:
///
/// ```
/// use markov_chain_monte_carlo::prelude::testing::{
///     Proposal, ProposalDensityError, verify_proposal_density,
/// };
/// use rand::{Rng, RngExt, distr::Open01};
///
/// struct Increasing;
/// impl Proposal<f64> for Increasing {
///     fn propose<R: Rng + ?Sized>(&self, _: &f64, rng: &mut R) -> f64 {
///         rng.sample::<f64, _>(Open01).sqrt()
///     }
///     fn log_q_ratio(&self, current: &f64, proposed: &f64) -> f64 {
///         current.ln() - proposed.ln()
///     }
/// }
/// let (x, y) = (0.25_f64, 0.75_f64);
/// let report = verify_proposal_density(
///     (2.0 * y).ln(), (2.0 * x).ln(), Increasing.log_q_ratio(&x, &y), 1e-12,
/// )?;
/// assert!(report.residual().abs() < 1e-12);
/// # Ok::<(), ProposalDensityError>(())
/// ```
pub fn verify_proposal_density(
    forward_log_density: f64,
    reverse_log_density: f64,
    reported_log_ratio: f64,
    tolerance: f64,
) -> Result<ProposalDensityReport, ProposalDensityError> {
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(ProposalDensityError::InvalidTolerance { tolerance });
    }
    if !forward_log_density.is_finite() {
        return Err(ProposalDensityError::InvalidForwardDensity {
            log_density: forward_log_density,
        });
    }
    if reverse_log_density.is_nan() || reverse_log_density == f64::INFINITY {
        return Err(ProposalDensityError::InvalidReverseDensity {
            log_density: reverse_log_density,
        });
    }
    if reported_log_ratio.is_nan() || reported_log_ratio == f64::INFINITY {
        return Err(ProposalDensityError::InvalidLogRatio {
            log_ratio: reported_log_ratio,
        });
    }
    let expected_log_ratio = reverse_log_density - forward_log_density;
    if reverse_log_density.is_finite() && !expected_log_ratio.is_finite() {
        return Err(ProposalDensityError::ExpectedLogRatioOverflow {
            forward_log_density,
            reverse_log_density,
        });
    }
    let residual =
        if expected_log_ratio == f64::NEG_INFINITY && reported_log_ratio == f64::NEG_INFINITY {
            0.0
        } else {
            reported_log_ratio - expected_log_ratio
        };
    if expected_log_ratio.is_finite() && reported_log_ratio.is_finite() && !residual.is_finite() {
        return Err(ProposalDensityError::ResidualOverflow {
            expected_log_ratio,
            reported_log_ratio,
        });
    }
    let report = ProposalDensityReport {
        forward_log_density,
        reverse_log_density,
        reported_log_ratio,
        expected_log_ratio,
        residual,
        tolerance,
    };
    if residual.abs() > tolerance {
        return Err(ProposalDensityError::Violation { report });
    }
    Ok(report)
}

/// Evidence from a simultaneous check of sampled proposal-bin probabilities.
///
/// Obtain this report from [`verify_proposal_bins`], either on success or from
/// [`ProposalBinsError::Violation`] when a bin exceeds the tolerance. The
/// residual and tolerance are absolute probabilities, not log probabilities.
/// Interpret the statistical bound under the sampling assumptions documented
/// on [`verify_proposal_bins`]; a report alone does not certify a passing check.
///
/// # Examples
///
/// Locate the largest discrepancy even when the histogram check fails:
///
/// ```
/// use markov_chain_monte_carlo::prelude::testing::{
///     ProposalBinsError, verify_proposal_bins,
/// };
///
/// let report = match verify_proposal_bins(&[800, 200], &[0.5, 0.5], 0.01) {
///     Ok(report) | Err(ProposalBinsError::Violation { report, .. }) => report,
///     Err(error) => return Err(error),
/// };
/// assert_eq!(report.worst_bin(), 0);
/// assert!(report.max_residual() > report.tolerance());
/// # Ok::<(), ProposalBinsError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct ProposalBinsReport {
    samples: usize,
    bins: usize,
    worst_bin: usize,
    max_residual: f64,
    tolerance: f64,
    false_positive_rate: f64,
}

impl ProposalBinsReport {
    /// Total number of independent proposal draws in the histogram.
    #[must_use]
    pub const fn samples(&self) -> usize {
        self.samples
    }

    /// Number of tested bins, including zero-probability bins.
    #[must_use]
    pub const fn bins(&self) -> usize {
        self.bins
    }

    /// Zero-based bin with the largest absolute residual; first in a tie.
    #[must_use]
    pub const fn worst_bin(&self) -> usize {
        self.worst_bin
    }

    /// Largest absolute difference between observed and expected bin probability.
    #[must_use]
    pub const fn max_residual(&self) -> f64 {
        self.max_residual
    }

    /// Simultaneous Hoeffding tolerance in probability units.
    #[must_use]
    pub const fn tolerance(&self) -> f64 {
        self.tolerance
    }

    /// Caller-selected upper bound on false rejection for this one histogram.
    ///
    /// This is not a p-value or a probability that the proposal is correct.
    #[must_use]
    pub const fn false_positive_rate(&self) -> f64 {
        self.false_positive_rate
    }
}

/// Invalid histogram inputs or disagreement with the reference bin probabilities.
///
/// Returned by [`verify_proposal_bins`]. [`Self::Violation`] carries a
/// [`ProposalBinsReport`]; [`Self::SupportViolation`] identifies an observation
/// that the reference declares impossible regardless of the statistical bound.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum ProposalBinsError {
    /// At least two counts and equally many expected probabilities are required.
    #[non_exhaustive]
    InvalidBinCount {
        /// Number of supplied counts.
        counts: usize,
        /// Number of supplied probabilities.
        probabilities: usize,
    },
    /// The false-positive rate must be finite and strictly between zero and one.
    #[non_exhaustive]
    InvalidFalsePositiveRate {
        /// Supplied rate.
        rate: f64,
    },
    /// A bin probability must be finite and in `[0, 1]`.
    #[non_exhaustive]
    InvalidProbability {
        /// Zero-based bin index.
        bin: usize,
        /// Supplied probability.
        probability: f64,
    },
    /// Probabilities must sum to one within absolute roundoff allowance `1e-12`.
    #[non_exhaustive]
    InvalidProbabilitySum {
        /// Compensated sum of the supplied probabilities.
        sum: f64,
    },
    /// Every bin count is zero; collect independent proposal draws first.
    NoSamples,
    /// Accumulating bin counts overflowed `usize` before checking the precision limit.
    #[non_exhaustive]
    SampleCountOverflow {
        /// Zero-based index of the first bin whose addition overflows.
        bin: usize,
        /// Total of all preceding bin counts.
        partial_sum: usize,
        /// Count in the overflowing bin.
        count: usize,
    },
    /// The total fits `usize` but exceeds the exact integer range of `f64`.
    #[non_exhaustive]
    SampleCountTooLarge {
        /// Total number of supplied observations.
        samples: usize,
        /// Largest accepted total, `2^53`.
        max_samples: usize,
    },
    /// The simultaneous tolerance is at least one; increase the sample budget.
    #[non_exhaustive]
    UninformativeBound {
        /// Total sample count.
        samples: usize,
        /// Computed probability tolerance.
        tolerance: f64,
    },
    /// An observation occurred in a bin declared impossible by the reference.
    #[non_exhaustive]
    SupportViolation {
        /// Zero-based impossible bin index.
        bin: usize,
        /// Number of observations in that bin.
        count: usize,
    },
    /// At least one bin exceeds the simultaneous tolerance.
    #[non_exhaustive]
    Violation {
        /// Largest discrepancy and the configured statistical bound.
        report: ProposalBinsReport,
    },
}

impl fmt::Display for ProposalBinsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBinCount {
                counts,
                probabilities,
            } => write!(
                f,
                "need at least two matching bins, got {counts} counts and {probabilities} probabilities"
            ),
            Self::InvalidFalsePositiveRate { rate } => {
                write!(
                    f,
                    "false-positive rate must be strictly between zero and one, got {rate}"
                )
            }
            Self::InvalidProbability { bin, probability } => {
                write!(
                    f,
                    "bin {bin} probability must be finite and in [0, 1], got {probability}"
                )
            }
            Self::InvalidProbabilitySum { sum } => {
                write!(
                    f,
                    "bin probabilities must sum to one within 1e-12, got {sum}"
                )
            }
            Self::NoSamples => {
                f.write_str("histogram contains zero samples; collect independent proposal draws")
            }
            Self::SampleCountOverflow {
                bin,
                partial_sum,
                count,
            } => write!(
                f,
                "histogram sample count overflows usize at bin {bin}: {partial_sum} + {count}"
            ),
            Self::SampleCountTooLarge {
                samples,
                max_samples,
            } => write!(
                f,
                "histogram has {samples} samples, exceeding exact count limit {max_samples}"
            ),
            Self::UninformativeBound { samples, tolerance } => write!(
                f,
                "{samples} samples give uninformative bin tolerance {tolerance}; increase sample count"
            ),
            Self::SupportViolation { bin, count } => {
                write!(
                    f,
                    "zero-probability bin {bin} received {count} observations"
                )
            }
            Self::Violation { report } => write!(
                f,
                "bin {} probability residual {} exceeds simultaneous tolerance {}",
                report.worst_bin, report.max_residual, report.tolerance
            ),
        }
    }
}

impl Error for ProposalBinsError {}

/// Compare independent proposal draws with reference probabilities over fixed bins.
///
/// Supply a histogram over a disjoint, exhaustive partition of generated states
/// or proposal deltas. Fix the bins, sample budget, and `false_positive_rate`
/// before drawing samples. Include tails, failed proposals, and other outcomes
/// in explicit bins; do not discard them or renormalize a selected range.
/// Derive expected probabilities independently, for example from a known CDF.
/// They must sum to one within `1e-12` for roundoff and are not renormalized here.
/// The check takes `O(k)` time for `k` bins and uses constant additional space.
///
/// For `n` draws, `k` bins, and rate `alpha`, the probability tolerance is
/// `sqrt(log(2*k/alpha) / (2*n))`. Applying Hoeffding's inequality to each bin
/// indicator and a union bound gives false rejection probability at most
/// `alpha` for this histogram, assuming independent draws with the stated bin
/// probabilities (up to floating-point arithmetic). The bound is conservative,
/// needs no minimum expected count, and is not a p-value. A hit in an exactly
/// zero-probability bin always fails. No observed hits in a rare positive bin
/// need not fail: the report's tolerance describes the available resolution.
///
/// Sample repeatedly from a fixed endpoint with frozen proposal parameters.
/// Correlated MCMC output and online adaptation violate this error model.
/// For [`crate::ProposalMut`] restore state and proposal internals after each
/// draw, including failed checks; for [`crate::DelayedProposal`] classify plans
/// without committing. The caller owns collection and rollback. Counts avoid
/// imposing cloning, equality, scalar-state, or RNG requirements on proposals.
///
/// Passing checks only the chosen bin masses, not the distribution within bins,
/// Hastings ratios, detailed balance, or convergence. Use multiple preselected
/// endpoints/partitions as needed, allocating a total error budget across calls.
/// See [Hoeffding (1963)](https://doi.org/10.1080/01621459.1963.10500830).
///
/// # Errors
///
/// Returns:
///
/// - [`ProposalBinsError::InvalidBinCount`] unless `counts` and `probabilities`
///   have equal lengths of at least two. Corresponding entries describe the same bin.
/// - [`ProposalBinsError::InvalidFalsePositiveRate`] unless `false_positive_rate`
///   is finite and strictly between zero and one.
/// - [`ProposalBinsError::InvalidProbability`] if any probability is nonfinite
///   or outside `[0, 1]`.
/// - [`ProposalBinsError::InvalidProbabilitySum`] if the compensated probability
///   sum differs from one by more than `1e-12`.
/// - [`ProposalBinsError::NoSamples`] if every count is zero.
/// - [`ProposalBinsError::SampleCountOverflow`] if summing counts overflows
///   `usize`, identifying the first overflowing bin and the addition's operands.
/// - [`ProposalBinsError::SampleCountTooLarge`] if the total fits `usize` but
///   exceeds `2^53`. This limit keeps count-to-`f64` conversion exact.
/// - [`ProposalBinsError::SupportViolation`] if any positive count occurs in a
///   zero-probability bin, even when the statistical tolerance would allow it.
/// - [`ProposalBinsError::UninformativeBound`] if the tolerance is at least one
///   and no support violation occurred. Increase the sample budget.
/// - [`ProposalBinsError::Violation`] if any absolute bin residual exceeds the
///   tolerance. The error retains a report with the largest discrepancy.
///
/// Shape, rate, and probability validation precede count validation. Integer
/// overflow is checked before the precision limit; support violations take
/// precedence over statistical bounds once all inputs are valid.
///
/// # Examples
///
/// Sample a uniform random-walk delta from a fixed endpoint, including tails:
///
/// ```
/// use markov_chain_monte_carlo::prelude::testing::{
///     ProposalBinsError, verify_proposal_bins,
/// };
/// use rand::{RngExt, SeedableRng, rngs::StdRng};
///
/// let mut rng = StdRng::seed_from_u64(42);
/// let mut counts = [0; 3];
/// for _ in 0..10_000 {
///     let delta = rng.random_range(-1.0..1.0);
///     let bin = if !(-1.0..1.0).contains(&delta) { 2 }
///         else { usize::from(delta >= 0.0) };
///     counts[bin] += 1;
/// }
/// let report = verify_proposal_bins(&counts, &[0.5, 0.5, 0.0], 1e-6)?;
/// assert!(report.max_residual() <= report.tolerance());
/// # Ok::<(), ProposalBinsError>(())
/// ```
pub fn verify_proposal_bins(
    counts: &[usize],
    probabilities: &[f64],
    false_positive_rate: f64,
) -> Result<ProposalBinsReport, ProposalBinsError> {
    if counts.len() < 2 || counts.len() != probabilities.len() {
        return Err(ProposalBinsError::InvalidBinCount {
            counts: counts.len(),
            probabilities: probabilities.len(),
        });
    }
    if !false_positive_rate.is_finite() || false_positive_rate <= 0.0 || false_positive_rate >= 1.0
    {
        return Err(ProposalBinsError::InvalidFalsePositiveRate {
            rate: false_positive_rate,
        });
    }
    for (bin, &probability) in probabilities.iter().enumerate() {
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err(ProposalBinsError::InvalidProbability { bin, probability });
        }
    }
    let sum = compensated_sum(probabilities.iter().copied());
    if (sum - 1.0).abs() > 1e-12 {
        return Err(ProposalBinsError::InvalidProbabilitySum { sum });
    }
    let samples = counts
        .iter()
        .enumerate()
        .try_fold(0_usize, |partial_sum, (bin, &count)| {
            partial_sum
                .checked_add(count)
                .ok_or(ProposalBinsError::SampleCountOverflow {
                    bin,
                    partial_sum,
                    count,
                })
        })?;
    if samples == 0 {
        return Err(ProposalBinsError::NoSamples);
    }
    let max_samples = usize::try_from(1_u64 << 53).unwrap_or(usize::MAX);
    if samples > max_samples {
        return Err(ProposalBinsError::SampleCountTooLarge {
            samples,
            max_samples,
        });
    }
    let n = count_as_f64(samples);
    // Log-space evaluation avoids overflow for subnormal alpha.
    let log_factor =
        std::f64::consts::LN_2 + count_as_f64(counts.len()).ln() - false_positive_rate.ln();
    let tolerance = (log_factor / (2.0 * n)).sqrt();
    let mut report = ProposalBinsReport {
        samples,
        bins: counts.len(),
        worst_bin: 0,
        max_residual: 0.0,
        tolerance,
        false_positive_rate,
    };
    for (bin, (&count, &probability)) in counts.iter().zip(probabilities).enumerate() {
        if probability == 0.0 && count != 0 {
            return Err(ProposalBinsError::SupportViolation { bin, count });
        }
        let residual = (count_as_f64(count) / n - probability).abs();
        if residual > report.max_residual {
            report.max_residual = residual;
            report.worst_bin = bin;
        }
    }
    if tolerance >= 1.0 {
        return Err(ProposalBinsError::UninformativeBound { samples, tolerance });
    }
    if report.max_residual > tolerance {
        return Err(ProposalBinsError::Violation { report });
    }
    Ok(report)
}
