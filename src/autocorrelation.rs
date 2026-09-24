//! Autocorrelation diagnostics for finite, regularly sampled scalar traces.

use std::{error::Error, fmt};

/// Errors from scalar autocorrelation diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AutocorrelationError {
    /// At least two samples are needed to estimate a variance.
    #[non_exhaustive]
    InsufficientSamples {
        /// Number of supplied samples.
        count: usize,
    },
    /// The requested inclusive maximum lag is not smaller than the sample count.
    #[non_exhaustive]
    InvalidMaxLag {
        /// Requested inclusive maximum lag.
        max_lag: usize,
        /// Number of supplied samples.
        sample_count: usize,
    },
    /// A sample is NaN or infinite.
    #[non_exhaustive]
    NonFiniteSample {
        /// Zero-based index in the supplied slice.
        index: usize,
    },
    /// All samples are equal, so normalized autocorrelation is undefined.
    ConstantTrace,
    /// No nonpositive pair was found within the available complete lag pairs.
    ///
    /// Increase the maximum lag or collect a longer trace. This error never
    /// substitutes a sum truncated only by the caller's lag limit.
    #[non_exhaustive]
    TruncationNotFound {
        /// Largest available lag, including any unpaired final lag.
        max_lag: usize,
    },
    /// The estimated integrated time is zero or negative.
    ///
    /// This can occur for short or strongly anticorrelated traces. No floor
    /// or absolute-value correction is applied.
    NonPositiveTime,
}

impl fmt::Display for AutocorrelationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InsufficientSamples { count } => {
                write!(f, "autocorrelation needs at least two samples, got {count}")
            }
            Self::InvalidMaxLag {
                max_lag,
                sample_count,
            } => {
                write!(
                    f,
                    "maximum lag {max_lag} must be less than sample count {sample_count}"
                )
            }
            Self::NonFiniteSample { index } => write!(f, "sample at index {index} is not finite"),
            Self::ConstantTrace => write!(f, "autocorrelation is undefined for a constant trace"),
            Self::TruncationNotFound { max_lag } => {
                write!(
                    f,
                    "no autocorrelation truncation pair found through lag {max_lag}"
                )
            }
            Self::NonPositiveTime => write!(
                f,
                "estimated integrated autocorrelation time is not positive"
            ),
        }
    }
}

impl Error for AutocorrelationError {}

/// Estimated autocorrelation function (ACF) of one scalar observable.
///
/// Use [`Self::estimate`] for an in-memory observable slice or a numeric column
/// loaded from CSV. Analyze each chain separately, in time order, after burn-in,
/// with a fixed interval between samples. Keep rejected steps and no-proposal
/// self-loops: dropping repeated states changes the sampled process. These
/// temporal assumptions cannot be checked from a scalar slice.
///
/// The result owns its computed estimates independently of the input samples.
/// The input buffer can be changed or dropped after estimation; [`Self::values`]
/// borrows the estimates from this result.
///
/// The estimator uses the sample mean and the **biased** autocovariance:
/// `rho[k] = sum((x[i] - mean) * (x[i+k] - mean)) / sum((x[i] - mean)^2)`.
/// Both sums use the same implicit divisor `N`, not `N-k`. Lag zero is exactly
/// one. Large-lag estimates are noisy; summing every lag is not a useful
/// integrated-time estimate.
///
/// # Examples
///
/// Select one observable by name with [`crate::Trace::observable_values`],
/// retaining every production row for one chain. This example's rows are
/// inserted in time order at a fixed recording interval. The short synthetic
/// trace demonstrates the workflow, not adequate length or convergence.
///
/// ```
/// use markov_chain_monte_carlo::prelude::{
///     Autocorrelation, ChainId, Trace, TraceError, TraceRecord, TraceStepOutcome,
/// };
///
/// let chain_id = ChainId::new(0);
/// let mut trace = Trace::new(["energy"])?;
/// for (index, energy) in [1.0, 2.0, 3.0, 4.0].into_iter().enumerate() {
///     trace.push(TraceRecord::new(
///         chain_id, index + 1, TraceStepOutcome::accepted(), -energy, vec![energy],
///     ))?;
/// }
/// let energy: Vec<_> = trace.observable_values(chain_id, "energy")?.copied().collect();
/// let max_lag = energy.len().saturating_sub(1).min(1_000);
/// let result = Autocorrelation::estimate(&energy, max_lag)
///     .and_then(|acf| acf.integrated_time());
/// assert!(matches!(result, Ok(time)
///     if (time.estimate() - 1.5).abs() < 1e-14
///         && time.window() == 1
///         && time.sample_count() == 4), "{result:?}");
/// # Ok::<(), TraceError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct Autocorrelation {
    values: Vec<f64>,
    sample_count: usize,
}

impl Autocorrelation {
    /// Estimate lags `0..=max_lag` by direct summation.
    ///
    /// Cost is `O(N * (max_lag + 1))` time and `O(N + max_lag)` memory. No FFT
    /// dependency is required. Choose a bounded lag budget for long traces.
    /// Shifted, scaled centering avoids overflow and preserves small variations
    /// around large offsets; compensated sums reduce accumulation error.
    /// Rounding and loss of tiny contributions at extreme mixed scales remain.
    ///
    /// # Examples
    ///
    /// ```
    /// use markov_chain_monte_carlo::{Autocorrelation, AutocorrelationError};
    ///
    /// let acf = {
    ///     let samples = vec![1.0, 2.0, 3.0, 4.0];
    ///     Autocorrelation::estimate(&samples, 3)?
    /// }; // The sample buffer has been dropped; the estimates remain available.
    /// assert_eq!(acf.sample_count(), 4);
    /// assert_eq!(acf.values()[0], 1.0);
    /// assert!((acf.values()[1] - 0.25).abs() < 1e-14);
    /// # Ok::<(), AutocorrelationError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`AutocorrelationError::InsufficientSamples`] for fewer than two
    /// samples, [`AutocorrelationError::InvalidMaxLag`] for `max_lag >= N`,
    /// [`AutocorrelationError::NonFiniteSample`] for NaN or infinity, or
    /// [`AutocorrelationError::ConstantTrace`] for zero variance, in that order.
    pub fn estimate(samples: &[f64], max_lag: usize) -> Result<Self, AutocorrelationError> {
        let sample_count = samples.len();
        if sample_count < 2 {
            return Err(AutocorrelationError::InsufficientSamples {
                count: sample_count,
            });
        }
        if max_lag >= sample_count {
            return Err(AutocorrelationError::InvalidMaxLag {
                max_lag,
                sample_count,
            });
        }
        if let Some(index) = samples.iter().position(|value| !value.is_finite()) {
            return Err(AutocorrelationError::NonFiniteSample { index });
        }
        let origin = samples[0];
        let mut centered: Vec<_> = samples.iter().map(|value| value - origin).collect();
        if centered.iter().any(|value| !value.is_finite()) {
            // Opposite-sign extreme values can overflow subtraction. Halving
            // before subtracting is safe here; subsequent scaling cancels it.
            for (delta, value) in centered.iter_mut().zip(samples) {
                *delta = origin.mul_add(-0.5, value * 0.5);
            }
        }
        let scale = centered
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);
        if scale == 0.0 {
            return Err(AutocorrelationError::ConstantTrace);
        }
        for value in &mut centered {
            *value /= scale;
        }
        let mean = compensated_sum(centered.iter().copied()) / count_as_f64(sample_count);
        for value in &mut centered {
            *value -= mean;
        }
        let variance_sum = compensated_sum(centered.iter().map(|value| value * value));
        let mut values = Vec::with_capacity(max_lag + 1);
        values.push(1.0);
        // Independent lag sums can run together without reordering the terms
        // within any sum. Keep the final incomplete batch on the scalar path.
        let batched_lags = max_lag / 4 * 4;
        for first_lag in (1..=batched_lags).step_by(4) {
            values.extend(lag_covariances(&centered, first_lag).map(|sum| sum / variance_sum));
        }
        for lag in batched_lags + 1..=max_lag {
            let covariance_sum =
                compensated_sum(centered.iter().zip(&centered[lag..]).map(|(a, b)| a * b));
            values.push(covariance_sum / variance_sum);
        }
        Ok(Self {
            values,
            sample_count,
        })
    }

    /// Borrow the ACF, indexed by lag, including lag zero.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Number of input samples, including repeated states.
    #[must_use]
    pub const fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// Estimate integrated autocorrelation time using Geyer's initial monotone sequence.
    ///
    /// Form pairs `P[j] = rho[2*j] + rho[2*j+1]`, starting at lag zero. Stop
    /// **before** the first nonpositive pair, then replace each retained pair
    /// with the minimum of itself and all earlier pairs. Return
    /// `tau = -1 + 2 * sum(P)`. An unpaired final lag is ignored. The ACF itself
    /// is unchanged. See [Geyer's method and assumptions](https://www.stat.umn.edu/geyer/mcmc/library/mcmc/html/initseq.html).
    ///
    /// This uses the convention `tau = 1 + 2 * sum(rho[k], k >= 1)`, in
    /// **recorded-sample intervals**, for which independent samples have true
    /// `tau = 1`. Positive estimates below one are retained for anticorrelation.
    /// Multiply by the recording interval to express time in transition steps.
    /// For thinned traces, this only changes the units of the retained-series
    /// estimate; it does not reconstruct the unthinned chain's integrated time.
    ///
    /// The sequence justification assumes a stationary, reversible chain with
    /// finite variance and summable autocorrelations. A finite estimate does not
    /// establish stationarity, convergence, or adequate length. Compare results
    /// across longer runs and lag budgets; short traces can severely underestimate
    /// slow modes even when a truncation pair is found.
    /// Pair signs and the final positivity check use rounded `f64` estimates;
    /// results near zero can be sensitive to roundoff or underflow.
    ///
    /// # Examples
    ///
    /// For integrated time alone, chain the two fallible stages:
    ///
    /// ```
    /// use markov_chain_monte_carlo::{Autocorrelation, AutocorrelationError};
    ///
    /// let time = Autocorrelation::estimate(&[1.0, 2.0, 3.0, 4.0], 3)?.integrated_time()?;
    /// assert!((time.estimate() - 1.5).abs() < 1e-14);
    /// assert_eq!(time.window(), 1);
    /// // This four-sample arithmetic example is not evidence of convergence.
    /// # Ok::<(), AutocorrelationError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`AutocorrelationError::TruncationNotFound`] if no complete
    /// nonpositive pair is available, including ACFs containing only lag zero.
    /// Returns [`AutocorrelationError::NonPositiveTime`] if the resulting time
    /// is zero or negative. Neither condition is silently clamped or accepted.
    pub fn integrated_time(&self) -> Result<IntegratedAutocorrelationTime, AutocorrelationError> {
        let pairs = self.values.as_chunks::<2>().0;
        let pair_count = pairs
            .iter()
            .position(|&[even, odd]| even + odd <= 0.0)
            .ok_or(AutocorrelationError::TruncationNotFound {
                max_lag: self.values.len() - 1,
            })?;
        let sum = compensated_sum(pairs[..pair_count].iter().scan(
            f64::INFINITY,
            |minimum_pair, &[even, odd]| {
                *minimum_pair = minimum_pair.min(even + odd);
                Some(*minimum_pair)
            },
        ));
        let estimate = sum.mul_add(2.0, -1.0);
        if estimate <= 0.0 {
            return Err(AutocorrelationError::NonPositiveTime);
        }
        Ok(IntegratedAutocorrelationTime {
            estimate,
            window: 2 * pair_count - 1,
            sample_count: self.sample_count,
        })
    }
}

/// A positive initial-monotone-sequence estimate and its retained lag window.
///
/// Constructed only by [`Autocorrelation::integrated_time`]. This is an
/// estimate for one observable, not a convergence certificate for a chain.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct IntegratedAutocorrelationTime {
    estimate: f64,
    window: usize,
    sample_count: usize,
}

impl IntegratedAutocorrelationTime {
    /// Estimated time in recorded-sample intervals; independent samples have true time one.
    #[must_use]
    pub const fn estimate(self) -> f64 {
        self.estimate
    }

    /// Largest retained lag (odd); the following complete pair caused truncation.
    #[must_use]
    pub const fn window(self) -> usize {
        self.window
    }

    /// Number of input samples used to estimate the ACF.
    #[must_use]
    pub const fn sample_count(self) -> usize {
        self.sample_count
    }
}

/// Reduce accumulation error in the ACF and integrated-time formulas using Kahan summation.
///
/// Callers supply scaled finite terms so summation stays finite; compensation
/// helps limit rounding error in the signed covariance sums.
fn compensated_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut accumulator = CompensatedSum::default();
    for value in values {
        accumulator.add(value);
    }
    accumulator.sum
}

/// Independent Kahan state for one ordered sum of scaled finite terms.
#[derive(Clone, Copy, Default)]
struct CompensatedSum {
    sum: f64,
    correction: f64,
}

impl CompensatedSum {
    fn add(&mut self, value: f64) {
        let adjusted = value - self.correction;
        let next = self.sum + adjusted;
        self.correction = (next - self.sum) - adjusted;
        self.sum = next;
    }
}

/// Sum four successive lag covariances in their original per-lag order.
///
/// The estimator supplies centered finite samples and `first_lag + 3 < N`.
/// The common prefix advances four independent sums; at most three trailing
/// terms per sum complete the unequal lag lengths without padding or reassociation.
fn lag_covariances(centered: &[f64], first_lag: usize) -> [f64; 4] {
    let mut sums = [CompensatedSum::default(); 4];
    let right = &centered[first_lag..];
    let common_len = right.len() - 3;
    for (&left, window) in centered.iter().zip(right.array_windows::<4>()) {
        for (sum, &right) in sums.iter_mut().zip(window) {
            sum.add(left * right);
        }
    }
    for (offset, sum) in sums.iter_mut().enumerate() {
        let lag = first_lag + offset;
        for (&left, &right) in centered[common_len..]
            .iter()
            .zip(&centered[common_len + lag..])
        {
            sum.add(left * right);
        }
    }
    sums.map(|sum| sum.sum)
}

#[expect(
    clippy::cast_precision_loss,
    reason = "allocated sample counts fit f64 at practical trace sizes"
)]
/// Convert an allocated trace length for the sample-mean denominator.
///
/// Keep the integer-to-float boundary in one place; practical trace lengths
/// stay within the exact integer range of `f64`.
const fn count_as_f64(count: usize) -> f64 {
    count as f64
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::{Autocorrelation, compensated_sum, lag_covariances};

    #[test]
    fn batched_covariances_preserve_scalar_order_and_unequal_tails() {
        let half_ulp = f64::EPSILON / 2.0;
        let samples = [
            1.0, half_ulp, -1.0, half_ulp, 0.25, -0.5, -half_ulp, 0.0, 0.75, -1.0, 1.0, half_ulp,
        ];
        // Exercise a one-term common prefix as well as longer prefixes. Each
        // batch has tails of three, two, one and zero terms; adding padding
        // or reassociating cancellation-sensitive terms can change the bits.
        for count in 5..=samples.len() {
            let input = &samples[..count];
            for first_lag in 1..count - 3 {
                for (offset, actual) in lag_covariances(input, first_lag).into_iter().enumerate() {
                    let lag = first_lag + offset;
                    let scalar = compensated_sum(
                        input[..count - lag]
                            .iter()
                            .zip(&input[lag..])
                            .map(|(left, right)| left * right),
                    );
                    assert_eq!(
                        actual.to_bits(),
                        scalar.to_bits(),
                        "count={count}, first_lag={first_lag}, lag={lag}"
                    );
                }
            }
        }
    }

    #[test]
    fn compensated_sum_accepts_iterables_and_preserves_small_terms() {
        // An array is IntoIterator but not Iterator. Two half-ULP terms
        // would both disappear in a naive left-to-right sum starting at 1.
        let half_ulp = f64::EPSILON / 2.0;
        let sum = compensated_sum([1.0, half_ulp, half_ulp, -1.0]);
        assert_eq!(sum.to_bits(), f64::EPSILON.to_bits());
    }

    #[test]
    fn monotone_pairs_and_ignored_unpaired_tail() {
        // Raw pairs: 1.5, 0.5, 0.75, 0.875, -0.25, 0.5. Both rising
        // pairs must use the cumulative minimum 0.5, not just their neighbor.
        // The negative pair stops the sequence despite later positive values.
        let values = [
            1.0, 0.5, 0.25, 0.25, 0.5, 0.25, 0.5, 0.375, -0.5, 0.25, 0.25, 0.25, 0.75,
        ];
        let acf = Autocorrelation {
            values: values.to_vec(),
            sample_count: 100,
        };
        let time = acf.integrated_time().unwrap();
        assert_relative_eq!(time.estimate(), 5.0);
        assert_eq!(time.window(), 7);
        assert_eq!(time.sample_count(), 100);
        assert_eq!(acf.values(), values);
        assert_eq!(acf.integrated_time().unwrap(), time);
    }

    #[test]
    fn exactly_zero_pair_stops_before_later_positive_pairs() {
        let acf = Autocorrelation {
            values: vec![1.0, 0.5, 0.25, -0.25, 0.25, 0.25],
            sample_count: 100,
        };
        let time = acf.integrated_time().unwrap();
        assert_relative_eq!(time.estimate(), 2.0);
        assert_eq!(time.window(), 1);
        assert_eq!(time.sample_count(), 100);
    }
}
