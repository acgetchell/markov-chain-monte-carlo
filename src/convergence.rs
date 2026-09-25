//! Classical split R-hat for comparable, finite scalar chains.
//!
//! [`SplitRhat`] compares within-half and between-half variation, while
//! [`SplitRhatError`] keeps invalid and unresolved inputs explicit. Its types are
//! re-exported at the crate root and in [`crate::prelude`].

use std::{error::Error, fmt};

use crate::numerics::{compensated_sum, count_as_f64};

/// Invalid or numerically unresolved input to [`SplitRhat::estimate`].
///
/// Every `chain_index` is a zero-based position in the supplied slice, not a
/// [`crate::ChainId`]. Keep the identifiers alongside those slices when reporting
/// failures for a [`crate::Trace`]. A `half` of zero selects the first half and
/// one selects the last half; sample indices refer to the original unsplit chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SplitRhatError {
    /// At least two original chains are required, before splitting.
    #[non_exhaustive]
    InsufficientChains {
        /// Number of original chains.
        count: usize,
    },
    /// Every original chain needs at least four samples.
    #[non_exhaustive]
    InsufficientSamples {
        /// Index of the short chain in the input slice.
        chain_index: usize,
        /// Number of samples in that chain.
        count: usize,
    },
    /// Chains must have equal lengths; no implicit trimming is performed.
    #[non_exhaustive]
    UnequalLengths {
        /// Index of the chain with a different length.
        chain_index: usize,
        /// Length of the first chain.
        expected: usize,
        /// Length of this chain.
        actual: usize,
    },
    /// A sample is NaN or infinite, including an otherwise unused middle draw.
    #[non_exhaustive]
    NonFiniteSample {
        /// Index of the original chain.
        chain_index: usize,
        /// Index within that original chain.
        sample_index: usize,
    },
    /// A split chain has no variation; a stuck chain must not imply convergence.
    #[non_exhaustive]
    ConstantSplitChain {
        /// Index of the original chain.
        chain_index: usize,
        /// Zero for the first half, one for the second.
        half: usize,
    },
    /// A nonconstant half's variance is not positive and finite after scaling.
    ///
    /// Unlike [`Self::ConstantSplitChain`], the original samples do vary.
    /// Extreme scale differences can make that variation numerically unresolved.
    #[non_exhaustive]
    UnresolvedVariance {
        /// Index of the original chain containing the unresolved half.
        chain_index: usize,
        /// Zero for the first half, one for the second.
        half: usize,
    },
    /// The final R-hat estimate is not representable as a positive finite value.
    ///
    /// Every half passed the individual variance checks before this failure.
    NumericalFailure,
}

impl fmt::Display for SplitRhatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InsufficientChains { count } => {
                write!(
                    f,
                    "split R-hat needs at least two original chains, got {count}"
                )
            }
            Self::InsufficientSamples { chain_index, count } => {
                write!(
                    f,
                    "chain {chain_index} needs at least four samples, got {count}"
                )
            }
            Self::UnequalLengths {
                chain_index,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "chain {chain_index} has {actual} samples; expected {expected}"
                )
            }
            Self::NonFiniteSample {
                chain_index,
                sample_index,
            } => {
                write!(f, "chain {chain_index} sample {sample_index} is not finite")
            }
            Self::ConstantSplitChain { chain_index, half } => {
                write!(f, "chain {chain_index} half {half} is constant")
            }
            Self::UnresolvedVariance { chain_index, half } => write!(
                f,
                "chain {chain_index} half {half} has varying samples but its variance is unresolved after scaling"
            ),
            Self::NumericalFailure => {
                f.write_str("final split R-hat estimate is not positive and finite")
            }
        }
    }
}

impl Error for SplitRhatError {}

/// Classical split potential scale reduction statistic for one scalar observable.
///
/// Use [`Self::estimate`] to compare chains for location disagreement or drift.
/// The result owns a scalar estimate and count metadata; it does not borrow or
/// retain the input observations. Inspect it with [`Self::value`], and use
/// [`crate::IntegratedAutocorrelationTime::effective_sample_size`] separately
/// for single-chain mean precision. Both diagnostics are available without
/// enabling any Cargo feature.
///
/// Supply at least two independently initialized original chains targeting the
/// same distribution, with the same observable, units, recording interval, and
/// warmup policy. Discard warmup first, preserve time order, and retain rejected
/// steps and no-proposal self-loops. Scalar slices cannot verify those assumptions.
///
/// Each chain is split into first and last halves of length `n = floor(N/2)`;
/// the middle draw of an odd-length chain is omitted. With `m = 2 * chains`,
/// let `W` be the mean unbiased within-half sample variance, and `B = n` times
/// the unbiased sample variance of the half means. Return
/// `sqrt(((n-1)/n * W + B/n) / W)`. Values below one are retained.
/// See the [classical split R-hat definition](https://mc-stan.org/docs/2_29/reference-manual/notation-for-samples-chains-and-draws.html).
///
/// This raw-moment estimator assumes finite marginal mean and variance. It is
/// **not rank-normalized or folded R-hat** and can miss scale or tail differences.
/// A value near one is not proof of convergence, adequate length, or exploration
/// of all modes. Use longer runs, dispersed starts, ESS, and trace inspection;
/// consider rank-normalized diagnostics for heavy tails or scale differences.
///
/// # Examples
///
/// Select the same named observable from each original [`crate::Trace`] chain,
/// then borrow the collected columns for estimation. The rows below are short
/// synthetic production records for demonstrating selection, not evidence of
/// adequate sampling. Real inputs must satisfy the comparability and warmup
/// requirements above; selection preserves insertion order and checks neither.
///
/// ```
/// use markov_chain_monte_carlo::prelude::{
///     ChainId, SplitRhat, Trace, TraceError, TraceRecord, TraceStepOutcome,
/// };
///
/// let chain_ids = [ChainId::new(10), ChainId::new(20)];
/// let mut trace = Trace::new(["energy"])?;
/// for (id, energies) in chain_ids.into_iter().zip([
///     [1.0, 2.0, 3.0, 4.0],
///     [3.0, 4.0, 5.0, 6.0],
/// ]) {
///     for (index, energy) in energies.into_iter().enumerate() {
///         trace.push(TraceRecord::new(
///             id, index + 1, TraceStepOutcome::accepted(), -energy, vec![energy],
///         ))?;
///     }
/// }
/// let columns: Result<Vec<Vec<f64>>, TraceError> = chain_ids.iter().map(|&id| {
///     trace.observable_values(id, "energy").map(|values| values.copied().collect())
/// }).collect();
/// let columns = columns?;
/// let chains: Vec<&[f64]> = columns.iter().map(Vec::as_slice).collect();
/// let result = SplitRhat::estimate(&chains);
/// assert!(matches!(result, Ok(rhat)
///     if rhat.chain_count() == 2 && (rhat.value() - (35.0_f64 / 6.0).sqrt()).abs() < 1e-14));
/// # Ok::<(), TraceError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct SplitRhat {
    /// Positive finite result of the classical split variance ratio.
    estimate: f64,
    /// Number of original chains, before constructing twice as many halves.
    chain_count: usize,
    /// Original length, retained to report whether splitting omitted a draw.
    samples_per_chain: usize,
}

impl SplitRhat {
    /// Estimate classical split R-hat from borrowed, equally long scalar chains.
    ///
    /// Supply at least two original chains, each with at least four draws,
    /// using the observable and sampling prerequisites described on [`Self`].
    /// Each chain contributes its first and last `floor(N/2)` draws; an odd
    /// middle draw is omitted from moments but still checked for finiteness.
    /// These minimum counts make the formula defined, not statistically reliable.
    /// The returned summary is independent of the input buffers, which are unchanged.
    ///
    /// Time is `O(chains * samples)` and auxiliary memory is `O(chains)`.
    /// Common shifted scaling and compensated sums avoid overflow; centering
    /// within each half preserves small local variation at large offsets.
    /// Extremely different scales can still underflow within-half variances.
    ///
    /// # Examples
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::{SplitRhat, SplitRhatError};
    ///
    /// let left = [0.0, 2.0, 0.0, 2.0];
    /// let right = [2.0, 4.0, 2.0, 4.0];
    /// let rhat = SplitRhat::estimate(&[&left, &right])?;
    /// assert!((rhat.value() - (7.0_f64 / 6.0).sqrt()).abs() < 1e-14);
    /// assert_eq!(rhat.chain_count(), 2);
    /// assert_eq!(rhat.samples_per_split_chain(), 2);
    /// # Ok::<(), SplitRhatError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - [`SplitRhatError::InsufficientChains`]: fewer than two original chains.
    /// - [`SplitRhatError::InsufficientSamples`]: a chain has fewer than four draws.
    /// - [`SplitRhatError::UnequalLengths`]: a chain differs in length from the first.
    /// - [`SplitRhatError::NonFiniteSample`]: any supplied draw is NaN or infinite,
    ///   including a middle draw omitted by splitting.
    /// - [`SplitRhatError::ConstantSplitChain`]: all draws in a half are exactly
    ///   equal as represented by `f64`, even if other halves vary.
    /// - [`SplitRhatError::UnresolvedVariance`]: a nonconstant half's computed
    ///   variance is nonpositive or nonfinite, including underflow after common
    ///   scaling. The error identifies the original chain and half.
    /// - [`SplitRhatError::NumericalFailure`]: the final estimate is nonpositive
    ///   or nonfinite after all individual halves passed their variance checks.
    ///
    /// Chain count is checked first. Then lengths are checked in input order,
    /// with minimum length before equality for each chain, followed by finiteness
    /// of every original sample. Moment calculation then checks each half for
    /// constancy and numerical failure. Chain indices in errors are input-slice
    /// positions; they do not identify [`crate::ChainId`] values.
    pub fn estimate(chains: &[&[f64]]) -> Result<Self, SplitRhatError> {
        let input = SplitRhatInput::parse(chains)?;
        let sample_count = input.samples_per_chain;
        let half_length = sample_count / 2;
        let halves = input
            .chains
            .iter()
            .flat_map(|chain| [&chain[..half_length], &chain[sample_count - half_length..]]);
        let origin = input.chains[0][0];
        // Use a common factor for every half; separate per-chain scaling would
        // erase between-chain differences and change the diagnostic.
        let (minimum, maximum) = halves
            .clone()
            .flatten()
            .fold((origin, origin), |(low, high), &value| {
                (low.min(value), high.max(value))
            });
        let halve = !(maximum - minimum).is_finite();
        let difference = |value: f64, anchor: f64| {
            if halve {
                anchor.mul_add(-0.5, value * 0.5)
            } else {
                value - anchor
            }
        };
        let scale = halves
            .clone()
            .flatten()
            .map(|&value| difference(value, origin).abs())
            .fold(0.0_f64, f64::max);
        let n = count_as_f64(half_length);
        let mut moments = Vec::with_capacity(2 * input.chains.len());
        for (index, half) in halves.enumerate() {
            let anchor = half[0];
            #[expect(
                clippy::float_cmp,
                reason = "constant means exactly equal represented observations, including signed zero"
            )]
            let constant = half.iter().all(|&value| value == anchor);
            if constant {
                return Err(SplitRhatError::ConstantSplitChain {
                    chain_index: index / 2,
                    half: index % 2,
                });
            }
            let delta = |value| difference(value, anchor) / scale;
            let mean_delta = compensated_sum(half.iter().map(|&value| delta(value))) / n;
            let variance = compensated_sum(half.iter().map(|&value| {
                let residual = delta(value) - mean_delta;
                residual * residual
            })) / (n - 1.0);
            if !variance.is_finite() || variance <= 0.0 {
                return Err(SplitRhatError::UnresolvedVariance {
                    chain_index: index / 2,
                    half: index % 2,
                });
            }
            moments.push((difference(anchor, origin) / scale + mean_delta, variance));
        }
        let m = count_as_f64(moments.len());
        let mean = compensated_sum(moments.iter().map(|&(mean, _)| mean)) / m;
        let between_over_n = compensated_sum(
            moments
                .iter()
                .map(|&(half_mean, _)| (half_mean - mean).powi(2)),
        ) / (m - 1.0);
        let within = compensated_sum(moments.into_iter().map(|(_, variance)| variance)) / m;
        // Taking square roots before division avoids overflowing B/W when
        // R-hat itself is still representable.
        let estimate = ((n - 1.0) / n).mul_add(within, between_over_n).sqrt() / within.sqrt();
        if !estimate.is_finite() || estimate <= 0.0 {
            return Err(SplitRhatError::NumericalFailure);
        }
        Ok(Self {
            estimate,
            chain_count: input.chains.len(),
            samples_per_chain: sample_count,
        })
    }

    /// Estimated dimensionless classical split R-hat, without a floor at one.
    ///
    /// Always positive and finite after successful estimation. A value near
    /// one means the estimated variances agree; it does not prove convergence.
    #[must_use]
    pub const fn value(self) -> f64 {
        self.estimate
    }

    /// Number of original chains, before splitting.
    #[must_use]
    pub const fn chain_count(self) -> usize {
        self.chain_count
    }

    /// Supplied sample count per original chain, including an omitted middle draw.
    #[must_use]
    pub const fn samples_per_chain(self) -> usize {
        self.samples_per_chain
    }

    /// Used sample count per split chain; half the original length, rounded down.
    #[must_use]
    pub const fn samples_per_split_chain(self) -> usize {
        self.samples_per_chain / 2
    }
}

/// Borrowed chains with the shape and finite-input preconditions for R-hat moments.
///
/// Checking two or more chains and equal lengths of at least four makes the
/// estimator's indexing, half slicing, and `n - 1`/`m - 1` denominators valid.
/// The shared borrow preserves these facts throughout estimation. It does not
/// establish independence, stationarity, or a common target. Constant halves
/// and numerical failures are classified later during moment calculation.
struct SplitRhatInput<'a> {
    chains: &'a [&'a [f64]],
    samples_per_chain: usize,
}

impl<'a> SplitRhatInput<'a> {
    /// Check lengths before values, with shortness before unequal length within
    /// each original chain. Check all samples, including omitted middle draws.
    fn parse(chains: &'a [&'a [f64]]) -> Result<Self, SplitRhatError> {
        if chains.len() < 2 {
            return Err(SplitRhatError::InsufficientChains {
                count: chains.len(),
            });
        }
        for (chain_index, chain) in chains.iter().enumerate() {
            if chain.len() < 4 {
                return Err(SplitRhatError::InsufficientSamples {
                    chain_index,
                    count: chain.len(),
                });
            }
            if chain.len() != chains[0].len() {
                return Err(SplitRhatError::UnequalLengths {
                    chain_index,
                    expected: chains[0].len(),
                    actual: chain.len(),
                });
            }
        }
        for (chain_index, chain) in chains.iter().enumerate() {
            if let Some(sample_index) = chain.iter().position(|value| !value.is_finite()) {
                return Err(SplitRhatError::NonFiniteSample {
                    chain_index,
                    sample_index,
                });
            }
        }
        Ok(Self {
            chains,
            samples_per_chain: chains[0].len(),
        })
    }
}
